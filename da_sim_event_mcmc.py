import time
import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys

import numpy as np
import matplotlib.pylab as plt
import scipy.optimize as opt
import sklearn.cluster as cluster
import multiprocessing

import emcee
import corner

from track_fitting import ParticleAndPointDeposition, build_sim
from raw_viewer import raw_h5_file

if __name__ == '__main__':
    run_number, event_num = sys.argv[1:]
    particle_type = '4He'
    run_number = int(run_number)
    event_num = int(event_num)
    #folder = '/mnt/analysis/e21072/gastest_h5_files/'
    # folder = '../../shared/Run_Data/'
    folder = '/egr/research-tpc/dopferjo/'
    experiment = 'e24joe'

    #MCMC priors
    class GaussianVar:
        def __init__(self, mu, sigma):
            self.mu, self.sigma  = mu, sigma

        def log_likelihood(self, val):
            return -np.log(np.sqrt(2*np.pi*self.sigma**2)) - (val - self.mu)**2/2/self.sigma**2

    E_from_ic = build_sim.get_energy_from_ic(experiment, run_number, event_num)
    E_from_ic_simga = build_sim.get_detector_E_sigma(experiment, run_number, E_from_ic)
    E_prior = GaussianVar(E_from_ic, E_from_ic_simga)
    rho0 = build_sim.get_gas_density(experiment, run_number)
    density_scale_prior = GaussianVar(1, 0.1)#TODO: decide on density range

    source_sim = build_sim.create_multi_particle_event(experiment, run_number, event_num, ['4He','4He'])
    source_sim.sims[0].initial_energy = 6
    source_sim.sims[1].initial_energy = 6
    source_sim.sims[0].initial_point = (0,0,20)
    source_sim.sims[1].initial_point = (0,0,20)
    source_sim.sims[0].theta = 0.785398
    source_sim.sims[1].theta = 2.35619
    source_sim.sims[0].phi = 3.92699
    source_sim.sims[1].phi = 0.785398
    source_sim.sims[0].sigma_xy = source_sim.sims[1].sigma_xy  = 2.0
    source_sim.sims[0].sigma_z = source_sim.sims[1].sigma_z  = 3.0
    source_sim.simulate_event()

    h5file = build_sim.get_rawh5_object(experiment, run_number)
    #set zmax to length of trimmed traces
    temp_sim = build_sim.create_multi_particle_event(experiment, run_number, event_num, ['4He','4He'])
    zmax = temp_sim.num_trace_bins*temp_sim.zscale
    
    zmax = 500
    
    x_real, y_real, z_real, e_real = source_sim.get_xyze(threshold=h5file.length_counts_threshold, traces=source_sim.traces_to_fit)
    xmin, xmax = np.min(x_real), np.max(x_real)
    ymin, ymax = np.min(y_real), np.max(y_real)
    zmin = 0

    track_center, track_direction_vec = h5file.get_track_axis(event_num)
    track_direction_vec = track_direction_vec[0]

    def get_sim(params):
        E, Ea_frac, xp, yp, zp, xa, ya, za, theta_p, phi_p, theta_a, phi_a, sigma_xy_p, sigma_z_p, sigma_xy_a, sigma_z_a = params
        Ep = E*(1-Ea_frac)
        Ea = E*Ea_frac
        trace_sim = build_sim.create_multi_particle_event(experiment, run_number, event_num, ['4He','4He'])
        trace_sim.sims[0].initial_energy = Ep
        trace_sim.sims[1].initial_energy = Ea
        trace_sim.sims[0].initial_point = (xp,yp,zp)
        trace_sim.sims[1].initial_point = (xa,ya,za)
        trace_sim.sims[0].sigma_xy = sigma_xy_p
        trace_sim.sims[0].sigma_z = sigma_z_p
        trace_sim.sims[1].sigma_xy = sigma_xy_a
        trace_sim.sims[1].sigma_z = sigma_z_a
        trace_sim.sims[0].theta = theta_p
        trace_sim.sims[0].phi = phi_p
        trace_sim.sims[1].theta = theta_a
        trace_sim.sims[1].phi = phi_a
        
        trace_sim.set_real_data(source_sim.pads_to_sim, source_sim.sim_traces, trim_threshold=100, trim_pad=10)
        build_sim.open_gui(trace_sim)
        
        for sim in trace_sim.sims:
            sim.load_srim_table(sim.particle, 'P10', rho0)#*rho_scale
        trace_sim.simulate_event()
        return trace_sim

    def log_likelihood(params, print_out=False):
        trace_sim = get_sim(params)
        to_return = trace_sim.log_likelihood()
        if print_out:
            print(params, to_return)
        #print('E=%f MeV, (x,y,z)=(%f, %f, %f) mm, theta = %f deg, phi=%f deg, sigma_xy, sigma_z, LL=%e'%(E, x,y,z,np.degrees(theta), np.degrees(phi), sigma_xy, sigma_z, to_return))
        return to_return#/len(trace_sim.pads_to_sim)#(2.355*shaping_time*clock_freq)

    def log_priors(params, direction):
        E, Ea_frac, xp, yp, zp, xa, ya, za, theta_p, phi_p, theta_a, phi_a, sigma_xy_p, sigma_z_p, sigma_xy_a, sigma_z_a = params
        #uniform priors
        if Ea_frac < 0 or Ea_frac > 1:
            return -np.inf
        if xp**2 + yp**2 > 60**2:
            return -np.inf
        if zp < zmin or zp >zmax:
            return -np.inf
        if xa**2 + ya**2 > 60**2:
            return -np.inf
        if za < zmin or za >zmax:
            return -np.inf
        vhat = np.array([np.sin(theta_p)*np.cos(phi_p), np.sin(theta_p)*np.sin(phi_p), np.cos(theta_p)])
        if np.dot(vhat, direction*track_direction_vec) < 0 or theta_p > np.pi or theta_p < 0 or np.abs(phi_p)>np.pi:
            return -np.inf 
        if theta_p < 0 or theta_p >= np.pi or phi_p < 0 or phi_p>2*np.pi:
            return -np.inf 
        if theta_a < 0 or theta_a >= np.pi or phi_a < 0 or phi_a>2*np.pi:
            return -np.inf
        if sigma_xy_p < 0 or sigma_xy_p > 40:
            return -np.inf
        if sigma_z_p < 0 or sigma_z_p > 40:
            return -np.inf
        if sigma_xy_a < 0 or sigma_xy_a > 40:
            return -np.inf
        if sigma_z_a < 0 or sigma_z_a > 40:
            return -np.inf
        #gaussian prior for energy, and assume uniform over solid angle
        return E_prior.log_likelihood(E)  + np.log(np.abs(np.sin(theta_a))) + np.log(np.abs(np.sin(theta_p))) #+ density_scale_prior.log_likelihood(rho_scale)

    def log_posterior(params, direction, print_out=False):
        to_return = log_priors(params, direction)
        if to_return != -np.inf:
            to_return +=  log_likelihood(params)
        if np.isnan(to_return):
            to_return = -np.inf
        if print_out:
            print('log posterior: %e'%to_return, params)
        return to_return

    fit_start_time = time.time()
    # ntemps = 20 # default increases temp by sqrt(2), so highest temp is 1024 (sigma_T = 32*sigma = 3.2)
    nwalkers = 400
    steps = 500
    ndim = 16

    def get_init_walker_pos(direction):
        #initialize E per priors
        #start walkers in a small ball at far end of the track from where the particle will stop, given selected direction
        d_best, best_point = np.inf, None #distance along track in direction of particle motion. Make as negative as possible
        for x, y, z in zip(x_real, y_real, z_real):
            delta = np.array([x,y,z]) - track_center
            dist = np.dot(delta, track_direction_vec*direction)
            if  dist < d_best:
                d_best= dist
                best_point = np.array([x,y,z])
        best_point = np.array([0,0,0])
        # points = np.array([x_real,y_real,z_real])
        # candidates = points[sp.spatial.ConvexHull(points).vertices]
        # dist_mat = sp.spatial.distance_matrix(candidates,candidates)
        # i,j,k = np.unravel_index(dist_mat.argmax(),dist_mat.shape)
        # print(candidates[i], candidates[j], candidates[k])
        # best_point = np.array([candidates[i], candidates[j], candidates[k]])

        # dxy, dz, angle = h5file.get_track_length_angle(event_num)
        # xs, ys, zs, es = h5file.get_xyze(event_num, h5file.length_counts_threshold, include_veto_pads=False)
        # angle = np.arctan2(np.sqrt(track_direction_vec[0]**2 + track_direction_vec[1]**2),np.abs(track_direction_vec[2]))

        # points = np.concatenate((xs[:, np.newaxis], 
        #                ys[:, np.newaxis], 
        #                zs[:, np.newaxis]), 
        #               axis=1)
        # rbar = points - track_center
        # rdotv = np.dot(rbar, track_direction_vec)
        # first_point = points[np.argmin(rdotv)]
        # last_point = points[np.argmax(rdotv)]
        # middle_point = (last_point - first_point) / 2
        # # find voxel closest to middle point in the list of voxels that fired
        # dist_prev = np.inf
        # for x, y, z in zip(x_real, y_real, z_real):
        #     delta = np.array([x,y,z]) - middle_point
        #     dist = np.sqrt(delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2]*0.65*0.65) # TODO: change 0.65 to accurate zscale
        #     if  dist < dist_prev:
        #         best_point = np.array([x,y,z])
        #     dist_prev = dist

        #start theta, phi in a small ball around track direction from svd
        vhat = track_direction_vec*direction
        theta = np.arctan2( np.sqrt(vhat[0]**2 + vhat[1]**2), vhat[2])
        phi = np.arctan2(vhat[1], vhat[0])

        #start sigma_xy, sigma_z, and c in a small ball around an initial guess
        sigma_guess = 10
        pos_ball_size = 20 # default value is 1
        angle_ball_size = 1*np.pi/180

        max_veto_pad_counts, dxy, dz, measured_counts, angle, pads_railed = h5file.process_event(event_num)
        track_length = np.sqrt(dxy**2 + dz**2)
        
        # delete this when not testing MCMC on fitted events
        track_length = 20
        
        Ep_guess = source_sim.sims[0].srim_table.get_energy_w_stopping_distance(track_length - sigma_guess) 
        Ea_frac_guess = 1-Ep_guess/E_prior.mu
        assert Ea_frac_guess >0


        print('initial_guess:', (E_prior.mu, Ea_frac_guess, best_point, theta, phi, sigma_guess))

        return [(E_prior.sigma*np.random.randn() + E_prior.mu, Ea_frac_guess + np.random.randn()*0.01,
                            best_point[0] + np.random.randn()*pos_ball_size,
                            best_point[1] + np.random.randn()*pos_ball_size,
                            best_point[2] + np.random.randn()*pos_ball_size,
                            best_point[0] + np.random.randn()*pos_ball_size,
                            best_point[1] + np.random.randn()*pos_ball_size,
                            best_point[2] + np.random.randn()*pos_ball_size,
                            min(np.pi, max(0,theta + np.random.randn()*angle_ball_size)),
                            min(np.pi, max(-np.pi,phi + np.random.randn()*angle_ball_size)),
                            np.random.uniform(0, np.pi), np.random.uniform(-np.pi, np.pi),
                            sigma_guess + np.random.randn()*pos_ball_size, sigma_guess + np.random.randn()*pos_ball_size,
                            sigma_guess + np.random.randn()*pos_ball_size, sigma_guess + np.random.randn()*pos_ball_size,
                            ) for w in range(nwalkers)]


    # We'll track how the average autocorrelation time estimate changes
    directory = 'run%d_dalpha_mcmc/event%d'%(run_number, event_num)
    if not os.path.exists(directory):
        os.makedirs(directory)


    with multiprocessing.Pool() as pool:
        for direction in [-1, 1]:
            if direction == 1:
                backend_fname = 'forward.h5'
            elif direction == -1:
                backend_fname = 'backward.h5'
            else:
                assert False

            init_walker_pos = get_init_walker_pos(direction)
            backend_file = os.path.join(directory, backend_fname)
            backend = emcee.backends.HDFBackend(backend_file)
            backend.reset(nwalkers, ndim)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[direction],backend=backend, 
                                            moves=[(emcee.moves.KDEMove(), 1)],
                                            pool=pool)

            for sample in sampler.sample(init_walker_pos, iterations=steps, progress=True):
                tau = sampler.get_autocorr_time(tol=0)
                xs = sampler.get_chain()[-1]
                Ea = xs[:, 0]*xs[:, 1]
                Ep = xs[:, 0]*(1-xs[:, 1])
                print('Ea = ', np.percentile(Ea, [0,16, 50, 84,100]))
                print('Ep = ', np.percentile(Ep, [0,16, 50, 84,100]))
                print(backend_fname, ', tau=', tau, ', accept fraction=', np.average(sampler.acceptance_fraction))
                lls = sampler.get_log_prob()[-1]
                
                print(np.percentile(xs, [50], axis=0))
                print('log prob:',np.percentile(lls, [0,16, 50, 84,100]))
        