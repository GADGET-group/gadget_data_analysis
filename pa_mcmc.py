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
    particle_type = 'proton'
    run_number = int(run_number)
    event_num = int(event_num)

    experiment = 'e25058'

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
    density_scale_prior = GaussianVar(1, 0.05)#TODO: decid on density range

    h5file = build_sim.get_rawh5_object(experiment, run_number)
    #set zmax to length of trimmed traces
    temp_sim = build_sim.create_multi_particle_decay(experiment, run_number, event_num, ['1H', '4He'], [1.,4.], '16O', 16. )
    zmax = temp_sim.num_trace_bins*temp_sim.zscale

    x_real, y_real, z_real, e_real = temp_sim.get_xyze(threshold=h5file.length_counts_threshold, traces=temp_sim.traces_to_fit)
    xmin, xmax = np.min(x_real), np.max(x_real)
    ymin, ymax = np.min(y_real), np.max(y_real)
    zmin = 0

    track_center, track_direction_vec = h5file.get_track_axis(event_num)
    track_direction_vec = track_direction_vec[0]

    def get_sim(params):
        # Unpack 14 parameters (swapped 4 angles for 6 Cartesian coords)
        E, Ea_frac, x, y, z, p_x, p_y, p_z, a_x, a_y, a_z, sigma_xy, sigma_z, rho_scale = params
        
        Ep = E*(1-Ea_frac)
        Ea = E*Ea_frac
        
        # Convert Cartesian to Spherical for the simulation
        r_p = np.sqrt(p_x**2 + p_y**2 + p_z**2)
        theta_p = np.arccos(p_z / r_p)
        phi_p = np.arctan2(p_y, p_x)
        
        r_a = np.sqrt(a_x**2 + a_y**2 + a_z**2)
        theta_a = np.arccos(a_z / r_a)
        phi_a = np.arctan2(a_y, a_x)

        trace_sim = build_sim.create_multi_particle_decay(experiment, run_number, event_num, ['1H', '4He'], [1.,4.], '16O', 16. )
        trace_sim.sims[0].initial_energy = Ep
        trace_sim.sims[1].initial_energy = Ea
        trace_sim.initial_point = (x,y,z)
        trace_sim.sigma_xy = sigma_xy
        trace_sim.sigma_z = sigma_z
        trace_sim.sims[0].theta = theta_p
        trace_sim.sims[0].phi = phi_p
        trace_sim.sims[1].theta = theta_a
        trace_sim.sims[1].phi = phi_a
        
        for sim in trace_sim.sims:
            sim.load_srim_table(sim.particle, 'P10', rho0*rho_scale)
        trace_sim.simulate_event()
        return trace_sim

    def log_likelihood(params, print_out=False):
        trace_sim = get_sim(params)
        to_return = trace_sim.log_likelihood()
        if print_out:
            print(params, to_return)
        return to_return

    def log_priors(params, direction):
        E, Ea_frac, x, y, z, p_x, p_y, p_z, a_x, a_y, a_z, sigma_p_xy, sigma_p_z, rho_scale = params
        
        if E > 1.5*E_prior.mu:
            return -np.inf
        if Ea_frac < 0 or Ea_frac > 1:
            return -np.inf
        if x**2 + y**2 > 40**2:
            return -np.inf
        if z < zmin or z > zmax:
            return -np.inf
        if sigma_p_xy < 0 or sigma_p_xy > 30:
            return -np.inf
        if sigma_p_z < 0 or sigma_p_z > 30:
            return -np.inf

        # Normalize proton direction to evaluate bimodal constraint
        r_p = np.sqrt(p_x**2 + p_y**2 + p_z**2)
        vhat_p = np.array([p_x, p_y, p_z]) / r_p
        
        # Keep bimodal sampling restricted to the correct hemisphere
        if np.dot(vhat_p, direction*track_direction_vec) < 0:
            return -np.inf
            
        # Standard normal prior for 3D Cartesian vectors ensures uniform spherical sampling
        cartesian_prior = -0.5 * (p_x**2 + p_y**2 + p_z**2 + a_x**2 + a_y**2 + a_z**2)

        return E_prior.log_likelihood(E) + density_scale_prior.log_likelihood(rho_scale) + cartesian_prior

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
    nwalkers = 400
    steps = 400
    ndim = 14 # Increased from 12 to 14

    def get_init_walker_pos(direction):
        d_best, best_point = np.inf, None 
        for x, y, z in zip(x_real, y_real, z_real):
            delta = np.array([x,y,z]) - track_center
            dist = np.dot(delta, track_direction_vec*direction)
            if  dist < d_best:
                d_best= dist
                best_point = np.array([x,y,z])
                
        # Direction vector for proton initialization
        vhat = track_direction_vec * direction

        sigma_guess = 2.5
        sigma_ball_size = 0.5
        pos_ball_size = 6
        vec_ball_size = 0.1 # Noise for Cartesian vectors

        max_veto_pad_counts, dxy, dz, measured_counts, angle, pads_railed = h5file.process_event(event_num)
        track_length = np.sqrt(dxy**2 + dz**2)
        Ep_guess = temp_sim.sims[0].srim_table.get_energy_w_stopping_distance(track_length - sigma_guess) 
        Ea_frac_guess = 1-Ep_guess/E_prior.mu
        if Ea_frac_guess < 0:
            Ea_frac_guess = 0.001

        print('initial_guess:', (E_prior.mu, Ea_frac_guess, best_point, vhat, sigma_guess, sigma_guess))

        # Yield initial positions with 14 parameters
        return [ (E_prior.sigma*np.random.randn() + E_prior.mu, 
                  Ea_frac_guess + np.random.randn()*0.01,
                  best_point[0] + np.random.randn()*pos_ball_size,
                  best_point[1] + np.random.randn()*pos_ball_size,
                  best_point[2] + np.random.randn()*pos_ball_size,
                  vhat[0] + np.random.randn()*vec_ball_size, # p_x
                  vhat[1] + np.random.randn()*vec_ball_size, # p_y
                  vhat[2] + np.random.randn()*vec_ball_size, # p_z
                  np.random.randn(), # a_x (standard normal)
                  np.random.randn(), # a_y
                  np.random.randn(), # a_z
                  sigma_guess + np.random.randn()*sigma_ball_size, 
                  sigma_guess + np.random.randn()*sigma_ball_size,
                  np.random.uniform(0.9, 2)
                  ) for w in range(nwalkers)]

    directory = '%s_mcmc/run%d_palpha_mcmc/event%d'%(experiment, run_number, event_num)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with multiprocessing.Pool() as pool:
        for direction in [1, -1]:
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
                                            #moves=[(emcee.moves.KDEMove(), 1)],
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