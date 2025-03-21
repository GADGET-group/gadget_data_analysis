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
import pymc as pm
import arviz as az

from track_fitting import SingleParticleEvent, build_sim

if __name__ == '__main__':
    particle_type, run_number, event_num = sys.argv[1:]
    run_number = int(run_number)
    event_num = int(event_num)
    experiment = 'e21072'
    if particle_type == '1H':
        particle_mass = 1
        recoil_mass = 19
        recoil_nuclues = '19Ne'
    elif particle_type == '4He':
        particle_mass = 4
        recoil_mass = 16
        recoil_nuclues = '16O'

    
    #MCMC priors
    class GaussianVar:
        def __init__(self, mu, sigma):
            self.mu, self.sigma  = mu, sigma

        def log_likelihood(self, val):
            return -np.log(np.sqrt(2*np.pi*self.sigma**2)) - (val - self.mu)**2/2/self.sigma**2
        
    E_from_ic = build_sim.get_energy_from_ic(experiment, run_number, event_num)
    E_prior = GaussianVar(E_from_ic, build_sim.get_detector_E_sigma(experiment, run_number, E_from_ic))
    density_rescale_prior = GaussianVar(1, 0.05)
    nominal_gas_density = build_sim.get_gas_density(experiment, run_number)

    h5file = build_sim.get_rawh5_object(experiment, run_number)
    temp_sim = build_sim.create_single_particle_sim('e21072', run_number, event_num, particle_type)
    x_real, y_real, z_real, e_real = temp_sim.get_xyze(threshold=h5file.length_counts_threshold, traces=temp_sim.traces_to_fit)
    xmin, xmax = np.min(x_real), np.max(x_real)
    ymin, ymax = np.min(y_real), np.max(y_real)
    zmin = 0
    #set zmax to length of trimmed traces
    zmax = temp_sim.num_trace_bins*h5file.zscale
    
    sigma_min, sigma_max = 0,30

    track_center, track_direction_vec = h5file.get_track_axis(event_num)
    track_direction_vec = track_direction_vec[0]

    def get_sim(params):
        E, x, y, z, theta, phi, sigma_xy, sigma_z, density_scale = params
        trace_sim = build_sim.create_multi_particle_decay(experiment, run_number, event_num, 
                                                          [particle_type], [particle_mass],
                                                          recoil_nuclues, recoil_mass)
        for sim in trace_sim.sims:
            sim.load_srim_table(sim.particle, sim.material, nominal_gas_density*density_scale)
        trace_sim.products[0].initial_energy = E
        trace_sim.initial_point = (x,y,z)
        trace_sim.products[0].theta = theta
        trace_sim.products[0].phi = phi
        trace_sim.sigma_xy = sigma_xy
        trace_sim.sigma_z = sigma_z
        trace_sim.simulate_event()
        #trace_sim.pad_gain_match_uncertainty = 0
        return trace_sim

    def log_likelihood(params, print_out=False):
        trace_sim = get_sim(params)
        to_return = trace_sim.log_likelihood()
        if print_out:
            print(params, to_return)
        #print('E=%f MeV, (x,y,z)=(%f, %f, %f) mm, theta = %f deg, phi=%f deg, sigma_xy, sigma_z, LL=%e'%(E, x,y,z,np.degrees(theta), np.degrees(phi), sigma_xy, sigma_z, to_return))
        return to_return/len(trace_sim.pads_to_sim)#trace_sim.num_trace_bins#(2.355*shaping_time*clock_freq)

    def log_priors(params, direction):
        E, x, y, z, theta, phi, sigma_xy, sigma_z, density_scale = params

        # Convert conditions to PyMC-friendly format using pm.math.switch()
        prior_x_y = pm.math.switch(x**2 + y**2 > 40**2, -np.inf, 0)
        prior_z = pm.math.switch(pm.math.or_(z < zmin, z > zmax), -np.inf, 0)
        prior_sigma_xy = pm.math.switch(pm.math.or_(sigma_xy < sigma_min, sigma_xy > sigma_max), -np.inf, 0)
        prior_sigma_z = pm.math.switch(pm.math.or_(sigma_z < sigma_min, sigma_z > sigma_max), -np.inf, 0)
        prior_density = pm.math.switch(density_scale < 0, -np.inf, 0)

        # Ensure particle motion is within 90 degrees of track axis
        vhat_x = pm.math.sin(theta) * pm.math.cos(phi)
        vhat_y = pm.math.sin(theta) * pm.math.sin(phi)
        vhat_z = pm.math.cos(theta)
        vhat = pm.math.stack([vhat_x, vhat_y, vhat_z])  # Convert to PyMC tensor
        dot_product = pm.math.dot(vhat, direction * track_direction_vec)

        prior_track_alignment = pm.math.switch(
            pm.math.or_(dot_product < 0, pm.math.or_(theta > np.pi, theta < 0)),
            -np.inf,
            0
        )

        # Gaussian prior for energy and uniform over solid angle
        prior_energy = E_prior.log_likelihood(E)
        prior_solid_angle = pm.math.log(pm.math.abs(pm.math.sin(theta)))
        prior_density_scale = density_rescale_prior.log_likelihood(density_scale)

        # Sum all priors
        priors = (
            prior_x_y + prior_z + prior_sigma_xy + prior_sigma_z + prior_density +
            prior_track_alignment + prior_energy + prior_solid_angle + prior_density_scale
        )

        return priors
    
    def log_posterior(params, direction, print_out=False):
        to_return = log_priors(params, direction)
        if to_return != -np.inf:
            to_return +=  log_likelihood(params)
        if np.isnan(to_return):
            to_return = -np.inf
        if print_out:
            print('log posterior: %e'%to_return, params)
        return to_return

    # fit_start_time = time.time()
    # nwalkers = 200
    # steps = 500
    # ndim = 9

    # def get_init_walker_pos(direction):
    #     #initialize E per priors
    #     #start walkers in a small ball at far end of the track from where the particle will stop, given selected direction
    #     d_best, best_point = np.inf, None #distance along track in direction of particle motion. Make as negative as possible
    #     for x, y, z in zip(x_real, y_real, z_real):
    #         delta = np.array([x,y,z]) - track_center
    #         dist = np.dot(delta, track_direction_vec*direction)
    #         if  dist < d_best:
    #             d_best= dist
    #             best_point = np.array([x,y,z])
    #     #start theta, phi in a small ball around track direction from svd
    #     vhat = track_direction_vec*direction
    #     theta = np.arctan2( np.sqrt(vhat[0]**2 + vhat[1]**2), vhat[2])
    #     phi = np.arctan2(vhat[1], vhat[0])
    #     #start sigma_xy, sigma_z, and c in a small ball around an initial guess
    #     sigma_guess = 7
    #     pos_ball_size = 1
    #     angle_ball_size = 1*np.pi/180

    #     x_guess = (E_prior)
    #     print('initial_guess:', (E_prior.mu, best_point, theta, phi, sigma_guess, sigma_guess))
    #     #print('starting likelihood maximization')
    #     #after_optimization = opt.minimize(lambda x: -log_posterior(x), x_guess, args=(direction,True))


    #     to_return = [(E_prior.sigma*np.random.randn() + E_prior.mu,
    #                         best_point[0] + np.random.randn()*pos_ball_size,
    #                         best_point[1] + np.random.randn()*pos_ball_size,
    #                         best_point[2] + np.random.randn()*pos_ball_size,
    #                         min(np.pi, max(0,theta + np.random.randn()*angle_ball_size)),
    #                         min(np.pi, max(-np.pi,phi + np.random.randn()*angle_ball_size)),
    #                         sigma_guess + np.random.randn()*pos_ball_size, sigma_guess + np.random.randn()*pos_ball_size,
    #                         density_rescale_prior.mu + np.random.randn()*density_rescale_prior.sigma
    #                         ) for w in range(nwalkers)]
    #     # for p in to_return:
    #     #     lp = log_posterior(p)
    #     #     print(p, lp)
    #     #     if not np.isfinite(lp):
    #     #         assert False
    #     return to_return
    # # We'll track how the average autocorrelation time estimate changes
    # directory = 'run%d_mcmc/event%d'%(run_number, event_num)
    # if not os.path.exists(directory):
    #     os.makedirs(directory)


    # with multiprocessing.Pool(nwalkers) as pool:
    #     for direction in [-1, 1]:
    #         if direction == 1:
    #             backend_fname = 'forward.h5'
    #         else:
    #             backend_fname = 'backward.h5'
    #         backend_file = os.path.join(directory, backend_fname)
    #         backend = emcee.backends.HDFBackend(backend_file)
    #         backend.reset(nwalkers, ndim)

    #         init_walker_pos = get_init_walker_pos(direction)

    #         sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[direction],backend=backend, 
    #                                          moves=[(emcee.moves.KDEMove(), 1)],
    #                                         pool=pool)

    #         for sample in sampler.sample(init_walker_pos, iterations=steps, progress=True):
    #             tau = sampler.get_autocorr_time(tol=0)
    #             print(backend_fname, ', tau=', tau, ', accept fraction=', np.average(sampler.acceptance_fraction))
    #             lls = sampler.get_log_prob()[-1]
    #             xs = sampler.get_chain()[-1]
    #             print(np.percentile(xs, [50], axis=0))
    #             print('log prob:',np.percentile(lls, [0,16, 50, 84,100]))

    directory = f'run{run_number}_mcmc/event{event_num}'
    os.makedirs(directory, exist_ok=True)

    for direction in [-1, 1]:
        if direction == 1:
            backend_fname = 'forward_smc.nc'
        else:
            backend_fname = 'backward_smc.nc'

        backend_file = os.path.join(directory, backend_fname)

        # ============================
        # PyMC Model Definition
        # ============================
        with pm.Model() as smc_model:
            E = pm.Normal("E", mu=E_prior.mu, sigma=E_prior.sigma)
            x = pm.Uniform("x", lower=xmin, upper=xmax)
            y = pm.Uniform("y", lower=ymin, upper=ymax)
            z = pm.Uniform("z", lower=zmin, upper=zmax)
            theta = pm.Uniform("theta", lower=0, upper=np.pi)
            phi = pm.Uniform("phi", lower=-np.pi, upper=np.pi)
            sigma_xy = pm.Uniform("sigma_xy", lower=sigma_min, upper=sigma_max)
            sigma_z = pm.Uniform("sigma_z", lower=sigma_min, upper=sigma_max)
            density_scale = pm.Normal("density_scale", mu=density_rescale_prior.mu, sigma=density_rescale_prior.sigma)

            # Custom logp function using the existing log_posterior
            def logp_func(E, x, y, z, theta, phi, sigma_xy, sigma_z, density_scale):
                params = [E, x, y, z, theta, phi, sigma_xy, sigma_z, density_scale]
                return log_posterior(params, direction)

            pm.Potential("logp", logp_func(E, x, y, z, theta, phi, sigma_xy, sigma_z, density_scale))

            # ============================
            # Run SMC Sampling
            # ============================
            trace_smc = pm.sample_smc(draws=2000, homepath=backend_file)

        # ============================
        # Save Results
        # ============================
        trace_smc.to_netcdf(backend_file)
        print(f"SMC Sampling Complete: Results saved to {backend_file}")
