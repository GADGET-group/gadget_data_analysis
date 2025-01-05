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

from track_fitting import SingleParticleEvent, build_sim

if __name__ == '__main__':
    particle_type, run_number, event_num = sys.argv[1:]
    run_number = int(run_number)
    event_num = int(event_num)
    experiment = 'e21072'

    
    #MCMC priors
    class GaussianVar:
        def __init__(self, mu, sigma):
            self.mu, self.sigma  = mu, sigma

        def log_likelihood(self, val):
            return -np.log(np.sqrt(2*np.pi*self.sigma**2)) - (val - self.mu)**2/2/self.sigma**2
        
    E_from_ic = build_sim.get_energy_from_ic(experiment, run_number, event_num)
    E_prior = GaussianVar(E_from_ic, build_sim.get_detector_E_sigma(experiment, run_number, E_from_ic))
    h5file = build_sim.get_rawh5_object(experiment, run_number)
    x_real, y_real, z_real, e_real = h5file.get_xyze(event_number=event_num)
    xmin, xmax = np.min(x_real), np.max(x_real)
    ymin, ymax = np.min(y_real), np.max(y_real)
    zmin = 0
    #set zmax to length of trimmed traces
    temp_sim = build_sim.create_single_particle_sim('e21072', run_number, event_num, particle_type)
    zmax = temp_sim.num_trace_bins*h5file.zscale
    num_stopping_points = temp_sim.get_num_stopping_points_for_energy(E_from_ic)

    def get_sim(params):
        E, x, y, z, theta, phi, sigma_xy, sigma_z, c = params
        trace_sim = build_sim.create_single_particle_sim(experiment, run_number, event_num, particle_type)
        trace_sim.initial_energy = E
        trace_sim.initial_point = (x,y,z)
        trace_sim.theta = theta
        trace_sim.phi = phi
        trace_sim.sigma_xy = sigma_xy
        trace_sim.sigma_z = sigma_z
        trace_sim.adaptive_stopping_power = False
        trace_sim.num_stopping_power_points = num_stopping_points
        trace_sim.simulate_event()
        #trace_sim.pad_gain_match_uncertainty = 0
        trace_sim.other_systematics = c
        return trace_sim

    def log_likelihood(params, print_out=False):
        trace_sim = get_sim(params)
        to_return = trace_sim.log_likelihood()
        if print_out:
            print(params, to_return)
        #print('E=%f MeV, (x,y,z)=(%f, %f, %f) mm, theta = %f deg, phi=%f deg, sigma_xy, sigma_z, LL=%e'%(E, x,y,z,np.degrees(theta), np.degrees(phi), sigma_xy, sigma_z, to_return))
        return to_return/len(trace_sim.pads_to_sim)#trace_sim.num_trace_bins#(2.355*shaping_time*clock_freq)

    def log_priors(params):
        E, x, y, z, theta, phi, sigma_xy, sigma_z, c = params
        #uniform priors
        if x**2 + y**2 > 40**2:
            return -np.inf
        if z < zmin or z >zmax:
            return -np.inf
        if theta < 0 or theta >= np.pi or phi < -2*np.pi or phi>2*np.pi:
            return -np.inf 
        if sigma_xy < 0 or sigma_xy > 20:
            return -np.inf
        if sigma_z < 0 or sigma_z > 20:
            return -np.inf
        if c <= 0 or c > 1000:
            return -np.inf
        #gaussian prior for energy, and assume uniform over solid angle
        return E_prior.log_likelihood(E) + np.log(np.abs(np.sin(theta)))

    def log_posterior(params, print_out=False):
        to_return = log_priors(params)
        if to_return != -np.inf:
            to_return +=  log_likelihood(params)
        if np.isnan(to_return):
            to_return = -np.inf
        if print_out:
            print('log posterior: %e'%to_return, params)
        return to_return

    fit_start_time = time.time()
    nwalkers = 200
    clustering_steps = 1000
    times_to_repeat_clustering = 1
    post_cluster_steps=0
    ndim = 9



    init_walker_pos = [(E_prior.sigma*np.random.randn() + E_prior.mu,
                            np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax), np.random.uniform(zmin, zmax),
                            np.random.uniform(0,np.pi), np.random.uniform(-np.pi, np.pi),
                            np.random.uniform(0, 30), np.random.uniform(0,30),
                            np.random.uniform(0, 100)) for w in range(nwalkers)]
    # We'll track how the average autocorrelation time estimate changes
    directory = 'run%d_mcmc/event%d'%(run_number, event_num)
    if not os.path.exists(directory):
        os.makedirs(directory)


    with multiprocessing.Pool(nwalkers) as pool:
        for step in range(times_to_repeat_clustering):
            backend_fname = 'clustering_run%d.h5'%step
            backend_file = os.path.join(directory, backend_fname)
            backend = emcee.backends.HDFBackend(backend_file)
            backend.reset(nwalkers, ndim)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, backend=backend, 
                                            #  moves=[
                                            #          (emcee.moves.DESnookerMove(), 0.2),
                                            #          (emcee.moves.DEMove(), 0.6),
                                            #          (emcee.moves.DEMove(gamma0=1.0), 0.2)
                                            #  ],
                                            pool=pool)

            for sample in sampler.sample(init_walker_pos, iterations=clustering_steps, progress=True):
                tau = sampler.get_autocorr_time(tol=0)
                print(backend_fname, ', tau=', tau, ', accept fraction=', np.average(sampler.acceptance_fraction))

            #cluster log likelihood into two clusters, and pick out the most recent samples from the best cluster 
            ll = sampler.get_log_prob()[-1]
            ll_to_cluster = ll[np.isfinite(ll)].reshape(-1,1)
            cluster_object = cluster.KMeans(2).fit(ll_to_cluster)
            clusters_to_propagate = cluster_object.labels_==np.argmax(cluster_object.cluster_centers_)
            samples_to_propagate = sampler.get_chain()[-1][np.isfinite(ll)][clusters_to_propagate]
            new_init_pos = list(samples_to_propagate)

            #randomly select samples to perturb and add to the list until we have the desired number of walkers
            while len(new_init_pos) < nwalkers:
                i = np.random.randint(0, len(samples_to_propagate))
                new_init_pos.append(samples_to_propagate[1] + .001*np.random.randn(ndim))
            init_walker_pos = new_init_pos

        #restart mcmc
        backend_file = os.path.join(directory, 'final_run.h5')
        backend = emcee.backends.HDFBackend(backend_file)
        backend.reset(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, backend=backend, pool=pool, moves=[(emcee.moves.KDEMove(), 1)],)
        for sample in sampler.sample(init_walker_pos, iterations=post_cluster_steps, progress=True):
            tau = sampler.get_autocorr_time(tol=0)
            print('after clustering iteration=', sampler.iteration, ', tau=', tau, ', accept fraction=', np.average(sampler.acceptance_fraction))