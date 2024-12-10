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
    #folder = '/mnt/analysis/e21072/gastest_h5_files/'
    folder = '../../shared/Run_Data/'

    experiment = 'e21072'

    #MCMC priors
    class GaussianVar:
        def __init__(self, mu, sigma):
            self.mu, self.sigma  = mu, sigma

        def log_likelihood(self, val):
            return -np.log(np.sqrt(2*np.pi*self.sigma**2)) - (val - self.mu)**2/2/self.sigma**2

    E_from_ic = build_sim.get_energy_from_ic(experiment, run_number, event_num)
    E_from_ic_simga = build_sim.get_detector_E_sigma(experiment, run_number, E_from_ic)
    E_prior = GaussianVar(E_from_ic, E_from_ic_simga)

    h5file = build_sim.get_rawh5_object(experiment, run_number)
    x_real, y_real, z_real, e_real = h5file.get_xyze(event_number=event_num)
    xmin, xmax = np.min(x_real), np.max(x_real)
    ymin, ymax = np.min(y_real), np.max(y_real)
    zmin = 0
    #set zmax to length of trimmed traces
    temp_sim = build_sim.create_pa_sim(experiment, run_number, event_num)
    zmax = temp_sim.num_trace_bins*temp_sim.zscale

    def get_sim(params):
        E, Ea_frac, x, y, z, theta_p, phi_p, theta_a, phi_a, sigma_p_xy, sigma_p_z, other_uncert = params
        Ep = E*(1-Ea_frac)
        Ea = E*Ea_frac
        trace_sim = build_sim.create_pa_sim(experiment, run_number, event_num)
        trace_sim.sims[0].initial_energy = Ep
        trace_sim.sims[1].initial_energy = Ea
        trace_sim.sims[0].initial_point = trace_sim.sims[1].initial_point = (x,y,z)
        trace_sim.sims[0].sigma_xy = sigma_p_xy
        trace_sim.sims[0].sigma_z = sigma_p_z
        trace_sim.sims[1].sigma_xy = sigma_p_xy
        trace_sim.sims[1].sigma_z = sigma_p_z
        trace_sim.sims[0].theta = theta_p
        trace_sim.sims[0].phi = phi_p
        trace_sim.sims[1].theta = theta_a
        trace_sim.sims[1].phi = phi_a
        #trace_sim.pad_gain_match_uncertainty = 0#gain_match
        trace_sim.other_systematics = other_uncert
        trace_sim.simulate_event()
        return trace_sim

    def log_likelihood(params, print_out=False):
        trace_sim = get_sim(params)
        to_return = trace_sim.log_likelihood()
        if print_out:
            print(params, to_return)
        #print('E=%f MeV, (x,y,z)=(%f, %f, %f) mm, theta = %f deg, phi=%f deg, sigma_xy, sigma_z, LL=%e'%(E, x,y,z,np.degrees(theta), np.degrees(phi), sigma_xy, sigma_z, to_return))
        return to_return#/len(trace_sim.pads_to_sim)#(2.355*shaping_time*clock_freq)

    def log_priors(params):
        E, Ea_frac, x, y, z, theta_p, phi_p, theta_a, phi_a, sigma_p_xy, sigma_p_z, other_uncert = params
        #uniform priors
        if Ea_frac < 0 or Ea_frac > 1:
            return -np.inf
        if x**2 + y**2 > 40**2:
            return -np.inf
        if z < zmin or z >zmax:
            return -np.inf
        if theta_p < 0 or theta_p >= np.pi or phi_p < 0 or phi_p>2*np.pi:
            return -np.inf 
        if theta_a < 0 or theta_a >= np.pi or phi_a < 0 or phi_a>2*np.pi:
            return -np.inf 
        if sigma_p_xy < 0 or sigma_p_xy > 40:
            return -np.inf
        if sigma_p_z < 0 or sigma_p_z > 40:
            return -np.inf
        if other_uncert < 0 or other_uncert > 4000:
            return -np.inf
        #gaussian prior for energy, and assume uniform over solid angle
        return E_prior.log_likelihood(E) + np.log(np.abs(np.sin(theta_a))) + np.log(np.abs(np.sin(theta_p)))

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
    nwalkers = 400
    clustering_steps = 1000
    times_to_repeat_clustering = 2
    post_cluster_steps=0
    ndim = 12



    init_walker_pos = [(E_prior.sigma*np.random.randn() + E_prior.mu, np.random.uniform(0,1),
                            np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax), np.random.uniform(zmin, zmax),
                            np.random.uniform(0,np.pi), np.random.uniform(-np.pi, np.pi), np.random.uniform(0,np.pi), np.random.uniform(-np.pi, np.pi),
                            np.random.uniform(0, 20), np.random.uniform(0,20),
                            #np.random.uniform(0, 1),
                              np.random.uniform(0,400)) for w in range(nwalkers)]
    # We'll track how the average autocorrelation time estimate changes
    directory = 'run%d_palpha_mcmc/event%d'%(run_number, event_num)
    if not os.path.exists(directory):
        os.makedirs(directory)


    with multiprocessing.Pool() as pool:
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
                                            moves=[(emcee.moves.KDEMove(), 1)],
                                                   #,(emcee.moves.StretchMove(), 0.5)],
                                            pool=pool)

            for sample in sampler.sample(init_walker_pos, iterations=clustering_steps, progress=True):
                tau = sampler.get_autocorr_time(tol=0)
                sample = sampler.get_chain()[-1]
                Ea = sample[:, 0]*sample[:, 1]
                Ep = sample[:, 0]*(1-sample[:, 1])
                print('Ea = ', np.percentile(Ea, [0,16, 50, 84,100]))
                print('Ep = ', np.percentile(Ep, [0,16, 50, 84,100]))
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
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, backend=backend, pool=pool)
        for sample in sampler.sample(init_walker_pos, iterations=post_cluster_steps, progress=True):
            tau = sampler.get_autocorr_time(tol=0)
            print('after clustering iteration=', sampler.iteration, ', tau=', tau, ', accept fraction=', np.average(sampler.acceptance_fraction))