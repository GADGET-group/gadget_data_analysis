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

from track_fitting import SingleParticleEvent
from raw_viewer import raw_h5_file

if __name__ == '__main__':
    particle_type, run_number, event_num = sys.argv[1:]
    run_number = int(run_number)
    event_num = int(event_num)
    #folder = '/mnt/analysis/e21072/gastest_h5_files/'
    folder = '../../shared/Run_Data/'


    run_h5_path = folder +'run_%04d.h5'%run_number

    if run_number == 124:
        adc_scale_mu = 86431./0.757 #counts/MeV, from fitting events with range 40-43 in run 0368 with p10_default
        detector_E_sigma = lambda E: (5631./adc_scale_mu)*np.sqrt(E/0.757) #sigma for above fit, scaled by sqrt energy

        #use theoretical zscale
        clock_freq = 50e6 #Hz, from e21062 config file on mac minis
        drift_speed = 54.4*1e6 #mm/s, from ruchi's paper
        zscale = drift_speed/clock_freq

        ic_threshold = 25
        h5file = raw_h5_file.raw_h5_file(file_path=run_h5_path,
                                        zscale=zscale,
                                        flat_lookup_csv='raw_viewer/channel_mappings/flatlookup4cobos.csv')
        h5file.background_subtract_mode='fixed window'
        h5file.data_select_mode='near peak'
        h5file.remove_outliers=True
        h5file.near_peak_window_width = 50
        h5file.require_peak_within= (-np.inf, np.inf)
        h5file.num_background_bins=(160, 250)
        h5file.zscale = zscale

        shaping_time = 70e-9 #s, from e21062 config file on mac minis

        pad_threshold = 65


        pressure = 860.3 #assuming current offset on MFC was present during experiment, and it was set to 800 torr

            
    rho0 = 1.5256 #mg/cm^3, P10 at 300K and 760 torr
    T = 20+273.15 #K
    get_gas_density = lambda P: rho0*(P/760)*(300./T)



    pads_to_fit, traces_to_fit = h5file.get_pad_traces(event_num, include_veto_pads=False)

    #MCMC priors
    class GaussianVar:
        def __init__(self, mu, sigma):
            self.mu, self.sigma  = mu, sigma

        def log_likelihood(self, val):
            return -np.log(np.sqrt(2*np.pi*self.sigma**2)) - (val - self.mu)**2/2/self.sigma**2

    max_veto_pad_counts, dxy, dz, measured_counts, angle, pads_railed = h5file.process_event(event_num)
    E_from_ic = measured_counts/adc_scale_mu
    E_prior = GaussianVar(E_from_ic, detector_E_sigma(E_from_ic))
    x_real, y_real, z_real, e_real = h5file.get_xyze(event_number=event_num)
    xmin, xmax = np.min(x_real), np.max(x_real)
    ymin, ymax = np.min(y_real), np.max(y_real)
    zmin = 0
    #set zmax to length of trimmed traces
    temp_sim = SingleParticleEvent.SingleParticleEvent(get_gas_density(pressure), particle_type)
    temp_sim.set_real_data(pads_to_fit, traces_to_fit, trim_threshold=50, trim_pad=10)
    zmax = temp_sim.num_trace_bins*zscale
    num_stopping_points = temp_sim.get_num_stopping_points_for_energy(E_from_ic)


    def log_likelihood(params):
        E, x, y, z, theta, phi, sigma_xy, sigma_z, m, c = params
        trace_sim = SingleParticleEvent.SingleParticleEvent(get_gas_density(pressure), particle_type)
        trace_sim.pad_gain_match_uncertainty = m
        trace_sim.other_systematics = c
        trace_sim.zscale = zscale
        trace_sim.counts_per_MeV = adc_scale_mu
        trace_sim.set_real_data(pads_to_fit, traces_to_fit, trim_threshold=50, trim_pad=10)#match trim threshold used for systematics determination\
        trace_sim.load_srim_table(particle_type, get_gas_density(pressure))
        trace_sim.initial_energy = E
        trace_sim.initial_point = (x,y,z)
        trace_sim.theta = theta
        trace_sim.phi = phi
        trace_sim.sigma_xy = sigma_xy
        trace_sim.sigma_z = sigma_z
        trace_sim.adaptive_stopping_power = False
        trace_sim.num_stopping_power_points = num_stopping_points
        
        trace_sim.simulate_event()
        to_return = trace_sim.log_likelihood()
        #print('E=%f MeV, (x,y,z)=(%f, %f, %f) mm, theta = %f deg, phi=%f deg, sigma_xy, sigma_z, LL=%e'%(E, x,y,z,np.degrees(theta), np.degrees(phi), sigma_xy, sigma_z, to_return))
        return to_return

    def log_priors(params):
        E, x, y, z, theta, phi, sigma_xy, sigma_z, m, c = params
        #uniform priors
        if m < 0 or m > 1:
            return -np.inf
        if c < 0 or c > 4000:
            return -np.inf
        if x**2 + y**2 > 40**2:
            return -np.inf
        if z < zmin or z >zmax:
            return -np.inf
        if theta < 0 or theta >= np.pi or phi < -2*np.pi or phi>2*np.pi:
            return -np.inf 
        if sigma_xy < 0 or sigma_xy > 40:
            return -np.inf
        if sigma_z < 0 or sigma_z > 40:
            return -np.inf
        #gaussian prior for energy, and assume uniform over solid angle
        return E_prior.log_likelihood(E) + np.log(np.abs(np.sin(theta)))

    def log_posterior(params):
        to_return = log_priors(params)
        if to_return != -np.inf:
            to_return +=  log_likelihood(params)
        if np.isnan(to_return):
            to_return = -np.inf
        #print('log posterior: %e'%to_return)
        return to_return

    fit_start_time = time.time()
    nwalkers = 250
    clustering_steps = 200
    times_to_repeat_clustering = 4
    post_cluster_steps=6000
    ndim = 10

    init_walker_pos = [[E_prior.mu + E_prior.sigma*np.random.randn(), np.random.uniform(xmin, xmax), 
                                np.random.uniform(ymin, ymax), np.random.uniform(zmin, zmax), np.random.uniform(0, np.pi), 
                                np.random.uniform(-np.pi, np.pi), np.random.uniform(0,40), np.random.uniform(0,40),
                                np.random.uniform(0, 1), np.random.uniform(0,4000)] for i in range(nwalkers)]

    # We'll track how the average autocorrelation time estimate changes
    directory = 'run%d_mcmc/event%d'%(run_number, event_num)
    if not os.path.exists(directory):
        os.makedirs(directory)


    with multiprocessing.Pool() as pool:
        for step in range(times_to_repeat_clustering):
            backend_file = os.path.join(directory, 'clustering_run%d.h5'%step)
            backend = emcee.backends.HDFBackend(backend_file)
            backend.reset(nwalkers, ndim)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, backend=backend, 
                                            #  moves=[
                                            #          (emcee.moves.DESnookerMove(), 0.2),
                                            #          (emcee.moves.StretchMove(), 0.6),
                                            #          (emcee.moves.DEMove(gamma0=1.0), 0.2)
                                            #  ],
                                            pool=pool)

            for sample in sampler.sample(init_walker_pos, iterations=clustering_steps, progress=True):
                tau = sampler.get_autocorr_time(tol=0)
                print('iteration=', sampler.iteration, ', tau=', tau, ', accept fraction=', np.average(sampler.acceptance_fraction))

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