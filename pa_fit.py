#turn off numpy multithreading to avoid conflicts with emcee
import os
os.environ["OMP_NUM_THREADS"] = "1"

import time
import os
import multiprocessing

import numpy as np
import matplotlib.pylab as plt
import scipy.optimize as opt
import corner
import sklearn.cluster as cluster
import emcee

from track_fitting import ParticleAndPointDeposition
from raw_viewer import raw_h5_file


#folder = '/mnt/analysis/e21072/gastest_h5_files/'
folder = '../../shared/Run_Data/'
run_number = 124
event_num = 51777
run_h5_path = folder +'run_%04d.h5'%run_number

if folder == '../../shared/Run_Data/':#folder == '/mnt/analysis/e21072/h5test/':
    if run_number == 124:
        adc_scale_mu = 86431./0.757 #counts/MeV, from fitting events with range 40-43 in run 0368 with p10_default
        detector_E_sigma = lambda E: (5631./adc_scale_mu)*np.sqrt(E/0.757) #sigma for above fit, scaled by sqrt energy

        #use theoretical zscale
        clock_freq = 50e6 #Hz, from e21062 config file on mac minis
        drift_speed = 54.4*1e6 #mm/s, from ruchi's paper
        zscale = drift_speed/clock_freq

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
        h5file.ic_counts_threshold = 24

        shaping_time = 70e-9 #s, from e21062 config file on mac minis
        shaping_width  = shaping_time*clock_freq*2.355

        pressure = 860.3 #assuming current offset on MFC was present during experiment, and it was set to 800 torr
        charge_spread = 2. #mm, value used when fitting systematics

        
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
zmin, zmax = 5, 400

pad_gain_match_uncertainty = 0.381959476
other_systematics = 16.86638095

#do MCMC
def log_likelihood_mcmc(params):
    E, Ea_frac, x, y, z, theta, phi = params
    trace_sim = ParticleAndPointDeposition.ParticleAndPointDeposition(get_gas_density(pressure), 'proton')
    trace_sim.shaping_width = shaping_width
    trace_sim.zscale = zscale
    trace_sim.counts_per_MeV = adc_scale_mu
    trace_sim.simulate_event()
    trace_sim.set_real_data(pads_to_fit, traces_to_fit, trim_threshold=50)#match trim threshold used for systematics determination
    trace_sim.align_pad_traces()
    trace_sim.initial_energy = E*(1-Ea_frac)
    trace_sim.point_energy_deposition = E*Ea_frac
    trace_sim.initial_point = (x,y,z)
    trace_sim.theta = theta
    trace_sim.phi = phi
    trace_sim.charge_spreading_sigma = charge_spread
    trace_sim.shaping_width = shaping_width
    trace_sim.counts_per_MeV = adc_scale_mu
    trace_sim.pad_gain_match_uncertainty = pad_gain_match_uncertainty
    trace_sim.other_systematics = other_systematics
    trace_sim.simulate_event()
    trace_sim.align_pad_traces()
    to_return = trace_sim.log_likelihood()
    #print('E=%f MeV, (x,y,z)=(%f, %f, %f) mm, theta = %f deg, phi=%f deg, cs=%f mm, shaping=%f, P=%f torr, adc_scale=%f, LL=%e'%(E, x,y,z,np.degrees(theta), np.degrees(phi), charge_spread, shaping_width, P, adc_scale, to_return))
    return to_return

def log_priors(params):
    E, Ea_frac, x, y, z, theta, phi = params
    
    #uniform priors
    if x**2 + y**2 > 40**2:
        return -np.inf
    if z < 0 or z >400:
        return -np.inf
    if theta < 0 or theta >= np.pi or phi < -np.pi or phi>np.pi:
        return -np.inf 
    if shaping_width <=0 or shaping_width > 20:
        return -np.inf
    if charge_spread < 0:
        return -np.inf
    if Ea_frac < 0 or Ea_frac > 1:
        return -np.inf
    #gaussian prior for energy, and assume uniform over solid angle
    return E_prior.log_likelihood(E) + np.log(np.abs(np.sin(theta)))


def log_posterior(params, beta):
    to_return = log_priors(params)
    if to_return != -np.inf:
        to_return +=  log_likelihood_mcmc(params)*beta
    if np.isnan(to_return):
        to_return = -np.inf
    #print('log posterior: %e'%to_return)
    return to_return


nwalkers = 200
max_n = 5000
ndim = 7

init_walker_pos = [[E_prior.mu + E_prior.sigma*np.random.randn(), np.random.uniform(0,1),np.random.uniform(xmin, xmax), 
                            np.random.uniform(ymin, ymax), np.random.uniform(zmin, zmax), np.random.uniform(0, np.pi), 
                            np.random.uniform(-np.pi, np.pi)] for i in range(nwalkers)]

# We'll track how the average autocorrelation time estimate changes
index = 0

beta_profile =  (3**0.5)**np.arange(-20, 1)
steps_per_beta = np.ones(len(beta_profile), dtype=np.int64)*100
#steps_per_beta[-1] = 1000


directory = 'run%d_palpha_mcmc/event%d'%(run_number, event_num)
if not os.path.exists(directory):
    os.makedirs(directory)
with multiprocessing.Pool() as pool:
    for steps, beta in zip(steps_per_beta, beta_profile):
        print(steps, beta)
        if beta == beta_profile[0]:
            p = init_walker_pos
        else:
            p = sampler.get_chain()[-1,:,:]
        #reset phi to be between -pi and pi
        p = np.array(p)
        p[:,5] -= np.trunc(p[:, 5]/np.pi)*np.pi
        
        backend_file = os.path.join(directory, 'beta%f.h5'%(beta) )
        backend = emcee.backends.HDFBackend(backend_file)
        backend.reset(nwalkers, ndim)
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, backend=backend, pool=pool, args=(beta,))

        for sample in sampler.sample(p, iterations=steps, progress=True):
            tau = sampler.get_autocorr_time(tol=0)
            print('beta=', beta, 'iteration=', sampler.iteration, ', tau=', tau, ', accept fraction=', np.average(sampler.acceptance_fraction))

        samples = sampler.get_chain()
        labels = ['E', 'Ea_frac', 'x','y','z','theta', 'phi']
        fig, axes = plt.subplots(len(labels), figsize=(10, 7), sharex=True)#len(labels)
        plt.title('beta=%f'%beta)
        for i in range(len(labels)):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number")
        plt.savefig(os.path.join(directory, 'beta%f.png'%(  beta)))

        flat_samples = sampler.get_chain(discard=int(steps/2), flat=True)
        corner.corner(flat_samples, labels=labels)
        plt.savefig(os.path.join(directory, 'beta%f_corner_plot.png'%beta))

if False:
    #cluster log likelihood into two clusters, and pick out the most recent samples from the best cluster 
    ll_to_cluster = sampler.get_log_prob()[-1].reshape(-1,1)
    cluster_object = cluster.KMeans(2).fit(ll_to_cluster)
    clusters_to_propagate = cluster_object.labels_==np.argmax(cluster_object.cluster_centers_)
    samples_to_propagate = sampler.get_chain()[-1][clusters_to_propagate]
    new_init_pos = list(samples_to_propagate)

    #randomly select samples to perturb and add to the list until we have the desired number of walkers
    nwalkers = 100
    while len(new_init_pos) < nwalkers:
        i = np.random.randint(0, len(samples_to_propagate))
        new_init_pos.append(samples_to_propagate[1] + .001*np.random.randn(ndim))

    #restart mcmc
    backend_file = os.path.join(directory, 'after_clustering.h5')
    backend = emcee.backends.HDFBackend(backend_file)
    backend.reset(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, backend=backend, 
                                        moves=[
                                            (emcee.moves.StretchMove(), 0.5), 
                                            (emcee.moves.DEMove(), 0.4),
                                            (emcee.moves.DESnookerMove(), 0.1),
                                        ])
    for sample in sampler.sample(new_init_pos, iterations=steps, progress=True):
        tau = sampler.get_autocorr_time(tol=0)
        print('after clustering iteration=', sampler.iteration, ', tau=', tau, ', accept fraction=', np.average(sampler.acceptance_fraction))
