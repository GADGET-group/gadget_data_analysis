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
event_num = 68129
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

def log_priors(params, phi_lim):
    E, Ea_frac, x, y, z, theta, phi = params
    
    #uniform priors
    if x**2 + y**2 > 40**2:
        return -np.inf
    if z < 0 or z >400:
        return -np.inf
    if theta < 0 or theta >= np.pi:# or phi < -np.pi or phi>np.pi:
        return -np.inf
    if phi < phi_lim[0] or phi>phi_lim[1]:
        return -np.inf 
    if shaping_width <=0 or shaping_width > 20:
        return -np.inf
    if charge_spread < 0:
        return -np.inf
    if Ea_frac < 0 or Ea_frac > 1:
        return -np.inf
    #gaussian prior for energy, and assume uniform over solid angle
    return E_prior.log_likelihood(E) + np.log(np.abs(np.sin(theta)))


def log_posterior(params, phi_lim):
    to_return = log_priors(params, phi_lim)
    if to_return != -np.inf:
        to_return +=  log_likelihood_mcmc(params)
    if np.isnan(to_return):
        to_return = -np.inf
    #print('log posterior: %e'%to_return)
    return to_return


nwalkers = 200
max_n = 5000
ndim = 7

# We'll track how the average autocorrelation time estimate changes
directory = 'run%d_palpha_mcmc/event%d'%(run_number, event_num)
if not os.path.exists(directory):
    os.makedirs(directory)

def do_mcmc(init_pos, steps, save_name, phi_lim=(-np.pi, np.pi)):
    with multiprocessing.Pool(nwalkers) as pool:
        backend_file = os.path.join(directory, '%s.h5'%(save_name) )
        backend = emcee.backends.HDFBackend(backend_file)
        backend.reset(nwalkers, ndim)
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, backend=backend, pool=pool, args=(phi_lim,))

        for sample in sampler.sample(init_pos, iterations=steps, progress=True):
            tau = sampler.get_autocorr_time(tol=0)
            print(save_name, 'iteration=', sampler.iteration, ', tau=', tau, ', accept fraction=', np.average(sampler.acceptance_fraction))

        samples = sampler.get_chain()
        labels = ['E', 'Ea_frac', 'x','y','z','theta', 'phi']
        fig, axes = plt.subplots(len(labels), figsize=(10, 7), sharex=True)#len(labels)
        for i in range(len(labels)):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number")
        plt.savefig(os.path.join(directory, '%s.png'%(save_name)))

        flat_samples = sampler.get_chain(discard=int(steps/2), flat=True)
        corner.corner(flat_samples, labels=labels)
        plt.savefig(os.path.join(directory, '%s_corner_plot.png'%save_name))

    return emcee.backends.HDFBackend(filename=backend_file, read_only=True)

#run 1000 samples
init_walker_pos = [[E_prior.mu + E_prior.sigma*np.random.randn(), np.random.uniform(0,1),np.random.uniform(xmin, xmax), 
                            np.random.uniform(ymin, ymax), np.random.uniform(zmin, zmax), np.random.uniform(0, np.pi), 
                            np.random.uniform(-np.pi, np.pi)] for i in range(nwalkers)]
init_run = do_mcmc(init_walker_pos, 1000, 'initial_run')

samples = init_run.get_chain()
log_prob = init_run.get_log_prob()
thetas = samples[-1][:, -2]
phis = samples[-1][:, -1]
plt.figure()
plt.title("before clustering")
plt.scatter(np.degrees(thetas), np.degrees(phis), c=log_prob[-1])
plt.colorbar(label="log prob")
plt.xlabel('theta (deg)')
plt.ylabel('phi (deg)')
plt.savefig(os.path.join(directory,'before_clustering.png'))

#cluster by direction vector, to avoid issues at phi=0/pi
#keep all clusters of size >10
zhat = np.cos(thetas)
xhat = np.sin(thetas)*np.cos(phis)
yhat = np.sin(thetas)*np.sin(phis)
cluster_obj = cluster.DBSCAN(0.1).fit(np.vstack((xhat, yhat, zhat)).T)
cluster_label, cluster_counts = np.unique(cluster_obj.labels_, return_counts=True)
clusters_to_keep = cluster_label[(cluster_label>=0) & (cluster_counts>10)]
to_keep = np.in1d(cluster_obj.labels_, clusters_to_keep)
plt.figure()
plt.title("clusters to fit")
plt.scatter(np.degrees(thetas)[to_keep], np.degrees(phis)[to_keep], c=cluster_obj.labels_[to_keep])
plt.colorbar(label="cluster id")
plt.xlabel('theta (deg)')
plt.ylabel('phi (deg)')
plt.savefig(os.path.join(directory,'clusters.png'))

starting_points = [] #list of initial walker positions for each cluster
phi_limits = []
for c in clusters_to_keep:
    this_cluster = cluster_obj.labels_ == c
    plt.figure()
    plt.title("cluster %d"%c)
    plt.scatter(np.degrees(thetas[this_cluster]), np.degrees(phis[this_cluster]), c=log_prob[-1][this_cluster])
    plt.colorbar(label="log prob")
    plt.xlabel('theta (deg)')
    plt.ylabel('phi (deg)')

    #add slightly perturbed data points until init points has required number
    samples_in_cluster = samples[-1][this_cluster]
    init_points = list(samples_in_cluster)
    while len(init_points) < nwalkers:
        random_point = samples_in_cluster[np.random.randint(0, len(samples_in_cluster))]
        init_points.append(random_point + .001*np.random.randn(ndim))
    starting_points.append(init_points)
    phi = init_points[0][-1]
    phi_limits.append((phi - np.pi, phi+np.pi))

#run sampler on each cluster
for i in range(len(clusters_to_keep)):
    do_mcmc(init_pos=starting_points[i],
            steps=1000, save_name='cluster%d'%clusters_to_keep[i], 
            phi_lim=phi_limits[i])