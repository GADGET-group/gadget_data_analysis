import time

#import os
#os.environ['OPENBLAS_NUM_THREADS'] = '3'
import numpy as np

import matplotlib.pylab as plt
import scipy.optimize as opt

from track_fitting import SingleParticleEvent
from raw_viewer import raw_h5_file

run_h5_path = '/mnt/analysis/e21072/gastest_h5_files/run_0368.h5'
event_num = 5

if event_num == 5:
    E_guess = 6.212
    init_position_guess = (-12, 13, 50)
    charge_spreading_guess = 3
    theta_guess = np.radians(90)
    phi_guess = np.radians(-30)
    P_guess = 1157

adc_scale_mu = 371902/6.288 #counts/MeV, from fitting events with range 40-43 in run 0368 with p10_default
adc_scale_sigma = 1401/6.288 #uncertainty in peak postiion from chi^2 fit
detector_E_sigma = 19506/adc_scale_mu #sigma for above fit

#use theoretical zscale
clock_freq = 50e6 #Hz
drift_speed = 54.4*1e6 #mm/s, from ruchi's paper
zscale = drift_speed/clock_freq

shaping_time = 117e-9 #s
shaping_width  = shaping_time*clock_freq*2.355

ic_threshold = 25


h5file = raw_h5_file.raw_h5_file(file_path=run_h5_path,
                                  zscale=zscale,
                                  flat_lookup_csv='raw_viewer/channel_mappings/flatlookup2cobos.csv')
h5file.background_subtract_mode='fixed window'
h5file.data_select_mode='near peak'
h5file.remove_outliers=True
h5file.near_peak_window_width = 50
h5file.require_peak_within= (-np.inf, np.inf)
h5file.num_background_bins=(400,500)

rho0 = 1.5256 #mg/cm^3, P10 at 300K and 760 torr
T = 20+273.15 #K
get_gas_density = lambda P: rho0*(P/760)*(300./T)
trace_sim = SingleParticleEvent.SingleParticleEvent(get_gas_density(P_guess), 'alpha')
trace_sim.shaping_width = shaping_width
trace_sim.zscale = zscale
trace_sim.counts_per_MeV = adc_scale_mu

trace_sim.initial_energy = E_guess
trace_sim.phi = phi_guess
trace_sim.theta = theta_guess
trace_sim.initial_point = init_position_guess

trace_sim.simulate_event()
pads_to_fit, traces_to_fit = h5file.get_pad_traces(event_num, include_veto_pads=False)
trace_sim.set_real_data(pads_to_fit, traces_to_fit, fit_threshold=ic_threshold, trim_pad = 20)
trace_sim.align_pad_traces()




#do initial minimization before starting MCMC
def neg_log_likelihood_init_min(params):
    E, x, y, theta, phi, charge_spread, shaping_width = params
    P = P_guess
    z = init_position_guess[2]
    trace_sim.load_srim_table('alpha', get_gas_density(P))
    trace_sim.initial_energy = E
    trace_sim.initial_point = (x,y,z)
    trace_sim.theta = theta
    trace_sim.phi = phi
    trace_sim.charge_spreading_sigma = charge_spread
    trace_sim.shaping_width = shaping_width
    trace_sim.simulate_event()
    trace_sim.align_pad_traces()
    to_return = -trace_sim.log_likelihood()
    print('E=%f MeV, (x,y,z)=(%f, %f, %f) mm, theta = %f deg, phi=%f deg, cs=%f mm, shaping=%f, P=%f torr,  LL=%e'%(E, x,y,z,np.degrees(theta), np.degrees(phi), charge_spread, shaping_width, P, to_return))
    return to_return


fit_start_time = time.time()

initial_guess = (E_guess, *init_position_guess[0:2], theta_guess, phi_guess, charge_spreading_guess, shaping_width)

#get log likilihood within 0.1%
'''res = opt.minimize(fun=neg_log_likelihood_init_min, x0=initial_guess, method="Powell", options={'disp':True, 'ftol':0.01, 'xtol':1})


print(res)
neg_log_likelihood_init_min(res.x)


print('total fit time: %f s'%(time.time() - fit_start_time))

plot_traces(trace_sim.traces_to_fit, 'clipped real traces')
plot_traces(trace_sim.aligned_sim_traces, 'simulated traces')
plot_residuals()
plot_residuals_3d()

show_simulated_3d_data(mode='aligned', threshold=100)
h5file.plot_3d_traces(event_num, threshold=100)
'''


#do MCMC
import emcee

shaping_best_fit = 10.126
charge_spread_best_fit = 4.179261

class GaussianVar:
    def __init__(self, mu, sigma):
        self.mu, self.sigma  = mu, sigma

    def log_likelihood(self, val):
        return -np.log(np.sqrt(2*np.pi*self.sigma**2)) - (val - self.mu)**2/2/self.sigma

adc_scale_prior = GaussianVar(adc_scale_mu, adc_scale_sigma)
max_veto_pad_counts, dxy, dz, measured_counts, angle, pads_railed = h5file.process_event(event_num)
E_from_ic = measured_counts/adc_scale_mu
E_prior = GaussianVar(E_from_ic, detector_E_sigma)
P_prior = GaussianVar(P_guess, P_guess*0.01)#assumes pressure transducer accuracy of 1%, should check what this really should be
x_real, y_real, z_real, e_real = h5file.get_xyze(event_number=event_num)
xmin, xmax = np.min(x_real), np.max(x_real)
ymin, ymax = np.min(y_real), np.max(y_real)
zmin, zmax = 5, 400

def log_likelihood_mcmc(params):
    E, x,y,theta, phi = params

    z,P, adc_scale = 50., P_guess, adc_scale_mu
    charge_spread, shaping_width = charge_spread_best_fit, shaping_best_fit

    trace_sim.load_srim_table('alpha', get_gas_density(P))
    trace_sim.initial_energy = E
    trace_sim.initial_point = (x,y,z)
    trace_sim.theta = theta
    trace_sim.phi = phi
    trace_sim.charge_spreading_sigma = charge_spread
    trace_sim.shaping_width = shaping_width
    trace_sim.counts_per_MeV = adc_scale
    trace_sim.simulate_event()
    trace_sim.align_pad_traces()
    to_return = trace_sim.log_likelihood()
    #print('E=%f MeV, (x,y,z)=(%f, %f, %f) mm, theta = %f deg, phi=%f deg, cs=%f mm, shaping=%f, P=%f torr, adc_scale=%f, LL=%e'%(E, x,y,z,np.degrees(theta), np.degrees(phi), charge_spread, shaping_width, P, adc_scale, to_return))
    return to_return




def log_priors(params):
    E, x,y,theta, phi = params
    #uniform priors
    if x < xmin or x > xmax or y < ymin or y > ymax:
        print('fail1')
        return -np.inf
    if theta < 0 or theta >= np.pi:
        print('fail2')
        return -np.inf 
    if phi <= -np.pi or phi > np.pi:
        print('fail3')
        return -np.inf
    if shaping_width <=0 or shaping_width > 20:
        print('fail4')
        return -np.inf
    #gaussian priors
    return E_prior.log_likelihood(E) 

def log_posterior(params):
    to_return = log_priors(params) + log_likelihood_mcmc(params)
    if np.isnan(to_return):
        to_return = -np.inf
    print('log posterior: %e'%to_return)
    return to_return
    

#use previous optimization for start pos
#E=6.496048 MeV, (x,y,z)=(-12.865501, 12.899337, 50.000000) mm, theta = 86.718415 deg, phi=-29.475943 deg, cs=4.179261 mm, shaping=10.126000, P=1157.000000 torr,  LL=7.633177e+06

start_pos = [6.496048, -12.8865501,12.89937,np.radians(86.718415), np.radians(-29.475943)]
nwalkers = 125
ndim = 5
#init_walker_pos =  [np.array(start_pos) + .001*np.random.randn(ndim) for i in range(nwalkers)]
init_walker_post = [(E_prior.mu + E_prior.sigma*np.random.randn(), np.random.uniform(xmin, xmax), np.random.uniform]

backend_file = "run368_event%d_samples_E_x_y_theta_phi.h5"%(event_num)
backend = emcee.backends.HDFBackend(backend_file)
backend.reset(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, backend=backend)
max_n = 100000
# We'll track how the average autocorrelation time estimate changes
index = 0
autocorr = np.empty(max_n)
# This will be useful to testing convergence
old_tau = np.inf
# Now we'll sample for up to max_n steps
for sample in sampler.sample(init_walker_pos, iterations=max_n, progress=True):
    # Only check convergence every 10 steps
    #if sampler.iteration % 100:
    #    continue

    # Compute the autocorrelation time so far
    # Using tol=0 means that we'll always get an estimate even
    # if it isn't trustworthy
    tau = sampler.get_autocorr_time(tol=0)
    autocorr[index] = np.mean(tau)
    index += 1
    print('iteration=', sampler.iteration, ', tau=', tau, ', accept fraction=', np.average(sampler.acceptance_fraction))

    # Check convergence
    converged = np.all(tau * 100 < sampler.iteration)
    #converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
    if converged:
        break
    old_tau = tau


#sampler.run_mcmc(init_walker_pos, 100, progress=True)

samples = sampler.get_chain()
labels = ['E', 'x','y','z','theta', 'phi', 'charge_spread', 'shaping_width', 'P', 'adc_scale', 'logf']
fig, axes = plt.subplots(len(labels), figsize=(10, 7), sharex=True)#len(labels)
for i in range(len(labels)):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")

import corner
tau = sampler.get_autocorr_time()
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
log_prior_samples = sampler.get_blobs(discard=burnin, flat=True, thin=thin)

print("burn-in: {0}".format(burnin))
print("thin: {0}".format(thin))
print("flat chain shape: {0}".format(samples.shape))
print("flat log prob shape: {0}".format(log_prob_samples.shape))
print("flat log prior shape: {0}".format(log_prior_samples.shape))

all_samples = np.concatenate(
    (samples, log_prob_samples[:, None], log_prior_samples[:, None]), axis=1
)

corner.corner(all_samples, labels=labels)
plt.show()