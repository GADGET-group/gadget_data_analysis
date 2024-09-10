import time
import numpy as np

import matplotlib.pylab as plt
import scipy.optimize as opt

from track_fitting import SingleParticleEvent
from raw_viewer import raw_h5_file


#folder = '/mnt/analysis/e21072/gastest_h5_files/'
folder = '../../shared/Run_Data/'
run_number = 124
event_num = 4
run_h5_path = folder +'run_%04d.h5'%run_number

init_by_priors = True
resume_previous_run = False

if folder == '/mnt/analysis/e21072/gastest_h5_files/':
    if run_number == 368:
        particle_type = 'alpha'
        adc_scale_mu = 371902/6.288 #counts/MeV, from fitting events with range 40-43 in run 0368 with p10_default
        detector_E_sigma = 19506/adc_scale_mu #sigma for above fit

        #use theoretical zscale
        clock_freq = 50e6 #Hz
        drift_speed = 54.4*1e6 #mm/s, from ruchi's paper
        zscale = drift_speed/clock_freq

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

        shaping_time = 117e-9 #s
        shaping_width  = shaping_time*clock_freq*2.355

        pressure = 1157
        if event_num == 5:
            init_position_guess = (-12, 13, 100)
            theta_guess = np.radians(90)
            phi_guess = np.radians(-30)
        elif event_num == 331:
            init_position_guess = (36, 1.5, 100)
            theta_guess = np.radians(50)
            phi_guess = np.radians(120)
        elif event_num == 12:
            init_position_guess = (28, 20, 100)
            theta_guess = np.radians(90+46)
            phi_guess = np.radians(160)
elif '../../shared/Run_Data/':#folder == '/mnt/analysis/e21072/h5test/':
    if run_number == 124:
        #TODO: make energy resolution energy dependent
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
        shaping_width  = shaping_time*clock_freq*2.355

        pressure = 860.3 #assuming current offset on MFC was present during experiment, and it was set to 800 torr
        charge_spread = 2. #mm, value used when fitting before

        #757 keV proton events
        if event_num == 4:
            #this one is very vertical
            particle_type = 'proton'
            init_position_guess = (-11, -3, 200)
            theta_guess = np.radians(0)
            phi_guess = np.radians(0)
        if event_num == 17:
            particle_type = 'proton'
            init_position_guess = (-21, 11, 200)
            theta_guess = np.radians(80)
            phi_guess = np.radians(-45)
        if event_num == 29:
            particle_type = 'proton'
            init_position_guess = (-29, -8, 200)
            theta_guess = np.radians(80)
            phi_guess = np.radians(-45)
        #1587 keV or 1574 keV protons
        if event_num == 55:
            #note: this one covers a pad which is missing pad mapping
            #which currently prevents the sim from running
            particle_type = 'proton'
            init_position_guess = (11, 15, 200)
            theta_guess = np.radians(36)
            phi_guess = np.radians(-180)
        if event_num == 108:
            particle_type = 'proton'
            init_position_guess = (0,0, 200)
            theta_guess = np.radians(82)
            phi_guess = np.radians(-45)
        if event_num == 132:
            particle_type = 'proton'
            init_position_guess = (3,-24, 200)
            theta_guess = np.radians(40)
            phi_guess = np.radians(200)

        
rho0 = 1.5256 #mg/cm^3, P10 at 300K and 760 torr
T = 20+273.15 #K
get_gas_density = lambda P: rho0*(P/760)*(300./T)
trace_sim = SingleParticleEvent.SingleParticleEvent(get_gas_density(pressure), particle_type)
trace_sim.shaping_width = shaping_width
trace_sim.zscale = zscale
trace_sim.counts_per_MeV = adc_scale_mu

trace_sim.simulate_event()
pads_to_fit, traces_to_fit = h5file.get_pad_traces(event_num, include_veto_pads=False)
trace_sim.set_real_data(pads_to_fit, traces_to_fit, trim_threshold=50)#match trim threshold used for systematics determination
trace_sim.align_pad_traces()

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

pad_gain_match_uncertainty = 2.978e+00
other_systematics = 8.315e+00

#do initial minimization before starting MCMC
def neg_log_likelihood_init_min(params):
    E, x, y, z, theta, phi = params
    #P = P_guess
    #z = init_position_guess[2]
    trace_sim.load_srim_table(particle_type, get_gas_density(P))
    trace_sim.initial_energy = E
    trace_sim.initial_point = (x,y,z)
    trace_sim.theta = theta
    trace_sim.phi = phi
    trace_sim.charge_spreading_sigma = charge_spread
    trace_sim.shaping_width = shaping_width
    
    trace_sim.simulate_event()
    trace_sim.align_pad_traces()
    to_return = -trace_sim.log_likelihood()
    print('E=%f MeV, (x,y,z)=(%f, %f, %f) mm, theta = %f deg, phi=%f deg, cs=%f mm, shaping=%f, P=%f torr, sigma=%f, LL=%e'%(E, x,y,z,np.degrees(theta), np.degrees(phi), charge_spread, shaping_width, P, likelihood_sigma, to_return))
    return to_return


fit_start_time = time.time()

initial_guess = (E_from_ic, *init_position_guess, theta_guess, phi_guess)

if not init_by_priors and not resume_previous_run:
    #get log likilihood within 0.1%
    res = opt.minimize(fun=neg_log_likelihood_init_min, x0=initial_guess, method="Powell", options={'disp':True})#, 'ftol':0.001, 'xtol':1})


    print(res)
    print('neg log likilihood: %e'%neg_log_likelihood_init_min(res.x))


    print('total fit time: %f s'%(time.time() - fit_start_time))

    trace_sim.plot_traces(trace_sim.traces_to_fit, 'clipped real traces')
    trace_sim.plot_traces(trace_sim.aligned_sim_traces, 'simulated traces')
    trace_sim.plot_residuals()

    trace_sim.plot_simulated_3d_data(mode='aligned', threshold=100)
    h5file.plot_3d_traces(event_num, threshold=100, block=False)
    trace_sim.plot_residuals_3d(energy_threshold=20)
    plt.show()
    
    if False: #if true, terminate after initial fit
        import sys
        sys.exit(0)

#do MCMC
import emcee

def log_likelihood_mcmc(params):
    E, x, y, z, theta, phi = params

    trace_sim.load_srim_table(particle_type, get_gas_density(pressure))
    trace_sim.initial_energy = E
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
    E, x, y, z, theta, phi = params
    
    #uniform priors
    if x**2 + y**2 > 40**2:
        return -np.inf
    if z < 0 or z >400:
        return -np.inf
    if theta < 0 or theta >= np.pi:
        return -np.inf 
    if shaping_width <=0 or shaping_width > 20:
        return -np.inf
    if charge_spread < 0:
        return -np.inf
    #gaussian prior for energy, and assume uniform over solid angle
    return E_prior.log_likelihood(E) + np.log(np.abs(np.sin(theta)))

beta = 0 #inverse temperature for tempering

def log_posterior(params):
    to_return = log_priors(params)
    if to_return != -np.inf:
        ll =  log_likelihood_mcmc(params)
        assert ll <= 0
        to_return -= np.abs(ll)**beta
    if np.isnan(to_return):
        to_return = -np.inf
    #print('log posterior: %e'%to_return)
    return to_return


nwalkers = 50
max_n = 5000
ndim = 6

if not resume_previous_run:
    if not init_by_priors:
        Efit, xfit, yfit, zfit, thetafit, phifit = res.x
        if thetafit < 0:
            thetafit = -thetafit
            phitfit = phifit + np.pi
            if phifit > 2*np.pi:
                phifit -= 2*np.pi
        start_pos = [Efit, xfit,yfit,zfit,thetafit, phifit]
        init_walker_pos =  [np.array(start_pos) + .001*np.random.randn(ndim) for i in range(nwalkers)]
    else:
        init_walker_pos = [[E_prior.mu + E_prior.sigma*np.random.randn(), np.random.uniform(xmin, xmax), 
                            np.random.uniform(ymin, ymax), np.random.uniform(zmin, zmax), np.random.uniform(0, np.pi), 
                            np.random.uniform(-np.pi, np.pi)] for i in range(nwalkers)]


if init_by_priors:
    backend_file = 'run%d_event%d_init_by_priors.h5'%(run_number, event_num) 
else:
    backend_file = 'run%d_event%d_init_by_best_fit.h5'%(run_number, event_num)
backend = emcee.backends.HDFBackend(backend_file)

if not resume_previous_run:
    backend.reset(nwalkers, ndim)

#sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, backend=backend)
if resume_previous_run:
    init_walker_pos = sampler.get_last_sample()


# We'll track how the average autocorrelation time estimate changes
index = 0

beta_profile = [0.1,0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
steps_per_beta = 100
for b in beta_profile:
    beta = b
    if b == beta_profile[0]:
        p = init_walker_pos
    else:
        p = sampler.get_chain()[-1,:,:]
    #print(p)
    #print(np.shape(p))
    

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)

    for sample in sampler.sample(p, iterations=steps_per_beta, progress=True):
        tau = sampler.get_autocorr_time(tol=0)
        print('beta=', beta, 'iteration=', sampler.iteration, ', tau=', tau, ', accept fraction=', np.average(sampler.acceptance_fraction))

    samples = sampler.get_chain()
    labels = ['E', 'x','y','z','theta', 'phi']
    fig, axes = plt.subplots(len(labels), figsize=(10, 7), sharex=True)#len(labels)
    plt.title('beta=%f'%beta)
    for i in range(len(labels)):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")

    plt.show(block=True)


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
