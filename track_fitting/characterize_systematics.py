'''
1. Find events to fit in different catagories
2. Do a least squares fit on the events, with the following free parameters:
    a. charge spreading
    b. x, y, z, E, theta, phi
    c. Pressure
3. Print mean and standard deviation of presssure and charge spreading?
4. MCMC charge spreading, pressure, gain match, and other systematics
'''

import time
import multiprocessing
import pickle

import numpy as np
import matplotlib.pylab as plt
import scipy.optimize as opt
import emcee

from track_fitting import SingleParticleEvent
from raw_viewer import raw_h5_file


start_time = time.time()

folder = '../../shared/Run_Data/'
run_number = 124
run_h5_path = folder +'run_%04d.h5'%run_number

adc_scale_mu = 86431./0.757 #counts/MeV, from fitting events with range 40-43 in run 0368 with p10_default
detector_E_sigma = lambda E: (5631./adc_scale_mu)*np.sqrt(E/0.757) #sigma for above fit, scaled by sqrt energy

#use theoretical zscale
clock_freq = 50e6 #Hz, from e21062 config file on mac minis
drift_speed = 54.4*1e6 #mm/s, from ruchi's paper
zscale = drift_speed/clock_freq

h5file = raw_h5_file.raw_h5_file(file_path=run_h5_path,
                                zscale=0.9, #use same zscale as was used for cut
                                flat_lookup_csv='raw_viewer/channel_mappings/flatlookup4cobos.csv')
h5file.length_counts_threshold=100
h5file.ic_counts_threshold = 25
h5file.background_subtract_mode='fixed window'
h5file.data_select_mode='near peak'
h5file.remove_outliers=True
h5file.near_peak_window_width = 50
h5file.require_peak_within= (-np.inf, np.inf)
h5file.num_background_bins=(160, 250)
h5file.ic_counts_threshold = 25

shaping_time = 70e-9 #s, from e21062 config file on mac minis
shaping_width  = shaping_time*clock_freq*2.355

#assuming current offset on MFC was present during experiment, and it was set to 800 torr
#which seems to be a good assumtion. See c48097904a89edc1131d7aa2d216ca6d045b5137
Pguess = 800 #torr

rho0 = 1.5256 #mg/cm^3, P10 at 300K and 760 torr
T = 20+273.15 #K
get_gas_density = lambda P: rho0*(P/760)*(300./T)

#factors to scale parameters by, so they are all of order 1
angle_scale = 3
distance_scale = 50
e_scale = 2
p_scale = 700
cs_scale = 1

def fit_event(pads_to_fit, traces_to_fit, particle_type, trim_threshold=50, return_key=None, return_dict=None, debug_plots=False, method='Powell'):
    trace_sim = SingleParticleEvent.SingleParticleEvent(get_gas_density(Pguess), particle_type)
    trace_sim.shaping_width = shaping_width
    trace_sim.zscale = zscale
    trace_sim.counts_per_MeV = adc_scale_mu
    trace_sim.set_real_data(pads_to_fit, traces_to_fit, trim_threshold=trim_threshold, trim_pad=int(shaping_width))
    #want max likilihood to just be least squares for this fit
    trace_sim.pad_gain_match_uncertainty = 0
    trace_sim.other_systematics = 1
    #to get initial guess
    #Energy: from integrated charge
    #z: center of the detector
    #x,y: Find the pixel where the brag peak occured by finding the brightest pixel.
    #     Then find the pixel farthest from there.
    #theta, phi: from guessed decay location to brag peak
    Eguess = np.sum(traces_to_fit)/adc_scale_mu
    #find guess for Brag peak location
    max_pad, max_val, max_index = 0,0,0
    for pad, trace in zip(pads_to_fit, traces_to_fit):
        this_max_index = np.argmax(trace)
        if trace[this_max_index] > max_val:
            max_pad, max_val, max_index = pad, trace[this_max_index], this_max_index
    brag_x, brag_y = trace_sim.pad_to_xy[max_pad]
    brag_z = zscale*max_index
    #find pad which fired which is furthest from the brag peak
    x_guess, y_guess = brag_x, brag_y

    furthest_dist_sqrd = 0
    for pad, trace in zip(pads_to_fit, traces_to_fit):
        x,y = trace_sim.pad_to_xy[pad]
        dist = (x-brag_x)**2 + (y - brag_y)**2
        if dist > furthest_dist_sqrd:
            furthest_dist_sqrd = dist
            x_guess, y_guess = x,y
            furthest_pad_trace = trace
    dz = brag_z - np.argmax(furthest_pad_trace)*zscale
    theta_guess = np.arctan2(np.sqrt(furthest_dist_sqrd), dz)
    phi_guess = np.arctan2(brag_y-y_guess, brag_x-x_guess)
    if debug_plots:
        print('max pad:', max_pad)
        print('brag x,y:', brag_x, brag_y)
        z_guess, charge_spead_guess = 200, 1
        print('guess:',x_guess, y_guess, z_guess, theta_guess, phi_guess, Eguess, Pguess)
        trace_sim.theta, trace_sim.phi = theta_guess, phi_guess
        trace_sim.initial_point = (x_guess,y_guess,z_guess)
        trace_sim.charge_spreading_sigma = charge_spead_guess
        trace_sim.initial_energy = Eguess
        trace_sim.load_srim_table(particle=particle_type, gas_density=get_gas_density(Pguess))
        trace_sim.simulate_event()
        trace_sim.align_pad_traces()
        trace_sim.plot_residuals()
        trace_sim.plot_residuals_3d(energy_threshold=25)
        trace_sim.plot_simulated_3d_data(threshold=25)
        plt.show(block=True)

    def neg_log_likelihood(params):
        theta, phi, x,y,z, E, P, charge_spread = params
        theta, phi = theta*angle_scale, phi*angle_scale
        x,y,z = x*distance_scale, y*distance_scale, z*distance_scale
        P = P*p_scale
        E = E*e_scale
        charge_spread = charge_spread*cs_scale
        if z > 400 or z<10:
            return np.inf
        if x**2 + y**2 > 40**2:
            return np.inf
        if charge_spread <0 or charge_spread>10:
            return np.inf
        if P < 700 or P > 1000:
            return np.inf
        if E < 0 or E > 10: #stopping power tables currently only go to 10 MeV
            return np.inf
        if particle_type == 'proton' and E > 6:
            return np.inf #greater than 6 MeV protons always escape the detector at these pressures
        trace_sim.theta, trace_sim.phi = theta, phi
        trace_sim.initial_point = (x,y,z)
        trace_sim.charge_spreading_sigma = charge_spread
        trace_sim.initial_energy = E
        trace_sim.load_srim_table(particle=particle_type, gas_density=get_gas_density(P))
        trace_sim.simulate_event()
        trace_sim.align_pad_traces()
        to_return = -trace_sim.log_likelihood()
        if debug_plots:
            print('%e'%to_return, params)
        return to_return
    
    init_guess = (theta_guess/angle_scale, phi_guess/angle_scale, x_guess/distance_scale, y_guess/distance_scale, 200/distance_scale, Eguess/e_scale, Pguess/p_scale,1/cs_scale)
    if method == 'Nelder-Mead':
        res = opt.minimize(fun=neg_log_likelihood, x0=init_guess, method="Nelder-Mead", options={'adaptive': True, 'maxfev':10000, 'maxiter':10000})
    elif method == 'Powell':
        res = opt.minimize(fun=neg_log_likelihood, x0=init_guess, method="Powell", options={'ftol':0.001, 'xtol':0.01})#, options={'maxiter':5, 'disp':True})
    else:
        assert False
    if return_dict != None:
        return_dict[return_key] = res
        print(return_key, res)
        print('total completed:', len(return_dict.keys()))
    if debug_plots:
        print(res)
        trace_sim.plot_residuals_3d(title=str(return_key)+particle_type, energy_threshold=20)
        trace_sim.plot_simulated_3d_data(title=str(return_key)+particle_type, threshold=20)
        trace_sim.plot_residuals()
        plt.show()
    return res

#55, 108, 132
#pads, traces = h5file.get_pad_traces(108, False)
#fit_event(pads, traces, 'proton', debug_plots=True)

events_in_catagory = [[],[],[],[]]
events_per_catagory = 5
processes = []

def classify(range, counts):
    if counts > 2e5 and counts < 4e5 and range < 25 and range > 15:
        return 0 #low energy alpha
    if counts > 4e5 and counts < 1e6 and range >20 and range < 100:
        return 1 #high energy alpha
    if counts > 7e4 and counts < 1e5 and range > 13 and range < 22:
        return 2 #757 keV proton
    if counts > 1.54e5 and counts < 2.1e5 and range > 34 and range < 55:
        return 3 #higher energy proton
    return -1 

veto_threshold = 300

fit_in_parrallel = True

n = h5file.get_event_num_bounds()[0]
manager = multiprocessing.Manager()
fit_results_dict = manager.dict()
while np.min([len(x) for x in events_in_catagory]) < events_per_catagory:
    max_veto_counts, dxy, dz, counts, angle, pads_railed = h5file.process_event(n)
    l = np.sqrt(dxy**2 + dz**2)
    event_catagory = classify(l, counts)
    #print(n, event_catagory, counts, l, max_veto_counts < veto_threshold )
    if max_veto_counts < veto_threshold and event_catagory >= 0 and len(events_in_catagory[event_catagory]) < events_per_catagory:
        if event_catagory in [0, 1]:
            particle_type = 'alpha'
        else:
            particle_type = 'proton'
        pads, traces  = h5file.get_pad_traces(n, include_veto_pads=False)
        if fit_in_parrallel:
            processes.append(multiprocessing.Process(target=fit_event, args=(pads, traces, particle_type, 50, n, fit_results_dict)))
            processes[-1].start()
        else:
            fit_event(pads, traces, particle_type, 50, n, fit_results_dict)
        events_in_catagory[event_catagory].append(n)
        print([len(x) for x in events_in_catagory])
    n += 1
#wait for all processes to end
for p in processes:
    p.join()

#save results objects
pickle_fname = 'run%d_results_objects.dat'%run_number
with open(pickle_fname, 'wb') as f:
    pickle.dump(fit_results_dict, f)
    

print('fitting took %f s'%(time.time() - start_time))

evts, thetas, phis,xs,ys,zs, charge_spreads, lls, cats, Es, Ps, nfev = [], [],[],[],[],[],[],[],[],[],[],[]
for cat in range(len(events_in_catagory)):
    for evt in events_in_catagory[cat]:
        if evt not in fit_results_dict:
            print('evt %d (cat %d)not in results dict'%(evt, cat))
            continue
        res = fit_results_dict[evt]
        if not res.success:
            print('evt %d (cat %d)not succesfully fit'%(evt, cat))
            continue
        thetas.append(res.x[0]*angle_scale)
        phis.append(res.x[1]*angle_scale)
        xs.append(res.x[2]*distance_scale)
        ys.append(res.x[3]*distance_scale)
        zs.append(res.x[4]*distance_scale)
        Es.append(res.x[5]*e_scale)
        Ps.append(res.x[6]*p_scale)
        charge_spreads.append(res.x[7]*cs_scale)
        lls.append(res.fun)
        cats.append(cat)
        evts.append(evt)
        nfev.append(res.nfev)

lls = np.array(lls)
Ps = np.array(Ps)
evts = np.array(evts)
cats = np.array(cats)

Pgood = Ps[(cats==1)|(cats==3)]
Pbest = Pgood[(np.abs(Pgood - np.mean(Pgood)) < np.std(Pgood))]
ptypes = ['proton' if cat in [2,3] else 'alpha' for cat in cats]

#save results arrays

def show_fit(evt):
    i = np.where(evt==evts)[0][0]
    sim=SingleParticleEvent.SingleParticleEvent(get_gas_density(Ps[i]), ptypes[i])
    sim.initial_energy = Es[i]
    sim.initial_point = (xs[i], ys[i], zs[i])
    sim.theta = thetas[i]
    sim.phi = phis[i]
    sim.charge_spreading_sigma = charge_spreads[i]
    sim.zscale = zscale
    sim.shaping_width = shaping_width
    sim.counts_per_MeV = adc_scale_mu
    pads, traces = h5file.get_pad_traces(evt, False)
    sim.set_real_data(pads, traces, 50, int(shaping_width))
    sim.simulate_event()
    sim.align_pad_traces()
    sim.plot_simulated_3d_data(mode='aligned', threshold=25)
    sim.plot_residuals_3d(energy_threshold=25)
    sim.plot_residuals()
    plt.show()

pressure = 860.3#need to nail this down better later, for now assume current offset #np.mean(Pbest)


ll_cutoff = [np.median(lls[cats==cat]) for cat in [0,1,2,3]]
evts_to_fit = []
trace_sims = []
for i in range(len(evts)):
    if lls[i] <= ll_cutoff[cats[i]]:
        evts_to_fit.append(evts[i])
        new_sim = SingleParticleEvent.SingleParticleEvent(get_gas_density(pressure), particle=ptypes[i])
        new_sim.shaping_width = shaping_width
        new_sim.zscale = zscale
        new_sim.counts_per_MeV = adc_scale_mu
        new_sim.initial_energy = Es[i]
        new_sim.initial_point = (xs[i], ys[i], zs[i])
        new_sim.theta = thetas[i]
        new_sim.phi = phis[i]
        pads, traces = h5file.get_pad_traces(evts[i], False)
        new_sim.set_real_data(pads, traces, trim_threshold=50)
        trace_sims.append(new_sim)

#cs_mu, cs_sigma = np.mean(charge_spreads), np.std(charge_spreads)
#now do MCMC to characterize systematics
def log_priors(params):
    #uninformed priors, just require each parameter to be >=0
    m, c, charge_spread = params
    if m < 0 or c < 0  or m > 10 or c>4000 or charge_spread < 0 or charge_spread > 10:
        return -np.inf
    return 0

def log_likelihood(params):
    m, c, charge_spread = params
    to_return = 0
    for sim in trace_sims:
        sim.charge_spreading_simga = charge_spread
        sim.pad_gain_match_uncertainty = m
        sim.other_systematics = c
        sim.simulate_event()
        sim.align_pad_traces()
        to_return += sim.log_likelihood()
    return to_return

def log_posterior(params):
    prior =  log_priors(params)
    if prior == -np.inf: #don't bother simulating if -inf anyway
        return prior
    to_return =  log_likelihood(params) + prior
    if np.isnan(to_return):
        to_return = -np.inf
    print(params, '%e'%to_return)
    return to_return

systematics_fit = opt.minimize(lambda params: -log_posterior(params), (0.3, 20, 2))


if False:
    #turn off numpy threading to avoid conflicts with emcee
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    import emcee



    from multiprocessing import Pool
    nwalkers = 50
    nsteps=500
    ndim = 3

    backend_file = 'run%d_systematics.h5'%run_number
    backend = emcee.backends.HDFBackend(backend_file)
    backend.reset(nwalkers, ndim)

    emcee_start_time = time.time()
    initial = [(np.random.uniform(0,1), np.random.uniform(0,1000), np.random.uniform(0,10)) for x in range(nwalkers)]
    with Pool(50) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, pool=pool, backend=backend)
        sampler.run_mcmc(initial, nsteps, progress=True)
    print('emcee took %f s'%(time.time() - emcee_start_time))