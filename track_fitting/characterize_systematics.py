'''
1. Find events to fit in different catagories
2. Do a least squares fit on the events, with the following free parameters:
    a. charge spreading
    b. x, y, z, E, theta, phi
    c. Pressure
3. Print mean and standard deviation of presssure and charge spreading?
4. MCMC charge spreading, pressure, gain match, and other systematics
'''
load_previous_fit = False

import time
import multiprocessing
import pickle
import os
# if not load_previous_fit:
#     os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
import scipy.optimize as opt

from track_fitting import SingleParticleEvent, build_sim

start_time = time.time()

m_guess, c_guess = 0,25 #guesses for pad gain match uncertainty and other systematics

run_number = 124
experiment = 'e21072'
pickle_fname = '%s_run%d_results_objects_m%d_c%d.dat'%(experiment,run_number, m_guess, c_guess)

h5file = build_sim.get_rawh5_object('e21072', run_number)

def fit_event(run, event, particle_type, trim_threshold=50, return_key=None, 
              return_dict=None, debug_plots=False):
    trace_sim = build_sim.create_single_particle_sim('e21072', run, event, particle_type)
    if trace_sim.num_trace_bins > 100:
        print('evt ', return_key, ' has %d bins, not fitting event since this is unexpected'%trace_sim.num_trace_bins)
        return 
    #want max likilihood to just be least squares for this fit
    trace_sim.pad_gain_match_uncertainty = m_guess
    trace_sim.other_systematics = c_guess
    #to get initial guess
    #Energy: from integrated charge
    #z: center of track
    #x,y,z: Find the pixel where the brag peak occured by finding the brightest pixel.
    #     Then find the pixel farthest from there.
    #theta, phi: from guessed decay location to brag peak
    #sigma_xy and sigma_z: just guess
    sigma_xy_guess = 4
    sigma_z_guess = 4


    #find guess for Brag peak location
    max_pad, max_val, max_index = 0,0,0
    for pad in trace_sim.traces_to_fit:
        trace = trace_sim.traces_to_fit[pad]
        this_max_index = np.argmax(trace)
        if trace[this_max_index] > max_val:
            max_pad, max_val, max_index = pad, trace[this_max_index], this_max_index
    brag_x, brag_y = trace_sim.pad_to_xy[max_pad]
    brag_z = trace_sim.zscale*max_index
    #find pad which fired which is furthest from the brag peak
    x_guess, y_guess = brag_x, brag_y

    furthest_dist_sqrd = 0
    for pad in trace_sim.traces_to_fit:
        trace = trace_sim.traces_to_fit[pad]
        x,y = trace_sim.pad_to_xy[pad]
        dist = (x-brag_x)**2 + (y - brag_y)**2
        if dist > furthest_dist_sqrd:
            furthest_dist_sqrd = dist
            x_guess, y_guess = x,y
            z_guess = np.argmax(trace)*trace_sim.zscale
            furthest_pad_trace = trace
    dz = brag_z - np.argmax(furthest_pad_trace)*trace_sim.zscale
    theta_guess = np.arctan2(np.sqrt(furthest_dist_sqrd), dz)
    phi_guess = np.arctan2(brag_y-y_guess, brag_x-x_guess)
    Eguess = build_sim.get_energy_from_ic(experiment, run, event)
    if debug_plots:
        print('num pads to simultate: ', len(trace_sim.pads_to_sim))
        print('max pad:', max_pad)
        print('brag x,y:', brag_x, brag_y)
        print('guess:',x_guess, y_guess, z_guess, np.degrees(theta_guess), 
              np.degrees(phi_guess), Eguess)
        trace_sim.theta, trace_sim.phi = theta_guess, phi_guess
        trace_sim.initial_point = (x_guess,y_guess,z_guess)
        trace_sim.initial_energy = Eguess
        trace_sim.sigma_xy = sigma_xy_guess
        trace_sim.sigma_z = sigma_z_guess
        trace_sim.simulate_event()
        trace_sim.plot_residuals()
        trace_sim.plot_residuals_3d(threshold=25)
        trace_sim.plot_simulated_3d_data(threshold=25)
        plt.show(block=True)

    def neg_log_likelihood(params):
        theta, phi, x,y,z, E, sigma_xy, sigma_z = params
        theta, phi = theta, phi
        x,y,z = x, y, z
        E = E
        if z > 400 or z<10:
            return np.inf
        if x**2 + y**2 > 40**2:
            return np.inf
        if E < 0 or E > 10: #stopping power tables currently only go to 10 MeV
            return np.inf
        if particle_type == 'proton' and E > 6:
            return np.inf #greater than 6 MeV protons always escape the detector at these pressures
        trace_sim.theta, trace_sim.phi = theta, phi
        trace_sim.initial_point = (x,y,z)
        trace_sim.sigma_xy = sigma_xy
        trace_sim.sigma_z = sigma_z
        trace_sim.initial_energy = E
        trace_sim.simulate_event()
        to_return = -trace_sim.log_likelihood()
        if debug_plots:
            print('%e'%to_return, params)
        if np.isnan(to_return):
            to_return = np.inf
        return to_return
    
    init_guess = (theta_guess, phi_guess, x_guess, y_guess, z_guess, Eguess, sigma_xy_guess, sigma_z_guess)
    
    res = opt.minimize(fun=neg_log_likelihood, x0=init_guess, method='BFGS')
    #if method == 'Nelder-Mead':
    #res = opt.minimize(fun=neg_log_likelihood, x0=init_guess, method="Nelder-Mead", options={'adaptive': True, 'maxfev':5000, 'maxiter':5000})
    #elif method == 'Powell':
    #res = opt.minimize(fun=neg_log_likelihood, x0=init_guess, method="Powell")#, options={'ftol':0.001, 'xtol':0.001})
    if return_dict != None:
        return_dict[return_key] = res
        print(return_key, res)
        print('total completed:', len(return_dict.keys()))
    if debug_plots:
        print(res)
        trace_sim.plot_residuals_3d(title=str(return_key)+particle_type, threshold=20)
        trace_sim.plot_simulated_3d_data(title=str(return_key)+particle_type, threshold=20)
        trace_sim.plot_residuals()
        plt.show()
    return res

if False: #try fitting one event to make sure it looks ok
    fit_event(124,108, 'proton', debug_plots=True)

events_in_catagory = [[],[],[],[]]
events_per_catagory = 50
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

if not load_previous_fit:
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
            if fit_in_parrallel:
                processes.append(multiprocessing.Process(target=fit_event, args=(run_number, n, particle_type, 50, n, fit_results_dict)))
                processes[-1].start()
            else:
                fit_event(run_number, n, particle_type, 50, n, fit_results_dict)
            events_in_catagory[event_catagory].append(n)
            print([len(x) for x in events_in_catagory])
        n += 1
    #wait for all processes to end
    for p in processes:
        p.join()
    
    print('fitting took %f s'%(time.time() - start_time))

    #save results objects
    fit_results_dict = {k:fit_results_dict[k] for k in fit_results_dict}
    with open(pickle_fname, 'wb') as f:
        pickle.dump(fit_results_dict, f)
else:
    with open(pickle_fname, 'rb') as f:
        fit_results_dict = pickle.load(f)
    events_in_catagory = [[],[],[],[]]
    for evt in fit_results_dict:
        max_veto_counts, dxy, dz, counts, angle, pads_railed = h5file.process_event(evt)
        l = np.sqrt(dxy**2 + dz**2)
        event_catagory = classify(l, counts)
        events_in_catagory[event_catagory].append(evt)

evts, thetas, phis,xs,ys,zs, lls, cats, Es, nfev, sigma_xys, sigma_zs = [], [],[],[],[],[],[],[],[],[],[],[]
for cat in range(len(events_in_catagory)):
    for evt in events_in_catagory[cat]:
        if evt not in fit_results_dict:
            print('evt %d (cat %d)not in results dict'%(evt, cat))
            continue
        res = fit_results_dict[evt]
        if not res.success and res.message != 'Desired error not necessarily achieved due to precision loss.':
            print('evt %d (cat %d)not succesfully fit: %s'%(evt, cat, res.message))
            continue
        thetas.append(res.x[0])
        phis.append(res.x[1])
        xs.append(res.x[2])
        ys.append(res.x[3])
        zs.append(res.x[4])
        Es.append(res.x[5])
        sigma_xys.append(res.x[6])
        sigma_zs.append(res.x[7])
        lls.append(res.fun)
        cats.append(cat)
        evts.append(evt)
        nfev.append(res.nfev)

lls = np.array(lls)
evts = np.array(evts)
cats = np.array(cats)

ptypes = ['proton' if cat in [2,3] else 'alpha' for cat in cats]

def show_fit(evt):
    i = np.where(evt==evts)[0][0]
    sim = build_sim.create_single_particle_sim(experiment, run_number, evt)
    sim.initial_energy = Es[i]
    sim.initial_point = (xs[i], ys[i], zs[i])
    sim.theta = thetas[i]
    sim.phi = phis[i]
    sim.sigma_xy = sigma_xys[i]
    sim.sigma_z = sigma_zs[i]
    sim.simulate_event()
    sim.plot_simulated_3d_data(threshold=25)
    sim.plot_residuals_3d(threshold=25)
    sim.plot_residuals()
    h5file.plot_3d_traces(evt, threshold=25)
    plt.show()

#pressure = 860.3#need to nail this down better later, for now assume current offset #np.mean(Pbest)
ll_thresh = [2*np.median(lls[cats==cat]) for cat in range(4)]

evts_to_fit = []
trace_sims = []
cats_to_fit = []
for i in range(len(evts)):
    #if lls[i] <= ll_cutoff[cats[i]]:
    #if i == 127:
    new_sim = build_sim.create_single_particle_sim(experiment, run_number, evts[i], ptypes[i])
    new_sim.sigma_xy = sigma_xys[i]
    new_sim.sigma_z = sigma_zs[i]
    new_sim.initial_energy = Es[i]
    new_sim.initial_point = (xs[i], ys[i], zs[i])
    new_sim.theta = thetas[i]
    new_sim.phi = phis[i]
    new_sim.pad_gain_match_uncertainty = m_guess
    new_sim.other_systematics = c_guess
    
    new_sim.simulate_event()
    pads, traces = h5file.get_pad_traces(evts[i], False)
    max_trace = np.max(traces)
    residuals_dict = new_sim.get_residuals()
    max_residual_percent = 0
    for pad in residuals_dict:
        val = np.max(np.abs(residuals_dict[pad]))/max_trace
        if  val > max_residual_percent:
            max_residual_percent = val
    print(evts[i], max_residual_percent)
    #only fit events with residuals no more than 40% of traces
    #and no more than 2x the median for the catagory
    if  new_sim.log_likelihood() < ll_thresh[cats[i]]:#higher energy proton #: 
        trace_sims.append(new_sim)
        evts_to_fit.append(evts[i])
        cats_to_fit.append(cats[i])
        #normalizations[evts[i]] = new_sim.log_likelihood()

print('num events to fit:', len(evts_to_fit))
print('catagories, counts:', np.unique(cats_to_fit, return_counts=True))

peak_residuals_fraction = []
peak_vals = []
peak_threshold = 400
for evt, sim in zip(evts_to_fit, trace_sims):
    simulated_traces = sim.sim_traces
    for pad in simulated_traces:
        if pad not in sim.traces_to_fit:
            continue
        sim_peak_index = np.argmax(simulated_traces[pad])
        act_peak_index = np.argmax(sim.traces_to_fit[pad])
        #don't use peaks if not well alligned
        if np.abs(sim_peak_index - act_peak_index) > 3: 
            continue
        sim_peak = simulated_traces[pad][sim_peak_index]
        observed_peak  = sim.traces_to_fit[pad][act_peak_index]
        if sim_peak >= peak_threshold and observed_peak > peak_threshold:
            peak_residuals_fraction.append(2*(sim_peak - observed_peak)/(sim_peak + observed_peak))
            peak_vals.append((sim_peak + observed_peak)/2)
peak_residuals_fraction = np.array(peak_residuals_fraction)
peak_vals = np.array(peak_vals)

pad_gain_match_uncertainty = np.std(peak_residuals_fraction)
print('gain match uncertainty: ', pad_gain_match_uncertainty)

min_trace_peak = np.inf
for sim in trace_sims:
    for pad in sim.traces_to_fit:
        x = np.max(sim.traces_to_fit[pad])
        if x < min_trace_peak and x > 0:
            min_trace_peak = x
print('suggested pad threshold = %f'%min_trace_peak)
        

def to_minimize(params):
    c = params[0]
    m = pad_gain_match_uncertainty
    to_return = 0
    for evt, sim in zip(evts_to_fit, trace_sims):
        sim.other_systematics = c
        sim.pad_gain_match_uncertainty = m
        to_add = -sim.log_likelihood()
        to_return += to_add
    print('==================',to_return, m, c, '===================')
    return to_return

if True:
    systematics_results = opt.minimize(to_minimize, (c_guess, ))
    other_systematics = systematics_results.x[0]
else:
    pad_gain_match_uncertainty,other_systematics = pad_gain_match_uncertainty, c_guess

'''
9bf15dc842c2ef3ec797b1ebdab942e48dc63a7b: After ll update, but with old pad threshold of 64. Gives m = 0.10459277119010803, c=24.99114302506084
'''

plt.figure()
plt.hist2d(peak_vals, peak_residuals_fraction, 100, norm=mpl.colors.LogNorm())
plt.xlabel('peak value (adc counts)')
plt.ylabel('peak residual fraction')
plt.colorbar()

plt.figure()
plt.hist2d(peak_vals, peak_residuals_fraction*peak_vals, 100, norm=mpl.colors.LogNorm())
plt.xlabel('peak value (adc counts)')
plt.ylabel('peak residual (adc counts)')
plt.colorbar()

dcounts = 100
counts = np.arange(150, 3050, dcounts)

mean_res, median_res, std_res = [],[],[]
mean_res_frac, median_res_frac, std_res_frac = [],[],[]
for c in counts:
    filter = (c -dcounts/2 < peak_vals) & (peak_vals < (c+dcounts/2))
    mean_res.append(np.mean(peak_vals[filter]*peak_residuals_fraction[filter]))
    median_res.append(np.median(peak_vals[filter]*peak_residuals_fraction[filter]))
    std_res.append(np.std(peak_vals[filter]*peak_residuals_fraction[filter]))
    mean_res_frac.append(np.mean(peak_residuals_fraction[filter]))
    median_res_frac.append(np.median(peak_residuals_fraction[filter]))
    std_res_frac.append(np.std(peak_residuals_fraction[filter]))

plt.figure()
plt.scatter(counts, mean_res, label='mean residuals')
plt.scatter(counts, median_res, label='median residuals')
plt.scatter(counts, std_res, label='residuals standard deviation')
plt.legend()

plt.figure()
plt.scatter(counts, mean_res_frac, label='mean residual fraction')
plt.scatter(counts, median_res_frac, label='median residual fraction')
plt.scatter(counts, std_res_frac, label='residual fraction standard deviation')
plt.legend()

plt.show()


for sim in trace_sims:
    sim.pad_gain_match_uncertainty = pad_gain_match_uncertainty
    sim.other_systematics = other_systematics

def plot_likelihood(sim:SingleParticleEvent.SingleParticleEvent, var:str, ds:np.array):
    x0 = sim.__dict__[var]
    xs = x0 + ds
    lls = []
    for x in xs:
        sim.__dict__[var] = x
        sim.simulate_event()
        lls.append(sim.log_likelihood())
    plt.figure()
    plt.scatter(ds, lls)
    plt.show()
    sim.__dict__[var] = x0#set it back when done