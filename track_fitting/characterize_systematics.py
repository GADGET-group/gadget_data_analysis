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


run_number = 124
experiment = 'e24joe'

m_guess, c_guess = 0.1004, 22.5
use_likelihood = False #if false, uses least squares
fit_adc_count_per_MeV = False #use known energy rather than fitting it as a free parameter, and instead fit adc_counts_per_MeV
fix_energy = False
if use_likelihood:
    pickle_fname = '%s_run%d_m%f_c%f_results_objects.dat'%(experiment,run_number, m_guess, c_guess)
else:
    if fit_adc_count_per_MeV:
        pickle_fname = '%s_run%d_adc_counts_free.dat'%(experiment,run_number)
    else:
        if fix_energy:
            pickle_fname = '%s_run%d_results_objects.dat'%(experiment,run_number)
        else:
            pickle_fname = '%s_run%d_energy_free.dat'%(experiment,run_number)
    

h5file = build_sim.get_rawh5_object(experiment, run_number)




def fit_event(run, event, particle_type, include_recoil, direction, Eknown, return_key=None, 
              return_dict=None, debug_plots=False):
    if include_recoil:
        if particle_type == '1H': 
            recoil_name, recoil_mass, product_mass = '19Ne', 19, 1
        elif particle_type == '4He':
            recoil_name, recoil_mass, product_mass = '16O', 16, 4
        trace_sim = build_sim.create_multi_particle_decay(experiment, run, event, [particle_type], [product_mass], recoil_name, recoil_mass)
        particle = trace_sim.sims[0]
    else:
        trace_sim = build_sim.create_single_particle_sim(experiment, run, event, particle_type)
        particle = trace_sim
    if trace_sim.num_trace_bins > 100:
        print('evt ', return_key, ' has %d bins, not fitting event since this is unexpected'%trace_sim.num_trace_bins)
        return 
    
    #trace_sim.counts_per_MeV *= 1.058
    trace_sim.pad_gain_match_uncertainty = m_guess
    trace_sim.other_systematics = c_guess

    x_real, y_real, z_real, e_real = trace_sim.get_xyze(threshold=h5file.length_counts_threshold, traces=trace_sim.traces_to_fit)
    zmin = 0
    #set zmax to length of trimmed traces
    zmax = trace_sim.num_trace_bins*trace_sim.zscale

    track_center, track_direction_vec = h5file.get_track_axis(event)
    track_direction_vec = track_direction_vec[0]

    d_best, best_point = np.inf, None #distance along track in direction of particle motion. Make as negative as possible
    for x, y, z in zip(x_real, y_real, z_real):
        delta = np.array([x,y,z]) - track_center
        dist = np.dot(delta, track_direction_vec*direction)
        if  dist < d_best:
            d_best= dist
            best_point = np.array([x,y,z])
    #start theta, phi in a small ball around track direction from svd
    vhat = track_direction_vec*direction
    #print('vhat:',vhat)
    theta_guess = np.arctan2(np.sqrt(vhat[0]**2 + vhat[1]**2), vhat[2])
    phi_guess = np.arctan2(vhat[1], vhat[0])

    #start sigma_xy, sigma_z, and c in a small ball around an initial guess
    sigma_guess = 3
    
    if fit_adc_count_per_MeV:
        init_guess = np.array((theta_guess, phi_guess, *best_point, trace_sim.counts_per_MeV, sigma_guess, sigma_guess))
    else:
        init_guess = np.array((theta_guess, phi_guess, *best_point, Eknown, sigma_guess, sigma_guess))

    def to_minimize(params, least_squares):
        theta, phi, x,y,z, E_or_m, sigma_xy, sigma_z = params
        if fit_adc_count_per_MeV:
            trace_sim.counts_per_MeV = E_or_m
            particle.initial_energy = Eknown
        else:
            if fix_energy:
                particle.initial_energy = Eknown
            else:
                particle.initial_energy = E_or_m
            trace_sim.counts_per_MeV = 129600. #using mean value fit when this was a free parameter
        

        #enforce particle direction
        vhat = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
        if np.dot(vhat, direction*track_direction_vec) < 0:
            return np.inf

        if z > trace_sim.num_trace_bins*trace_sim.zscale or z<0:
            return np.inf
        if x**2 + y**2 > 40**2:
            return np.inf
        if particle.initial_energy < 0 or particle.initial_energy  > 10: #stopping power tables currently only go to 10 MeV
            return np.inf
        
        particle.theta, particle.phi = theta, phi
        trace_sim.initial_point = (x,y,z)
        trace_sim.sigma_xy = sigma_xy
        trace_sim.sigma_z = sigma_z
        
        trace_sim.simulate_event()
        if least_squares:
            residuals_dict = trace_sim.get_residuals()
            for pad in residuals_dict:
                #don't penalize for traces which don't go above threshold if pad didnt fire
                if pad not in trace_sim.traces_to_fit and np.max(trace_sim.sim_traces[pad] < trace_sim.pad_threshold): 
                    residuals_dict[pad] = residuals_dict[pad] - trace_sim.pad_threshold
                    residuals_dict[pad][residuals_dict[pad] < 0] = 0
            residuals = np.array([residuals_dict[p] for p in residuals_dict])
            to_return  = np.sum(residuals*residuals)
        else:
            to_return = -trace_sim.log_likelihood()
        #to_return = -trace_sim.log_likelihood()
        if debug_plots:
            print('%e'%to_return, params)
        if np.isnan(to_return):
            to_return = np.inf
        return to_return
    

    if debug_plots:
        print('guess:', init_guess)
        to_minimize(init_guess, True)
        #print('recoil E, point, theta, phi:', trace_sim.recoil.initial_energy, trace_sim.recoil.initial_point, trace_sim.recoil.theta, trace_sim.recoil.phi)
        trace_sim.plot_residuals()
        trace_sim.plot_residuals_3d(threshold=25)
        trace_sim.plot_simulated_3d_data(threshold=25)
        plt.show(block=True)

    res = opt.minimize(fun=to_minimize, x0=init_guess, args=(True,))
    if use_likelihood:
        res = opt.minimize(fun=to_minimize, x0=res.x, args=(False,))

    if return_dict != None:
        to_minimize(res.x, use_likelihood) #make sure sim is updated with best params
        return_dict[return_key] = (res, trace_sim)
        print(return_key, res)
        print('total completed in direction %d:'%direction, len(return_dict.keys()))
    if debug_plots:
        print(res)
        trace_sim.plot_residuals_3d(title=str(return_key)+particle_type, threshold=20)
        trace_sim.plot_simulated_3d_data(title=str(return_key)+particle_type, threshold=20)
        trace_sim.plot_residuals()
        plt.show()
    return res

if False: #try fitting one event to make sure it looks ok
    #fit_event(124,108, '1H', debug_plots=True)
    #fit_event(124,145, '4He', True, direction=1, debug_plots=True)
    fit_event(124,145, '4He', True, Eknown=4.434,direction=-1, debug_plots=True)

events_in_catagory = [[] for i in range(8)]
events_per_catagory = 20
processes = []

'''
| cat # | description                                    |
|   0   | 770 keV p + 19Ne from 20 Mg                    |
|   1   | ~1600 keV p + 19Ne from 20Mg                   |
|   2   | 538 keV 16O recoil from 20Na decay at cathode  |
|   3   | 1108 keV 16O recoil from 20Na decay at cathode |
|   4   | 2153 keV alpha + 538 keV 16O recoil            |
|   5   | 2153 keV alpha w/o recoil from cathode         |
|   6   | 4434 keV alpha + 1108 16O recoil               |
|   7   | 4434 keV alpha w/o recoil from cathode         |

TODO: do a better job of outlier removal for adc gain fit

'''
def classify(range, counts):
    if counts > 8.33e4 and  range > 16.93 and counts < 9.45e4 and range < 22.95:
        return 0
    elif counts > 1.738e5 and range>34.08 and counts < 2.032e5 and range < 59.33:
        return 1
    elif counts > 4.16e4 and range > 11.56 and counts < 5.37e4 and range < 13.12:
        return 2
    elif counts > 1.061e5 and range < 15.75 and counts < 1.188e5 and range > 13.99:
        return 3
    elif counts > 2.912e5 and range < 23.29 and counts < 3.365e5 and range > 18.03:
        return 4
    elif counts > 2.335e5 and range > 16.97 and counts < 2.721e5 and range < 21.64:
        return 5
    elif counts  > 5.91e5 and range < 46.8 and counts < 7.3e5 and range > 31:
        return 6
    elif counts > 4.7e5 and range > 24.1 and counts < 5.52e5 and range < 40.5:
        return 7
    return -1 

ptype_and_recoil_dict = {
    0:('1H', True, 0.770),
    1:('1H', True, 1.590),
    2:('16O', False, 0.5384),
    3:('16O', False, 1.1087),
    4:('4He', True,2.1536),
    5:('4He', False, 2.1536),
    6:('4He', True, 4.4347),
    7:('4He', False, 4.4347)
}

veto_threshold = 300

if not load_previous_fit:
    n = h5file.get_event_num_bounds()[0]
    manager = multiprocessing.Manager()
    forward_fit_results_dict = manager.dict()
    backward_fit_results_dict = manager.dict()
    while np.min([len(x) for x in events_in_catagory]) < events_per_catagory and n < h5file.get_event_num_bounds()[1]:
        max_veto_counts, dxy, dz, counts, angle, pads_railed = h5file.process_event(n)
        l = np.sqrt(dxy**2 + dz**2)
        event_catagory = classify(l, counts)
        #print(n, event_catagory, counts, l, max_veto_counts < veto_threshold )
        if max_veto_counts < veto_threshold and event_catagory >= 0 and len(events_in_catagory[event_catagory]) < events_per_catagory:
            particle_type, include_recoil, E = ptype_and_recoil_dict[event_catagory]
            processes.append(multiprocessing.Process(target=fit_event, 
                                                        args=(run_number, n, particle_type, include_recoil, 1, E,
                                                            n, forward_fit_results_dict)))
            processes[-1].start()
            processes.append(multiprocessing.Process(target=fit_event, 
                                                        args=(run_number, n, particle_type, include_recoil, -1, E,
                                                        n, backward_fit_results_dict)))
            processes[-1].start()
            events_in_catagory[event_catagory].append(n)
            print(n, event_catagory, [len(x) for x in events_in_catagory])
        n += 1
    #wait for all processes to end
    for p in processes:
        p.join()
    
    print('fitting took %f s'%(time.time() - start_time))

    #pick the best of each direction, and save it
    fit_results_dict = {k:forward_fit_results_dict[k] for k in forward_fit_results_dict}
    for k in backward_fit_results_dict:
        if (k in fit_results_dict and backward_fit_results_dict[k][0].fun < fit_results_dict[k][0].fun) or k not in fit_results_dict: 
            fit_results_dict[k] = backward_fit_results_dict[k]
    with open(pickle_fname, 'wb') as f:
        pickle.dump(fit_results_dict, f)
else:
    with open(pickle_fname, 'rb') as f:
        fit_results_dict = pickle.load(f)
    events_in_catagory =  [[] for i in range(8)]
    for evt in fit_results_dict:
        max_veto_counts, dxy, dz, counts, angle, pads_railed = h5file.process_event(evt)
        l = np.sqrt(dxy**2 + dz**2)
        event_catagory = classify(l, counts)
        events_in_catagory[event_catagory].append(evt)

evts, thetas, phis,xs,ys,zs, lls, cats, Es, Erecs, nfev, sigma_xys, sigma_zs = [], [],[],[],[],[],[],[],[],[],[],[],[]
trace_sims = []
for cat in range(len(events_in_catagory)):
    for evt in events_in_catagory[cat]:
        if evt not in fit_results_dict:
            print('evt %d (cat %d)not in results dict'%(evt, cat))
            continue
        res, sim = fit_results_dict[evt]
        trace_sims.append(sim)
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
Es = np.array(Es)
lls = np.array(lls)
evts = np.array(evts)
cats = np.array(cats)

def show_fit(evt):
    i = np.where(evt==evts)[0][0]
    sim = trace_sims[i]
    sim.plot_simulated_3d_data(threshold=25)
    sim.plot_residuals_3d(threshold=25)
    sim.plot_residuals()
    h5file.plot_3d_traces(evt, threshold=25)
    plt.show()

#pressure = 860.3#need to nail this down better later, for now assume current offset #np.mean(Pbest)
ll_thresh = [2*np.median(lls[cats==cat]) for cat in range(4)]

evts_to_fit = []
cats_to_fit = []
for i in range(len(evts)):
    #if lls[i] <= ll_cutoff[cats[i]]:
    #if i == 127:
    pads, traces = h5file.get_pad_traces(evts[i], False)
    sim = trace_sims[i]
    max_trace = np.max(traces)
    residuals_dict = trace_sims[i].get_residuals()
    max_residual_fraction = 0
    for pad in residuals_dict:
        val = np.max(np.abs(residuals_dict[pad]))/max_trace
        if  val > max_residual_fraction:
            max_residual_fraction = val
    print(evts[i], max_residual_fraction)
    #only fit events with residuals no more than 40% of traces
    #and no more than 2x the median for the catagory
    if  max_residual_fraction < 0.4:#higher energy proton #: 
        trace_sims.append(sim)
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

if False:
    c_guess = 20
    systematics_results = opt.minimize(to_minimize, (c_guess, ))
    other_systematics = systematics_results.x[0]
else:
    pad_gain_match_uncertainty,other_systematics = pad_gain_match_uncertainty, c_guess

'''
0.1749208523362717 38.38687105327825
suggested pad threshold = 54.863968

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

temp = h5file.background_subtract_mode
h5file.background_subtract_mode='none'
stds = []
for evt in evts:
    pads, traces = h5file.get_pad_traces(evt)
    for trace in traces:
        stds.append(np.std(trace[400: 450]))
h5file.background_subtract_mode = temp
print('median noise:', np.median(stds))
print('mean noise:', np.mean(stds))

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


plt.figure()
bins=np.linspace(3,6,30)
e_from_ic = np.array([build_sim.get_energy_from_ic(experiment, 124,e) for e in evts])  
plt.hist(e_from_ic[cats==6], bins=bins, label='4434 keV alpha + 1108 16O recoil', alpha=0.5)
plt.hist(e_from_ic[cats==7], bins=bins, label='4434 keV alpha w/o recoil from cathode', alpha=0.5)
plt.legend()
plt.xlabel('energy from IC (MeV)')
plt.figure()
e_from_ic = np.array([build_sim.get_energy_from_ic(experiment, 124,e) for e in evts])  
plt.hist(Es[cats==6], bins=bins, label='4434 keV alpha + 1108 16O recoil', alpha=0.5)
plt.hist(Es[cats==7], bins=bins, label='4434 keV alpha w/o recoil from cathode', alpha=0.5)
plt.legend()
plt.xlabel('alpha energy from fit (MeV)')
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