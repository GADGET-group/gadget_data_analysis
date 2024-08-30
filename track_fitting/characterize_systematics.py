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

import numpy as np
import matplotlib.pylab as plt
import scipy.optimize as opt

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
                                zscale=zscale,
                                flat_lookup_csv='raw_viewer/channel_mappings/flatlookup4cobos.csv')
h5file.background_subtract_mode='fixed window'
h5file.data_select_mode='near peak'
h5file.remove_outliers=True
h5file.near_peak_window_width = 50
h5file.require_peak_within= (-np.inf, np.inf)
h5file.num_background_bins=(160, 250)
h5file.ic_counts_threshold = 25

shaping_time = 70e-9 #s, from e21062 config file on mac minis
shaping_width  = shaping_time*clock_freq*2.355

#pressure = 860.3 #torr, assuming current offset on MFC was present during experiment, and it was set to 800 torr
p_guess = 860
rho0 = 1.5256 #mg/cm^3, P10 at 300K and 760 torr
T = 20+273.15 #K
get_gas_density = lambda P: rho0*(P/760)*(300./T)


def fit_event(pads_to_fit, traces_to_fit, particle_type, trim_threshold=50, return_key=None, return_dict=None):
    trace_sim = SingleParticleEvent.SingleParticleEvent(get_gas_density(p_guess), particle_type)
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
    brag_x, brag_y = trace_sim.pad_to_xy[max_index]
    brag_z = zscale*max_index
    #find pad which fired which is furthest from the brag peak
    x_guess, y_guess = brag_x, brag_y
    furthest_dist_sqrd = 0
    for pad, trace in zip(pads_to_fit, traces_to_fit):
        x,y = trace_sim.pad_to_xy[pad]
        dist = (x-x_guess)**2 + (y - y_guess)**2
        if dist > furthest_dist_sqrd:
            furthest_dist_sqrd = dist
            x_guess, y_guess = x,y
            furthest_pad_trace = pad
    dz = brag_z - np.argmax(furthest_pad_trace)*zscale
    theta_guess = np.arctan2(dz, np.sqrt(furthest_dist_sqrd))
    phi_guess = np.arctan2(brag_y-y_guess, brag_x-x_guess)
    def neg_log_likelihood(params):
        P, x,y,z,theta, phi, charge_spread = params
        if charge_spread < 0 or P < 760 or P>900:
            return np.inf
        if z > 400 or z<10:
            return np.inf
        if x**2 + y**2 > 40**2:
            return np.inf
        if charge_spread <0:
            return np.inf
        trace_sim.load_srim_table(particle=particle_type, gas_density=get_gas_density(P))
        trace_sim.theta, trace_sim.phi = theta, phi
        trace_sim.initial_point = (x,y,z)
        trace_sim.charge_spreading_sigma = charge_spread
        trace_sim.simulate_event()
        trace_sim.align_pad_traces()
        return -trace_sim.log_likelihood()
    
    init_guess = (p_guess, x_guess, y_guess, 200, theta_guess, phi_guess, 1)
    res = opt.minimize(fun=neg_log_likelihood, x0=init_guess, method="Powell")
    if return_dict != None:
        return_dict[return_key] = res
        print(return_key, res.x)
    return res

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

n = h5file.get_event_num_bounds()[0]
manager = multiprocessing.Manager()

fit_results_dict = manager.dict()
while np.min([len(x) for x in events_in_catagory]) < events_per_catagory:
    max_veto_counts, dxy, dz, counts, angle, pads_railed = h5file.process_event(n)
    l = np.sqrt(dxy**2 + dz**2)
    event_catagory = classify(l, counts)
    #print(n, event_catagory, counts, l, max_veto_counts < veto_threshold )
    if max_veto_counts < veto_threshold and event_catagory >= 0 and len(events_in_catagory[event_catagory]) < events_per_catagory:
        pads, traces  = h5file.get_pad_traces(n, include_veto_pads=False)
        processes.append(multiprocessing.Process(target=fit_event, args=(pads, traces, 'alpha', 50, n, fit_results_dict)))
        processes[-1].start()
        events_in_catagory[event_catagory].append(n)
        print([len(x) for x in events_in_catagory])
    n += 1
#wait for all processes to end
for p in processes:
    p.join()

print('fitting took %f s'%(time.time() - start_time))

results_by_cat = {}
for cat in range(len(events_in_catagory)):
    results_by_cat[cat] = []
    for evt in events_in_catagory[cat]:
        if evt not in fit_results_dict:
            print('evt %d (cat %d)not in results dict'%(evt, cat))
            continue
        res = fit_results_dict[evt]
        if not res.success:
            print('evt %d (cat %d)not succesfully fit'%(evt, cat))
        results_by_cat[cat].append(res)


Ps = np.array([fit_results_dict[k].x[0] for k in fit_results_dict])
Pgood = Ps[(np.abs(Ps - np.mean(Ps)) < np.std(Ps))]
Pbest = Pgood[(np.abs(Pgood - np.mean(Pgood)) < np.std(Pgood))]
print(np.mean(Pbest), np.std(Pbest))
plt.figure()
plt.hist(Pbest, 20)
plt.show()