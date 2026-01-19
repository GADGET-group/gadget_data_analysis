import os
#os.environ["OMP_NUM_THREADS"] = "1"
import pickle
import time

import numpy as np
import matplotlib.pylab as plt
import scipy.optimize as opt
import sklearn.cluster as cluster
import multiprocessing

from track_fitting import build_sim

particle_type = 'proton'

class GaussianVar:
    def __init__(self, mu, sigma):
        self.mu, self.sigma  = mu, sigma

    def log_likelihood(self, val):
        return -np.log(np.sqrt(2*np.pi*self.sigma**2)) - (val - self.mu)**2/2/self.sigma**2

def get_sims_and_param_bounds(experiment, run_number, events, flip_angle=False):
    sims, bounds, epriors, guesses = [],[], [], []
    for event_num in events:
        new_sim = build_sim.create_pa_sim(experiment, run_number, event_num)
        E_from_ic = build_sim.get_energy_from_ic(experiment, run_number, event_num)
        xmin, xmax, ymin, ymax, biggest_peak = np.inf, -np.inf, np.inf, -np.inf, -np.inf
        brightest_pad = None
        for pad in new_sim.pads_to_sim:
            x,y = new_sim.pad_to_xy[pad]
            if x < xmin:
                xmin = x
            if y < ymin:
                ymin = y
            if x > xmax:
                xmax = x
            if y > ymax:
                ymax = y
        for pad in new_sim.traces_to_fit:
            peak_val = np.max(new_sim.traces_to_fit[pad])
            if peak_val > biggest_peak:
                brightest_pad = pad
                biggest_peak = peak_val
        zmin = 0
        zmax = new_sim.num_trace_bins
        
        
        sims.append(new_sim)
        Esigma = build_sim.get_detector_E_sigma(experiment, run_number, E_from_ic)
        bounds.append(((E_from_ic-5*Esigma, E_from_ic+5*Esigma), (0,1), 
                       (xmin - new_sim.pad_width, xmax+new_sim.pad_width), 
                       (ymin - new_sim.pad_width, ymax+new_sim.pad_width),
                       (zmin, zmax),
                       (0, np.pi), (0, 2*np.pi), (0, np.pi), (0, 2*np.pi),
                       (2.2, 20), (2.2, 20), (0.1,100)))
        epriors.append(GaussianVar(E_from_ic, Esigma))
        
        #guess decay location is at peak with largest
        xguess, yguess = new_sim.pad_to_xy[brightest_pad]
        zguess = new_sim.zscale*np.argmax(new_sim.traces_to_fit[brightest_pad])
        guesses.append([E_from_ic,0.5, 
                        xguess, yguess, zguess, 
                        0,0,0,0,10,10,15])
        
    return sims, guesses, bounds, epriors


def apply_params(sim, params):
    E, Ea_frac, x, y, z, theta_p, phi_p, theta_a, phi_a, sigma_xy, sigma_z, c = params
    sim.sims[0].initial_energy = E*(1-Ea_frac)
    sim.sims[1].initial_energy = E*Ea_frac
    sim.sims[0].initial_point = sim.sims[1].initial_point = (x,y,z)
    sim.sims[0].theta, sim.sims[0].phi= theta_p, phi_p
    sim.sims[1].theta, sim.sims[1].phi= theta_a, phi_a
    sim.sims[0].sigma_xy = sim.sims[1].sigma_xy =sigma_xy
    sim.sims[0].sigma_z = sim.sims[1].sigma_z = sigma_z
    sim.other_systematics = c
    sim.simulate_event()

def fit_event(sim, guess, bounds, Eprior, fit_results_dict=None, results_key=None, workers=1):
    #tries to maximize posterior of each of the sims passed in subject to the given parameter bounds
    def to_minimize(params):
        apply_params(sim, params)
        return -(sim.log_likelihood() + Eprior.log_likelihood(params[0]))
    #res =  opt.shgo(to_minimize, bounds, sampling_method='halton', options={'ftol':0.1}, workers=workers)
    #res =  opt.shgo(to_minimize, bounds, options={'ftol':0.1}, workers=workers)
    #res =  opt.direct(to_minimize, bounds)
    #res =  opt.differential_evolution(to_minimize, bounds)
    res = opt.minimize(to_minimize, guess, bounds=bounds)
    if fit_results_dict != None:
        fit_results_dict[results_key]=res
        print(results_key, res)
    return res


def fit_events(experiment, run_num, events, timeout=3600):
    manager = multiprocessing.Manager()
    fit_results_dict = manager.dict()
    processes = []
    sims, guesses, bounds, epriors = get_sims_and_param_bounds(experiment,run_num, events)
    for i in range(len(sims)):
        processes.append(multiprocessing.Process(target=fit_event, args=(sims[i], guesses[i], bounds[i], epriors[i], fit_results_dict, events[i])))
        processes[-1].start()
    #for p in processes:
    #    p.join()
    start = time.time()
    while time.time() - start <= timeout:
        if not any(p.is_alive() for p in processes):
            # All the processes are done, break now.
            break

        time.sleep(.1)  # Just to avoid hogging the CPU
    else:
        # We only enter this if we didn't 'break' above.
        print("timed out, killing all processes")
        for p in processes:
            p.terminate()
            p.join()
    fit_results_dict = {k:fit_results_dict[k] for k in fit_results_dict}
    return fit_results_dict

def get_cnn_events(run_num):
    data_dir = '/egr/research-tpc/shared/Run_Data/run_%04d/'%run_num
    for dir in os.listdir(data_dir):
        if 'Boxes' in dir:
            cut_dir = data_dir + dir
    images = os.listdir(cut_dir)
    good_events = np.load(data_dir + 'good_events.npy') #maps image numbers to GET event nuber
    events = []
    for img in images:
        if '.png' not in img:
            continue
        events.append(good_events[int(img.split('_')[2])])
    return events
    


#from track_fitting.pa_fit import *

#res_dict = fit_events(124, [87480,19699,51777,68192,68087, 21640, 96369, 21662, 26303, 50543])
run_num = 124
if False:
    #events_to_fit = get_cnn_events(run_num)
    events_to_fit = [87480, 19699, 51777, 68192, 68087, 10356, 21640, 96369, 21662, 26303, 50543, 27067, 74443, 25304, 38909, 104723, 43833, 52010, 95644, 98220]
    res_dict = fit_events('e21072', run_num, events_to_fit, timeout=12*3600)
    with open('run_%d_hand_picked_events_local_minimizer.dat'%run_num,'wb') as f:
        pickle.dump(res_dict, f)
else:
    #with open('run_%d_cnn_palpha_fits_w_direct.dat'%run_num,'rb') as f:
    with open('run_%d_hand_picked_events_local_minimizer.dat'%run_num, 'rb') as f:
        res_dict = pickle.load(f)
Ea = np.array([res_dict[k].x[0]*res_dict[k].x[1] for k in res_dict])
Ep = np.array([res_dict[k].x[0]*(1-res_dict[k].x[1]) for k in res_dict])
ll = np.array([res_dict[k].fun for k in res_dict])
evt_nums = [k for k in res_dict]

filter = ll<1.5e5
plt.scatter(Ea[filter], Ep[filter], c=ll[filter])
plt.xlabel('alpha energy (MeV)')
plt.ylabel('proton energy (MeV)')
plt.colorbar()

plt.figure()
plt.hist2d(Ea, Ep, 50)
plt.xlabel('alpha energy (MeV)')
plt.ylabel('proton energy (MeV)')
plt.colorbar()
plt.show()

def show_fit(event_num):
    sim, bounds, eprior = get_sims_and_param_bounds('e21072', run_num, [event_num])
    sim = sim[0]
    params = res_dict[event_num].x
    apply_params(sim, params)
    sim.plot_residuals_3d(threshold=20)
    sim.plot_simulated_3d_data(threshold=20)
    sim.plot_real_data_3d(threshold=20)
    plt.show(block=False)

