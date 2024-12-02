import os
os.environ["OMP_NUM_THREADS"] = "1"
import pickle
import time

import numpy as np
import matplotlib.pylab as plt
import scipy.optimize as opt
import sklearn.cluster as cluster
import multiprocessing

from track_fitting import ParticleAndPointDeposition, build_sim
from raw_viewer import raw_h5_file

particle_type = 'proton'

class GaussianVar:
    def __init__(self, mu, sigma):
        self.mu, self.sigma  = mu, sigma

    def log_likelihood(self, val):
        return -np.log(np.sqrt(2*np.pi*self.sigma**2)) - (val - self.mu)**2/2/self.sigma**2

def get_sims_and_param_bounds(experiment, run_number, events):
    sims, bounds, epriors = [],[], []
    for event_num in events:
        new_sim = build_sim.create_pa_sim(experiment, run_number)
        E_from_ic = new_sim.get
        x_real, y_real, z_real, ic_real = new_sim.get

        xmin, xmax = np.min(x_real), np.max(x_real)
        ymin, ymax = np.min(y_real), np.max(y_real)
        zmin = 0
        #set zmax to length of trimmed traces
        
        
        sims.append(new_sim)
        bounds.append(((E_from_ic/2, E_from_ic*2), (0,1), 
                       (xmin, xmax), (ymin, ymax),(zmin, zmax),
                       (0, np.pi), (0, 2*np.pi),
                       (2.2, 20), (2.2, 20)))
        epriors.append(GaussianVar(E_from_ic, detector_E_sigma(E_from_ic)))
    return sims, bounds, epriors


def apply_params(sim, params):
    E, Ea_frac, x, y, z, theta, phi, sigma_xy, sigma_z = params
    sim.initial_energy = E*(1-Ea_frac)
    sim.point_energy_deposition = E*Ea_frac
    sim.initial_point = (x,y,z)
    sim.theta = theta
    sim.phi = phi
    sim.sigma_xy = sigma_xy
    sim.sigma_z = sigma_z
    sim.simulate_event()

def fit_event(sim:ParticleAndPointDeposition.ParticleAndPointDeposition, bounds, Eprior, fit_results_dict=None, results_key=None):
    #tries to maximize posterior of each of the sims passed in subject to the given parameter bounds
    def to_minimize(params):
        apply_params(sim, params)
        return -(sim.log_likelihood() + Eprior.log_likelihood(params[0]))
    #res =  opt.shgo(to_minimize, bounds, sampling_method='halton')
    #res =  opt.direct(to_minimize, bounds)
    res =  opt.differential_evolution(to_minimize, bounds)
    if fit_results_dict != None:
        fit_results_dict[results_key]=res
        print(results_key, res)
    return res


def fit_events(run_num, events, timeout=3600):
    manager = multiprocessing.Manager()
    fit_results_dict = manager.dict()
    results = []
    processes = []
    sims, bounds, epriors = get_sims_and_param_bounds(run_num, events)
    for i in range(len(sims)):
        processes.append(multiprocessing.Process(target=fit_event, args=(sims[i], bounds[i], epriors[i], fit_results_dict, events[i])))
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
    res_dict = fit_events(run_num, events_to_fit, timeout=12*3600)
    with open('run_%d_tylersevts_palpha_fits_w_devolution.dat'%run_num,'wb') as f:
        pickle.dump(res_dict, f)
else:
    #with open('run_%d_cnn_palpha_fits_w_direct.dat'%run_num,'rb') as f:
    with open('run_%d_tylersevts_palpha_fits_w_direct.dat'%run_num,'rb') as f:
        res_dict = pickle.load(f)
Ea = np.array([res_dict[k].x[0]*res_dict[k].x[1] for k in res_dict])
Ep = np.array([res_dict[k].x[0]*(1-res_dict[k].x[1]) for k in res_dict])
ll = np.array([res_dict[k].fun for k in res_dict])

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
    sim, bounds, eprior = get_sims_and_param_bounds(run_num, [event_num])
    sim = sim[0]
    params = res_dict[event_num].x
    apply_params(sim, params)
    sim.plot_residuals_3d(threshold=20)
    sim.plot_simulated_3d_data(threshold=20)
    plt.show()
