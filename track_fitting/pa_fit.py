import time
import os
os.environ["OMP_NUM_THREADS"] = "1"
import pickle

import numpy as np
import matplotlib.pylab as plt
import scipy.optimize as opt
import sklearn.cluster as cluster
import multiprocessing

from track_fitting import ParticleAndPointDeposition
from raw_viewer import raw_h5_file

particle_type = 'proton'

class GaussianVar:
    def __init__(self, mu, sigma):
        self.mu, self.sigma  = mu, sigma

    def log_likelihood(self, val):
        return -np.log(np.sqrt(2*np.pi*self.sigma**2)) - (val - self.mu)**2/2/self.sigma**2

def get_sims_and_param_bounds(run_number, events):
    #returns [sims], [parameter_bounds], energy_prior
    folder = '../../shared/Run_Data/'


    run_h5_path = folder +'run_%04d.h5'%run_number

    if run_number == 124:
        adc_scale_mu = 86431./0.757 #counts/MeV, from fitting events with range 40-43 in run 0368 with p10_default
        adc_scale_sigma = 5631.
        
    elif run_number == 270:
        adc_scale_mu = 151984/0.757 #counts/MeV, from fitting events with range 40-43 in run 0368 with p10_default
        adc_scale_sigma = 8485.
    detector_E_sigma = lambda E: (adc_scale_sigma/adc_scale_mu)*np.sqrt(E/0.757) #sigma for above fit, scaled by sqrt energy
    #use theoretical zscale
    clock_freq = 50e6 #Hz, from e21062 config file on mac minis
    drift_speed = 54.4*1e6 #mm/s, from ruchi's paper
    zscale = drift_speed/clock_freq

    ic_threshold = 25

    pad_threshold = 70
    gain_match_uncertainty = 0.28
    other_systematics = 13.5


    pressure = 860.3 #assuming current offset on MFC was present during experiment, and it was set to 800 torr

            
    rho0 = 1.5256 #mg/cm^3, P10 at 300K and 760 torr
    T = 20+273.15 #K
    get_gas_density = lambda P: rho0*(P/760)*(300./T)


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

    sims, bounds, epriors = [],[], []
    for event_num in events:
        pads_to_fit, traces_to_fit = h5file.get_pad_traces(event_num, include_veto_pads=False)
        max_veto_pad_counts, dxy, dz, measured_counts, angle, pads_railed = h5file.process_event(event_num)
        x_real, y_real, z_real, e_real = h5file.get_xyze(event_number=event_num)

        E_from_ic = measured_counts/adc_scale_mu
        xmin, xmax = np.min(x_real), np.max(x_real)
        ymin, ymax = np.min(y_real), np.max(y_real)
        zmin = 0
        #set zmax to length of trimmed traces
        new_sim = ParticleAndPointDeposition.ParticleAndPointDeposition(get_gas_density(pressure), particle_type)
        new_sim.set_real_data(pads_to_fit, traces_to_fit, trim_threshold=50, trim_pad=10)
        zmax = new_sim.num_trace_bins*zscale
        new_sim.num_stopping_power_points = new_sim.get_num_stopping_points_for_energy(E_from_ic)
        new_sim.adaptive_stopping_power = False
        new_sim.pad_gain_match_uncertainty = gain_match_uncertainty
        new_sim.other_systematics = other_systematics
        new_sim.counts_per_MeV = adc_scale_mu
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
    res =  opt.direct(to_minimize, bounds, vol_tol=0, len_tol=1e-4)
    #res =  opt.differential_evolution(to_minimize, bounds)
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
run_num = 270
if False:
    events_to_fit = get_cnn_events(run_num)
    res_dict = fit_events(270, get_cnn_events(run_num), timeout=3600)
    with open('run_%d_cnn_palpha_fits_w_direct.dat'%run_num,'wb') as f:
        pickle.dump(res_dict, f)
else:
    with open('run_%d_cnn_palpha_fits.dat'%run_num,'rb') as f:
        res_dict = pickle.load(f)
Ea = np.array([res_dict[k].x[0]*res_dict[k].x[1] for k in res_dict])
Ep = np.array([res_dict[k].x[0]*(1-res_dict[k].x[1]) for k in res_dict])
ll = np.array([res_dict[k].fun for k in res_dict])

plt.scatter(Ea[ll<0.8e6], Ep[ll<1e5], c=ll[ll<0.8e6])
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
