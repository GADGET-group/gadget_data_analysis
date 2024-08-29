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
import numpy as np

import matplotlib.pylab as plt
import scipy.optimize as opt

from track_fitting import SingleParticleEvent
from raw_viewer import raw_h5_file

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
h5file.data_select_mode='all data'
h5file.remove_outliers=True
#h5file.near_peak_window_width = 50
#h5file.require_peak_within= (-np.inf, np.inf)
h5file.num_background_bins=(160, 250)
h5file.ic_counts_threshold = 25

shaping_time = 70e-9 #s, from e21062 config file on mac minis
shaping_width  = shaping_time*clock_freq*2.355

pressure = 860.3 #torr, assuming current offset on MFC was present during experiment, and it was set to 800 torr
rho0 = 1.5256 #mg/cm^3, P10 at 300K and 760 torr
T = 20+273.15 #K
get_gas_density = lambda P: rho0*(P/760)*(300./T)


def fit_event(pads_to_fit, traces_to_fit, particle_type, trim_threshold=50):
    trace_sim = SingleParticleEvent.SingleParticleEvent(get_gas_density(pressure), particle_type)
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
        x,y,z,theta, phi, charge_spread = params
        trace_sim.theta, trace_sim.phi = theta, phi
        trace_sim.initial_point = (x,y,z)
        trace_sim.charge_spreading_sigma = charge_spread
        trace_sim.simulate_event()
        trace_sim.align_pad_traces()
        return -trace_sim.log_likelihood()
    
    init_guess = (x_guess, y_guess, 200, theta_guess, phi_guess, 1)
    res = opt.minimize(fun=neg_log_likelihood, x0=init_guess, method="Powell")
    return res.x

pads_to_fit, traces_to_fit = h5file.get_pad_traces(17, include_veto_pads=False)
print(fit_event(pads_to_fit, traces_to_fit, 'proton', trim_threshold=50))

