'''
File for building sims, that can then be used for MCMC, fitting, or visualization.
'''
import socket
import configparser

import numpy as np
import emcee 

from raw_viewer.raw_h5_file import raw_h5_file
from track_fitting.ParticleAndPointDeposition import ParticleAndPointDeposition
from track_fitting.SingleParticleEvent import SingleParticleEvent
from track_fitting.MultiParticleEvent import MultiParticleEvent
from track_fitting.SimGui import SimGui
#########################################################################
# Functions for getting gain, pressure, etc which may vary between runs #
#########################################################################
#detector settings
#list of 2 point calibrations, inexed by experiment and then run number.
#contents of the dictionairy should be a tuple of adc counts, followed by energies in MeV, followed by width of the peaks in adc counts
calibration_points = {'e21072': #from 770 keV and 1.596 MeV protons, adjusted to include recoilling nucleus from Tyler
                        {124:((183193, 86431),(1.623, 0.779))}}

def get_adc_counts_per_MeV(experiment:str, run:int)->float:
    adc_counts, MeV = calibration_points[experiment][run]
    return (adc_counts[1] - adc_counts[0])/(MeV[1] - MeV[0])

def get_integrated_charge_energy_offset(experiment:str, run:int)->float:
    adc_counts, MeV = calibration_points[experiment][run]
    return MeV[1] - adc_counts[1]/get_adc_counts_per_MeV(experiment, run)

def get_detector_E_sigma(experiment:str, run:int, MeV):
    if experiment == 'e21072':
        #assume energy calibraiton goes as sqrt energy, and use 770 keV protons
        if run == 124:
            return (5631/86431)*0.779*(MeV/0.779)**0.5

def get_gas_density(experiment:str, run:int)->float:
    if experiment == 'e21072':
        rho0 = 1.5256 #mg/cm^3, P10 at 300K and 760 torr
        T = 20+273.15 #K
        P = 860.3 #torr
        return rho0*(P/760)*(300./T)
    
def get_zscale(experiment:str, run:int):
    if experiment == 'e21072':
        clock_freq = 50e6 #Hz, from e21062 config file on mac minis
        drift_speed = 54.4*1e6 #mm/s, from ruchi's paper
    return drift_speed/clock_freq

#raw h5 data location and processing settings
def get_raw_h5_path(experiment:str, run:int):
    if experiment == 'e21072':
        if socket.gethostname() == 'tpcgpu':
            return "/egr/research-tpc/shared/Run_Data/" + ('run_%04d.h5'%run)

def get_rawh5_object(experiment:str, run:int)->raw_h5_file:
    '''
    Get a raw_h5_file object
    override_dict: override member variables (background subtract mode, etc)
    '''
    if experiment == 'e21072':
        h5file = raw_h5_file(file_path=get_raw_h5_path(experiment, run),
                                    zscale=get_zscale(experiment, run),
                                    flat_lookup_csv='raw_viewer/channel_mappings/flatlookup4cobos.csv')
        h5file.background_subtract_mode='fixed window'
        h5file.data_select_mode='near peak'
        h5file.remove_outliers=True
        h5file.near_peak_window_width = 50
        h5file.require_peak_within= (-np.inf, np.inf)
        h5file.num_background_bins=(40,50)#(160, 250)
        return h5file
    assert False
    
def apply_config_to_object(config_file, object):
    pass #TODO


pads_and_traces = {}#indexed by experiment, run, event
def get_pads_and_traces(experiment, run, event):
    if (experiment, run, event) not in pads_and_traces:
        h5file = get_rawh5_object(experiment, run)
        pads_and_traces[(experiment, run, event)] = h5file.get_pad_traces(event, False)
    return pads_and_traces[(experiment, run, event)]

energies_from_ic = {}
def get_energy_from_ic(experiment, run, event):
    if (experiment, run, event) not in energies_from_ic:
        h5file = get_rawh5_object(experiment, run)
        max_veto_pad_counts, dxy, dz, measured_counts, angle, pads_railed = h5file.process_event(event)
        energies_from_ic[(experiment, run, event)] = measured_counts/get_adc_counts_per_MeV(experiment, run) + get_integrated_charge_energy_offset(experiment, run)
    return energies_from_ic[(experiment, run, event)]


########################################################
# Functions to creating and manipulating sim objects
########################################################

def create_single_particle_sim(experiment:str, run:int, event:int, particle_type:str):
    '''
    sim_constructor: assumed to take the same parameters as single particle event
    '''
    
    pads, traces = get_pads_and_traces(experiment, run, event)
    E_from_ic = get_energy_from_ic(experiment, run, event)

    if experiment == 'e21072':
        sim = SingleParticleEvent(get_gas_density(experiment, run), particle_type)
        sim.zscale = get_zscale(experiment, run)
        sim.set_real_data(pads, traces, trim_threshold=50, trim_pad=10, pads_to_sim_select='adjacent')#'unchanged')#
        sim.counts_per_MeV = get_adc_counts_per_MeV(experiment, run)
        
        sim.adaptive_stopping_power = False #TODO: see if I can set this to True

        sim.num_stopping_power_points = sim.get_num_stopping_points_for_energy(E_from_ic)
        sim.pad_gain_match_uncertainty, sim.other_systematics = 0,8.876#0.3286, 8.876
        sim.pad_threshold = 50
        return sim

def create_pa_sim(experiment:str, run:int, event:int):
    proton = create_single_particle_sim(experiment, run, event, 'proton')
    alpha = create_single_particle_sim(experiment, run, event, 'alpha')
    sims = [proton, alpha]
    to_return =  MultiParticleEvent(sims)
    pads, traces = pads_and_traces[(experiment, run, event)]
    to_return.set_real_data(pads, traces, trim_threshold=50, trim_pad=10, pads_to_sim_select='adjacent')
    to_return.pad_threshold = proton.pad_threshold
    to_return.pad_gain_match_uncertainty = proton.pad_gain_match_uncertainty
    to_return.other_systematics = proton.other_systematics
    return to_return
    

def set_params_and_simulate(sim, param_dict:dict):
    '''
    Modifies each of the member variables listed in param_dict, and then runs the sim.
    '''
    for param in param_dict:
        sim.__dict__[param] = param_dict[param]
    sim.simulate_event()

def load_pa_mcmc_results(run:int, event:int, mcmc_name='final_run')->ParticleAndPointDeposition:
    sim = create_pa_sim('e21072', run, event)
    reader = emcee.backends.HDFBackend(filename='run%d_palpha_mcmc/event%d/%s.h5'%(run, event, mcmc_name), read_only=True)
    #reader = emcee.backends.HDFBackend(filename='run%d_palpha_mcmc_likelihood_div_by_num_pads/event%d/%s.h5'%(run, event, mcmc_name), read_only=True)
    
    samples = reader.get_chain()
    ll = reader.get_log_prob()
    best_params = samples[np.unravel_index(np.argmax(ll), ll.shape)]
    E, Ea_frac, x, y, z, theta_p, phi_p, theta_a, phi_a, sigma_p_xy, sigma_p_z, sigma_a_xy, sigma_a_z = best_params
    Ep = E*(1-Ea_frac)
    Ea = E*Ea_frac
    trace_sim = create_pa_sim('e21072', run, event)
    trace_sim.sims[0].initial_energy = Ep
    trace_sim.sims[1].initial_energy = Ea
    trace_sim.sims[0].initial_point = trace_sim.sims[1].initial_point = (x,y,z)
    trace_sim.sims[0].sigma_xy = sigma_p_xy
    trace_sim.sims[0].sigma_z = sigma_p_z
    trace_sim.sims[1].sigma_xy = sigma_a_xy
    trace_sim.sims[1].sigma_z = sigma_a_z
    trace_sim.sims[0].theta = theta_p
    trace_sim.sims[0].phi = phi_p
    trace_sim.sims[1].theta = theta_a
    trace_sim.sims[1].phi = phi_a
    trace_sim.simulate_event()
    return trace_sim

def show_results(event:int):
    sim = load_pa_mcmc_results(124,event, 'clustering_run2')
    import matplotlib.pylab as plt
    sim.plot_residuals_3d(threshold=20)
    sim.plot_simulated_3d_data(threshold=20)

    h5 = get_rawh5_object('e21072', 124)
    h5.plot_3d_traces(event,threshold=20)
    #plt.show()

def open_gui(sim:SingleParticleEvent):
    import tkinter as tk
    root = tk.Tk()
    SimGui(root, sim).grid()
    root.mainloop()
