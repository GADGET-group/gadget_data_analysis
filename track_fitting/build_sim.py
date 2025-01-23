'''
File for building sims, that can then be used for MCMC, fitting, or visualization.
'''
import socket
import configparser
import pickle

import numpy as np
import emcee 
import matplotlib.pylab as plt

from raw_viewer.raw_h5_file import raw_h5_file
from track_fitting.ParticleAndPointDeposition import ParticleAndPointDeposition
from track_fitting.SingleParticleEvent import SingleParticleEvent
from track_fitting.MultiParticleEvent import MultiParticleEvent, ProtonAlphaEvent, DoubleAlphaEvent
from track_fitting.SimGui import SimGui

read_data_mode = 'adjacent'

#########################################################################
# Functions for getting gain, pressure, etc which may vary between runs #
#########################################################################
#detector settings
#list of 2 point calibrations, inexed by experiment and then run number.
#contents of the dictionairy should be a tuple of adc counts, followed by energies in MeV, followed by width of the peaks in adc counts
calibration_points = {'e21072': #from 770 keV and 1.596 MeV protons, adjusted to include recoilling nucleus from Tyler
                        {124:((183193, 86431),(1.623, 0.779))},
                    'e24joe':
                        {124:((5.4e5, 6.9e5),(6.288, 8.7849))}
                    }

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
        
    if experiment == 'e24joe':
        #using same process as e21072 for now
        #TODO change this to better reflect the likely larger sigma in e24joe
        # if run == 124:
        return (5631/86431)*0.779*(MeV/0.779)**0.5

def get_gas_density(experiment:str, run:int)->float:
    if experiment == 'e21072':
        rho0 = 1.5256 #mg/cm^3, P10 at 300K and 760 torr
        T = 20+273.15 #K
        P = 860.3 #torr
        return rho0*(P/760)*(300./T)
    
    if experiment == 'e24joe':
        rho0 = 1.5256 #mg/cm^3, P10 at 300K and 760 torr
        T = 19+273.15 #K
        P = 2000 #torr
        return rho0*(P/760)*(300./T)
    
def get_zscale(experiment:str, run:int):
    if experiment == 'e21072':
        clock_freq = 50e6 #Hz, from e21062 config file on mac minis
        drift_speed = 54.4*1e6 #mm/s, from ruchi's paper
        return drift_speed/clock_freq
    
    if experiment == 'e24joe':
        return 0.65 #TODO: use correct value
        clock_freq = 50e6 #Hz, from e24joe config file on mac minis
        drift_speed = 54.4*1e6 #mm/s, from ruchi's paper
        return drift_speed/clock_freq

#raw h5 data location and processing settings
def get_raw_h5_path(experiment:str, run:int):
    if experiment == 'e21072':
        if socket.gethostname() == 'tpcgpu':
            return "/egr/research-tpc/shared/Run_Data/" + ('run_%04d.h5'%run)
    if experiment == 'e24joe':
        if socket.gethostname() == 'pike' or socket.gethostname() == 'steelhead' or socket.gethostname() == 'flagtail':
            return "/mnt/daqtesting/protondet2024/interesting_events_without_run_number_in_event_name_without_event_447.h5"
            return "/mnt/daqtesting/protondet2024/h5/" + ('run_%04d.h5'%run)
        if socket.gethostname() == 'tpcgpu':
            print("Make sure double alpha data is transferred to the tpcgpu machine!")
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
        h5file.num_background_bins=(160, 250)#(40,50)#
        h5file.zscale = get_zscale(experiment, run)
        return h5file
    if experiment == 'e24joe':
        h5file = raw_h5_file(file_path=get_raw_h5_path(experiment, run),
                                    zscale=get_zscale(experiment, run),
                                    flat_lookup_csv='raw_viewer/channel_mappings/flatlookup2cobos.csv')
        h5file.background_subtract_mode='all data'
        h5file.data_select_mode='smart'
        h5file.ic_counts_threshold = 9
        h5file.remove_outliers=True
        h5file.near_peak_window_width = 50
        h5file.require_peak_within= (-np.inf, np.inf)
        h5file.num_background_bins=(400, 500)#(40,50)#
        h5file.zscale = get_zscale(experiment, run)
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
        sim.set_real_data(pads, traces, trim_threshold=100, trim_pad=10, pads_to_sim_select=read_data_mode)
        sim.counts_per_MeV = get_adc_counts_per_MeV(experiment, run)
        
        sim.adaptive_stopping_power = False
        sim.points_per_bin = 5
        sim.num_stopping_power_points = sim.get_num_stopping_points_for_energy(E_from_ic)

        sim.pad_gain_match_uncertainty, sim.other_systematics = 0.1046, 24.99
        sim.pad_threshold = 50.4

        with open('./raw_viewer/h5_utils/timing_offsets_e21072_run%d.pkl'%run, 'rb') as f:
            sim.timing_offsets = pickle.load(f)
        for pad in sim.timing_offsets:
            if pad != 1:
                sim.timing_offsets[pad] -= sim.timing_offsets[1] #give pad 1 an offset of 0
        sim.timing_offsets[1] = 0
        return sim
    if experiment == 'e24joe':
        sim = SingleParticleEvent(get_gas_density(experiment, run), particle_type)
        sim.zscale = get_zscale(experiment, run)
        sim.set_real_data(pads, traces, trim_threshold=100, trim_pad=10, pads_to_sim_select=read_data_mode)
        sim.counts_per_MeV = get_adc_counts_per_MeV(experiment, run)
        
        sim.adaptive_stopping_power = False
        sim.points_per_bin = 5
        sim.num_stopping_power_points = sim.get_num_stopping_points_for_energy(E_from_ic)

        sim.pad_gain_match_uncertainty, sim.other_systematics = 0.1046, 24.99
        sim.pad_threshold = 50.4

        with open('./raw_viewer/h5_utils/timing_offsets.pkl', 'rb') as f:
            sim.timing_offsets = pickle.load(f)
        for pad in sim.timing_offsets:
            if pad != 1:
                sim.timing_offsets[pad] -= sim.timing_offsets[1] #give pad 1 an offset of 0
        sim.timing_offsets[1] = 0
        return sim

def create_pa_sim(experiment:str, run:int, event:int):
    proton = create_single_particle_sim(experiment, run, event, 'proton')
    alpha = create_single_particle_sim(experiment, run, event, 'alpha')
    sims = [proton, alpha]
    to_return =  MultiParticleEvent(sims)
    pads, traces = pads, traces = get_pads_and_traces(experiment, run, event)
    to_return.set_real_data(pads, traces, trim_threshold=100, trim_pad=10, pads_to_sim_select=read_data_mode)
    to_return.pad_threshold = proton.pad_threshold
    to_return.pad_gain_match_uncertainty = proton.pad_gain_match_uncertainty
    to_return.other_systematics = proton.other_systematics
    return to_return

def create_da_sim(experiment:str, run:int, event:int):
    alpha1 = create_single_particle_sim(experiment, run, event, 'alpha')
    alpha2 = create_single_particle_sim(experiment, run, event, 'alpha')
    to_return =  DoubleAlphaEvent(alpha1, alpha2)
    pads, traces = get_pads_and_traces(experiment, run, event)
    to_return.set_real_data(pads, traces, trim_threshold=100, trim_pad=10, pads_to_sim_select=read_data_mode)
    to_return.pad_threshold = alpha1.pad_threshold
    to_return.pad_gain_match_uncertainty = alpha1.pad_gain_match_uncertainty
    to_return.other_systematics = alpha1.other_systematics
    return to_return
    

def set_params_and_simulate(sim, param_dict:dict):
    '''
    Modifies each of the member variables listed in param_dict, and then runs the sim.
    '''
    for param in param_dict:
        sim.__dict__[param] = param_dict[param]
    sim.simulate_event()

def load_pa_mcmc_results(run:int, event:int, mcmc_name='final_run', step=-1):
    reader = emcee.backends.HDFBackend(filename='run%d_palpha_mcmc/event%d/%s.h5'%(run, event, mcmc_name), read_only=True)
    
    samples = reader.get_chain()[step]
    ll = reader.get_log_prob()[step]
    best_params = samples[np.argmax(ll)]
    E, Ea_frac, x, y, z, theta_p, phi_p, theta_a, phi_a, sigma_p_xy, sigma_p_z, c = best_params
    rho_scale = 1
    Ep = E*(1-Ea_frac)
    Ea = E*Ea_frac
    trace_sim = create_pa_sim('e21072', run, event)
    trace_sim.sims[0].initial_energy = Ep
    trace_sim.sims[1].initial_energy = Ea
    trace_sim.sims[0].initial_point = trace_sim.sims[1].initial_point = (x,y,z)
    trace_sim.sims[0].sigma_xy = sigma_p_xy
    trace_sim.sims[0].sigma_z = sigma_p_z
    trace_sim.sims[1].sigma_xy = sigma_p_xy
    trace_sim.sims[1].sigma_z = sigma_p_z
    trace_sim.sims[0].theta = theta_p
    trace_sim.sims[0].phi = phi_p
    trace_sim.sims[1].theta = theta_a
    trace_sim.sims[1].phi = phi_a
    trace_sim = ProtonAlphaEvent(*trace_sim.sims)
    pads, traces = pads, traces = get_pads_and_traces('e21072', run, event)
    trace_sim.set_real_data(pads, traces, trim_threshold=100, trim_pad=10, pads_to_sim_select=read_data_mode)
    trace_sim.pad_gain_match_uncertainty, trace_sim.other_systematics = trace_sim.proton.pad_gain_match_uncertainty, trace_sim.proton.other_systematics
    trace_sim.gas_density = rho_scale*trace_sim.proton.gas_density
    #trace_sim.pad_gain_match_uncertainty = m
    trace_sim.other_systematics = c
    trace_sim.name = '%s run %d event %d %s'%('e21072', run, event, mcmc_name)
    return trace_sim

def load_da_mcmc_results(run:int, event:int, mcmc_name='final_run', step=-1):
    reader = emcee.backends.HDFBackend(filename='run%d_dalpha_mcmc/event%d/%s.h5'%(run, event, mcmc_name), read_only=True)
    
    samples = reader.get_chain()[step]
    ll = reader.get_log_prob()[step]
    best_params = samples[np.argmax(ll)]
    E, Ea_frac, x, y, z, x_1, y_1, z_1, theta_p, phi_p, theta_a, phi_a, sigma_p_xy, sigma_p_z, c = best_params
    rho_scale = 1
    Ep = E*(1-Ea_frac)
    Ea = E*Ea_frac
    trace_sim = create_da_sim('e24joe', run, event)
    trace_sim.sims[0].initial_energy = Ep
    trace_sim.sims[1].initial_energy = Ea
    trace_sim.sims[0].initial_point = (x,y,z)
    trace_sim.sims[1].initial_point = (x_1,y_1,z_1)
    trace_sim.sims[0].sigma_xy = sigma_p_xy
    trace_sim.sims[0].sigma_z = sigma_p_z
    trace_sim.sims[1].sigma_xy = sigma_p_xy
    trace_sim.sims[1].sigma_z = sigma_p_z
    trace_sim.sims[0].theta = theta_p
    trace_sim.sims[0].phi = phi_p
    trace_sim.sims[1].theta = theta_a
    trace_sim.sims[1].phi = phi_a
    trace_sim = DoubleAlphaEvent(*trace_sim.sims)
    pads, traces = pads, traces = get_pads_and_traces('e24joe', run, event)
    trace_sim.set_real_data(pads, traces, trim_threshold=100, trim_pad=10, pads_to_sim_select=read_data_mode)
    # I changed proton to alpha in the next 2 lines, idk if this is correct
    trace_sim.pad_gain_match_uncertainty, trace_sim.other_systematics = trace_sim.alpha.pad_gain_match_uncertainty, trace_sim.alpha.other_systematics
    trace_sim.gas_density = rho_scale*trace_sim.alpha.gas_density
    #trace_sim.pad_gain_match_uncertainty = m
    trace_sim.other_systematics = c
    trace_sim.name = '%s run %d event %d %s'%('e24joe', run, event, mcmc_name)
    return trace_sim

def load_single_particle_mcmc_result(run:int, event:int, particle='proton', mcmc_name='final_run', step=-1, select_model='best')->SingleParticleEvent:
    filename='run%d_mcmc/event%d/%s.h5'%(run, event, mcmc_name)
    print('loading: ', filename)
    reader = emcee.backends.HDFBackend(filename=filename, read_only=True)
    
    samples = reader.get_chain()[step]
    ll = reader.get_log_prob()[step]

    if select_model == 'best':
        best_params = samples[np.argmax(ll)]
    else:
        best_params = samples[select_model]
    E, x, y, z, theta, phi, sigma_xy, sigma_z = best_params

    trace_sim = create_single_particle_sim('e21072', run, event, particle)
    trace_sim.initial_energy = E
    trace_sim.initial_point = (x,y,z)
    trace_sim.sigma_xy = sigma_xy
    trace_sim.sigma_z = sigma_z
    trace_sim.theta = theta
    trace_sim.phi = phi
    #trace_sim.other_systematics = c
    pads, traces = pads, traces = get_pads_and_traces('e21072', run, event)
    trace_sim.set_real_data(pads, traces, trim_threshold=100, trim_pad=10, pads_to_sim_select=read_data_mode)
    #trace_sim.gas_density = rho_scale*trace_sim.proton.gas_density
    #trace_sim.pad_gain_match_uncertainty = m
    #trace_sim.other_systematics = c
    trace_sim.name = '%s run %d event %d %s'%('e21072', run, event, mcmc_name)
    return trace_sim

def show_results(event:int):
    sim = load_da_mcmc_results(124,event, 'clustering_run2')
    sim.plot_residuals_3d(threshold=20)
    sim.plot_simulated_3d_data(threshold=20)

    h5 = get_rawh5_object('e24joe', 124)
    h5.plot_3d_traces(event,threshold=20)
    #plt.show()

def open_gui(sim:SingleParticleEvent):
    import tkinter as tk
    root = tk.Tk()
    if 'name' in sim.__dict__:
        root.title(sim.name)
    if type(sim) == DoubleAlphaEvent:
        expose_arrays={'alpha1_initial_point':float, 'alpha2_initial_point':float}
    else:
        expose_arrays={'initial_point':float}
    SimGui(root, sim, expose_arrays).grid()
    root.mainloop()

def show_3d_plots(sim, view_thresh = 20):
    sim.plot_real_data_3d(threshold=view_thresh)
    sim.plot_simulated_3d_data(threshold=view_thresh)
    sim.plot_residuals_3d(threshold=view_thresh)
    plt.show(block=False)
