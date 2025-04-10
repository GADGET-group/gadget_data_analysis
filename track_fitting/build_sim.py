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
from track_fitting.SimulatedEvent import SimulatedEvent
from track_fitting.SingleParticleEvent import SingleParticleEvent
from track_fitting.MultiParticleEvent import MultiParticleEvent, MultiParticleDecay
from track_fitting.SimGui import SimGui

read_data_mode = 'unchanged'

#########################################################################
# Functions for getting gain, pressure, etc which may vary between runs #
#########################################################################
#detector settings
#list of 2 point calibrations, inexed by experiment and then run number.
#contents of the dictionairy should be a tuple of adc counts, followed by energies in MeV, followed by width of the peaks in adc counts
calibration_points = {'e21072': #calibration points are for proton + recoiling 19Ne. Energies only include that which is deposited as ionization
                        {124:((90625 , 192102 ),(0.7856, 1.633))}}

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

def get_stopping_material(experiment:str, run:int):
    if experiment == 'e21072':
        return 'P10'

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
        h5file.background_subtract_mode='smart'
        h5file.data_select_mode='all data' 
        h5file.remove_outliers=True
        h5file.smart_bins_away_to_check = 50
        h5file.num_smart_background_ave_bins = 50
        h5file.require_peak_within= (20, 300)
        h5file.ic_counts_threshold = 9
        h5file.length_counts_threshold = 60
        h5file.num_background_bins=(160, 250) #not used for "smart" background subtraction
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
def configure_sim_for_event(sim:SimulatedEvent, experiment:str, run:int, event:int):
    '''
    Load data from h5 file, and set sim variables
    '''
    if experiment == 'e21072':
        sim.zscale = get_zscale(experiment, run)
        pads, traces = get_pads_and_traces(experiment, run, event)
        sim.set_real_data(pads, traces, trim_threshold=100, trim_pad=10, pads_to_sim_select=read_data_mode)
        sim.pad_gain_match_uncertainty, sim.other_systematics = 0.0706, 4.77
        sim.pad_threshold = 54.8
        if run == 124:
            sim.counts_per_MeV = 129600.

        with open('./raw_viewer/h5_utils/timing_offsets_e21072_run%d.pkl'%run, 'rb') as f:
            sim.timing_offsets = pickle.load(f)
        for pad in sim.timing_offsets:
            if pad != 1:
                sim.timing_offsets[pad] -= sim.timing_offsets[1] #give pad 1 an offset of 0
        sim.timing_offsets[1] = 0


def create_single_particle_sim(experiment:str, run:int, event:int, particle_type:str, load_data=True)->SingleParticleEvent:
    '''
    load_data:
    '''
    E_from_ic = get_energy_from_ic(experiment, run, event)
    sim = SingleParticleEvent(get_gas_density(experiment, run), particle_type, get_stopping_material(experiment, run))
    adaptive_stopping_power = False
    sim.points_per_bin = 5
    sim.num_stopping_power_points = sim.get_num_stopping_points_for_energy(E_from_ic)
    if load_data:
        configure_sim_for_event(sim, experiment, run, event)
    return sim

def create_multi_particle_event(experiment:str, run:int, event:int, particle_types:str, load_data=True)->MultiParticleEvent:
    individual_sims = [create_single_particle_sim(experiment, run, event, ptype, False) for ptype in particle_types]
    to_return = MultiParticleEvent(individual_sims)
    if load_data:
        configure_sim_for_event(to_return, experiment, run, event)
    return to_return

def create_multi_particle_decay(experiment:str, run:int, event:int, product_names:list[str], prodcut_masses:float, 
                                recoil_name:str, recoil_mass:float, load_data=True)->MultiParticleDecay:
    product_sims = [create_single_particle_sim(experiment, run, event, ptype, False) for ptype in product_names]
    recoil_sim = create_single_particle_sim(experiment, run, event, recoil_name, False)
    to_return = MultiParticleDecay(product_sims, prodcut_masses, recoil_sim, recoil_mass)
    if load_data:
        configure_sim_for_event(to_return, experiment, run, event)
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

def load_single_particle_mcmc_result(run:int, event:int, particle='1H', mcmc_name='final_run', step=-1, select_model='best')->SingleParticleEvent:
    filename='run%d_mcmc/event%d/%s.h5'%(run, event, mcmc_name)
    print('loading: ', filename)
    reader = emcee.backends.HDFBackend(filename=filename, read_only=True)
    
    samples = reader.get_chain()[step]
    ll = reader.get_log_prob()[step]

    if select_model == 'best':
        best_params = samples[np.argmax(ll)]
    else:
        best_params = samples[select_model]
    E, x, y, z, theta, phi, sigma_xy, sigma_z, density_scale = best_params

    if particle == '1H':
        recoil_name = '19Ne'
        recoil_mass = 19
        product_mass = 1

    trace_sim = create_multi_particle_decay('e21072', run, event, [particle], [product_mass], recoil_name, recoil_mass)
    for sim in trace_sim.sims:
        sim.load_srim_table(sim.particle, sim.material, sim.gas_density*density_scale)
    trace_sim.products[0].initial_energy = E
    trace_sim.initial_point = trace_sim.products[0].initial_point = (x,y,z)
    trace_sim.sigma_xy = sigma_xy
    trace_sim.sigma_z = sigma_z
    trace_sim.products[0].theta = theta
    trace_sim.products[0].phi = phi
    #trace_sim.other_systematics = c
    pads, traces = pads, traces = get_pads_and_traces('e21072', run, event)
    trace_sim.set_real_data(pads, traces, trim_threshold=100, trim_pad=10, pads_to_sim_select=read_data_mode)
    #trace_sim.gas_density = rho_scale*trace_sim.proton.gas_density
    #trace_sim.pad_gain_match_uncertainty = m
    #trace_sim.other_systematics = c
    trace_sim.name = '%s run %d event %d %s'%('e21072', run, event, mcmc_name)
    return trace_sim

def show_results(event:int):
    sim = load_pa_mcmc_results(124,event, 'clustering_run2')
    sim.plot_residuals_3d(threshold=20)
    sim.plot_simulated_3d_data(threshold=20)

    h5 = get_rawh5_object('e21072', 124)
    h5.plot_3d_traces(event,threshold=20)
    #plt.show()

def open_gui(sim:SingleParticleEvent):
    import tkinter as tk
    root = tk.Tk()
    if 'name' in sim.__dict__:
        root.title(sim.name)
    SimGui(root, sim).grid()
    root.mainloop()

def show_3d_plots(sim, view_thresh = 20):
    sim.plot_real_data_3d(threshold=view_thresh)
    sim.plot_simulated_3d_data(threshold=view_thresh)
    sim.plot_residuals_3d(threshold=view_thresh)
    plt.show(block=False)
