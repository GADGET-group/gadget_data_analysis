'''
File for building sims, that can then be used for MCMC, fitting, or visualization.
'''
import socket

import numpy as np
import emcee 

from raw_viewer.raw_h5_file import raw_h5_file
from track_fitting.ParticleAndPointDeposition import ParticleAndPointDeposition
class SimBuilder:
    def get_raw_h5_file(run:int):
        '''
        should return a raw_h5_file object for the given run number
        '''
        pass

#########################################################################
# Functions for getting gain, pressure, etc which may vary between runs #
#########################################################################
#detector settings
def get_adc_scale(experiment, run):
    if experiment == 'e21072':
        if run == 124:
            return 86431./0.757 #counts/MeV

def get_gas_density(experiment, run):
    if experiment == 'e21072':
        rho0 = 1.5256 #mg/cm^3, P10 at 300K and 760 torr
        T = 20+273.15 #K
        P = 860.3 #torr
        return rho0*(P/760)*(300./T)
    
def get_zscale(experiment, run):
    if experiment == 'e21072':
        clock_freq = 50e6 #Hz, from e21062 config file on mac minis
        drift_speed = 54.4*1e6 #mm/s, from ruchi's paper
    return drift_speed/clock_freq

#systematics to use for likelihood funciton
def get_systematics(experiment, run):
    '''
    Returns gain match uncertainty, other systematics
    '''
    if experiment == 'e21072':
        if run == 124:
            return 0.3286, 8.876

#raw h5 data location and processing settings
def get_raw_h5_path(experiment, run):
    if experiment == 'e21072':
        if socket.gethostname() == 'tpcgpu':
            return "/egr/research-tpc/shared/Run_Data/" + ('run_%04d.h5'%run)

def get_rawh5_object(experiment, run):
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
        h5file.num_background_bins=(160, 250)
        return h5file

#################
# Functions to creating and manipulating sim objects
#######
def create_pa_sim(experiment, run, event):
    '''
    Gets a particle and point energy deposition sim object, and configures it for fitting a specific event.
    '''
    if experiment == 'e21072':
        h5file = get_rawh5_object(experiment, run)
        pads, traces = h5file.get_pad_traces(event, False)
        sim = ParticleAndPointDeposition(get_gas_density(experiment, run), 'proton')
        sim.zscale = get_zscale(experiment, run)
        sim.set_real_data(pads, traces, trim_threshold=50, trim_pad=10)
        sim.adaptive_stopping_power = True #todo: fix this
        sim.counts_per_MeV = get_adc_scale(experiment, run)
        return sim
    

def set_params_and_simulate(sim, param_dict):
    '''
    Modifies each of the member variables listed in param_dict, and then runs the sim.
    '''
    for param in param_dict:
        sim.__dict__[param] = param_dict[param]
    sim.simulate_event()

def load_pa_mcmc_results(sim, run, event, mcmc_name='final_run'):
    reader = emcee.backends.HDFBackend(filename='run%d_palpha_mcmc/event%d/%s.h5'%(run, event, mcmc_name), read_only=True)
    samples = reader.get_chain()
    ll = reader.get_log_prob()
    best_params = samples[np.unravel_index(np.argmax(ll), ll.shape)]
    E, Ea_frac, x, y, z, theta, phi, sigma_xy, sigma_z = best_params
    Ep = E*(1-Ea_frac)
    Ea = E*Ea_frac
    sim.initial_energy = Ep
    sim.point_energy_deposition = Ea
    sim.initial_point = (x,y,z)
    sim.theta = theta
    sim.phi = phi
    sim.sigma_xy = sigma_xy
    sim.sigma_z = sigma_z
    sim.simulate_event()

event = 74443
sim = create_pa_sim('e21072', 124, event)
load_pa_mcmc_results(sim, 124,event)
import matplotlib.pylab as plt
sim.plot_residuals_3d(threshold=20)
sim.plot_simulated_3d_data(threshold=20)
plt.show()