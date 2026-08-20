'''
File which stores information about runs
'''
from pathlib import Path
import os
import pickle

import pandas as pd
import numpy as np
from skimage.measure import points_in_poly

from raw_viewer import process_runs, ddas_interface
from track_fitting import srim_interface, build_sim

experiment = 'e23035'

current_dir = Path(__file__).parent.resolve()
run_df = pd.read_csv(os.path.join(current_dir, 'runs.csv'))
run_df['GET'] = pd.to_numeric(run_df['GET'], errors='coerce', downcast='integer')
run_df['DDAS'] = pd.to_numeric(run_df['DDAS'], errors='coerce', downcast='integer')
run_df['degrader angle'] = pd.to_numeric(run_df['degrader angle'], errors='coerce', downcast='float')

def get_ddas_60_Ga_runs(good_gamma, good_low_energy_tpc, good_long_tracks_tpc, final_beam_settings, tpc_data_valid=True):
    '''
    good_gamma: runs for which I beleive the gamma array was performing well
    good_low_energy_tpc: Runs during which the SCA threshold was behaving correctly. Important for protons with energy <~1.2 MeV
    good_long_tracks_tpc: Runs with get settings which properly captured long tracks Only important above ~2.2 MeV, and likely only for protons.
    final_beam_settings: only include runs with final beam settings (38 degree degrader and reduced momentum acceptance)
    TPC data valid: requires GET data to be in the merged tree
    '''
    runs = []
    for run, get_run in zip(run_df['DDAS'][(run_df['Run Type']=='60Ga')], run_df['GET'][(run_df['Run Type']=='60Ga')]):
        if tpc_data_valid:
            if not np.isfinite(get_run):
                continue
        if np.isnan(run):
            continue
        if final_beam_settings and run <149:
            continue
        if good_gamma and (run <150 or run in [174, 205, 237] or (run>=182 and run<=191) or run in [218, 238, 163]):
            continue
            #169-173: beam disruptions, and following short runs. Include
            #174: attenuated beam
            #180, 181: grow in after PID, including    
            #205 doesn't have matching GET run
            #Runs 182-191 also have poor beharior. Run 187 was LN2 fill, but reason for other runs is unknown.
            #237 has some odds, remove
            #218, 238, 163 are too short to gain match
            #run not in [162,163,203,204,209, 213,217, 218, 238] and #runs which previously were missing h5 files
            #TODO LN2 fill runs
        if good_low_energy_tpc and (run <241 and run > 208):
            continue #SCA was set to ~1 MeV during these runs
        if good_long_tracks_tpc and run < 238:
            #run 238 has max readout depth and final gate delay. Only important above ~2.2 MeV
            continue
        if os.path.exists(ddas_interface.get_merged_root_file_path(experiment, run)):
            runs.append(run)
    return runs

def get_ddas_59_Zn_runs(good_gamma, good_low_energy_tpc, good_long_tracks_tpc, final_beam_settings, tpc_data_valid=True):
    '''
    good_gamma: runs for which I beleive the gamma array was performing well
    good_low_energy_tpc: Runs during which the SCA threshold was behaving correctly. Important for protons with energy <~1.2 MeV
    good_long_tracks_tpc: Runs with get settings which properly captured long tracks Only important above ~2.2 MeV, and likely only for protons.
    final_beam_settings: only include runs with final beam settings (38 degree degrader and reduced momentum acceptance)
    TPC data valid: requires GET data to be in the merged tree
    '''
    runs = []

    if tpc_data_valid:
        run_selector = (run_df['Run Type']=='59Zn')
    else:
        run_selector = (run_df['Run Type']=='59Zn') & (run_df['Field Cage Functional?']=='yes')

    for run, get_run in zip(run_df['DDAS'][run_selector], run_df['GET'][run_selector]):
        if tpc_data_valid:
            if not np.isfinite(get_run):
                continue        
        if np.isnan(run):
            continue
        if final_beam_settings and False: #TODO: currently this doesn't do anything, since the degrader was adjusted so many times
            continue
        if good_gamma and False: #TODO
            continue
        #SCA and GET settings are correct for all 59Zn runs, so don't need to exclude any runs for high/low proton energy
        if os.path.exists(ddas_interface.get_merged_root_file_path(experiment, run)):
            runs.append(run)
    return runs

def is_iterable(obj):
    try:
        iter(obj)
        return True
    except:
        return False

def get_GET_run_number(ddas_run_number):
    if is_iterable(ddas_run_number):
        return np.array([get_GET_run_number(i) for i in ddas_run_number])
    return run_df['GET'][run_df['DDAS']==ddas_run_number].iloc[0]

def get_DDAS_run_number(get_run_number):
    if is_iterable(get_run_number):
        return np.array([get_DDAS_run_number(i) for i in get_run_number])
    return run_df['DDAS'][run_df['GET']==get_run_number].iloc[0]


def get_veto_mask(get_run=None, endpoints=None, max_veto_counts=None):
    if max_veto_counts is not None:
        veto_mask = max_veto_counts < 200
    else:
        if get_run is None:
            raise ValueError("Must provide either max_veto_counts or get_run")
        veto_thresholds = np.ones(process_runs.raw_h5_file.NUM_PADS)*np.inf
        for pad in process_runs.raw_h5_file.VETO_PADS:
                veto_thresholds[pad] = 200
        
        if is_iterable(get_run):
            veto_mask = process_runs.get_veto_mask(experiment, get_run, veto_thresholds)
        else:
            veto_mask = process_runs.get_veto_mask(experiment, [get_run], veto_thresholds)
            
    if endpoints is None:
        if get_run is None:
            raise ValueError("Must provide either endpoints or get_run")
        if not is_iterable(get_run):
            get_run = [get_run]
        endpoints = process_runs.get_quantity('endpoints', experiment, get_run)
        
    min_z = np.min(endpoints[:,:,2], axis=1)
    return veto_mask & (min_z > 5)

def get_pad_gains():
    #gain_match_path = '/egr/research-tpc/adamsa52/gadget_analysis/raw_viewer/plots/e23035_prep_runs61to63_gm.pkl'
    gain_match_path = '/egr/research-tpc/adamsa52/gadget_analysis/raw_viewer/pad_gain_match/gain_match_results/gm_old/fft6_res3.pkl'
    #return np.ones(1024)*5.47e-6
    with open(gain_match_path, 'rb') as f:
        gain_match_result = pickle.load(f)
    #return gain_match_result.x[:1024]
    return gain_match_result.pad_gains

def get_length_mm(get_run):
    if not is_iterable(get_run):
        get_run = [get_run]
    return process_runs.get_lengths(experiment, get_run)

def get_energy_MeV(get_run, num_workers=1):
    if not is_iterable(get_run):
        get_run = [get_run]
    return process_runs.get_gm_ic(experiment, get_run, get_pad_gains(), num_workers=num_workers)

def get_proton_mask_min_max_range(get_run, energies:np.ndarray):
    if not is_iterable(get_run):
        get_run = [get_run]
    stopping_power_path = 'track_fitting/stopping_powers/%s_in_%s.txt'%('1H', 'P10')
    proton_srim_table = srim_interface.SRIM_Table(stopping_power_path, build_sim.get_gas_density('e23035', get_run[0]))
    expected_proton_length = proton_srim_table.get_stopping_distance(energies)
    lower_band = expected_proton_length - 57
    upper_band = expected_proton_length + 20

    x1, x2 = 0.81, 2.14
    y1 = proton_srim_table.get_stopping_distance(x1)-2
    y2 = proton_srim_table.get_stopping_distance(x2)-57
    lower_band[energies<x2] = y1 + (y2-y1)/(x2-x1)*(energies[energies<x2]-x1)

    xa, xb = 0.42, 1
    ya = y1 + (y2-y1)/(x2-x1)*(xa-x1)
    yb = proton_srim_table.get_stopping_distance(xb)+20
    upper_band[energies<xb] = ya + (yb-ya)/(xb-xa)*(energies[energies<xb]-xa)
    return lower_band, upper_band

def get_proton_mask(get_run, lengths=None, energy=None, veto_mask=None):
    if not is_iterable(get_run):
        get_run = [get_run]
    if lengths is None:
        lengths = get_length_mm(get_run)
    if energy is None:
        energy = get_energy_MeV(get_run)
    if veto_mask is None:
        veto_mask = get_veto_mask(get_run)
    lower_band, upper_band = get_proton_mask_min_max_range(get_run, energy)
    return veto_mask & (lengths < upper_band) & (lengths > lower_band)

def get_alpha_mask_min_max_range(get_run, energies:np.ndarray):
    if not is_iterable(get_run):
        get_run = [get_run]
    stopping_power_path = 'track_fitting/stopping_powers/%s_in_%s.txt'%('4He', 'P10')
    alpha_srim_table = srim_interface.SRIM_Table(stopping_power_path, build_sim.get_gas_density('e23035', get_run[0]))
    expected_alpha_length = alpha_srim_table.get_stopping_distance(energies)
    lower_band = np.zeros(len(energies))
    lower_proton, upper_proton = get_proton_mask_min_max_range(get_run, energies)
    upper_band = np.min([lower_proton-10, expected_alpha_length + 22], axis=0)
    return lower_band, upper_band

def get_alpha_mask(get_run, lengths=None, energy=None, veto_mask=None):
    if not is_iterable(get_run):
        get_run = [get_run]
    if lengths is None:
        lengths = get_length_mm(get_run)
    if energy is None:
        energy = get_energy_MeV(get_run)
    if veto_mask is None:
        veto_mask = get_veto_mask(get_run)
    lower_band, upper_band = get_alpha_mask_min_max_range(get_run, energy)    
    return veto_mask & (lengths > lower_band) & (lengths < upper_band)

pid_cuts = {}
def get_pid_cut(ddas_run, species):
    global pid_cuts
    name = 'run%d_%s_cut'%(ddas_run, species)
    if name in pid_cuts:
        return pid_cuts[name]
    
    if species == '60Ga':
        points = [(-6.22262e-7,6715.99),(-6.20756e-7,6829.88),(-6.19298e-7,6768.19),(-6.19465e-7,6635.32),(-6.2066e-7,6540.42),(-6.21903e-7,6583.12)]
    elif species == '59Zn':
        points=[(-6.20086e-7,6417.04),(-6.18461e-7,6554.65),(-6.16429e-7,6438.47),(-6.16429e-7,6307.9),(-6.18556e-7,6208.25),(-6.19943e-7,6269.94)]
    points.append(points[0])
    if ddas_run == 113:
        timing_offset = 0
    elif ddas_run == 177 or ddas_run == 240:
        timing_offset = (-6.24407e-7 + 6.22262e-7)
    elif ddas_run == 262:
        timing_offset = -6.2222e-7 + 6.20086e-7
    points_arr = np.array(points, dtype=np.float64)
    xs = np.ascontiguousarray(points_arr[:,0], dtype=np.float64) + timing_offset
    ys = np.ascontiguousarray(points_arr[:,1], dtype=np.float64)
    to_return =  ROOT.TCutG(name, len(xs), xs, ys)
    to_return.SetVarX("cross_scint_b2_t - db_5_scint_t")
    to_return.SetVarY("msx100_e")
    pid_cuts[name] = to_return
    return to_return

def get_counts_in_pid_cut(ddas_run, species):
    global current_run
    global current_file
    global current_data
    if current_run != ddas_run:
        current_run = ddas_run
        current_file = ROOT.TFile(get_merged_root_file_path(ddas_run), 'READ')
        current_data = current_file.Get('merged_data')
    cut = get_pid_cut(ddas_run, species)
    cut_name = 'run%d_%s_cut'%(ddas_run, species)
    return current_data.Draw('msx100_e:(cross_scint_b2_t - db_5_scint_t)', 'cross_scint_b2_m==1 && db_5_scint_m==1 &&msx100_m==1 && %s'%cut_name, 'goff')