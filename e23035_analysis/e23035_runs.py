'''
File which stores information about runs
'''
from pathlib import Path
import os
import pickle

import pandas as pd
import numpy as np
from skimage.measure import points_in_poly

from raw_viewer import process_runs
from track_fitting import srim_interface, build_sim

experiment = 'e23035'

current_dir = Path(__file__).parent.resolve()
run_df = pd.read_csv(os.path.join(current_dir, 'runs.csv'))
run_df['GET'] = pd.to_numeric(run_df['GET'], errors='coerce', downcast='integer')
run_df['DDAS'] = pd.to_numeric(run_df['DDAS'], errors='coerce', downcast='integer')
run_df['degrader angle'] = pd.to_numeric(run_df['degrader angle'], errors='coerce', downcast='float')

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


def get_veto_mask(get_run):
    if is_iterable(get_run):
        return np.concatenate([get_veto_mask(i) for i in get_run])
    max_pad_counts = process_runs.get_quantity('pad_max', experiment, [get_run])
    veto_thresholds = np.ones(process_runs.raw_h5_file.NUM_PADS)*np.inf
    for pad in process_runs.raw_h5_file.VETO_PADS:
            veto_thresholds[pad] = 500#260
    veto_thresholds[509]=500
    # if get_run<285:
        

    #     # veto_thresholds[253]=170
    #     # veto_thresholds[254]=170
    #     # veto_thresholds[508]=200
    #     # veto_thresholds[509]=600
    #     # veto_thresholds[763]=280
    #     # veto_thresholds[764]=260


    #     # veto_thresholds[253]=110
    #     # veto_thresholds[254]=110
    #     # veto_thresholds[508]=130
    #     # veto_thresholds[509]=390
    #     # veto_thresholds[763]=182
    #     # veto_thresholds[764]=168

        
    #     # start_veto = False
    #     # for pad in process_runs.OUTER_RING_PADS:
    #     #     if pad == 272:
    #     #         start_veto = True
    #     #     if pad == 465:
    #     #         start_veto = False
    #     #     if start_veto:
    #     #         veto_thresholds[pad] = 50
    # else:
    #     veto_thresholds[253]=500
    #     veto_thresholds[254]=500
    #     veto_thresholds[508]=500
    #     veto_thresholds[509]=750
    #     veto_thresholds[763]=500
    #     veto_thresholds[764]=500
    #     start_veto = False
    #     for pad in process_runs.OUTER_RING_PADS:
    #         if pad == 272:
    #             start_veto = True
    #         if pad == 465:
    #             start_veto = False
    #         if start_veto:
    #             veto_thresholds[pad] = 50
    return np.all(max_pad_counts<veto_thresholds, axis=1)

def get_pad_gains(get_run):
    #gain_match_path = '/egr/research-tpc/adamsa52/gadget_analysis/raw_viewer/plots/e23035_prep_runs61to63_gm.pkl'
    gain_match_path = '/egr/research-tpc/adamsa52/gadget_analysis/fft6_res3.pkl'
    #return np.ones(1024)*5.47e-6
    with open(gain_match_path, 'rb') as f:
        gain_match_result = pickle.load(f)
    #return gain_match_result.x[:1024]
    return gain_match_result.pad_gains

def get_length_mm(get_run):
    if not is_iterable(get_run):
        get_run = [get_run]
    return process_runs.get_lengths(experiment, get_run)

def get_energy_MeV(get_run):
    if not is_iterable(get_run):
        get_run = [get_run]
    to_return = []
    for i in get_run:
        to_return.append(process_runs.get_gm_ic(experiment, [i], get_pad_gains(i)))
    return np.concatenate(to_return)

def get_proton_mask_min_max_range(get_run, energies:np.ndarray):
    if not is_iterable(get_run):
        get_run = [get_run]
    stopping_power_path = 'track_fitting/stopping_powers/%s_in_%s.txt'%('1H', 'P10')
    proton_srim_table = srim_interface.SRIM_Table(stopping_power_path, build_sim.get_gas_density('e23035', get_run[0]))
    expected_proton_length = proton_srim_table.get_stopping_distance(energies)
    lower_band = expected_proton_length - 37
    upper_band = expected_proton_length + 20

    x1, x2 = 0.81, 2
    y1 = proton_srim_table.get_stopping_distance(x1)
    y2 = proton_srim_table.get_stopping_distance(x2)-37
    lower_band[energies<x2] = y1 + (y2-y1)/(x2-x1)*(energies[energies<x2]-x1)

    xa, xb = 0.4, 1
    ya = y1 + (y2-y1)/(x2-x1)*(xa-x1)
    yb = proton_srim_table.get_stopping_distance(xb)+20
    upper_band[energies<xb] = ya + (yb-ya)/(xb-xa)*(energies[energies<xb]-xa)
    return lower_band, upper_band

def get_proton_mask(get_run):
    if not is_iterable(get_run):
        get_run = [get_run]
    lengths = get_length_mm(get_run)
    energy = get_energy_MeV(get_run)
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
    upper_band = np.min([lower_proton-10, expected_alpha_length + 30], axis=0)
    return lower_band, upper_band

def get_alpha_mask(get_run):
    if not is_iterable(get_run):
        get_run = [get_run]
    lengths = get_length_mm(get_run)
    energy = get_energy_MeV(get_run)
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