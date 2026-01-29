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
    if get_run<285:
        veto_thresholds[253]=170
        veto_thresholds[254]=170
        veto_thresholds[508]=200
        veto_thresholds[509]=600
        veto_thresholds[763]=280
        veto_thresholds[764]=260

        # veto_thresholds[253]=110
        # veto_thresholds[254]=110
        # veto_thresholds[508]=130
        # veto_thresholds[509]=390
        # veto_thresholds[763]=182
        # veto_thresholds[764]=168

        
        start_veto = False
        for pad in process_runs.OUTER_RING_PADS:
            if pad == 272:
                start_veto = True
            if pad == 465:
                start_veto = False
            if start_veto:
                veto_thresholds[pad] = 50
    else:
        veto_thresholds[253]=500
        veto_thresholds[254]=500
        veto_thresholds[508]=500
        veto_thresholds[509]=750
        veto_thresholds[763]=500
        veto_thresholds[764]=500
        start_veto = False
        for pad in process_runs.OUTER_RING_PADS:
            if pad == 272:
                start_veto = True
            if pad == 465:
                start_veto = False
            if start_veto:
                veto_thresholds[pad] = 50
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

def get_proton_mask(get_run):
    if not is_iterable(get_run):
        get_run = [get_run]
    lengths = get_length_mm(get_run)
    energy = get_energy_MeV(get_run)
    veto_mask = get_veto_mask(get_run)
    
    stopping_power_path = 'track_fitting/stopping_powers/%s_in_%s.txt'%('1H', 'P10')
    proton_srim_table = srim_interface.SRIM_Table(stopping_power_path, build_sim.get_gas_density('e23035', get_run))
    expected_proton_length = proton_srim_table.get_stopping_distance(energy)
    return veto_mask & (lengths < expected_proton_length+20) & (lengths > expected_proton_length-37)

    #old
    polygon = np.array([(3.78,215.8),(0.56,25.3),(0.56,11.8),(1.56,17.4),(3.01,56.1),(3.95,187.4)])
    points = np.vstack([energy, lengths]).T
    return veto_mask&points_in_poly(points, polygon)

def get_alpha_mask(get_run):
    if not is_iterable(get_run):
        get_run = [get_run]
    lengths = get_length_mm(get_run)
    energy = get_energy_MeV(get_run)
    veto_mask = get_veto_mask(get_run)
    
    stopping_power_path = 'track_fitting/stopping_powers/%s_in_%s.txt'%('1H', 'P10')
    proton_srim_table = srim_interface.SRIM_Table(stopping_power_path, build_sim.get_gas_density('e23035', get_run))
    expected_proton_length = proton_srim_table.get_stopping_distance(energy)
    return veto_mask & (lengths < expected_proton_length-37 - 30)