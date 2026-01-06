'''
File which stores information about runs
'''
from pathlib import Path
import os

import pandas as pd
import numpy as np

from raw_viewer import process_runs

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
        return np.concatenate([get_GET_run_number(i) for i in ddas_run_number])
    return run_df['GET'][run_df['DDAS']==ddas_run_number].iloc[0]

def get_DDAS_run_number(get_run_number):
    if is_iterable(get_run_number):
        return np.concatenate([get_DDAS_run_number(i) for i in get_run_number])
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
    else:
        veto_thresholds[253]=500
        veto_thresholds[254]=500
        veto_thresholds[508]=500
        veto_thresholds[509]=750
        veto_thresholds[763]=500
        veto_thresholds[764]=500
    return np.all(max_pad_counts<veto_thresholds, axis=1)

