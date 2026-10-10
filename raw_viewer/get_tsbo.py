'''
Time since beam off for each GET event, cached in one ROOT file per GET run so it loads as fast as the process_runs
quantities and lines up with them event by event.

The GET-DDAS event matching is the one made for the TPC friend trees (ddas_interface.make_tpc_friend_file, which stores
get_run_id and get_event_id for every DDAS entry); it is inverted here and time_since_beam_off is taken from the merged
DDAS tree. Only that branch is read from the merged files, since their tpc_* branches can be stale.

Cache: <process_runs save path>/<experiment>_run<GET run>_get_tsbo.root, tree 'events', one entry per GET event in the
order of process_runs.get_run_and_event_numbers:
    time_since_beam_off   s, NaN for GET events without a matched DDAS trigger
    ddas_run              DDAS run of the matched trigger, -1 if none
    ddas_entry            entry of the matched trigger in that DDAS run's merged tree (and friend tree), -1 if none
plus a 'metadata' tree with the friend files the matching came from.
'''
import os
import importlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import uproot

from raw_viewer import process_runs, ddas_interface


def get_tsbo_file_path(experiment, get_run):
    return os.path.join(process_runs.get_save_path(experiment), f'{experiment}_run{int(get_run)}_get_tsbo.root')


def make_tsbo_file(experiment, get_run, tpc_ini_filename=""):
    '''
    Write the cache file of one GET run and return its arrays. tpc_ini_filename picks the friend trees to take the
    matching from (the matching itself does not depend on the processing config). Merged files and friend trees are
    never made here (a merged file of a long DDAS run takes hours): if one of the GET run's DDAS runs lacks either, its
    events get NaN and no cache file is written, so the next call tries again.
    '''
    get_run = int(get_run)
    exp_runs = importlib.import_module(f"{experiment}_analysis.{experiment}_runs")
    run_df = exp_runs.run_df
    ddas_runs = sorted({int(d) for d in run_df['DDAS'][run_df['GET'] == get_run] if np.isfinite(d)})
    _, event_numbers = process_runs.get_run_and_event_numbers(experiment, [get_run], config_filename=tpc_ini_filename)
    num_events = len(event_numbers)
    first_event = int(event_numbers[0]) if num_events > 0 else 0
    tsbo = np.full(num_events, np.nan)
    ddas_run = np.full(num_events, -1, dtype=np.int32)
    ddas_entry = np.full(num_events, -1, dtype=np.int64)
    sources, complete = [], True
    for d in ddas_runs:
        friend_path = ddas_interface.get_tpc_friend_file_path(experiment, d, tpc_ini_filename)
        merged_path = ddas_interface.get_ddas_root_file_path(experiment, d)
        if not (os.path.exists(friend_path) and os.path.exists(merged_path)):
            print(f'WARNING: GET run {get_run}: no merged file or TPC friend tree for DDAS run {d}; its events get NaN and the time since beam off is not cached')
            complete = False
            continue
        with uproot.open(friend_path) as f:
            friend = f['tpc_data'].arrays(['get_run_id', 'get_event_id'], library='np')
        with uproot.open(merged_path) as f:
            merged_tsbo = f['merged_data']['time_since_beam_off'].array(library='np')
        entries = np.nonzero((friend['get_run_id'] == get_run) & (friend['get_event_id'] >= 0))[0]
        idx = friend['get_event_id'][entries].astype(np.int64) - first_event
        assert np.all((idx >= 0) & (idx < num_events)), f'GET run {get_run}: friend tree of DDAS {d} has events outside the h5 file'
        tsbo[idx] = merged_tsbo[entries]
        ddas_run[idx] = d
        ddas_entry[idx] = entries
        sources.append(friend_path)
    arrays = {'time_since_beam_off': tsbo, 'ddas_run': ddas_run, 'ddas_entry': ddas_entry}
    if complete:
        with uproot.recreate(get_tsbo_file_path(experiment, get_run)) as f:
            f['events'] = arrays
            f['metadata'] = {'friend_files': np.array([';'.join(sources)])}
    return arrays


def get_time_since_beam_off(experiment, get_runs, tpc_ini_filename="", num_workers=1, return_ddas_entries=False):
    '''
    Time since beam off (s) of every event of get_runs, concatenated in the order of process_runs quantities.
    Missing cache files are made first. With return_ddas_entries, also returns the matched DDAS run and entry arrays.
    '''
    get_runs = [int(r) for r in get_runs]

    def load(run):
        path = get_tsbo_file_path(experiment, run)
        if not os.path.exists(path):
            return make_tsbo_file(experiment, run, tpc_ini_filename)
        with uproot.open(path) as f:
            return f['events'].arrays(['time_since_beam_off', 'ddas_run', 'ddas_entry'], library='np')

    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(load, get_runs))
    else:
        results = [load(run) for run in get_runs]
    tsbo = np.concatenate([r['time_since_beam_off'] for r in results])
    if return_ddas_entries:
        return tsbo, np.concatenate([r['ddas_run'] for r in results]), np.concatenate([r['ddas_entry'] for r in results])
    return tsbo
