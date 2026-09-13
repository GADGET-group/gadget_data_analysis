
import sys
import hashlib
import os
import re
import pickle
import gzip
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

import concurrent.futures
import tqdm

import ROOT
import numpy as np

from e23035_analysis import e23035_runs, energy_calibration_tools
from raw_viewer import process_runs, ddas_interface

NUM_SLOTS = 10
CH_PER_SLOT = 16
NUM_TOTAL_CH = NUM_SLOTS*CH_PER_SLOT

class CH_MAP:
    MESH_PRE_AMP = 7*16+6
    GET_TRIG_ACCEPTED = 7*16+7
    SCA_LOGIC = 7*16+8
    VETO_LOGIC = 7*16+9
    #beam off is signal from ARIS used to switch GG
    BEAM_ON = 7*16+10
    BEAM_OFF = 7*16+11
    #chopper is actual control of accelerator
    CHOPPER_ON = 7*16+12
    CHOPPER_OFF = 7*16+13

    #PID channels
    MSX100 = 8*16
    MSX40 = 8*16+1
    CROSS_SCINT_B2 = 13
    CROSS_SCINT_T2 = 14
    DB_5_SCINT = 16
    DB_3_SCINT_L = 16 + 2

    #clovers
    CLOVER_1A = 3*16 + 0
    CLOVER_1B = 3*16 + 1
    CLOVER_1C = 3*16 + 2
    CLOVER_1D = 3*16 + 3
    CLOVER_1 = np.array([CLOVER_1A, CLOVER_1B, CLOVER_1C, CLOVER_1D])
    CLOVER_2A = 3*16 + 4
    CLOVER_2B = 3*16 + 5
    CLOVER_2C = 3*16 + 6
    CLOVER_2D = 3*16 + 7
    CLOVER_2 = np.array([CLOVER_2A, CLOVER_2B, CLOVER_2C, CLOVER_2D])
    CLOVER_3A = 3*16 + 8
    CLOVER_3B = 5*16 + 4
    CLOVER_3C = 5*16 + 5
    CLOVER_3D = 3*16 + 11
    CLOVER_3 = np.array([CLOVER_3A, CLOVER_3B, CLOVER_3C, CLOVER_3D])
    #clover 4 not installed
    CLOVER_5A = 4*16 + 0
    CLOVER_5B = 4*16 + 1
    CLOVER_5C = 4*16 + 2
    CLOVER_5D = 4*16 + 3
    CLOVER_5 = np.array([CLOVER_5A, CLOVER_5B, CLOVER_5C, CLOVER_5D])
    CLOVER_6A = 4*16 + 4
    CLOVER_6B = 4*16 + 5
    CLOVER_6C = 4*16 + 6
    CLOVER_6D = 4*16 + 7
    CLOVER_6_INDECIES = np.array([CLOVER_6A, CLOVER_6B, CLOVER_6C, CLOVER_6D])
    CLOVER_7A = 5*16 + 6
    CLOVER_7B = 5*16 + 7
    CLOVER_7C = 5*16 + 8
    CLOVER_7D = 4*16 + 11
    CLOVER_7 = np.array([CLOVER_7A, CLOVER_7B, CLOVER_7C, CLOVER_7D])
    #clover 8 not installed
    CLOVER_9A = 9*16 + 0
    CLOVER_9B = 9*16 + 1
    CLOVER_9C = 9*16 + 2
    CLOVER_9D = 9*16 + 3
    CLOVER_9 = np.array([CLOVER_9A, CLOVER_9B, CLOVER_9C, CLOVER_9D])
    CLOVER_10A = 9*16 + 4
    CLOVER_10B = 9*16 + 5
    CLOVER_10C = 9*16 + 6
    CLOVER_10D = 9*16 + 7
    CLOVER_10 = np.array([CLOVER_10A, CLOVER_10B, CLOVER_10C, CLOVER_10D])
    CLOVER_11A = 9*16 + 8
    CLOVER_11B = 9*16 + 9
    CLOVER_11C = 9*16 + 10
    CLOVER_11D = 9*16 + 11
    CLOVER_11 = np.array([CLOVER_11A, CLOVER_11B, CLOVER_11C, CLOVER_11D])
    #list of all germnaium channels
    GE_INDECIES = np.concatenate([CLOVER_1, CLOVER_2, CLOVER_3, CLOVER_5,
                              CLOVER_6_INDECIES, CLOVER_7, CLOVER_9, CLOVER_10,
                              CLOVER_11])
    #list of crystal ids corresponding to the above inecies
    GE_CALIBRATION_INDEXES = np.array([0,1,2,3,4,5,6,7,8,9,10,11,16,17,18,19,20,21,22,23,24,25,26,27,32,33,34,35,36,37,38,39,40,41,42,43])

def get_root_file_path(experiment, run):
    base_path = f'/egr/research-tpc/shared/proc_runs/{experiment}/ddas/'
    file_name = 'run-%04d.root'%run
    return base_path + file_name

def get_gadget_root_file_path(experiment, run):
    base_path = f'/egr/research-tpc/shared/proc_runs/{experiment}/ddas/'
    file_name = 'run-%04d_gadget.root'%run
    return base_path + file_name

def extract_get_event_data(experiment, run):
    save_path = '/egr/research-tpc/shared/proc_runs/%s/proc_pkl'%experiment
    pkl_fname = os.path.join(save_path, '%s_run%d_ddas_gadget.pkl.gz'%(experiment, run))
    if os.path.exists(pkl_fname):
        print('run %d previously extracted, loading previous results'%run)
        with gzip.open(pkl_fname, 'rb') as file:
            es = pickle.load(file)
            ts = pickle.load(file)
            ms = pickle.load(file)
    else:
        file = ROOT.TFile(get_root_file_path(experiment, run), "READ")
        tree = file.Get("tree")
        energies, times, multiplicities = np.zeros(NUM_TOTAL_CH, dtype=np.int32), np.zeros(NUM_TOTAL_CH), np.zeros(NUM_TOTAL_CH,dtype=np.int32)
        tree.SetBranchAddress("energies", energies)
        tree.SetBranchAddress("times", times)
        tree.SetBranchAddress("multiplicity", multiplicities)
        shape = (tree.GetEntries(), NUM_TOTAL_CH)
        es, ts, ms = [],[],[]
        for i in tqdm.tqdm(range(tree.GetEntries())):
            tree.GetEntry(i)

            if np.max(multiplicities[CH_MAP.MESH_PRE_AMP:CH_MAP.CHOPPER_OFF+1]) > 0:
                es.append(np.copy(energies))
                ts.append(np.copy(times))
                ms.append(np.copy(multiplicities))
        es, ts, ms = np.array(es), np.array(ts), np.array(ms)
        with gzip.open(pkl_fname, 'wb') as save_file:
            pickle.dump(es, save_file)
            pickle.dump(ts, save_file)
            pickle.dump(ms, save_file)
    return es, ts, ms


def get_time_since_beam_off(experiment, run):
    '''
    Get time since beam as turned off for each accepted trigger
    '''
    save_path = '/egr/research-tpc/shared/proc_runs/%s/proc_pkl'%experiment
    pkl_fname = os.path.join(save_path, '%s_run%d_tsbo.pkl.gz'%(experiment, run))
    if os.path.exists(pkl_fname):
        with gzip.open(pkl_fname, 'rb') as file:
            return pickle.load(file)
    else:
        es, ts, ms = extract_get_event_data(experiment, run)
        beam_off_times = ts[:, CH_MAP.CHOPPER_OFF]
        beam_off_times = beam_off_times[beam_off_times >= 0]
        event_times = ts[:, CH_MAP.GET_TRIG_ACCEPTED]
        event_times = event_times[event_times >= 0]
        
        to_return = np.zeros(len(event_times))
        i, j = 0,0
        while i < len(event_times):
            while (j < len(beam_off_times) - 1) and (beam_off_times[j+1] < event_times[i]):
                j += 1
            to_return[i] = event_times[i] - beam_off_times[j]
            i += 1

        to_return /= 1e9
        with gzip.open(pkl_fname, 'wb') as save_file:
            pickle.dump(to_return, save_file)
        return to_return

def get_ddas_root_file_path(experiment, ddas_run):
    root_file_path = get_root_file_path(experiment=experiment, run=ddas_run)
    to_return =  os.path.join(os.path.split(root_file_path)[0], f'run{int(ddas_run)}_merged.root')
    if experiment == 'e25058':
        to_return += '_alex'
    return to_return

def get_tpc_friend_file_path(experiment, ddas_run, tpc_ini_filename=""):
    root_file_path = get_root_file_path(experiment=experiment, run=ddas_run)
    ini_prefix = os.path.splitext(tpc_ini_filename)[0]
    to_return =  os.path.join(os.path.split(root_file_path)[0], f'run{int(ddas_run)}_tpc_{ini_prefix}.root')
    if experiment == 'e25058':
        to_return += '_alex'
    return to_return

def make_ddas_root_file(experiment, ddas_run):
    '''
    Create a root file with a new TTree called "merged_data".
    '''
    root_file_path = get_root_file_path(experiment=experiment, run=ddas_run)
    
    log_path = os.path.join(os.path.split(root_file_path)[0], f'run{int(ddas_run)}_merge.log')
    if experiment == 'e25058':
        log_path += '_alex'
    output_path = get_ddas_root_file_path(experiment, ddas_run)
    # Write to a temporary path and rename once the merge finishes, so a failed merge doesn't leave a file at output_path
    # The temporary path is unique to this process, so two processes making the same file don't write to or rename each other's output
    partial_output_path = f'{output_path}.{os.getpid()}.partial'
    with ROOT.TFile(root_file_path, "READ") as input_file, open(log_path, 'w') as log_file, ROOT.TFile(partial_output_path, "RECREATE") as output_file:
        # Run git in the repository rather than the current working directory, which may be outside it
        git_version = subprocess.run(['git', 'rev-parse', '--verify', 'HEAD'], capture_output=True, text=True, check=True, cwd=BASE_DIR).stdout
        git_status = subprocess.run(['git', 'status'], capture_output=True, text=True, check=True, cwd=BASE_DIR).stdout
        git_diff = subprocess.run(['git', 'diff'], capture_output=True, text=True, check=True, cwd=BASE_DIR).stdout
        log_file.write('preparing to process ddas run %d \n'%ddas_run)
        log_file.write('git commit %s\n'%git_version)
        log_file.write('git status: %s\n'%git_status)
        log_file.write('git diff: %s\n'%git_diff)

        ddas_ch_map_path = os.path.join(BASE_DIR, f'{experiment}_analysis', 'channel_map.csv')
        log_file.write('loading DDAS channel map from %s\n'%ddas_ch_map_path)
        chmap = np.genfromtxt(ddas_ch_map_path,delimiter=', ', dtype=str, skip_header=1)
        ch_indexes = np.array(chmap[:,0], dtype=int)
        ch_names = chmap[:,1]
        slopes, offsets = np.array(chmap[:,2], dtype=float), np.array(chmap[:,3], dtype=float)

        log_file.write('ch_names:')
        log_file.write(str(ch_names))
        log_file.write('\n slopes: %s\n'%str(slopes))
        
        log_file.write('Opening ROOT file\n')
        in_tree = input_file.Get("tree")
        energies, times, multiplicities = np.zeros(NUM_TOTAL_CH, dtype=np.int32), np.zeros(NUM_TOTAL_CH), np.zeros(NUM_TOTAL_CH,dtype=np.int32)
        in_tree.SetBranchAddress("energies", energies)
        in_tree.SetBranchAddress("times", times)
        in_tree.SetBranchAddress("multiplicity", multiplicities)

        log_file.write('Setting up tree in which merged data will be stored\n')
        out_tree = ROOT.TTree("merged_data", "merged_data")
        branch_evals = [np.array([0], dtype=np.float64) for i in ch_names]
        branch_tvals = [np.array([0], dtype=np.float64) for i in ch_names]
        branch_mvals = [np.array([0], dtype=np.int32) for i in ch_names]
        branch_counts = [np.array([0], dtype=np.int32) for i in ch_names]
        branch_counts_ss = [np.array([0], dtype=np.float64) for i in ch_names] #sliding scale method counts
        
        for i in range(len(ch_names)):
            out_tree.Branch(ch_names[i]+'_e', branch_evals[i], ch_names[i]+'_e/D')
            out_tree.Branch(ch_names[i]+'_t', branch_tvals[i], ch_names[i]+'_t/D')
            out_tree.Branch(ch_names[i]+'_m', branch_mvals[i], ch_names[i]+'_m/I')
            out_tree.Branch(ch_names[i]+'_c', branch_counts[i], ch_names[i]+'_c/I')
            out_tree.Branch(ch_names[i]+'_cr', branch_counts_ss[i], ch_names[i]+'_cr/D')
            
        tsbo = np.array([np.nan], dtype=np.float64)
        out_tree.Branch('time_since_beam_off', tsbo, 'time_since_beam_off/D')
        tsco = np.array([np.nan], dtype=np.float64)
        out_tree.Branch('time_since_chopper_off', tsco, 'time_since_chopper_off/D')

        log_file.write('Starting process \n')
        last_beam_off_time = np.nan
        last_chopper_off_time = np.nan

        for ddas_index in tqdm.tqdm(range(in_tree.GetEntries())):
            #copy over ddas values with calibration factors applied
            in_tree.GetEntry(ddas_index)
            for i in range(len(ch_names)):
                branch_mvals[i][0] = multiplicities[ch_indexes[i]]
                if branch_mvals[i][0] > 0:
                    branch_evals[i][0] = energies[ch_indexes[i]]*slopes[i] + offsets[i]
                    branch_tvals[i][0] = times[ch_indexes[i]]/1e9 #store all times in seconds
                    branch_counts[i][0] = energies[ch_indexes[i]]
                    branch_counts_ss[i][0] = energies[ch_indexes[i]] + np.random.uniform(-0.5, 0.5)
                else:
                    branch_evals[i][0] = 0
                    branch_tvals[i][0] = np.nan
                    branch_counts[i][0] = -1 #set to -1 when multiplicity is 0
                    branch_counts_ss[i][0] = -1
                
                if ch_names[i] == 'beam_off' and multiplicities[ch_indexes[i]] == 1:
                    last_beam_off_time = times[ch_indexes[i]]/1e9
                if ch_names[i] == 'chopper_off' and multiplicities[ch_indexes[i]] == 1:
                    last_chopper_off_time = times[ch_indexes[i]]/1e9
            
            tsbo[0] = np.max(times)/1e9 - last_beam_off_time
            tsco[0] = np.max(times)/1e9 - last_chopper_off_time
            out_tree.Fill()

        output_file.WriteObject(out_tree, "merged_data")
    os.replace(partial_output_path, output_path)

GET_DDAS_TIME_MATCH_THRESHOLD = 10e-6 # seconds
GET_DDAS_MAX_CLOCK_DRIFT = 1e-4 # fractional difference in the rates of the GET and DDAS clocks. This was 45 ppm in e23035 run 277
GET_DDAS_ANCHOR_GAPS = 5 # number of time gaps from the starting point which have to agree to line up the start of a GET run
GET_DDAS_ANCHOR_WINDOW = 10 # number of events after the starting point searched for these gaps
GET_DDAS_MAX_SKIPPED_GET_EVENTS = 20 # max GET events skipped in a row before searching for a new starting point
GET_DDAS_MAX_SKIPPED_DDAS_TRIGGERS = 1000 # max DDAS triggers skipped in a row before searching for a new starting point

def match_get_to_ddas(ddas_trigger_times, get_timestamps, get_run_ids, log_file):
    '''
    Pair GET events with DDAS events that have an accepted GET trigger, using the time between consecutive events
    to find triggers that were only recorded by one of the two systems.

    GET timestamps don't carry over from one GET run to the next, so the time between the last event of one GET run
    and the first event of the next can't be compared with DDAS. Instead, the first event of each GET run is lined up by
    searching forward from where the previous GET run left off for a starting point where at least GET_DDAS_ANCHOR_GAPS
    of the time gaps to the next few GET events agree with gaps to the next few DDAS triggers.

    Returns an array with the index of the matched GET event for each DDAS trigger, or -1 if it has no GET event.
    '''
    ddas_trigger_times, get_timestamps = np.asarray(ddas_trigger_times, dtype=float), np.asarray(get_timestamps, dtype=float)
    get_run_ids = np.asarray(get_run_ids)
    matches = np.full(len(ddas_trigger_times), -1, dtype=np.int64)

    def gaps_agree(ddas_gap, get_gap):
        return np.abs(ddas_gap - get_gap) <= GET_DDAS_TIME_MATCH_THRESHOLD + GET_DDAS_MAX_CLOCK_DRIFT*np.abs(ddas_gap)

    def find_starting_point(ddas_index, get_index, get_end):
        # Prefer the starting point which skips the fewest DDAS triggers, then the fewest GET events
        for ddas_skip in range(min(GET_DDAS_MAX_SKIPPED_DDAS_TRIGGERS, len(ddas_trigger_times) - ddas_index)):
            for get_skip in range(min(GET_DDAS_MAX_SKIPPED_GET_EVENTS, get_end - get_index)):
                i, j = ddas_index + ddas_skip, get_index + get_skip
                # Look a few events past GET_DDAS_ANCHOR_GAPS, so that an extra event on either side right after
                # the true starting point doesn't cause it to be skipped
                ddas_gaps = ddas_trigger_times[i+1:i+1+GET_DDAS_ANCHOR_WINDOW] - ddas_trigger_times[i]
                get_gaps = get_timestamps[j+1:min(j+1+GET_DDAS_ANCHOR_WINDOW, get_end)] - get_timestamps[j]
                if len(ddas_gaps) == 0 or len(get_gaps) == 0:
                    continue
                num_agreeing = np.sum(np.any(gaps_agree(ddas_gaps[np.newaxis, :], get_gaps[:, np.newaxis]), axis=1))
                if num_agreeing >= min(GET_DDAS_ANCHOR_GAPS, len(get_gaps)):
                    return i, j
        return None

    run_starts = np.flatnonzero(np.diff(get_run_ids) != 0) + 1
    ddas_index = 0 # first DDAS trigger which hasn't been matched or skipped
    for get_start, get_end in zip([0, *run_starts], [*run_starts, len(get_timestamps)]):
        run = int(get_run_ids[get_start])
        get_index = get_start
        last_match = None
        num_matched = 0
        while get_index < get_end and ddas_index < len(ddas_trigger_times):
            if last_match is None:
                start = find_starting_point(ddas_index, get_index, get_end)
                if start is None:
                    log_file.write(f'GET run {run}: could not line up GET events starting at index {get_index} with DDAS triggers starting at index {ddas_index}\n')
                    break
                if start != (ddas_index, get_index):
                    log_file.write(f'GET run {run}: lined up GET event index {start[1]} with DDAS trigger index {start[0]}, skipping {start[1] - get_index} GET events and {start[0] - ddas_index} DDAS triggers\n')
                ddas_index, get_index = start
            else:
                ddas_gap = ddas_trigger_times[ddas_index] - ddas_trigger_times[last_match[0]]
                get_gap = get_timestamps[get_index] - get_timestamps[last_match[1]]
                if not gaps_agree(ddas_gap, get_gap):
                    # The gaps are measured from the last matched pair, so whichever side has the smaller gap has an extra event
                    if ddas_gap < get_gap:
                        log_file.write(f'GET run {run}: DDAS trigger index {ddas_index} has no GET event\n')
                        ddas_index += 1
                    else:
                        log_file.write(f'GET run {run}: GET event index {get_index} has no DDAS trigger\n')
                        get_index += 1
                    if ddas_index - last_match[0] > GET_DDAS_MAX_SKIPPED_DDAS_TRIGGERS or get_index - last_match[1] > GET_DDAS_MAX_SKIPPED_GET_EVENTS:
                        log_file.write(f'GET run {run}: lost track of matching events after GET event index {last_match[1]}, searching for a new starting point\n')
                        ddas_index, get_index = last_match[0] + 1, last_match[1] + 1
                        last_match = None
                    continue
            matches[ddas_index] = get_index
            last_match = (ddas_index, get_index)
            num_matched += 1
            ddas_index += 1
            get_index += 1
        # Unmatched DDAS triggers after the last match may belong to the next GET run
        if last_match is not None:
            ddas_index = last_match[0] + 1
        num_events = get_end - get_start
        log_file.write(f'GET run {run}: matched {num_matched} of {num_events} GET events\n')
        if num_matched < 0.99*num_events:
            print(f'WARNING: only matched {num_matched} of {num_events} events in GET run {run} with DDAS triggers', file=sys.stderr)
    log_file.write(f'{np.sum(matches < 0)} of {len(matches)} DDAS triggers have no GET event\n')
    return matches

def make_tpc_friend_file(experiment, ddas_run, tpc_ini_filename=""):
    merged_path = get_ddas_root_file_path(experiment, ddas_run)
    if not os.path.exists(merged_path):
        make_ddas_root_file(experiment, ddas_run)
        
    output_path = get_tpc_friend_file_path(experiment, ddas_run, tpc_ini_filename)
    log_path = os.path.splitext(output_path)[0] + '.log'
    
    with ROOT.TFile(merged_path, "READ") as input_file, open(log_path, 'w') as log_file, ROOT.TFile(output_path, "RECREATE") as output_file:
        in_tree = input_file.Get("merged_data")
        
        import importlib
        try:
            exp_runs = importlib.import_module(f"{experiment}_analysis.{experiment}_runs")
            get_runs = np.sort(exp_runs.run_df['GET'][(exp_runs.run_df['DDAS']==ddas_run) & np.isfinite(exp_runs.run_df['GET'])] )
        except (ImportError, AttributeError):
            log_file.write(f'no runs module or compatible run_df found for {experiment}, skipping GET data merge\n')
            get_runs = []

        if len(get_runs)>0:
            log_file.write('found corresponding GET runs: %s\n'%str(get_runs))
            tpc_energy_MeV = exp_runs.get_energy_MeV(get_runs, tpc_ini_filename=tpc_ini_filename)
            proton_mask = exp_runs.get_proton_mask(get_runs, tpc_ini_filename=tpc_ini_filename)
            alpha_mask = exp_runs.get_alpha_mask(get_runs, tpc_ini_filename=tpc_ini_filename)
            track_lengths = exp_runs.get_length_mm(get_runs, tpc_ini_filename=tpc_ini_filename)
            track_angles = np.degrees(process_runs.get_angle(experiment, get_runs, config_filename=tpc_ini_filename))
            get_timestamps = process_runs.get_quantity('timestamps', experiment, get_runs, config_filename=tpc_ini_filename)
            veto_mask = exp_runs.get_veto_mask(get_runs, tpc_ini_filename=tpc_ini_filename)
            track_centroids = process_runs.get_quantity('track_center', experiment, get_runs, config_filename=tpc_ini_filename)
            get_run_ids, get_event_ids = process_runs.get_run_and_event_numbers(experiment, get_runs, config_filename=tpc_ini_filename)
        else:
            log_file.write('no corresponding GET runs found \n')
            get_timestamps = []
            track_centroids = []
            get_event_ids = []
            get_run_ids = []

        out_tree = ROOT.TTree("tpc_data", "tpc_data")
        tree_tpc_energy, tree_track_length = np.array([0.], dtype=np.float64), np.array([0.], dtype=np.float64)
        tree_track_angle = np.array([0.], dtype=np.float64)
        tree_ptype = np.array([0], dtype=np.int32)
        tree_should_veto = np.array([True], dtype=bool)
        tree_get_timestamp = np.array([np.nan])
        
        out_tree.Branch('tpc_energy', tree_tpc_energy, 'tpc_energy/D')
        out_tree.Branch('tpc_track_length', tree_track_length, 'tpc_track_length/D')
        out_tree.Branch('tpc_particle_id', tree_ptype, 'tpc_particle_id/I')
        out_tree.Branch('tpc_should_veto', tree_should_veto, 'tpc_should_veto/O')
        out_tree.Branch('tpc_track_angle', tree_track_angle, 'tpc_track_angle/D')
        out_tree.Branch('get_timestamp', tree_get_timestamp, 'get_timestamp/D')
        
        tree_track_centroid = np.array([0., 0., 0.], dtype=np.float64)
        out_tree.Branch('tpc_track_centroid', tree_track_centroid, 'tpc_track_centroid[3]/D')
        
        tree_get_event_id = np.array([0], dtype=np.int32)
        out_tree.Branch('get_event_id', tree_get_event_id, 'get_event_id/I')
        tree_get_run_id = np.array([0], dtype=np.int32)
        out_tree.Branch('get_run_id', tree_get_run_id, 'get_run_id/I')

        # Find the DDAS events with an accepted GET trigger, in entry order, and match them with GET events.
        # Events where the trigger channel fired more than once are included: in e23035 run 182 each of these matched a single GET event.
        matched_get_indexes = np.full(in_tree.GetEntries(), -1, dtype=np.int64)
        if len(get_timestamps) > 0:
            in_tree.SetEstimate(in_tree.GetEntries() + 1)
            num_triggers = in_tree.Draw('Entry$:get_trig_accepted_t', 'get_trig_accepted_m>=1', 'goff')
            trigger_entries = np.array([int(in_tree.GetV1()[k]) for k in range(num_triggers)], dtype=np.int64)
            trigger_times = np.array([in_tree.GetV2()[k] for k in range(num_triggers)])
            matched_get_indexes[trigger_entries] = match_get_to_ddas(trigger_times, get_timestamps, get_run_ids, log_file)

        for ddas_index in tqdm.tqdm(range(in_tree.GetEntries())):
            get_evt_index = matched_get_indexes[ddas_index]
            if get_evt_index >= 0:
                tree_tpc_energy[0] = tpc_energy_MeV[get_evt_index]*1000
                tree_track_length[0] = track_lengths[get_evt_index]
                tree_ptype[0] = 0
                if proton_mask[get_evt_index]:
                    tree_ptype[0] = 1
                if alpha_mask[get_evt_index]:
                    tree_ptype[0] = 2
                tree_should_veto[0] = not veto_mask[get_evt_index]
                tree_get_timestamp[0] = get_timestamps[get_evt_index]
                tree_track_angle[0] = track_angles[get_evt_index]

                tree_track_centroid[0] = track_centroids[get_evt_index][0]
                tree_track_centroid[1] = track_centroids[get_evt_index][1]
                tree_track_centroid[2] = track_centroids[get_evt_index][2]
                tree_get_event_id[0] = int(get_event_ids[get_evt_index])
                tree_get_run_id[0] = int(get_run_ids[get_evt_index])
            else: #no corresponding TPC event; set TPC quantities to NaN
                tree_get_timestamp[0] = tree_tpc_energy[0] = tree_track_length[0] = tree_track_angle[0] = np.nan
                tree_ptype[0] = -1
                tree_should_veto[0] = True
                
                tree_track_centroid[0] = tree_track_centroid[1] = tree_track_centroid[2] = np.nan
                tree_get_event_id[0] = -1
                tree_get_run_id[0] = -1
            out_tree.Fill()
            
        output_file.WriteObject(out_tree, "tpc_data")


current_run, current_file, current_data = np.nan, None, None
def show_pid(experiment, ddas_run):
    global current_run
    global current_file
    global current_data
    if current_run != ddas_run:
        current_run = ddas_run
        current_file = ROOT.TFile(get_ddas_root_file_path(experiment, ddas_run), 'READ')
        current_data = current_file.Get('merged_data')
    current_data.Draw('msx100_e:(cross_scint_b2_t - db_5_scint_t)>>(1000,-0.63e-6,-0.6e-6,1000,4000,8000)', 'cross_scint_b2_m==1 && db_5_scint_m==1 &&msx100_m==1', 'colz')

def _worker_get_cross_scint_counts(experiment, ddas_run):
    df = ROOT.RDataFrame('merged_data', get_ddas_root_file_path(experiment, ddas_run))
    return df.Sum('cross_scint_b2_m').GetValue()

rdataframes = {}
def get_cross_scint_counts(experiment, ddas_run, num_workers=None):
    if is_iterable_runs(ddas_run):
        run_list = list(ddas_run)
        total_counts = 0
        if num_workers is None or num_workers > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(_worker_get_cross_scint_counts, experiment, run) for run in run_list]
                for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(run_list), desc="Counting cross scintillator"):
                    total_counts += future.result()
        else:
            for run in tqdm.tqdm(run_list, desc="Counting cross scintillator"):
                total_counts += get_cross_scint_counts(experiment, run, num_workers=1)
        return total_counts

    global rdataframes
    if ddas_run not in rdataframes:
        rdataframes[ddas_run] = ROOT.RDataFrame('merged_data', get_ddas_root_file_path(experiment, ddas_run))
    return rdataframes[ddas_run].Sum('cross_scint_b2_m').GetValue()

def get_cross_scint_counts_during_get_run(experiment, get_run):
    pass #TODO

def get_ddas_run_duration(experiment, ddas_run):
    pass#TODO

def is_iterable_runs(obj):
    if isinstance(obj, (str, bytes)):
        return False
    try:
        iter(obj)
        return True
    except TypeError:
        return False

import os
import sys
import hashlib
import concurrent.futures
import tqdm
import ROOT

def get_file_fingerprint(path):
    '''
    Identify the version of a file on disk by its resolved path, size, and modification time. Cache keys
    include this so cached results are rebuilt when the files they were made from change.
    '''
    stat = os.stat(path)
    return (os.path.realpath(path), stat.st_size, stat.st_mtime_ns)

def check_friend_shadowing(tree, friend_tree, expressions, tree_file_path):
    '''
    ROOT looks up a branch name in the main tree before its friends, so a branch that exists in both trees
    silently hides the friend's copy. Raise if any of the expressions uses one of these names without a tree prefix.
    '''
    friend_branches = {b.GetName() for b in friend_tree.GetListOfBranches()}
    duplicated = sorted(friend_branches & {b.GetName() for b in tree.GetListOfBranches()})
    shadowed = [name for name in duplicated if any(re.search(rf'(?<![\w.]){re.escape(name)}(?!\w)', exp) for exp in expressions)]
    if shadowed:
        friend_name = friend_tree.GetName()
        raise ValueError(f"{tree_file_path} has its own copies of branches {shadowed}, which hide the ones in friend tree '{friend_name}'. "
                         f"Remake the file without these branches, or refer to them explicitly as {friend_name}.<branch>.")

def _worker_fill_run(experiment, run, binning, var_exp, selection, force_recreate, tpc_ini_filename=""):
    cache_dir = os.path.join(BASE_DIR, f'{experiment}_analysis', 'hist_cache')
    os.makedirs(cache_dir, exist_ok=True)

    # The input files have to exist before the cache key is made, since the key records which version of them was used
    data_file_path = get_ddas_root_file_path(experiment, run)
    if not os.path.exists(data_file_path):
        make_ddas_root_file(experiment, run)
    input_files = [data_file_path]

    needs_tpc = any(kw in var_exp or kw in selection for kw in ['tpc_', 'get_timestamp', 'get_event_id', 'get_run_id'])
    if needs_tpc:
        if not tpc_ini_filename:
            raise ValueError(f"tpc_ini_filename is required because TPC or GET variables are used in var_exp or selection (var_exp: '{var_exp}', selection: '{selection}')")
        tpc_friend_path = get_tpc_friend_file_path(experiment, run, tpc_ini_filename)
        if not os.path.exists(tpc_friend_path):
            try:
                make_tpc_friend_file(experiment, run, tpc_ini_filename)
            except Exception as e:
                import traceback
                import sys
                print(f"\n{'='*80}\n!!! CRITICAL FAILURE in make_tpc_friend_file for RUN {run} !!!\n{'='*80}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                print(f"{'='*80}\n", file=sys.stderr)

                if os.path.exists(tpc_friend_path):
                    try:
                        os.remove(tpc_friend_path)
                    except OSError:
                        pass
                raise
        input_files.append(tpc_friend_path)

    input_fingerprints = [get_file_fingerprint(path) for path in input_files]
    unique_string = str((run, tuple(binning), var_exp, selection, tpc_ini_filename, input_fingerprints)).encode('utf-8')
    hash_name = "h_" + hashlib.md5(unique_string).hexdigest()
    cache_file_path = os.path.join(cache_dir, f"{hash_name}.root")
    
    # --- LAYER 1: PROACTIVE CACHE HEALTH CHECK ---
    if not force_recreate and os.path.exists(cache_file_path):
        is_healthy = False
        try:
            cf = ROOT.TFile.Open(cache_file_path, 'READ')
            if cf and not cf.IsZombie():
                # Verify the histogram actually exists inside the file
                if cf.Get(hash_name): 
                    is_healthy = True
            if cf: 
                cf.Close()
        except OSError:
            pass
            
        if is_healthy:
            return cache_file_path, hash_name
        else:
            # The file exists but is corrupted/empty. Delete it.
            try:
                os.remove(cache_file_path)
            except OSError:
                pass

    # --- BUILD THE HISTOGRAM ---
    data_file = ROOT.TFile.Open(data_file_path, 'READ')
    
    if not data_file or data_file.IsZombie():
        raise FileNotFoundError(f"Could not open ROOT data file: {data_file_path}")
        
    tree = data_file.Get('merged_data')
    if not tree:
        data_file.Close()
        raise ValueError(f"Could not find TTree 'merged_data' in {data_file_path}.")

    tpc_friend_file = None
    if needs_tpc:
        tpc_friend_file = ROOT.TFile.Open(tpc_friend_path, 'READ')
        if not tpc_friend_file or tpc_friend_file.IsZombie():
            raise FileNotFoundError(f"Could not open TPC friend file: {tpc_friend_path}")
        tpc_tree = tpc_friend_file.Get('tpc_data')
        if not tpc_tree:
            raise ValueError(f"Could not find 'tpc_data' tree in file {tpc_friend_path}")
        tree.AddFriend(tpc_tree)
        check_friend_shadowing(tree, tpc_tree, [var_exp, selection], data_file_path)
        
    if ':' in var_exp:
        raw_hist = ROOT.TH2D(hash_name, "", *binning)
    else:
        raw_hist = ROOT.TH1D(hash_name, "", *binning)
        
    tree.Draw(f'{var_exp}>>{hash_name}', selection, 'goff')
    raw_hist.SetDirectory(0)
    data_file.Close()

    cache_file = ROOT.TFile.Open(cache_file_path, 'RECREATE')
    raw_hist.SetDirectory(cache_file)
    raw_hist.Write("", ROOT.TObject.kOverwrite)
    raw_hist.SetDirectory(0)
    cache_file.Close()

    return cache_file_path, hash_name


def get_histogram(experiment, ddas_run, binning, hist_name, hist_title, var_exp, selection="", force_recreate=False, num_workers=1, tpc_ini_filename=""):
    needs_tpc = any(kw in var_exp or kw in selection for kw in ['tpc_', 'get_timestamp', 'get_event_id', 'get_run_id'])
    if needs_tpc:
        import importlib
        try:
            exp_runs = importlib.import_module(f"{experiment}_analysis.{experiment}_runs")
            run_list_for_check = list(ddas_run) if is_iterable_runs(ddas_run) else [ddas_run]
            get_runs = np.unique(exp_runs.run_df['GET'][
                (exp_runs.run_df['DDAS'].isin(run_list_for_check)) & np.isfinite(exp_runs.run_df['GET'])
            ])
            if len(get_runs) > 0:
                process_runs.ensure_processed(experiment, get_runs, config_filename=tpc_ini_filename, show_progress=True)
        except (ImportError, AttributeError):
            pass

    # --- MULTIPLE RUNS LOGIC ---
    if is_iterable_runs(ddas_run):
        sum_hist = None
        # Drop repeated runs so they aren't added to the sum twice
        run_list = list(dict.fromkeys(ddas_run))
        
        if num_workers > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(_worker_fill_run, experiment, run, binning, var_exp, selection, force_recreate, tpc_ini_filename) 
                    for run in run_list
                ]
                
                # Forced stdout and dynamic columns
                for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(run_list), desc=f"Filling {hist_name} (Parallel)", file=sys.stdout, dynamic_ncols=True, leave=True):
                    cache_file_path, hash_name = future.result()
                    
                    # --- LAYER 2: RACE CONDITION FAILSAFE ---
                    cf = None
                    import time
                    for _ in range(5):
                        try:
                            cf = ROOT.TFile.Open(cache_file_path, 'READ')
                            if cf and not cf.IsZombie():
                                break
                        except OSError:
                            pass
                        time.sleep(1)

                    temp_hist = cf.Get(hash_name) if cf and not cf.IsZombie() else None
                    
                    if not temp_hist:
                        if cf: cf.Close()
                        print(f"\nWarning: Failsafe triggered. Cache {cache_file_path} missing histogram or file unreadable. Forcing recreate...")
                        cache_file_path, hash_name = _worker_fill_run(experiment, run, binning, var_exp, selection, force_recreate=True, tpc_ini_filename=tpc_ini_filename)
                        for _ in range(5):
                            try:
                                cf = ROOT.TFile.Open(cache_file_path, 'READ')
                                if cf and not cf.IsZombie():
                                    break
                            except OSError:
                                pass
                            time.sleep(1)
                        temp_hist = cf.Get(hash_name) if cf and not cf.IsZombie() else None
                        
                    hist = temp_hist.Clone(f"{hist_name}_temp") if temp_hist else None
                    if hist:
                        hist.SetDirectory(0)
                    if cf: cf.Close()
                    
                    if not hist:
                        raise RuntimeError(f"Failed to create or read histogram from {cache_file_path}")
                    
                    if sum_hist is None:
                        sum_hist = hist.Clone(hist_name)
                        sum_hist.SetTitle(hist_title)
                        sum_hist.SetDirectory(0)
                    else:
                        sum_hist.Add(hist)
        else:
            # Forced stdout and dynamic columns
            for run in tqdm.tqdm(run_list, desc=f"Filling {hist_name} (Sequential)", file=sys.stdout, dynamic_ncols=True, leave=True):
                temp_name = f"{hist_name}_run{int(run)}"
                hist = get_histogram(experiment, run, binning, temp_name, hist_title, var_exp, selection, force_recreate, num_workers=1, tpc_ini_filename=tpc_ini_filename)
                
                if sum_hist is None:
                    sum_hist = hist.Clone(hist_name)
                    sum_hist.SetDirectory(0)
                else:
                    sum_hist.Add(hist)
                    
        return sum_hist

    # --- SINGLE RUN LOGIC ---
    # Wrapped in a 1-step progress bar so you always see it
    with tqdm.tqdm(total=1, desc=f"Filling {hist_name} (Single)", file=sys.stdout, dynamic_ncols=True, leave=True) as pbar:
        cache_file_path, hash_name = _worker_fill_run(experiment, ddas_run, binning, var_exp, selection, force_recreate, tpc_ini_filename)
        pbar.update(1)
    
    # --- LAYER 2: SINGLE RUN FAILSAFE ---
    cf = ROOT.TFile.Open(cache_file_path, 'READ')
    temp_hist = cf.Get(hash_name)
    
    if not temp_hist:
        cf.Close()
        print(f"\nWarning: Failsafe triggered. Cache {cache_file_path} missing histogram. Forcing recreate...")
        cache_file_path, hash_name = _worker_fill_run(experiment, ddas_run, binning, var_exp, selection, force_recreate=True, tpc_ini_filename=tpc_ini_filename)
        cf = ROOT.TFile.Open(cache_file_path, 'READ')
        temp_hist = cf.Get(hash_name)
        
    final_hist = temp_hist.Clone(hist_name)
    final_hist.SetTitle(hist_title)
    final_hist.SetDirectory(0)
    cf.Close()
    
    return final_hist

def get_first_and_last_ddas_time(experiment, ddas_run):
    '''
    Get first and last time stamps in the run in seconds
    '''
    with ROOT.TFile(get_root_file_path(experiment, ddas_run), 'READ') as f:
        tree = f.Get('tree')
        times = np.zeros(NUM_TOTAL_CH)
        tree.SetBranchAddress('times', times)
        tree.GetEntry(0)
        start_time = np.min(times[times>0])
        tree.GetEntry(tree.GetEntries()-1)
        stop_time = np.max(times)
        return start_time/1e9, stop_time/1e9


#code used to generate "channel_map.csv"
# gamma_cal_table = np.genfromtxt('e23035_analysis/init_ge_cal.csv', delimiter=',', skip_header=1)
# gamma_slopes = gamma_cal_table[:, 2]
# gamma_offsets = gamma_cal_table[:, 1]
# for k in CH_MAP.__dict__:
#     v = CH_MAP.__dict__[k]
#     if type(v) == int:
#         if 'clover' in k.lower():
#             cal_index = CH_MAP.GE_CALIBRATION_INDEXES[np.where(v==CH_MAP.GE_INDECIES)]
#             slope, offset = gamma_slopes[cal_index], gamma_offsets[cal_index]
#         else:
#             slope, offset = 1,0
#         print('%d, %s, %f, %f'%(v, k.lower(), slope, offset)), 

def show_selected_event(experiment, ddas_run, selection, index):
    """
    Given a DDAS run and a ROOT selection string, retrieves the index-th matching event
    and displays its corresponding GET TPC data.
    """
    import ROOT
    from raw_viewer import process_runs
    import os
    
    root_file_path = get_ddas_root_file_path(experiment, ddas_run)
    if not os.path.exists(root_file_path):
        raise FileNotFoundError(f"Merged root file not found for {experiment} run {ddas_run}")
        
    df = ROOT.RDataFrame("merged_data", root_file_path)
    filtered_df = df.Filter(selection)
    
    count = filtered_df.Count().GetValue()
    if index >= count:
        print(f"Error: index {index} is out of bounds for {count} selected events.")
        return
        
    data = filtered_df.Range(index, index+1).AsNumpy(["get_event_id", "get_run_id"])
    evt = int(data["get_event_id"][0])
    run = int(data["get_run_id"][0])
    
    if evt < 0 or run < 0:
        print(f"Error: The selected event (index {index}) does not have corresponding GET TPC data.")
        return
        
    print(f'experiment {experiment} ddas_run {ddas_run} GET run {run} evt {evt}')
    
    h5file = process_runs.get_h5_file(experiment, run)
    h5file.show_2d_projection(evt, block=False)
    h5file.plot_3d_traces(evt, threshold=h5file.length_counts_threshold, block=False)
    h5file.plot_traces(evt, block=False)
    
    import matplotlib.pyplot as plt
    plt.show(block=False)