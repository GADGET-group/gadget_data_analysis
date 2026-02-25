import hashlib
import os
import pickle
import gzip
import re
import subprocess

import ROOT
import numpy as np
import matplotlib.pylab as plt
import matplotlib.colors
import tqdm

from e23035_analysis import e23035_runs
from raw_viewer import process_runs

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

def extract_all_data(experiment, run):
    file = ROOT.TFile(get_root_file_path(experiment, run), "READ")
    tree = file.Get("tree")
    energies, times, multiplicities = np.zeros(NUM_TOTAL_CH, dtype=np.int32), np.zeros(NUM_TOTAL_CH), np.zeros(NUM_TOTAL_CH,dtype=np.int32)
    tree.SetBranchAddress("energies", energies)
    tree.SetBranchAddress("times", times)
    tree.SetBranchAddress("multiplicity", multiplicities)
    shape = (tree.GetEntries(), NUM_TOTAL_CH)
    es, ts, ms = np.zeros(shape), np.zeros(shape), np.zeros(shape), 
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        es[i,:] = energies
        ts[i,:] = times
        ms[i,:] = multiplicities
    return es, ts, ms
    return np.array(es), np.array(ts), np.array(ms)

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

def time_since_beam_off(ts, ch_select):
    '''
    Get time since beam as turned off for each accepted trigger
    ''' 
    beam_off_times = ts[:, CH_MAP.BEAM_OFF]
    beam_off_times = beam_off_times[beam_off_times >= 0]
    event_times = np.max(ts[:, ch_select], axis=1)
    event_times = event_times[event_times >= 0]
    
    to_return = np.zeros(len(event_times))
    i, j = 0,0
    while i < len(event_times):
        while (j < len(beam_off_times) - 1) and (beam_off_times[j+1] < event_times[i]):
            j += 1
        to_return[i] = event_times[i] - beam_off_times[j]
        i += 1
    return to_return/1e9


def show_dE_dt_pid(es, ts, ms):
    dE_ch = CH_MAP.MSX100
    t_start_ch = CH_MAP.DB_5_SCINT
    t_stop_ch = CH_MAP.CROSS_SCINT_B2
    dt = ts[:, t_stop_ch] - ts[:,t_start_ch]
    dE = es[:, dE_ch]
    plt_mask = (dt<-580) & (dt>-680) & (dE>0)
    plt.figure()
    plt.hist2d(dt[plt_mask], dE[plt_mask], 1000,norm=matplotlib.colors.LogNorm())
    plt.colorbar()
    plt.show(block=False)

def show_gamma_energy_alignment_plot(es):
    energy_bins = np.linspace(0,2**22, 1000)
    image = np.zeros(len(CH_MAP.GE_INDECIES), len(energy_bins))

def get_calibrated_gamma_energies(es, file='e23035_analysis/init_ge_cal.csv'):
    '''
    [CLOVER_1_INDECIES, CLOVER_2_INDECIES, CLOVER_3_INDECIES, CLOVER_5_INDECIES,
                              CLOVER_6_INDECIES, CLOVER_7_INDECIES, CLOVER_9_INDECIES, CLOVER_10_INDECIES,
                              CLOVER_11_INDECIES])
    '''
    cal_table = np.genfromtxt(file, delimiter=',', skip_header=1)
    slopes = cal_table[:, 2]
    offsets = cal_table[:, 1]
    to_return = np.zeros((len(es), len(CH_MAP.GE_INDECIES)))
    for i in range(len(CH_MAP.GE_INDECIES)):
        cal_index = CH_MAP.GE_CALIBRATION_INDEXES[i]
        to_return[:, i] = offsets[cal_index] + slopes[cal_index]*es[:, CH_MAP.GE_INDECIES[i]]
    return to_return

def get_merged_root_file_path(ddas_run):
    root_file_path = get_root_file_path(experiment='e23035', run=ddas_run)
    return os.path.join(os.path.split(root_file_path)[0], 'run%d_merged.root'%ddas_run)

def make_merged_root_file(ddas_run):
    '''
    Merge GET data stream into an existing root file, adding a new TTree called "merged_data".
    The ddas root file is assumed to be in the uncalibrated raw format assumed by the above function, but the 
    file written will include rough energy calibration for all the gamma detectors, and branches will have friendly names rather than a 
    array of unlabeled channels.

    Branches will be added for each channel in channel_map.csv, and given the specified name with a _t ending for ddas time,
    _e for energy, or _m for multiplicity.
    Additionally, the following branches will be added for GET data:
    Energy, track length, particle type (0=uncatagorized, 1=proton, 2=alpha), should_veto, get_timetamp

    An initial energy calibraiton will be provided for DDAS channels using the slope and offset specified in the channel map.

    All times will be stored in seconds, and energies will be stored in keV if a calibraiton is available. Track lengths are in mm.

    '''
    root_file_path = get_root_file_path(experiment='e23035', run=ddas_run)

    log_path = os.path.join(os.path.split(root_file_path)[0], 'run%d_merge.log'%ddas_run)
    output_path = get_merged_root_file_path(ddas_run)
    with ROOT.TFile(root_file_path, "READ") as input_file, open(log_path, 'w') as log_file, ROOT.TFile(output_path, "RECREATE") as output_file:
        git_version = subprocess.run(['git', 'rev-parse', '--verify', 'HEAD'], capture_output=True, text=True, check=True).stdout
        git_status = subprocess.run(['git', 'status'], capture_output=True, text=True, check=True).stdout
        git_diff = subprocess.run(['git', 'diff'], capture_output=True, text=True, check=True).stdout
        log_file.write('preparing to merge ddas run %d with any corresponding GET runs\n'%ddas_run)
        log_file.write('git commit %s\n'%git_version)
        log_file.write('git status: %s\n'%git_status)
        log_file.write('git diff: %s\n'%git_diff)

        get_runs = np.sort(e23035_runs.run_df['GET'][(e23035_runs.run_df['DDAS']==ddas_run) & np.isfinite(e23035_runs.run_df['GET'])] )
        if len(get_runs)>0:
            log_file.write('found corresponding GET runs: %s\n'%str(get_runs))
            tpc_energy_MeV = e23035_runs.get_energy_MeV(get_runs)
            proton_mask = e23035_runs.get_proton_mask(get_runs)
            alpha_mask = e23035_runs.get_alpha_mask(get_runs)
            track_lengths = e23035_runs.get_length_mm(get_runs)
            get_timestamps = process_runs.get_quantity('timestamps', 'e23035', get_runs)
            veto_mask = e23035_runs.get_veto_mask(get_runs)
        else:
            log_file.write('no corresponding GET runs found \n')
            get_timestamps = []

        ddas_ch_map_path = 'e23035_analysis/channel_map.csv'
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
        tree_tpc_energy, tree_track_length = np.array([0.], dtype=np.float64), np.array([0.], dtype=np.float64)
        tree_ptype = np.array([0], dtype=np.int32)
        tree_should_veto = np.array([True], dtype=bool)
        tree_get_timestamp = np.array([np.nan])

        for i in range(len(ch_names)):
            out_tree.Branch(ch_names[i]+'_e', branch_evals[i], ch_names[i]+'_e/D')
            out_tree.Branch(ch_names[i]+'_t', branch_tvals[i], ch_names[i]+'_t/D')
            out_tree.Branch(ch_names[i]+'_m', branch_mvals[i], ch_names[i]+'_m/I')
        out_tree.Branch('tpc_energy', tree_tpc_energy, 'tpc_energy/D')
        out_tree.Branch('tpc_track_length', tree_track_length, 'tpc_track_length/D')
        out_tree.Branch('tpc_particle_id', tree_ptype, 'tpc_particle_id/I')
        out_tree.Branch('tpc_should_veto', tree_should_veto, 'tpc_should_veto/O')
        
        tsbo = np.array([np.nan], dtype=np.float64)
        out_tree.Branch('time_since_beam_off', tsbo, 'time_since_beam_off/D')

        log_file.write('Starting merge \n')
        ddas_index = 0
        get_evt_index = 0
        get_trig_accepted_index = ch_indexes[np.where(ch_names=='get_trig_accepted')][0]
        last_ddas_time, last_get_time = np.nan, np.nan
        GET_DDAS_TIME_MATCH_TRHESHOLD = 10e-6

        last_beam_off_time = np.nan

        for ddas_index in tqdm.tqdm(range(in_tree.GetEntries())):
            #copy over ddas values with calibration factors applied
            in_tree.GetEntry(ddas_index)
            for i in range(len(ch_names)):
                branch_mvals[i][0] = multiplicities[ch_indexes[i]]
                if branch_mvals[i][0] > 0:
                    branch_evals[i][0] = energies[ch_indexes[i]]*slopes[i] + offsets[i]
                    #print(branch_evals[i])
                    branch_tvals[i][0] = times[ch_indexes[i]]/1e9 #store all times in seconds
                else:
                    branch_evals[i][0] = 0
                    branch_tvals[i][0] = np.nan
                #check if beam just turned off
                #note that this uses the beam off signal and not chopper signal
                #and so is offset by 2 ms from the true beam off. This is desirable
                #because the chopper pules off briefl during beam on time for
                #diagnostic reasons
                if ch_names[i] == 'beam_off' and multiplicities[ch_indexes[i]] == 1:
                    last_beam_off_time = times[ch_indexes[i]]/1e9
            
            tsbo[0] = np.max(times)/1e9 - last_beam_off_time

            record_get_event = False
            if multiplicities[get_trig_accepted_index] == 1:
                if get_evt_index < len(get_timestamps):
                    get_time = get_timestamps[get_evt_index] - last_get_time
                    ddas_time = times[get_trig_accepted_index]/1e9
                    if last_get_time == np.nan: #first trigger
                        record_get_event = True
                    else: #check that delta between timestamps matches
                        if (get_time - last_get_time) - (ddas_time - last_ddas_time) > GET_DDAS_TIME_MATCH_TRHESHOLD:
                            log_file.write('GET event index %d doesn\'t match with  next DDAS event with a valid trigger; trigger likely not recorded in GET.'%get_evt_index)
                        elif (ddas_time - last_ddas_time)>(get_time - last_get_time) > GET_DDAS_TIME_MATCH_TRHESHOLD:
                            log_file.write('WARNING: GET event index %d not coppied into ROOT tree. No corresponding DDAS event!.'%get_evt_index)
                            print('WARNING: GET event index %d not coppied into ROOT tree. No corresponding DDAS event!.'%get_evt_index)
                            get_evt_index += 1
                        else:
                            record_get_event = True
                else:
                    log_file.write('detected GET trigger in DDAS data stream, but no remaining GET events to read')
            if record_get_event:
                tree_tpc_energy[0] = tpc_energy_MeV[get_evt_index]*1000
                tree_track_length[0] = track_lengths[get_evt_index]
                tree_ptype[0] = 0
                if proton_mask[get_evt_index]:
                    tree_ptype[0] = 1
                if alpha_mask[get_evt_index]:
                    tree_ptype[0] = 2
                tree_should_veto[0] = not veto_mask[get_evt_index]
                tree_get_timestamp[0] = get_time
                
                last_get_time = get_time
                last_ddas_time = ddas_time            
                tree_ptype
                get_evt_index += 1
                #print(tree_tpc_energy)
            else: #no corresponding TPC event; set TPC quantities to NaN
                tree_get_timestamp[0] = tree_tpc_energy[0] = tree_track_length[0] = np.nan
                tree_ptype[0] = -1
                tree_should_veto[0] = True
            out_tree.Fill()

        output_file.WriteObject(out_tree, "merged_data")

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


current_run, current_file, current_data = np.nan, None, None
def show_pid(ddas_run):
    global current_run
    global current_file
    global current_data
    if current_run != ddas_run:
        current_run = ddas_run
        current_file = ROOT.TFile(get_merged_root_file_path(ddas_run), 'READ')
        current_data = current_file.Get('merged_data')
    current_data.Draw('msx100_e:(cross_scint_b2_t - db_5_scint_t)>>(1000,-0.63e-6,-0.6e-6,1000,4000,8000)', 'cross_scint_b2_m==1 && db_5_scint_m==1 &&msx100_m==1', 'colz')

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

rdataframes = {}
def get_cross_scint_counts(ddas_run):
    global rdataframes
    if ddas_run not in rdataframes:
        rdataframes[ddas_run] = ROOT.RDataFrame('merged_data', get_merged_root_file_path(ddas_run))
    return rdataframes[ddas_run].Sum('cross_scint_b2_m').GetValue()

def get_cross_scint_counts_during_get_run(get_run):
    pass #TODO

def get_ddas_run_duration(ddas_run):
    pass#TODO

def get_summed_gamma_e_str():
    to_return = ''
    for num in [1,2,3,5,6,7,9,10,11]:
        for letter in ('a','b','c','d'):
            if len(to_return) > 0:
                to_return += ' + '
            to_return += ' clover_%d%s_e'%(num, letter)
    return to_return

def is_iterable_runs(obj):
    """Check if an object is iterable, explicitly excluding strings."""
    if isinstance(obj, (str, bytes)):
        return False
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def is_iterable_runs(obj):
    """Check if an object is iterable, explicitly excluding strings."""
    if isinstance(obj, (str, bytes)):
        return False
    try:
        iter(obj)
        return True
    except TypeError:
        return False

def get_histogram(ddas_run, binning, hist_name, hist_title, var_exp, selection="", force_recreate=False):
    '''
    Get a histogram from the merged data for a ddas run (or iterable of runs). 
    Individual histograms are cached natively in a ROOT file using hashed parameters.
    If an iterable of runs is provided, the function retrieves/caches each run individually 
    and returns their sum.

    ddas_run: run number to use (int/str) OR iterable of run numbers
    binning: tuple of (number of bins, low, high) for TH1D, or (nx, xmin, xmax, ny, ymin, ymax) for TH2D
    hist_name, hist_title: name and title to give the created histogram
    var_exp: Selection to pass to TTree.Draw. If a ':' is present, a TH2D is created.
    selection: Selection string or cut to pass to TTree.Draw. Defaults to "".
    force_recreate: If true, refill the histogram from the root file, bypassing the cache.
    '''
    
    # 1. Handle iterables of runs recursively
    if is_iterable_runs(ddas_run):
        sum_hist = None
        for run in ddas_run:
            temp_name = f"{hist_name}_run{run}"
            hist = get_histogram(run, binning, temp_name, hist_title, var_exp, selection, force_recreate)
            
            if sum_hist is None:
                sum_hist = hist.Clone(hist_name)
                sum_hist.SetDirectory(0)
            else:
                sum_hist.Add(hist)
                
        return sum_hist

    # 2. Setup Cache Directory and File
    cache_dir = 'e23035_analysis'
    os.makedirs(cache_dir, exist_ok=True)
    cache_file_path = os.path.join(cache_dir, 'hist_cache.root')
    
    # 3. Generate a unique, ROOT-safe hash for this specific histogram configuration
    unique_string = str((ddas_run, tuple(binning), var_exp, selection)).encode('utf-8')
    hash_name = "h_" + hashlib.md5(unique_string).hexdigest()

    # 4. Try to load from the ROOT cache file
    if not force_recreate and os.path.exists(cache_file_path):
        cache_file = ROOT.TFile.Open(cache_file_path, 'READ')
        if cache_file and not cache_file.IsZombie():
            cached_obj = cache_file.Get(hash_name)
            if cached_obj:
                final_hist = cached_obj.Clone(hist_name)
                final_hist.SetTitle(hist_title)
                final_hist.SetDirectory(0)  # Detach before closing cache_file
                cache_file.Close()
                return final_hist
            cache_file.Close()

    # 5. If not in cache (or force_recreate), fill from the raw merged data
    data_file_path = get_merged_root_file_path(ddas_run) 
    data_file = ROOT.TFile.Open(data_file_path, 'READ')
    
    if not data_file or data_file.IsZombie():
        raise FileNotFoundError(f"Could not open ROOT data file: {data_file_path}")
        
    tree = data_file.Get('merged_data')
    if not tree:
        data_file.Close()
        raise ValueError(f"Could not find TTree 'merged_data' in {data_file_path}.")
        
    if ':' in var_exp:
        raw_hist = ROOT.TH2D(hash_name, hist_title, *binning)
    else:
        raw_hist = ROOT.TH1D(hash_name, hist_title, *binning)
        
    # Draw directly into our hashed-name histogram
    tree.Draw(f'{var_exp}>>{hash_name}', selection, 'goff')
    
    # Detach from data_file BEFORE closing it
    raw_hist.SetDirectory(0)
    data_file.Close()

    # 6. Save the new histogram to the cache file using UPDATE mode
    cache_file = ROOT.TFile.Open(cache_file_path, 'UPDATE')
    raw_hist.SetDirectory(cache_file)
    raw_hist.Write("", ROOT.TObject.kOverwrite)
    
    # Detach from cache_file BEFORE closing it
    raw_hist.SetDirectory(0)
    cache_file.Close()

    # 7. Prepare the final histogram to return to the user
    raw_hist.SetName(hist_name)
    
    return raw_hist

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