import os
import pickle
import gzip

import ROOT
import numpy as np
import matplotlib.pylab as plt
import matplotlib.colors
import tqdm

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
    CLOVER_1_INDECIES = np.array([CLOVER_1A, CLOVER_1B, CLOVER_1C, CLOVER_1D])
    CLOVER_2A = 3*16 + 4
    CLOVER_2B = 3*16 + 5
    CLOVER_2C = 3*16 + 6
    CLOVER_2D = 3*16 + 7
    CLOVER_2_INDECIES = np.array([CLOVER_2A, CLOVER_2B, CLOVER_2C, CLOVER_2D])
    CLOVER_3A = 3*16 + 8
    CLOVER_3B = 5*16 + 4
    CLOVER_3C = 5*16 + 5
    CLOVER_3D = 3*16 + 11
    CLOVER_3_INDECIES = np.array([CLOVER_3A, CLOVER_3B, CLOVER_3C, CLOVER_3D])
    #clover 4 not installed
    CLOVER_5A = 4*16 + 0
    CLOVER_5B = 4*16 + 1
    CLOVER_5C = 4*16 + 2
    CLOVER_5D = 4*16 + 3
    CLOVER_5_INDECIES = np.array([CLOVER_5A, CLOVER_5B, CLOVER_5C, CLOVER_5D])
    CLOVER_6A = 4*16 + 4
    CLOVER_6B = 4*16 + 5
    CLOVER_6C = 4*16 + 6
    CLOVER_6D = 4*16 + 7
    CLOVER_6_INDECIES = np.array([CLOVER_6A, CLOVER_6B, CLOVER_6C, CLOVER_6D])
    CLOVER_7A = 5*16 + 6
    CLOVER_7B = 5*16 + 7
    CLOVER_7C = 5*16 + 8
    CLOVER_7D = 4*16 + 11
    CLOVER_7_INDECIES = np.array([CLOVER_7A, CLOVER_7B, CLOVER_7C, CLOVER_7D])
    #clover 8 not installed
    CLOVER_9A = 9*16 + 0
    CLOVER_9B = 9*16 + 1
    CLOVER_9C = 9*16 + 2
    CLOVER_9D = 9*16 + 3
    CLOVER_9_INDECIES = np.array([CLOVER_9A, CLOVER_9B, CLOVER_9C, CLOVER_9D])
    CLOVER_10A = 9*16 + 4
    CLOVER_10B = 9*16 + 5
    CLOVER_10C = 9*16 + 6
    CLOVER_10D = 9*16 + 7
    CLOVER_10_INDECIES = np.array([CLOVER_10A, CLOVER_10B, CLOVER_10C, CLOVER_10D])
    CLOVER_11A = 9*16 + 8
    CLOVER_11B = 9*16 + 9
    CLOVER_11C = 9*16 + 10
    CLOVER_11D = 9*16 + 11
    CLOVER_11_INDECIES = np.array([CLOVER_11A, CLOVER_11B, CLOVER_11C, CLOVER_11D])
    #list of all germnaium channels
    GE_INDECIES = np.concatenate([CLOVER_1_INDECIES, CLOVER_2_INDECIES, CLOVER_3_INDECIES, CLOVER_5_INDECIES,
                              CLOVER_6_INDECIES, CLOVER_7_INDECIES, CLOVER_9_INDECIES, CLOVER_10_INDECIES,
                              CLOVER_11_INDECIES])
    #list of crystal ids corresponding to the above inecies
    GE_CALIBRATION_INDEXES = [0,1,2,3,4,5,6,7,8,9,10,11,16,17,18,19,20,21,22,23,24,25,26,27,32,33,34,35,36,37,38,39,40,41,42,43]

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

def make_cal_root_file(run):
    pass