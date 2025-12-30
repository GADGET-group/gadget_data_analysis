import os
import pickle
import gzip

import ROOT
import numpy as np
import matplotlib.pylab as plt

NUM_SLOTS = 10
CH_PER_SLOT = 16
NUM_TOTAL_CH = NUM_SLOTS*CH_PER_SLOT

class CH_MAP:
    MESH_PRE_AMP = 150
    GET_TRIG_ACCEPTED = 151
    SCA_LOGIC = 152
    VETO_LOGIC = 153
    #beam off is signal from ARIS used to switch GG
    BEAM_ON = 154
    BEAM_OFF = 155
    #chopper is actual control of accelerator
    CHOPPER_ON = 156
    CHOPPER_OFF = 157

    #PID channels
    MSX100 = 160
    MSX40 = 161
    CROSS_SCINT_B2 = 2*16+13
    CROSS_SCINT_T2 = 2*16+14
    DB_5_SCINT = 3*16
    DB_3_SCINT_L = 3*16 + 2

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
        for i in range(tree.GetEntries()):
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


def get_time_since_beam_off(es, ts, ms):
    '''
    Get time since beam as turned off for each accepted trigger
    ''' 
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
    return to_return

def show_dE_dt_pid(es, ts, ms):
    dE_ch = CH_MAP.MSX100
    t_start_ch = CH_MAP.DB_3_SCINT_L
    t_stop_ch = CH_MAP.CROSS_SCINT_B2
    dt = ts[:, t_stop_ch] - ts[:,t_start_ch]
    dE = es[:, dE_ch]
    plt_mask = (dt>0) & (dE>0)
    plt.figure()
    plt.hist2d(dt[plt_mask], dE[plt_mask])
    plt.show(block=False)
