import ROOT
import numpy as np

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

def get_root_file_path(experiment, run):
    base_path = f'/egr/research-tpc/shared/proc_runs/{experiment}/ddas/'
    file_name = 'run-%04d_gadget.root'%run
    return base_path + file_name

def extract_data(experiment, run):
    file = ROOT.TFile(get_root_file_path(experiment, run), "READ")
    tree = file.Get("tree")
    energies, times, multiplicities = np.zeros(160, dtype=np.int32), np.zeros(160), np.zeros(160,dtype=np.int32)
    tree.SetBranchAddress("energies", energies)
    tree.SetBranchAddress("times", times)
    tree.SetBranchAddress("multiplicity", multiplicities)
    es, ts, ms = [], [], []
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        es.append(np.copy(energies))
        ts.append(np.copy(times))
        ms.append(np.copy(multiplicities))
    return np.array(es), np.array(ts), np.array(ms)


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
