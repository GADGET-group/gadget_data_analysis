import ROOT
import numpy as np

def get_root_file_path(experiment, run):
    base_path = f'/egr/research-tpc/shared/proc_runs/{experiment}/ddas/'
    file_name = 'run-%04d_gadget.root'%run
    return base_path + file_name

def extract_data(experiment, run):
    file = ROOT.TFile(get_root_file_path(experiment, run), "READ")
    tree = file.Get("tree")
    energies, times, multiplicities = np.zeros(160), np.zeros(160), np.zeros(160)
    es, ts, ms = [], [], []
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        es.append(np.copy(energies))
        ts.append(np.copy(times))
        ms.append(np.copy(multiplicities))
    return np.array(es), np.array(ts), np.array(ms)