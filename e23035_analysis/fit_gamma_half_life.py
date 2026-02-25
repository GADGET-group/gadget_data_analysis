import os

import ROOT
import numpy as np

from raw_viewer import ddas_interface, process_runs
from e23035_analysis import e23035_runs, fitting_tools

runs = range(241,255)
# merged_data = ROOT.TChain('merged_data_chain', 'merged_data_chain')
# for run in range(241, 255):
#     merged_data.Add('%s/merged_data'%ddas_interface.get_merged_root_file_path(run))
# f = ROOT.TFile(ddas_interface.get_merged_root_file_path(255), 'READ')
# merged_data = f.Get('merged_data')
gammas = ddas_interface.get_histogram(runs, (12000-1,1,12000), 'gammas', 'summed gamma spectrum', ddas_interface.get_summed_gamma_e_str(), num_workers=20)
s = ddas_interface.get_summed_gamma_e_str()
tsbo = ddas_interface.get_histogram(runs, (200, 0, 0.22), '60Ga_decay_times', 'Time since beam off for 60Ga gamma rays (1003 keV & 3848 keV)',
                                    'time_since_beam_off', 'time_since_beam_off < 0.22 && ((%s>3840 && %s < 3855) || (%s > 1002 && %s < 1006))'%(s,s,s,s), num_workers=20)

func_str = '[0] + [1]*exp(-log(2)*x/[2]) + [3]*exp(-log(2)*x/[4])'
init_vals = [0, 160, 0.06, 160, 0.180]
lims = [(0, np.inf), (0,np.inf), (0, np.inf), (0,np.inf), (0.1821, 0.1821)]
fit_window = (0,0.095)#(0.02,0.08)
res, rp, c1, h_sub, f_fit, h_resid = fitting_tools.fit_func(tsbo, func_str, init_vals, lims, fit_window)

if False:
    func_str = '[0] + [1]*exp(-log(2)*x/[2])'
    init_vals = [0, 160, 0.06]
    lims = [(0, np.inf), (-np.inf,np.inf), (0, np.inf)]
    fit_window = (0.1,0.2)
    res2, rp2, c12, h_sub2, f_fit2, h_resid2 = fitting_tools.fit_func(tsbo, func_str, init_vals, lims, fit_window)

#try fitting protons
