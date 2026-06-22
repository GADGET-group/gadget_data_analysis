import os

import ROOT
import numpy as np

from raw_viewer import ddas_interface, process_runs
from e23035_analysis import e23035_runs, fitting_tools

experiment = 'e23035'

if True: #all 59Zn
    runs = np.array(e23035_runs.run_df['DDAS'][(e23035_runs.run_df['Run Type']=='59Zn')])# & (e23035_runs.run_df['Field Cage Functional?'] == 'yes')])
    runs = runs[(runs != 275)&(runs!=276)&(runs!=279)]
    tsbo_bins = (600, 0, 0.600)
if False: #subset of 60Ga
    runs = range(241,255)
    tsbo_bins = (220, 0, 0.22)

n_workers = len(runs)

gammas = ddas_interface.get_histogram(experiment, runs, (12000-1,1,12000), 'gammas', 'summed gamma spectrum', ddas_interface.get_add_back_gamma_str(), num_workers=n_workers)
s = ddas_interface.get_add_back_gamma_str()
tsbo = ddas_interface.get_histogram(experiment, runs, tsbo_bins, '60Ga_decay_times', 'Time since beam off for 60Ga gamma rays (1003 keV & 3848 keV)',
                                    'time_since_beam_off', '((%s>3840 && %s < 3855) || (%s > 1002 && %s < 1006))'%(s,s,s,s), num_workers=n_workers)

func_str = '[0] + [1]*exp(-log(2)*x/[2]) + [3]*exp(-log(2)*x/[4]) + [5]*exp(-log(2)*x/[6])'
init_vals = [0, 160, 0.06, 160, 0.180, 160, 3]
lims = [(0, np.inf), (0,np.inf), (0.02, 0.1), (0,np.inf), (0.1821,0.1821), (0, np.inf), (0.2, np.inf)]# (0.1821, 0.1821)]
fit_window = (0,0.45)#(0.02,0.08)
res, rp, c1, h_sub, f_fit, h_resid = fitting_tools.fit_hist(tsbo, func_str, init_vals, lims, fit_window)

if False:
    func_str = '[0] + [1]*exp(-log(2)*x/[2])'
    init_vals = [0, 160, 0.06]
    lims = [(0, np.inf), (-np.inf,np.inf), (0, np.inf)]
    fit_window = (0.1,0.2)
    res2, rp2, c12, h_sub2, f_fit2, h_resid2 = fitting_tools.fit_hist(tsbo, func_str, init_vals, lims, fit_window)

#try fitting protons

