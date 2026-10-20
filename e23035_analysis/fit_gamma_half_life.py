import os

import ROOT
import numpy as np

from raw_viewer import ddas_interface, process_runs, degai
from e23035_analysis import e23035_runs, fitting_tools, root_vis_tools

experiment = 'e23035'

if False: #all 59Zn
    runs = np.array(e23035_runs.run_df['DDAS'][(e23035_runs.run_df['Run Type']=='59Zn')])# & (e23035_runs.run_df['Field Cage Functional?'] == 'yes')])
    runs = runs[(runs != 275)&(runs!=276)&(runs!=279)]
    tsbo_bins = (600, 0, 0.600)
if True: #subset of 60Ga
    runs = range(241,258)
    tsbo_bins = (220, 0, 0.22)

n_workers = len(runs)

addback_ethresh = 150
event_build_window = 500 #ns
adj_dict =  degai.crystal_adj_dict
cal_name = 'gm_511and2614_1'
tpc_config = 'smart1_veto2_rpr_fzp.csv'
nlc_name = 'c1'

gamma_bin_size = 0.25 #keV
upper_energy = 7000
gamma_binning = (int((upper_energy-addback_ethresh)/gamma_bin_size),addback_ethresh,upper_energy) 

tsbo_binning = (200, 0, 0.200)
tsbo_mesh = ddas_interface.get_histogram(experiment, runs, tsbo_binning, 'tsbo_mesh', 'time since beam off mesh', 
                                        'time_since_beam_off', "mesh_pre_amp_m>0", num_workers=n_workers, tpc_ini_filename=tpc_config)

tsbo_get = ddas_interface.get_histogram(experiment, runs, tsbo_binning, 'tsbo_get', 'tsbo_get',
                    'time_since_beam_off', 'get_timestamp>0', num_workers=n_workers, tpc_ini_filename=tpc_config)

tsbo_no_tpc_veto = ddas_interface.get_histogram(experiment, runs, tsbo_binning, 'tsbo_no_tpc_veto', 'tsbo_no_tpc_veto', 
                    'time_since_beam_off', 'get_timestamp>0 && !tpc_should_veto', num_workers=n_workers, tpc_ini_filename=tpc_config)
tsbo_tcp_veto = ddas_interface.get_histogram(experiment, runs, tsbo_binning, 'tsbo_tpc_veto', 'tsbo_tpc_veto', 
                    'time_since_beam_off', 'get_timestamp>0 && tpc_should_veto', num_workers=n_workers, tpc_ini_filename=tpc_config)

tsbo_hists = {'mesh':tsbo_mesh, 'GET no veto': tsbo_no_tpc_veto, 'GET veto': tsbo_tcp_veto, 'GET':tsbo_get}

single_exp_str = '[0] + [1]*exp(-log(2)*x/[2])'
fit_window = (0.015, 0.09)
init_vals = [0, 100, 0.069]
lims = [(0, 7000), (0, 1e6), (0.01, 0.2)]
fits = {} #name -> (res, rp, canvas, h_sub, f_fit, h_resid)
for name, h in tsbo_hists.items():
    fits[name] = fitting_tools.fit_hist(h, single_exp_str, init_vals, lims, fit_window)

print('half lives, fit window %g-%g s:'%fit_window)
for name, (res, rp, c1, h_sub, f_fit, h_resid) in fits.items():
    print('  %-12s %6.2f +- %4.2f ms  (valid=%s)'%(name, res.Parameter(2)*1e3, res.ParError(2)*1e3, res.IsValid()))

# gamma_energy_hist = degai.get_histogram(experiment, runs, adj_dict, cal_name, gamma_binning, 'gamma_hist', 'gamma spectrum', 
#                                    'addback_energy', '', event_build_window, addback_ethresh, True,
#                                   nonlinearity_correction_name=nlc_name)

tsbo_1004keV = degai.get_histogram(experiment, runs, adj_dict, cal_name, tsbo_binning, 'tsbo_1004keV', 'tsbo_1004keV', 
                                   'time', 'addback_energy>1003 && addback_energy<1005', event_build_window, addback_ethresh, True,
                                  nonlinearity_correction_name=nlc_name)


_ = root_vis_tools.draw_overlaid_histograms(tsbo_hists)
if False: #old
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

