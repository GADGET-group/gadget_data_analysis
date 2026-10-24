import os

import ROOT
import numpy as np

from tpc_and_decay_station import ddas_interface, degai
from tpc_data_processing import process_runs
from e23035_analysis import e23035_runs
from analysis_tools import fitting_tools, root_vis_tools

experiment = 'e23035'

if True: #all 59Zn
    runs = np.array(e23035_runs.run_df['DDAS'][(e23035_runs.run_df['Run Type']=='59Zn')])# & (e23035_runs.run_df['Field Cage Functional?'] == 'yes')])
    runs = runs[(runs != 275)&(runs!=276)&(runs!=279)]
    tsbo_binning = (1000, 0, 1)
    fit_window = (0.06, 0.49)
if False: #subset of 60Ga
    runs = range(241,258)
    tsbo_binning = (200, 0, 0.200)
    fit_window = (0.015, 0.09)
if False: #all 60Ga with valid gammas and TPC
    runs = e23035_runs.get_ddas_60_Ga_runs(True, False, False, True, True)
    tsbo_binning = (200, 0, 0.200)
    fit_window = (0.015, 0.09)

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

tsbo_mesh = ddas_interface.get_histogram(experiment, runs, tsbo_binning, 'tsbo_mesh', 'time since beam off mesh', 
                                        'time_since_beam_off', "mesh_pre_amp_m>0", num_workers=n_workers, tpc_ini_filename=tpc_config)

tsbo_get = ddas_interface.get_histogram(experiment, runs, tsbo_binning, 'tsbo_get', 'tsbo_get',
                    'time_since_beam_off', 'get_timestamp>0', num_workers=n_workers, tpc_ini_filename=tpc_config)

tsbo_no_tpc_veto = ddas_interface.get_histogram(experiment, runs, tsbo_binning, 'tsbo_no_tpc_veto', 'tsbo_no_tpc_veto', 
                    'time_since_beam_off', 'get_timestamp>0 && !tpc_should_veto', num_workers=n_workers, tpc_ini_filename=tpc_config)
tsbo_tcp_veto = ddas_interface.get_histogram(experiment, runs, tsbo_binning, 'tsbo_tpc_veto', 'tsbo_tpc_veto', 
                    'time_since_beam_off', 'get_timestamp>0 && tpc_should_veto', num_workers=n_workers, tpc_ini_filename=tpc_config)
tsbo_1004keV = degai.get_histogram(experiment, runs, adj_dict, cal_name, tsbo_binning, 'tsbo_1004keV', 'tsbo_1004keV', 
                                   'time_since_beam_off', 'addback_energy>1003 && addback_energy<1005', event_build_window, addback_ethresh, True,
                                  nonlinearity_correction_name=nlc_name)
tsbo_491keV = degai.get_histogram(experiment, runs, adj_dict, cal_name, tsbo_binning, 'tsbo_491keV', 'tsbo_491keV', 
                                   'time_since_beam_off', 'addback_energy>490 && addback_energy<492', event_build_window, addback_ethresh, True,
                                  nonlinearity_correction_name=nlc_name)
tsbo_hists = {'mesh':tsbo_mesh, 'GET no veto': tsbo_no_tpc_veto, 'GET veto': tsbo_tcp_veto, 'GET':tsbo_get, 
                '1004 keV gamma':tsbo_1004keV, '491 keV gamma':tsbo_491keV}

def set_fit_title(fit, title, window_title):
    #title the plot and canvas window made by fitting_tools.fit_hist, label its axes and hide the stats box
    res, rp, c1, h_sub, f_fit, h_resid = fit
    h_sub.SetTitle(title)
    h_sub.SetStats(0)
    h_sub.GetXaxis().SetTitle('time since beam off (s)') #the ratio plot's visible x axes copy their title from h_sub
    rp.GetUpperRefYaxis().SetTitle('counts / %g ms'%(h_sub.GetBinWidth(1)*1e3))
    rp.SetLeftMargin(0.13)
    rp.SetUpTopMargin(0.16)
    rp.SetLowBottomMargin(0.4)
    upper_pad = rp.GetUpperPad()
    upper_pad.GetListOfPrimitives().Remove(upper_pad.GetPrimitive('title')) #still sized for the old title, rebuilt on the next paint
    c1.SetTitle(window_title)
    upper_pad.Modified()
    rp.GetLowerPad().Modified()
    c1.Modified()
    c1.Update()

single_exp_str = '[0] + [1]*exp(-log(2)*x/[2])'
init_vals = [0, 100, 0.069]
lims = [(0, 7000), (0, 1e6), (0.01, 0.2)]
fits = {} #name -> (res, rp, canvas, h_sub, f_fit, h_resid)
for name, h in tsbo_hists.items():
    fits[name] = fitting_tools.fit_hist(h, single_exp_str, init_vals, lims, fit_window)

print('half lives, fit window %g-%g s:'%fit_window)
for name, (res, rp, c1, h_sub, f_fit, h_resid) in fits.items():
    print('  %-12s %6.2f +- %4.2f ms  (valid=%s, pvalue=%f)'%(name, res.Parameter(2)*1e3, res.ParError(2)*1e3, res.IsValid(), res.Prob()))
    set_fit_title(fits[name], '%s, exponential + constant: T_{1/2} = %.1f #pm %.1f ms, #chi^{2}/ndf = %.1f/%d'%
                  (name, res.Parameter(2)*1e3, res.ParError(2)*1e3, 2*res.MinFcnValue(), res.Ndf()), '%s: exponential + constant fit'%name)
if False:
    two_exp_str = '[0] + [1]*exp(-log(2)*x/[2]) + [3]*exp(-log(2)*x/[4])'
    init_vals2 = [0, 100, 0.069, 100, 0.11]
    lims2 = [(0, 7000), (0, 1e6), (0.01, 0.2), (0, 1e6), (0.01, 0.2)]
    fits_two_exp = {} #name -> (res, rp, canvas, h_sub, f_fit, h_resid)
    for name, h in tsbo_hists.items():
        #no Minos ('E'): with the two components degenerate it feeds the model nan half lives (GSL qags errors) and breaks the covariance
        fits_two_exp[name] = fitting_tools.fit_hist(h, two_exp_str, init_vals2, lims2, fit_window, fit_options='LS0QI')

    def exp_counts(amp, half_life, t_low, t_high, bin_width):
        #counts between t_low and t_high from amp*exp(-log(2)*t/half_life), with amp in counts per bin
        tau = half_life/np.log(2)
        return amp*tau*(np.exp(-t_low/tau) - np.exp(-t_high/tau))/bin_width

    def exp_counts_and_error(res, amp_index, half_life_index, t_low, t_high, bin_width):
        #error propagated from the fit covariance of the amplitude and half life
        amp, half_life = res.Parameter(amp_index), res.Parameter(half_life_index)
        dhl = 1e-3*half_life
        grad = np.array([exp_counts(1, half_life, t_low, t_high, bin_width),
                        (exp_counts(amp, half_life+dhl, t_low, t_high, bin_width) - exp_counts(amp, half_life-dhl, t_low, t_high, bin_width))/(2*dhl)])
        cov = np.array([[res.CovMatrix(i, j) for j in (amp_index, half_life_index)] for i in (amp_index, half_life_index)])
        return exp_counts(amp, half_life, t_low, t_high, bin_width), np.sqrt(grad @ cov @ grad)

    print('double exponential fit')
    for name, (res, rp, c1, h_sub, f_fit, h_resid) in fits_two_exp.items():
        print('  %-12s %6.2f +- %4.2f ms and  %6.2f +- %4.2f (valid=%s, pvalue=%f)'%
                (name, res.Parameter(2)*1e3, res.ParError(2)*1e3, res.Parameter(4)*1e3, res.ParError(4)*1e3, res.IsValid(), res.Prob()))
        #fraction of all counts in the fit window (data, constant background included) from each component
        t_low, t_high = h_sub.GetXaxis().GetXmin(), h_sub.GetXaxis().GetXmax()
        bin_width = h_sub.GetBinWidth(1)
        total = h_sub.Integral()
        n1, n1_err = exp_counts_and_error(res, 1, 2, t_low, t_high, bin_width)
        n2, n2_err = exp_counts_and_error(res, 3, 4, t_low, t_high, bin_width)
        n_bg, n_bg_err = res.Parameter(0)*h_sub.GetNbinsX(), res.ParError(0)*h_sub.GetNbinsX()
        print('  %-12s of %d counts in %g-%g s: %5.1f +- %4.1f %% and %5.1f +- %4.1f %% (constant %5.1f +- %4.1f %%)'%
                ('', total, t_low, t_high, 100*n1/total, 100*n1_err/total, 100*n2/total, 100*n2_err/total, 100*n_bg/total, 100*n_bg_err/total))
        set_fit_title(fits_two_exp[name], '#splitline{%s, two exponentials + constant: #chi^{2}/ndf = %.1f/%d}'
                    '{T_{1/2} = %.1f #pm %.1f ms (%.0f%%), %.1f #pm %.1f ms (%.0f%%), constant %.0f%%}'%
                    (name, 2*res.MinFcnValue(), res.Ndf(), res.Parameter(2)*1e3, res.ParError(2)*1e3, 100*n1/total,
                    res.Parameter(4)*1e3, res.ParError(4)*1e3, 100*n2/total, 100*n_bg/total), '%s: two exponentials + constant fit'%name)

    # gamma_energy_hist = degai.get_histogram(experiment, runs, adj_dict, cal_name, gamma_binning, 'gamma_hist', 'gamma spectrum', 
    #                                    'addback_energy', '', event_build_window, addback_ethresh, True,
    #                                   nonlinearity_correction_name=nlc_name)





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

