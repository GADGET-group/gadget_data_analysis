import os

import numpy as np
import ROOT

from raw_viewer import ddas_interface
from e23035_analysis import fitting_tools, root_vis_tools, e23035_runs, degai
from e23035_analysis.spectrum_fitter import spectrum_fitter

gamma_bin_size = 0.25 #keV
addback_ethresh = 150
upper_energy = 7000
gamma_binning = (int((upper_energy-addback_ethresh)/gamma_bin_size),addback_ethresh,upper_energy) #was 1-12000 w/ 1 keV bins
#run_candidates = e23035_runs.run_df['DDAS'][(e23035_runs.run_df['Run Type']=='60Ga')]
runs = e23035_runs.get_ddas_60_Ga_runs()

event_build_window = 500 #ns


adj_dict =  degai.crystal_adj_dict#degai.clover_adj_dict#
cal_name = 'gm_511and2614_1'
nlc_name = 'c1'
gamma_hist = degai.get_histogram(runs, adj_dict, cal_name, gamma_binning, 'gamma_hist', 'gamma spectrum', 'addback_energy', '', event_build_window, addback_ethresh, True,
                                  nonlinearity_correction_name=nlc_name)
gamma_beam_off_hist = degai.get_histogram(runs, adj_dict, cal_name, gamma_binning, 'beam_off_gammas', 'beam off gamma spectrum', 'addback_energy', 
                                          "time_since_beam_off<0.094", event_build_window, addback_ethresh, True,
                                        nonlinearity_correction_name=nlc_name)
gamma_beam_on_hist = degai.get_histogram(runs, adj_dict, cal_name, gamma_binning, 'beam_on_gammas', 'beam on gamma spectrum', 'addback_energy', 
                                          "(time_since_beam_off>0.1) && (time_since_beam_off<0.195)", event_build_window, addback_ethresh, True,
                                        nonlinearity_correction_name=nlc_name)
root_vis_tools.draw_overlaid_histograms({'beam off':gamma_beam_off_hist, 'beam on':gamma_beam_on_hist, 'all':gamma_hist})


fit_model = 'bg_shift_nemg'#'bg_shift_ngaus'#
sigma1_func = lambda E: 0.02078*np.sqrt(E + 742.9)
tau1_func = lambda E: 0.0001117*E + 0.157
sigma2_func = lambda E: 0.03037*np.sqrt(E + 498.5)
tau2_func = lambda E: max(0.01, 0.0003378*E - 0.2317)

#def fit_nemg_w_bg_shift(spectrum:ROOT.TH1D, e_guess:float|list, fit_window:tuple, num_emgs:int, data_source=None, param_bounds=None, fit_options='LS0QEI'): 
fitters = []
def fit_peak(spectrum, location, window_start, window_stop):
    f = spectrum_fitter(gamma_hist, fit_model) # This will now use N=2 by default
    f.nemg = f.ngaus = 2
    f.peaks_to_fit = [(location, window_start, window_stop)]
    f.param_bound_functions['sigma1'] = lambda E: (sigma1_func(E), sigma1_func(E))
    f.param_bound_functions['tau1'] = lambda E: (tau1_func(E), tau1_func(E))
    f.param_bound_functions['sigma2'] = lambda E: (sigma2_func(E), sigma2_func(E))
    f.param_bound_functions['tau2'] = lambda E: (tau2_func(E), tau2_func(E))
    f.fit_peaks()
    f.show_fit_results(0)
    fitters.append(f)



# Use a coarser binning for the 2D coincidence matrix to prevent ROOT's 1GB serialization limit
coincidence_bin_size = 1.0 # keV
coincidence_binning = (int((upper_energy-addback_ethresh)/coincidence_bin_size), addback_ethresh, upper_energy)
coincidence_hist = degai.get_addback_coincidence_spectrum(runs, adj_dict, cal_name, coincidence_binning, event_build_window, addback_ethresh, event_build_window, True, 
                                                nonlinearity_correction_name=nlc_name)
# h6134 = degai.get_bg_subtracted_projection(gg_hist, 6134, 4, 6180, 20)

#make 2D histogram of time vs energy
E_v_tsbo = degai.get_histogram(runs, adj_dict, cal_name, (200, 0, 0.200, *coincidence_binning), "E_vs_t", "energy (keV) vs time (s)", "addback_energy:time_since_beam_off", 
                    dt_window_ns=event_build_window, e_thresh=addback_ethresh, nonlinearity_correction_name=nlc_name)

E_v_tsco = degai.get_histogram(runs, adj_dict, cal_name, (2000, 0, 0.200, *coincidence_binning), "E_vs_t", "energy (keV) vs time (s)", "addback_energy:time_since_chopper_off", 
                    dt_window_ns=event_build_window, e_thresh=addback_ethresh, nonlinearity_correction_name=nlc_name)

fit_results = []
def fit_decay_curve(Egate, tgate, Egate_bg=None, source=E_v_tsbo):
    if Egate_bg is None:
        h_to_fit = degai.get_gated_projection(E_v_tsbo, np.average(Egate), Egate[1]-Egate[0])
    else:
        h_to_fit = degai.get_bg_subtracted_projection(E_v_tsbo, np.average(Egate), Egate[1]-Egate[0], np.average(Egate_bg), Egate_bg[1]-Egate_bg[0])
    fit_string = '[0]*exp(-0.693147*x/[1]) + [2]'
    fit_results.append(fitting_tools.fit_hist(h_to_fit, fit_string, [1,1, 0], ((0, np.inf), (0, np.inf), (0, np.inf)), tgate,fit_options='S0QEI', names=['A', 'half life (s)', 'bg']))

if False:
    chists = degai.get_crystal_histograms(runs, coincidence_binning, 'cal', 'gm_511and2614_1', nonlinearity_correction_name='c1')
    hupstream = ROOT.TH1D('upsteam', 'upsteam', *coincidence_binning)
    hcenter = ROOT.TH1D('center', 'center', *coincidence_binning)
    hdownstream = ROOT.TH1D('downstream', 'downstream', *coincidence_binning)
    upsteam_crystals = 0
    downstream_crystals = 0
    center_crystals = 0
    for clover, crystal in degai.clover_list:
        crystal = {1:'a', 2:'b', 3:'c', 4:'d'}[crystal]
        crystal_hist = chists[f'clover_{clover}{crystal}_keV']
        if clover in [1,2,3]:
            hupstream.Add(crystal_hist)
            upsteam_crystals += 1
        elif clover in [5,6,7]:
            hcenter.Add(crystal_hist)
            center_crystals += 1
        elif clover in [11,12,13]:
            hdownstream.Add(crystal_hist)
            downstream_crystals += 1
    hupstream *= (1/upsteam_crystals)
    hcenter *= (1/center_crystals)
    hdownstream *= (1/downstream_crystals)
    root_vis_tools.draw_overlaid_histograms({'upstream':hupstream, 'center':hcenter, 'downstream':hdownstream})

# cfitters = {}
# for c in chists:
#     f = spectrum_fitter(chists[c], 'bg_shift_gaus')
#     f.peaks_to_fit.append((198.3, 192, 204))
#     f.peaks_to_fit.append((846, 838, 852))
#     f.fit_peaks()
#     cfitters[c] = f