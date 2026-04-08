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
