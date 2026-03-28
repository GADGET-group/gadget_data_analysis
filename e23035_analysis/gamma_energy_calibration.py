import os

import numpy as np
import ROOT
#ROOT.EnableImplicitMT()

from raw_viewer import ddas_interface
from e23035_analysis import fitting_tools, root_vis_tools, e23035_runs, degai
from e23035_analysis.spectrum_fitter import spectrum_fitter

'''
Notes on peak finding an automatic fitting
https://root.cern/root/htmldoc/guides/spectrum/Spectrum.html
'''
gamma_bin_size = 1 #keV
addback_ethresh = 150
gamma_binning = (int((7000-0)/gamma_bin_size),addback_ethresh,7000) #was 1-12000 w/ 1 keV bins
#run_candidates = e23035_runs.run_df['DDAS'][(e23035_runs.run_df['Run Type']=='60Ga')]
runs = e23035_runs.get_ddas_60_Ga_runs()

event_build_window = 500 #ns

# from e23035_analysis.gamma_energy_calibration import *

adj_dict = degai.get_adjacency_dict(1) #use sum spectrum
cal_name = 'gm2'
gamma_hist = degai.get_addback_spectrum(runs, adj_dict, cal_name, gamma_binning, event_build_window, addback_ethresh, True)

f = spectrum_fitter(gamma_hist, 'bg_shift_gaus')
# f.peaks_to_fit = [511, 1003.5, 1555, 2293,2559,3848.3]
# for i in range(len(f.peaks_to_fit)):
#     f.peaks_to_fit[i] = (f.peaks_to_fit[i],f.peaks_to_fit[i]*0.99, f.peaks_to_fit[i]*1.01)

peaks = [('60Zn', 670.3, 0.3),
         ('59Zn', 491.4, 0.1), # ('59Zn', 914.2, 0.1), #exclude 914 peak since it overlaps with a 228Ac peak
         ('60Ga', 1003.7, 0.2), ('60Ga', 1554.9, 0.6), ('60Ga', 2293.0, 1.0), ('60Ga', 2559.0, 0.8), ('60Ga', 3848.3, 0.7)]
#f.find_peaks()
for i in range(len(peaks)):
    f.peaks_to_fit.append((peaks[i][1],peaks[i][1]-10, peaks[i][1]+10))
f.param_bound_functions['sigma']=lambda E: (0.1, 10)
f.fit_peaks()
#f.show_peak_locations()
mu, mu_err = f.get_fit_param('mu')
sigma, sigma_err = f.get_fit_param('sigma')
A, A_err = f.get_fit_param('amplitude')
#tau, tau_err = f.get_fit_param('tau')
probs = f.get_fit_probs()

#(label, energy, energy uncertainty)
# graph = ROOT.TGraphErrors(
#     len(peaks), 
#     np.array(mu, dtype=np.float64), 
#     np.array(sigma, dtype=np.float64), 
#     np.array(mu_err, dtype=np.float64), 
#     sigma_err
# )

# graph.SetMarkerStyle(20)
# graph.Draw()

#timing_hist = degai.get_adjacent_timing_spectrum(126, degai.get_adjacency_dict(180), (1500, 0, 15000))
if False:
    indiv_adj_dict = degai.get_adjacency_dict(1)
    gg_hist = degai.get_addback_coincidence_spectrum(runs, indiv_adj_dict, 'gm', gamma_binning, event_build_window, 0, event_build_window)
    gspec = degai.get_bg_subtracted_projection(gg_hist, 2614, 1, 2700, 2)
