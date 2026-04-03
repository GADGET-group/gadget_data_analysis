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
gamma_bin_size = 0.25 #keV
addback_ethresh = 150
upper_energy = 7000
gamma_binning = (int((upper_energy-addback_ethresh)/gamma_bin_size),addback_ethresh,upper_energy) #was 1-12000 w/ 1 keV bins
#run_candidates = e23035_runs.run_df['DDAS'][(e23035_runs.run_df['Run Type']=='60Ga')]
runs = e23035_runs.get_ddas_60_Ga_runs()

event_build_window = 500 #ns

# from e23035_analysis.gamma_energy_calibration import *

adj_dict =  degai.clover_adj_dict#degai.crystal_adj_dict
cal_name = 'gm_511and2614_1'
nlc_name = 'c1'
gamma_hist = degai.get_histogram(runs, adj_dict, cal_name, gamma_binning, 'gamma_hist', 'gamma spectrum', 'addback_energy', '', event_build_window, addback_ethresh, True,
                                  nonlinearity_correction_name=nlc_name)

# To use the new N-Gaussian model, you would add a new elif block in spectrum_fitter.py
# For example:
# elif self.peak_model.lower() == 'bg_shift_ngauss':
#     res = fitting_tools.fit_ngaussian_w_bg_shift(self.spectrum, loc_guess, fit_range, 
#                                     num_gaussians=self.num_gaussians_for_fit, # A new attribute on the class
#                                     param_bounds=param_bounds)

f = spectrum_fitter(gamma_hist, 'bg_shift_emg') # This will now use N=2 by default
#f.ngaus = 3
peaks = [('60Zn', 670.3, 0.3, 5),
         ('59Zn', 491.4, 0.1, 5), # ('59Zn', 914.2, 0.1), #exclude 914 peak since it overlaps with a 228Ac peak
         ('60Ga', 1003.7, 0.2, 5), ('60Ga', 1554.9, 0.6, 7), ('60Ga', 2293.0, 1.0, 8), ('60Ga', 2559.0, 0.8, 8), ('60Ga', 3848.3, 0.7, 8)]

for i in range(len(peaks)):
    f.peaks_to_fit.append((peaks[i][1],peaks[i][1]- peaks[i][3], peaks[i][1]+peaks[i][3]))
f.param_bound_functions['sigma']=lambda E: (0.1, 10)
f.fit_peaks()
#f.show_peak_locations()
mu, mu_err = f.get_fit_param('mu')
sigma, sigma_err = f.get_fit_param('sigma')
A, A_err = f.get_fit_param('amplitude')
#tau, tau_err = f.get_fit_param('tau')
probs = f.get_fit_probs()

#fit function for peak width
f_to_fit = ROOT.TF1("f_to_fit", "[0]*sqrt(x + [1])", 0, max(mu) * 1.2)
f_to_fit.SetParameters(0.05, 0.0) # initial guesses

graph = ROOT.TGraphErrors(
    len(mu), 
    np.array(mu, dtype=np.float64), 
    np.array(sigma, dtype=np.float64), 
    np.array(mu_err, dtype=np.float64), 
    np.array(sigma_err, dtype=np.float64)
)

c1 = ROOT.TCanvas("c_sigma", "Sigma vs Energy", 800, 600)
graph.SetTitle("Peak Width vs Energy;Energy (keV);Sigma (keV)")
graph.SetMarkerStyle(20)
graph.Draw("AP")
graph.Fit(f_to_fit)
c1.Update()

sigma_func = lambda E: 0.02456749968462633*(E + 905.0664550369642)**0.5 #sum spectrum
sigma_func = lambda E: 0.028389590048833593*(E + 781.8486169568944)**0.5 #clover add back spectrum
# f_seak = spectrum_fitter(gamma_hist, 'bg_shift_gaus')
# f_seak.param_bound_functions['sigma'] = lambda E: (sigma_func(E), sigma_func(E))
# f_seak.find_peaks(fit_sig=3)

# gg_hist = degai.get_addback_coincidence_spectrum(runs, degai.get_adjacency_dict(1), cal_name, gamma_binning, event_build_window, addback_ethresh, event_build_window, True, 
#                                                  nonlinearity_correction_name=nlc_name)
# h6134 = degai.get_bg_subtracted_projection(gg_hist, 6134, 4, 6180, 20)
