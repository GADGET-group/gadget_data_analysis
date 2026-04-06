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

f = spectrum_fitter(gamma_hist, 'bg_shift_nemg') # This will now use N=2 by default
f.nemg = 2
peaks = [('60Zn', 670.3, 0.3, 5),
         ('59Zn', 491.4, 0.1, 5), # ('59Zn', 914.2, 0.1), #exclude 914 peak since it overlaps with a 228Ac peak
         ('60Ga', 1003.7, 0.2, 5), ('60Ga', 1554.9, 0.6, 7), ('60Ga', 2293.0, 1.0, 4), ('60Ga', 2559.0, 0.8, 6), ('60Ga', 3848.3, 0.7, 7)]

for i in range(len(peaks)):
    f.peaks_to_fit.append((peaks[i][1],peaks[i][1]- peaks[i][3], peaks[i][1]+peaks[i][3]))
f.param_bound_functions['sigma']=lambda E: (0.1, 10)
f.fit_peaks()
#f.show_peak_locations()
mu, mu_err = f.get_fit_param('mu')
sigma1, sigma1_err = f.get_fit_param('sigma1')
sigma2, sigma2_err = f.get_fit_param('sigma2')
tau1, tau1_err = f.get_fit_param('tau1')
tau2, tau2_err = f.get_fit_param('tau2')
bg_shift, bg_shift_err = f.get_fit_param('bg_shift')

sigma_a, sigma_a_err, sigma_b, sigma_b_err, tau_a, tau_a_err, tau_b, tau_b_err = [], [], [], [], [], [], [], []
for i in range(len(mu)):
    if sigma1[i] < sigma2[i]:
        sigma_a.append(sigma1[i])
        sigma_a_err.append(sigma1_err[i])
        sigma_b.append(sigma2[i])
        sigma_b_err.append(sigma2_err[i])
        tau_a.append(tau1[i])
        tau_a_err.append(tau1_err[i])
        tau_b.append(tau2[i])
        tau_b_err.append(tau2_err[i])
    else:
        sigma_a.append(sigma2[i])
        sigma_a_err.append(sigma2_err[i])
        sigma_b.append(sigma1[i])
        sigma_b_err.append(sigma1_err[i])
        tau_a.append(tau2[i])
        tau_a_err.append(tau2_err[i])
        tau_b.append(tau1[i])
        tau_b_err.append(tau1_err[i])

A, A_err = f.get_fit_param('amplitude')
#tau, tau_err = f.get_fit_param('tau')
probs = f.get_fit_probs()

fit_results = []
sigma_a_vs_mu = ROOT.TGraphErrors(
    len(mu), 
    np.array(mu, dtype=np.float64), 
    np.array(sigma_a, dtype=np.float64), 
    np.array(mu_err, dtype=np.float64), 
    np.array(sigma_a_err, dtype=np.float64)
)
fit_results.append(fitting_tools.fit_graph(sigma_a_vs_mu, "[0]*sqrt(x + [1])", [1, 500], [(0, np.inf), (0, np.inf)]))

sigma_b_vs_mu = ROOT.TGraphErrors(
    len(mu), 
    np.array(mu, dtype=np.float64), 
    np.array(sigma_b, dtype=np.float64), 
    np.array(mu_err, dtype=np.float64), 
    np.array(sigma_b_err, dtype=np.float64)
)
fit_results.append(fitting_tools.fit_graph(sigma_b_vs_mu, "[0]*sqrt(x + [1])", [1, 500], [(0, np.inf), (0, np.inf)]))

tau_a_vs_mu = ROOT.TGraphErrors(
    len(mu), 
    np.array(mu, dtype=np.float64), 
    np.array(tau_a, dtype=np.float64), 
    np.array(mu_err, dtype=np.float64), 
    np.array(tau_a_err, dtype=np.float64)
)
fit_results.append(fitting_tools.fit_graph(tau_a_vs_mu, "[0]*x + [1]", [1, 500], [(0, np.inf), (0, np.inf)]))




sigma_func = lambda E: 0.02456749968462633*(E + 905.0664550369642)**0.5 #sum spectrum
sigma_func = lambda E: 0.028389590048833593*(E + 781.8486169568944)**0.5 #clover add back spectrum


# Use a coarser binning for the 2D coincidence matrix to prevent ROOT's 1GB serialization limit
coincidence_bin_size = 1.0 # keV
coincidence_binning = (int((upper_energy-addback_ethresh)/coincidence_bin_size), addback_ethresh, upper_energy)
coincidence_hist = degai.get_addback_coincidence_spectrum(runs, adj_dict, cal_name, coincidence_binning, event_build_window, addback_ethresh, event_build_window, True, 
                                                 nonlinearity_correction_name=nlc_name)
# h6134 = degai.get_bg_subtracted_projection(gg_hist, 6134, 4, 6180, 20)
