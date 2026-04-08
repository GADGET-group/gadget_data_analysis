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

adj_dict =  degai.crystal_adj_dict#degai.clover_adj_dict#
cal_name = 'gm_511and2614_1'
nlc_name = 'c1'
gamma_hist = degai.get_histogram(runs, adj_dict, cal_name, gamma_binning, 'gamma_hist', 'gamma spectrum', 'addback_energy', '', event_build_window, addback_ethresh, True,
                                  nonlinearity_correction_name=nlc_name)


fit_model = 'bg_shift_nemg'#'bg_shift_ngaus'#
f = spectrum_fitter(gamma_hist, fit_model) # This will now use N=2 by default
f.nemg = f.ngaus = 2
sigma1_func = lambda E: 0.02067*np.sqrt(E + 737.9)
tau1_func = lambda E: 0.0001117*E + 0.157
sigma2_func = lambda E: 0.03037*np.sqrt(E + 498.5)
tau2_func = lambda E: max(0.01, 0.0003378*E - 0.2317)
f.param_bound_functions['sigma1'] = lambda E: (sigma1_func(E), sigma1_func(E))
f.param_bound_functions['tau1'] = lambda E: (tau1_func(E), tau1_func(E))
f.param_bound_functions['sigma2'] = lambda E: (sigma2_func(E), sigma2_func(E))
f.param_bound_functions['tau2'] = lambda E: (tau2_func(E), tau2_func(E))

peaks = [#('60Zn', 670.3, 0.3, 665,677), #doesn't fit well, maybe another peak?
         ('59Zn', 491.4, 0.1, 487, 496), # ('59Zn', 914.2, 0.1), #exclude 914 peak since it overlaps with a 228Ac peak
        # ('60Ga', 1003.7, 0.2, 997, 1008), #didn't fit well with constrained tau2
         ('60Ga', 1554.9, 0.6, 1548,1563), #('60Ga', 2293.0, 1.0, 2287,2299), #this last one wants to have a very different sigma2 for some reason
        ('60Ga', 2559.0, 0.8, 2551, 2564), ('60Ga', 3848.3, 0.7, 3840,3856), #from ENSDF
         ('60Ga', 2996.8, 0.2, 2991, 3005), #2021OR01
         ('60Ga', 4850.2, 0.5, 4840, 4860)] #not enough statistics

for i in range(len(peaks)):
    f.peaks_to_fit.append((peaks[i][1],peaks[i][3], peaks[i][4]))
f.fit_peaks()
ROOT.gROOT.SetBatch(False)

mu, mu_err = f.get_fit_param('mu')
mu_vs_E = ROOT.TGraphErrors(
    len(f.peaks_to_fit), 
    np.array([peak[1] for peak in peaks], dtype=np.float64), 
    mu, 
    np.array([peak[2] for peak in peaks], dtype=np.float64), 
    mu_err)
fit_results = []
fit_results.append(fitting_tools.fit_graph(mu_vs_E, "[0]*x + [1]", [1,0], [(-1, 1), (-1, 1)]))
fit_results[-1][2].SetTitle('mu vs keV ')

tau2, tau2_err = f.get_fit_param('tau2')
tau2_vs_mu = ROOT.TGraphErrors(
    len(mu), 
    np.array(mu, dtype=np.float64), 
    np.array(tau2, dtype=np.float64), 
    np.array(mu_err, dtype=np.float64), 
    np.array(tau2_err, dtype=np.float64)
)
fit_results.append(fitting_tools.fit_graph(tau2_vs_mu, "[0]*x + [1]", [0,0], [(-1, 1), (-1, 1)]))
fit_results[-1][2].SetTitle('tau2 vs Energy (sigma1 & tau1 fixed)')


if True:
    #f.show_peak_locations()
    mu, mu_err = f.get_fit_param('mu')
    sigma1, sigma1_err = f.get_fit_param('sigma1')
    sigma2, sigma2_err = f.get_fit_param('sigma2')
    bg_shift, bg_shift_err = f.get_fit_param('bg_shift')
    if fit_model == 'bg_shift_nemg':
        tau1, tau1_err = f.get_fit_param('tau1')
        tau2, tau2_err = f.get_fit_param('tau2')
        tau_a, tau_a_err, tau_b, tau_b_err = [], [], [], []

    sigma_a, sigma_a_err, sigma_b, sigma_b_err,  = [], [], [], []

    for i in range(len(mu)):
        if sigma1[i] < sigma2[i]:
            sigma_a.append(sigma1[i])
            sigma_a_err.append(sigma1_err[i])
            sigma_b.append(sigma2[i])
            sigma_b_err.append(sigma2_err[i])
            if fit_model == 'bg_shift_nemg':
                tau_a.append(tau1[i])
                tau_a_err.append(tau1_err[i])
                tau_b.append(tau2[i])
                tau_b_err.append(tau2_err[i])
        else:
            sigma_a.append(sigma2[i])
            sigma_a_err.append(sigma2_err[i])
            sigma_b.append(sigma1[i])
            sigma_b_err.append(sigma1_err[i])
            if fit_model == 'bg_shift_nemg':
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
    fit_results[-1][2].SetTitle('simga_a vs Energy')

    sigma_b_vs_mu = ROOT.TGraphErrors(
        len(mu), 
        np.array(mu, dtype=np.float64), 
        np.array(sigma_b, dtype=np.float64), 
        np.array(mu_err, dtype=np.float64), 
        np.array(sigma_b_err, dtype=np.float64)
    )
    fit_results.append(fitting_tools.fit_graph(sigma_b_vs_mu, "[0]*sqrt(x + [1])", [1, 500], [(0, np.inf), (0, np.inf)]))
    fit_results[-1][2].SetTitle('simga_b vs Energy')

    if fit_model == 'bg_shift_nemg':
        tau_a_vs_mu = ROOT.TGraphErrors(
            len(mu), 
            np.array(mu, dtype=np.float64), 
            np.array(tau_a, dtype=np.float64), 
            np.array(mu_err, dtype=np.float64), 
            np.array(tau_a_err, dtype=np.float64)
        )
        fit_results.append(fitting_tools.fit_graph(tau_a_vs_mu, "[0]*x + [1]", [0,0], [(-1, 1), (-1, 1)]))
        fit_results[-1][2].SetTitle('tau_a vs Energy ')


        tau_b_vs_mu = ROOT.TGraphErrors(
            len(mu), 
            np.array(mu, dtype=np.float64), 
            np.array(tau_b, dtype=np.float64), 
            np.array(mu_err, dtype=np.float64), 
            np.array(tau_b_err, dtype=np.float64)
        )
        fit_results.append(fitting_tools.fit_graph(tau_b_vs_mu, "[0]*x + [1]", [0,0], [(-1, 1), (-1, 1)]))
        fit_results[-1][2].SetTitle('tau_b vs Energy ')




if False:
    f2 = spectrum_fitter(gamma_hist, 'bg_shift_nemg') # This will now use N=2 by default
    f2.nemg = 2
    f2.peaks_to_fit = f.peaks_to_fit.copy()
    f2.param_bound_functions['sigma1'] = lambda E: (sigma1_func(E), sigma1_func(E))
    f2.param_bound_functions['tau1'] = lambda E: (tau1_func(E), tau1_func(E))
    #f2.param_bound_functions['sigma2'] = lambda E: (sigma2_func(E), sigma2_func(E))
    #f2.param_bound_functions['tau2'] = lambda E: (tau2_func(E), tau2_func(E))


    #include 4850 peak now that there are fewer free parameters: ('60Ga', 4850.2, 0.5, 4840, 4860)
    f2.peaks_to_fit.append((4850.2, 4840, 4860))
    f2.fit_peaks()

    mu2, mu2_err = f2.get_fit_param('mu')

    sigma2, sigma2_err = f2.get_fit_param('sigma2')
    sigma2_vs_mu = ROOT.TGraphErrors(
        len(mu2), 
        np.array(mu2, dtype=np.float64), 
        np.array(sigma2, dtype=np.float64), 
        np.array(mu2_err, dtype=np.float64), 
        np.array(sigma2_err, dtype=np.float64)
    )
    fit_results.append(fitting_tools.fit_graph(sigma2_vs_mu, "[0]*sqrt(x + [1])", [1, 500], [(0, np.inf), (0, np.inf)]))
    fit_results[-1][2].SetTitle('sigma2 vs Energy (sigma1 & tau1 fixed)')

    tau2, tau2_err = f2.get_fit_param('tau2')
    tau2_vs_mu = ROOT.TGraphErrors(
        len(mu2), 
        np.array(mu2, dtype=np.float64), 
        np.array(tau2, dtype=np.float64), 
        np.array(mu2_err, dtype=np.float64), 
        np.array(tau2_err, dtype=np.float64)
    )
    fit_results.append(fitting_tools.fit_graph(tau2_vs_mu, "[0]*x + [1]", [0,0], [(-1, 1), (-1, 1)]))
    fit_results[-1][2].SetTitle('tau2 vs Energy (sigma1 & tau1 fixed)')





    # Use a coarser binning for the 2D coincidence matrix to prevent ROOT's 1GB serialization limit
    coincidence_bin_size = 1.0 # keV
    coincidence_binning = (int((upper_energy-addback_ethresh)/coincidence_bin_size), addback_ethresh, upper_energy)
    coincidence_hist = degai.get_addback_coincidence_spectrum(runs, adj_dict, cal_name, coincidence_binning, event_build_window, addback_ethresh, event_build_window, True, 
                                                    nonlinearity_correction_name=nlc_name)
    # h6134 = degai.get_bg_subtracted_projection(gg_hist, 6134, 4, 6180, 20)
