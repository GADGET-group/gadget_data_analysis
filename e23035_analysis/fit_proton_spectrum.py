import os

import ROOT
import numpy as np

from raw_viewer import ddas_interface, process_runs
from e23035_analysis import e23035_runs

# ddas_runs = [277,278]
# analysis_file_name = 'ddas_runs_277_278_proton_spectra.root'

# merged_data = ROOT.TChain('merged_data')
# for ddas_run in ddas_runs:
#     merged_data.Add(ddas_interface.get_merged_root_file_path(ddas_run))

# df = ROOT.RDataFrame(merged_data)
# good_protons = df.Filter( 'tpc_particle_id==1 && !tpc_should_veto')

# proton_spectrum = good_protons.Histo1D(('proton_hist', 'proton energy spectrum', 350, 500, 3500),'tpc_energy')
# proton_spectrum.Draw()

get_runs = e23035_runs.run_df['GET'][(e23035_runs.run_df['Run Type']=='59Zn') & (e23035_runs.run_df['Field Cage Functional?'] == 'yes')]
get_runs = get_runs[(get_runs != 298) & (get_runs != 297)] #TODO: need to merge these runs!!!
tpc_energy = e23035_runs.get_energy_MeV(get_runs)
angles = process_runs.get_angle('e23035', get_runs)
proton_mask = e23035_runs.get_proton_mask(get_runs)#&(np.degrees(angles)>15)
n_protons = len(tpc_energy[proton_mask])
print('total protons: ', n_protons)
proton_spectrum = ROOT.TH1D('proton_spectrum', 'proton spectrum', 3000, 0.5, 3.5)
proton_spectrum.FillN(n_protons, tpc_energy[proton_mask], np.ones(n_protons, dtype='float64'))

if False:
    proton_spectrum = ROOT.TH1D('proton_spectrum', 'proton spectrum', 3000, 0.5, 3.5)
    proton_spectrum.FillN(n_protons, tpc_energy[proton_mask], np.ones(n_protons, dtype='float64'))

    peak_location_guesses = [0.913,1.063,1.1264,1.1331,1.1376,1.1778,1.817,1.857]#,2025,2089,2182,2197,2250,2410,2455]#825, 910, 1054]#, 1183, 1262, 1376, 1792, 2060, 2157, 2429]
    sigma_guess = 0.100
    sigma_bounds = (.005,0.50)
    magnitude_guess = 100
    fit_range = (0.890,1.900)#(700, 2600)
    peak_location_wiggle = 0.030

    function_string = '[0] + [1]*x'
    for i in range(len(peak_location_guesses)):
        function_string += ' + [%d]*exp(-0.5*((x-[%d])/[2])^2)/([2] *sqrt(2*pi))*%f'%(2*i+3, 2*i+4, proton_spectrum.GetBinWidth(0))
    #use [2] for sigma
    f_to_fit = ROOT.TF1('to_fit', function_string, *fit_range)
    f_to_fit.SetParLimits(0, 0, np.inf)
    for i in range(len(peak_location_guesses)):
        f_to_fit.SetParameter(2*i+3, magnitude_guess) #magnitude
        f_to_fit.SetParLimits(2*i+3,0,np.inf)
        f_to_fit.SetParName(2*i+3, 'A_%d'%i)

        f_to_fit.SetParameter(2*i+4, peak_location_guesses[i])
        f_to_fit.SetParLimits(2*i+4,peak_location_guesses[i] - peak_location_wiggle,peak_location_guesses[i] + peak_location_wiggle)
        f_to_fit.SetParName(2*i+4, 'mu_%d'%i)

    f_to_fit.SetParameter(2, sigma_guess)
    f_to_fit.SetParLimits(2,*sigma_bounds)
    f_to_fit.SetParName(2, 'sigma')
    f_to_fit.SetNpx(3000)
    # for i in range(12):
    fit_res = proton_spectrum.Fit(f_to_fit, "LR")
    proton_spectrum.Draw()

def fit_single_peak(energy_guess, energy_wiggle, energy_window):
    function_string = '[0] + [1]*x + [3]*exp(-0.5*((x-[4])/[2])^2)/([2] *sqrt(2*pi))*%f'%proton_spectrum.GetBinWidth(0)
    f_to_fit = ROOT.TF1('to_fit', function_string, energy_guess-energy_window, energy_guess+energy_window)

    f_to_fit.SetParameter(3, 100) #magnitude
    f_to_fit.SetParLimits(3,0,np.inf)
    f_to_fit.SetParName(3, 'A')

    f_to_fit.SetParameter(4, energy_guess)
    f_to_fit.SetParLimits(4,energy_guess - energy_wiggle,energy_guess+energy_wiggle)
    f_to_fit.SetParName(4, 'mu')

    f_to_fit.SetParameter(2, 0.05)
    f_to_fit.SetParLimits(2,0.005, 0.5)
    f_to_fit.SetParName(2, 'sigma')
    f_to_fit.SetNpx(3000)

    f_to_fit.SetNpx(3000)
    done = False
    while not done:
        fit_res = proton_spectrum.Fit(f_to_fit, "LRS")
        done = fit_res.IsValid()
    proton_spectrum.Draw()
    return fit_res

def fit_multiple_peaks(energy_guesses, energy_wiggle, energy_window):
    function_string = '[0] + [1]*x'
    for i in range(len(energy_guesses)):
        function_string += ' + [%d]*exp(-0.5*((x-[%d])/[2])^2)/([2] *sqrt(2*pi))*%f'%(2*i+3, 2*i+4, proton_spectrum.GetBinWidth(0))
    #use [2] for sigma
    f_to_fit = ROOT.TF1('to_fit', function_string, energy_guesses[0] - energy_window, energy_guesses[-1]+energy_window)
    f_to_fit.SetParLimits(0, 0, np.inf)
    for i in range(len(energy_guesses)):
        f_to_fit.SetParameter(2*i+3, 100) #magnitude
        f_to_fit.SetParLimits(2*i+3,0,np.inf)
        f_to_fit.SetParName(2*i+3, 'A_%d'%i)

        f_to_fit.SetParameter(2*i+4, energy_guesses[i])
        f_to_fit.SetParLimits(2*i+4,energy_guesses[i] - energy_wiggle,energy_guesses[i] + energy_wiggle)
        f_to_fit.SetParName(2*i+4, 'mu_%d'%i)

    f_to_fit.SetParameter(2, 0.05)
    f_to_fit.SetParLimits(2,0.005, 0.5)
    f_to_fit.SetParName(2, 'sigma')
    f_to_fit.SetNpx(3000)

    f_to_fit.SetNpx(3000)
    done = False
    while not done:
        fit_res = proton_spectrum.Fit(f_to_fit, "LRS")
        done = fit_res.IsValid()
    proton_spectrum.Draw()
    return fit_res