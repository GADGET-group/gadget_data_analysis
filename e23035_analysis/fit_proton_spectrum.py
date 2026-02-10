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
get_runs = range(136,151)
tpc_energy = e23035_runs.get_energy_MeV(get_runs)
angles = process_runs.get_angle('e23035', get_runs)

proton_mask = e23035_runs.get_proton_mask(get_runs)#&(np.degrees(angles)>15)
n_protons = len(tpc_energy[proton_mask])
print('total protons: ', n_protons)
proton_spectrum = ROOT.TH1D('proton_spectrum', 'proton spectrum', 3000, 0.5, 3.5)
proton_spectrum.FillN(n_protons, tpc_energy[proton_mask], np.ones(n_protons, dtype='float64'))

alpha_mask = e23035_runs.get_alpha_mask(get_runs)#&(np.degrees(angles)>15)
n_alphas = len(tpc_energy[alpha_mask])
print('total alphas: ', n_alphas)
alpha_spectrum = ROOT.TH1D('alpha_spectrum', 'alpha spectrum', 350, 2, 9)
alpha_spectrum.FillN(n_alphas, tpc_energy[alpha_mask], np.ones(n_alphas, dtype='float64'))


def fit_single_peak(spectrum, energy_guess, energy_wiggle, energy_window):
    function_string = '[0] + [1]*x + [3]*exp(-0.5*((x-[4])/[2])^2)/([2] *sqrt(2*pi))*%f'%spectrum.GetBinWidth(0)
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
        fit_res = spectrum.Fit(f_to_fit, "LRS")
        done = fit_res.IsValid()
    spectrum.Draw()
    return fit_res

def fit_multiple_peaks(spectrum, energy_guesses, energy_wiggle, energy_window):
    function_string = '[0] + [1]*x'
    for i in range(len(energy_guesses)):
        function_string += ' + [%d]*exp(-0.5*((x-[%d])/[2])^2)/([2] *sqrt(2*pi))*%f'%(2*i+3, 2*i+4, spectrum.GetBinWidth(0))
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
        fit_res = spectrum.Fit(f_to_fit, "LRS")
        done = fit_res.IsValid()
    spectrum.Draw()
    return fit_res