import os

import ROOT
import numpy as np

from raw_viewer import ddas_interface

ddas_runs = [277,278]
analysis_file_name = 'ddas_runs_277_278_proton_spectra.root'

merged_data = ROOT.TChain('merged_data')
for ddas_run in ddas_runs:
    merged_data.Add(ddas_interface.get_merged_root_file_path(ddas_run))

df = ROOT.RDataFrame(merged_data)
good_protons = df.Filter( 'tpc_particle_id==1 && !tpc_should_veto')

proton_spectrum = good_protons.Histo1D(('proton_hist', 'proton energy spectrum', 350, 500, 3500),'tpc_energy')
proton_spectrum.Draw()

peak_location_guesses = [825, 910, 1054, 1183, 1262, 1376, 1792, 2060, 2157, 2429]
sigma_guess = 100
sigma_bounds = (10,500)
magnitude_guess = 100
fit_range = (700, 2600)
peak_location_wiggle = 30

function_string = '[0] + [1]*x'
for i in range(len(peak_location_guesses)):
    function_string += ' + gausn(%d)/%f'%(3*i+2, proton_spectrum.GetBinWidth(0))
f_to_fit = ROOT.TF1('to_fit', function_string, *fit_range)
for i in range(len(peak_location_guesses)):
    f_to_fit.SetParameter(3*i+2, magnitude_guess) #magnitude
    f_to_fit.SetParLimits(3*i+2,0,np.inf)
    f_to_fit.SetParName(3*i+2, 'A_%d'%i)

    f_to_fit.SetParameter(3*i+3, peak_location_guesses[i])
    f_to_fit.SetParLimits(3*i+3,peak_location_guesses[i] - peak_location_wiggle,peak_location_guesses[i] + peak_location_wiggle)
    f_to_fit.SetParName(3*i+3, 'mu_%d'%i)

    f_to_fit.SetParameter(3*i+4, sigma_guess)
    f_to_fit.SetParLimits(3*i+4,*sigma_bounds)
    f_to_fit.SetParName(3*i+4, 'sigma_%d'%i)
f_to_fit.SetNpx(3000)
proton_spectrum.Fit(f_to_fit, "LR")
proton_spectrum.Fit(f_to_fit, "LR")
proton_spectrum.Fit(f_to_fit, "LR")
