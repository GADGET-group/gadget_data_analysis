import os

import ROOT
import numpy as np

from raw_viewer import ddas_interface, process_runs
from e23035_analysis import e23035_runs, fitting_tools

print('loading 59Zn data')
get_runs_Zn = e23035_runs.run_df['GET'][(e23035_runs.run_df['Run Type']=='59Zn') & (e23035_runs.run_df['Field Cage Functional?'] == 'yes')]
get_runs_Zn = get_runs_Zn[(get_runs_Zn != 298) & (get_runs_Zn != 297)] #TODO: need to merge these runs!!!
tpc_energy_Zn = e23035_runs.get_energy_MeV(get_runs_Zn)
angles_Zn = process_runs.get_angle('e23035', get_runs_Zn)

proton_mask_Zn = e23035_runs.get_proton_mask(get_runs_Zn)#&(np.degrees(angles)>15)
n_protons_Zn = len(tpc_energy_Zn[proton_mask_Zn])
print('total protons in 59Zn runs: ', n_protons_Zn)
proton_spectrum = ROOT.TH1D('proton_spectrum_59Zn', 'Proton Spectrum from 59Zn Runs', 1000, 0.5, 3.5)#3000
proton_spectrum.FillN(n_protons_Zn, tpc_energy_Zn[proton_mask_Zn], np.ones(n_protons_Zn, dtype='float64'))

alpha_mask_Zn = e23035_runs.get_alpha_mask(get_runs_Zn)#&(np.degrees(angles)>15)
n_alphas_Zn = len(tpc_energy_Zn[alpha_mask_Zn])
print('total alphas in 59Zn runs: ', n_alphas_Zn)
alpha_spectrum = ROOT.TH1D('alpha_spectrum_59Zn', 'Alpha Spectrum from 59Zn Runs', 350, 2, 9)
alpha_spectrum.FillN(n_alphas_Zn, tpc_energy_Zn[alpha_mask_Zn], np.ones(n_alphas_Zn, dtype='float64'))


get_runs_Ga = range(275,279)#e23035_runs.run_df['GET'][(e23035_runs.run_df['Run Type']=='60Ga')]
tpc_energy_Ga  = e23035_runs.get_energy_MeV(get_runs_Ga )
angles_Ga = process_runs.get_angle('e23035', get_runs_Ga)

proton_mask_Ga = e23035_runs.get_proton_mask(get_runs_Ga)#&(np.degrees(angles)>15)
n_protons_Ga = len(tpc_energy_Ga[proton_mask_Ga])
print('total protons in 59Ga runs: ', n_protons_Ga)
proton_spectrum = ROOT.TH1D('proton_spectrum_59Ga', 'Proton Spectrum from 59Ga Runs', 1000, 0.5, 3.5)#3000
proton_spectrum.FillN(n_protons_Ga, tpc_energy_Ga[proton_mask_Ga], np.ones(n_protons_Ga, dtype='float64'))

alpha_mask_Ga = e23035_runs.get_alpha_mask(get_runs_Ga)#&(np.degrees(angles)>15)
n_alphas_Ga = len(tpc_energy_Ga[alpha_mask_Ga])
print('total alphas in 59Ga runs: ', n_alphas_Ga)
alpha_spectrum = ROOT.TH1D('alpha_spectrum_59Ga', 'Alpha Spectrum from 59Ga Runs', 350, 2, 9)
alpha_spectrum.FillN(n_alphas_Ga, tpc_energy_Ga[alpha_mask_Ga], np.ones(n_alphas_Ga, dtype='float64'))

peaks_to_fit = [[0.913], [1.063], [0.913, 1.063, 1.1, 1.264, 1.331,1.376]]
