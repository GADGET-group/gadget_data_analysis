import os

import ROOT
import numpy as np

from raw_viewer import ddas_interface, process_runs
from e23035_analysis import e23035_runs, fitting_tools

if True:
    get_runs = e23035_runs.run_df['GET'][(e23035_runs.run_df['Run Type']=='59Zn') & (e23035_runs.run_df['Field Cage Functional?'] == 'yes')]
    get_runs = get_runs[(get_runs != 298) & (get_runs != 297)] #TODO: need to merge these runs!!!
else:
    get_runs = range(264,279)#e23035_runs.run_df['GET'][(e23035_runs.run_df['Run Type']=='60Ga')]
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

peaks_to_fit = [[0.913], [1.063], ]
