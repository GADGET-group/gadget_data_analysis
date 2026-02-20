import os

import ROOT
import numpy as np

from raw_viewer import ddas_interface, process_runs
from e23035_analysis import e23035_runs, fitting_tools

print('loading 59Zn data')
get_runs_Zn = np.array(e23035_runs.run_df['GET'][(e23035_runs.run_df['Run Type']=='59Zn') & (e23035_runs.run_df['Field Cage Functional?'] == 'yes')])
get_runs_Zn = get_runs_Zn[(get_runs_Zn != 298) & (get_runs_Zn != 297)] #TODO: need to merge these runs!!!
tpc_energy_Zn = e23035_runs.get_energy_MeV(get_runs_Zn)
angles_Zn = process_runs.get_angle('e23035', get_runs_Zn)

print(get_runs_Zn)
proton_mask_Zn = e23035_runs.get_proton_mask(get_runs_Zn)#&(np.degrees(angles)>15)
n_protons_Zn = len(tpc_energy_Zn[proton_mask_Zn])
print('total protons in 59Zn runs: ', n_protons_Zn)
proton_spectrum_Zn = ROOT.TH1D('proton_spectrum_59Zn', 'Proton Spectrum from 59Zn Runs', 1000, 0.5, 3.5)#3000
proton_spectrum_Zn.FillN(n_protons_Zn, tpc_energy_Zn[proton_mask_Zn], np.ones(n_protons_Zn, dtype='float64'))

alpha_mask_Zn = e23035_runs.get_alpha_mask(get_runs_Zn)#&(np.degrees(angles)>15)
n_alphas_Zn = len(tpc_energy_Zn[alpha_mask_Zn])
print('total alphas in 59Zn runs: ', n_alphas_Zn)
alpha_spectrum_Zn = ROOT.TH1D('alpha_spectrum_59Zn', 'Alpha Spectrum from 59Zn Runs', 350, 2, 9)
alpha_spectrum_Zn.FillN(n_alphas_Zn, tpc_energy_Zn[alpha_mask_Zn], np.ones(n_alphas_Zn, dtype='float64'))

print('loading 60Ga data')
get_run_canidates_Ga = np.array(range(275,279))#np.array(e23035_runs.run_df['GET'][(e23035_runs.run_df['Run Type']=='60Ga')])#
get_runs_Ga = []
for run in get_run_canidates_Ga:
    if not np.isnan(run):
        if os.path.exists(process_runs.get_h5_path('e23035', run)):
            get_runs_Ga.append(run)
get_runs_Ga = np.array(get_runs_Ga)
get_runs_Ga = np.sort(get_runs_Ga)
tpc_energy_Ga  = e23035_runs.get_energy_MeV(get_runs_Ga )
angles_Ga = process_runs.get_angle('e23035', get_runs_Ga)

proton_mask_Ga = e23035_runs.get_proton_mask(get_runs_Ga)#&(np.degrees(angles)>15)
n_protons_Ga = len(tpc_energy_Ga[proton_mask_Ga])
print('total protons in 60Ga runs: ', n_protons_Ga)
proton_spectrum_Ga = ROOT.TH1D('proton_spectrum_60Ga', 'Proton Spectrum from 60Ga Runs', 1000, 0.5, 3.5)#3000
proton_spectrum_Ga.FillN(n_protons_Ga, tpc_energy_Ga[proton_mask_Ga], np.ones(n_protons_Ga, dtype='float64'))

alpha_mask_Ga = e23035_runs.get_alpha_mask(get_runs_Ga)#&(np.degrees(angles)>15)
n_alphas_Ga = len(tpc_energy_Ga[alpha_mask_Ga])
print('total alphas in 60Ga runs: ', n_alphas_Ga)
alpha_spectrum_Ga = ROOT.TH1D('alpha_spectrum_60Ga', 'Alpha Spectrum from 60Ga Runs', 350, 2, 9)
alpha_spectrum_Ga.FillN(n_alphas_Ga, tpc_energy_Ga[alpha_mask_Ga], np.ones(n_alphas_Ga, dtype='float64'))

zn_run_cross_scint_counts = 0
for ddas_run in np.array(e23035_runs.run_df['DDAS'][e23035_runs.run_df['GET'].isin(get_runs_Zn)]):
     zn_run_cross_scint_counts += ddas_interface.get_cross_scint_counts(ddas_run) #TODO: use only cross scintilator counts during this get run
ga_in_ga_runs = ddas_interface.get_counts_in_pid_cut(240, '60Ga')/ddas_interface.get_cross_scint_counts(240)*zn_run_cross_scint_counts
zn_in_ga_runs = ddas_interface.get_counts_in_pid_cut(240, '59Zn')/ddas_interface.get_cross_scint_counts(240)*zn_run_cross_scint_counts

ga_run_cross_scint_counts = 0
for ddas_run in np.array(e23035_runs.run_df['DDAS'][e23035_runs.run_df['GET'].isin(get_runs_Ga)]):
     ga_run_cross_scint_counts += ddas_interface.get_cross_scint_counts(ddas_run) #TODO: use only cross scintilator counts during this get run
ga_in_ga_runs = ddas_interface.get_counts_in_pid_cut(240, '60Ga')/ddas_interface.get_cross_scint_counts(240)*ga_run_cross_scint_counts
zn_in_ga_runs = ddas_interface.get_counts_in_pid_cut(240, '59Zn')/ddas_interface.get_cross_scint_counts(240)*ga_run_cross_scint_counts

peaks_to_fit = [[0.913], [1.063], [0.913, 1.063, 1.1, 1.264, 1.331,1.376]]
peaks_60Ga_to_fit = [0.72, 1.11, 1.2]
# fit_res, background, peaks, rp, canvas, spectrum_to_plot, f_to_fit, h_fit = fitting_tools.fit_peaks(proton_spectrum_Ga,[1.063,1.11,1.2,1.25,1.3,1.35, 1.4,1.5,1.62,1.77,1.85, 1.95,2.04],0.05,(1.05,2.16))

