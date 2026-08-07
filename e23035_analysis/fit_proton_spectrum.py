import os
from pathlib import Path

import ROOT
import numpy as np

from raw_viewer import ddas_interface, process_runs
from e23035_analysis import e23035_runs, fitting_tools, spectrum_fitter

experiment = 'e23035'
num_workers = 200
force_refit=True

# efficiencies with 0.100000 s implant time and 0.100000 s decay time
# Assumes 12 ms dead time at start of decay window + 2 ms at end
# These efficiencies are defined in terms of fractions of implanted nuclie which decay during the measurement window
# 61Ge efficiency =  0.3230255772737927
Zn59_cycle_efficiency =  0.41616841590773374
Ga60_cycle_efficiency =  0.37410064021102757

proton_binning = (4000//5, 0, 4000)
alpha_binning = (7000//10, 2000, 9000)

ddas_runs_protons_all_energies = e23035_runs.get_ddas_60_Ga_runs(good_gamma=False, final_beam_settings=True, good_low_energy_tpc=True, good_long_tracks_tpc=True)
pspec = ddas_interface.get_histogram(experiment, ddas_runs_protons_all_energies, proton_binning, "proton_spectrum", "proton_spectrum", "tpc_energy", "tpc_particle_id==1", num_workers=num_workers)
ddas_runs_low_energy_protons = e23035_runs.get_ddas_60_Ga_runs(good_gamma=False, final_beam_settings=True, good_low_energy_tpc=True, good_long_tracks_tpc=False)
pspec_low_energy = ddas_interface.get_histogram(experiment, ddas_runs_low_energy_protons, proton_binning, "proton_spectrum_low_energy", "proton_spectrum_low_energy", "tpc_energy", "tpc_particle_id==1", num_workers=num_workers)
ddas_runs_alphas_60Ga = e23035_runs.get_ddas_60_Ga_runs(good_gamma=False, final_beam_settings=True, good_long_tracks_tpc=False, good_low_energy_tpc=False)
aspec_60Ga = ddas_interface.get_histogram(experiment, ddas_runs_alphas_60Ga, alpha_binning, 'alpha_spectrum_60Ga', '60Ga run alpha spectrum', 'tpc_energy', 'tpc_particle_id==2', num_workers=num_workers)

alpha_energy_v_tsbo = ddas_interface.get_histogram(experiment, ddas_runs_alphas_60Ga, (100, 0, 0.100, 140, 2000, 9000), "alpha_energy_v_tsbo", "alpha energy vs time since beam off", "tpc_energy:time_since_beam_off", "tpc_particle_id==2", num_workers=num_workers)
proton_energy_v_tsbo = ddas_interface.get_histogram(experiment, ddas_runs_protons_all_energies, (100, 0, 0.100, 400, 0, 4000), "proton_energy_v_tsbo", "proton energy vs time since beam off", "tpc_energy:time_since_beam_off", "tpc_particle_id==1", num_workers=num_workers)

sigma_tpc = lambda E: (0.011107*E/1e3 + 0.008813049)*1e3

fit_path = 'e23035_analysis/tpc_spectrum_fitting/'
def get_save_path(save_name):
    return os.path.join(fit_path,save_name)
def fit_exists(save_name):
    return Path(get_save_path(save_name)+'.root').exists()
def get_fitter(save_name):
    return spectrum_fitter.load_spectrum_fitter_from_file(get_save_path(save_name)+'.root')

def fit_peaks(spectrum, peaks, save_name, zero_bg_shift=False, likelihood=True, force_refit=False, additional_param_bounds={}, loc_wiggle=10):
    '''
    manual_bounds: if False, use add peaks function to cluster peaks and set fitting bounds.
    '''
    if not force_refit and fit_exists(save_name):
        return get_fitter(save_name)
    f = spectrum_fitter.spectrum_fitter(spectrum, 'bg_shift_gaus') # This will now use N=2 by default
    #f.param_bound_functions['sigma'] = lambda E: (sigma_tpc(E), sigma_tpc(E))
    f.parameterizations = {
        'sigma': {
            'formula': '[sigma_c] + [sigma_m]*({mu})',
            'params': ['sigma_c', 'sigma_m'],
            'guesses': [0., 0.01],
            'bounds': [(-100, 100), (0.0001, 0.1)]
        }
    }
    f.spectrum.GetXaxis().UnZoom()
    f.peaks_to_fit = peaks
    f.location_wiggle = loc_wiggle
    f.shared_sigma = False
    if zero_bg_shift:
        f.param_bound_functions['bg_shift'] = lambda E: (0, 0)
    for p in additional_param_bounds:
        f.param_bound_functions[p] = additional_param_bounds[p]
    if not likelihood:
        f.fit_options = f.fit_options.replace('L','')
    f.fit_peaks()
    f.save(get_save_path(save_name))
    return f

ROOT.Math.MinimizerOptions.SetDefaultErrorDef(1)
proton_peak_guesses = [([725, 814, 913, 950, 1060],500,1000),
                             ([1060, 1109,1160, 1212, 1260],1000,1288),
                             ([1330, 1380, 1440, 1488, 1560], 1300, 1580),
                             ([1625, 1730, 1780, 1820, 1840], 1585, 1900),
                             ([2040, 2250, 2520, 2610, 3140], 1910, 3500)
                             #([1330, 1380, 1440, 1468, 1541, 1625, 1710, 1780, 1820, 1860, 1950, 2030, 2090, 2180, 2200, 2250, 2410, 2460, 2500, 3140],1300,3500)
                             ]
f_all_proton = fit_peaks(pspec, 
                         proton_peak_guesses,
                         'all_proton_energies', force_refit=force_refit
                        ,additional_param_bounds={'bg_slope':lambda E: (0,0) if E > 1000 else (-1,1), 'amplitude': lambda E:(1, 1e6)}, loc_wiggle=20)
f_proton_low_energy = fit_peaks(pspec_low_energy, 
                         proton_peak_guesses,
                         'low_energy_proton_energies', force_refit=force_refit
                        ,additional_param_bounds={'bg_slope':lambda E: (0,0) if E > 1000 else (-1,1)})
                        

f_alpha = fit_peaks(aspec_60Ga, [([3374, 3529, 3662,3810, 3890,4000, 4125], 2800, 4400)],
                    'alpha', force_refit=force_refit,
                    additional_param_bounds={'bg_slope':lambda E: (0,0), 'amplitude': lambda E:(1, 1e6)})#3356

f_alpha2 = fit_peaks(aspec_60Ga, [([3529, 3662,3810,3890,4000, 4125], 2800, 4400)],
                    '_a', force_refit=force_refit,
                    additional_param_bounds={'bg_slope':lambda E: (0,0)})#3356

f_proton2 = fit_peaks(pspec, 
                         [([1060, 1109,1160, 1212],1000,1288),
                             #([1330, 1380, 1440, 1468, 1541, 1625, 1710, 1780, 1820, 1860, 1950, 2030, 2090, 2180, 2200, 2250, 2410, 2460],1288,2800)
                             ],
                         '_p', force_refit=force_refit
                        ,additional_param_bounds={'bg_slope':lambda E: (0,0) if E > 1000 else (-0.1,0.1)})
#do 3 sigma
ROOT.Math.MinimizerOptions.SetDefaultErrorDef(9)
f_all_proton_3sigma = fit_peaks(pspec, 
                         [([725, 814, 913, 950, 1060],500,1000),
                             ([1060, 1109,1160, 1212, 1260],1000,1288),
                             #([1330, 1380, 1440, 1468, 1541, 1625, 1710, 1780, 1820, 1860, 1950, 2030, 2090, 2180, 2200, 2250, 2410, 2460],1288,2800)
                             ],
                         'all_proton_energies_3_sigma', force_refit=force_refit
                        ,additional_param_bounds={'bg_slope':lambda E: (0,0) if E > 1000 else (-0.1,0.1)})

f_alpha_3sigma  = fit_peaks(aspec_60Ga, [([3374, 3529, 3662,3810, 3890,4000, 4125], 2800, 4400)],
                    'alpha_3sigma', force_refit=force_refit,
                    additional_param_bounds={'bg_slope':lambda E: (0,0)})
ROOT.Math.MinimizerOptions.SetDefaultErrorDef(1)
# Ensure batch mode is off so any interactive plots will display
ROOT.gROOT.SetBatch(False)

#mesh_spectrum = ddas_interface.get_histogram(ddas_runs, (1000,0,10000), "mesh_spectrum", "mesh_spectrum", 'mesh_pre_amp_cr',"tpc_particle_id==1", num_workers=200)

# print('loading 59Zn data')
# get_runs_Zn = np.array(e23035_runs.run_df['GET'][(e23035_runs.run_df['Run Type']=='59Zn') & (e23035_runs.run_df['Field Cage Functional?'] == 'yes')])
# get_runs_Zn = get_runs_Zn[(get_runs_Zn != 298) & (get_runs_Zn != 297)] #TODO: need to merge these runs!!!
# tpc_energy_Zn = e23035_runs.get_energy_MeV(get_runs_Zn)
# angles_Zn = process_runs.get_angle('e23035', get_runs_Zn)

# print(get_runs_Zn)
# proton_mask_Zn = e23035_runs.get_proton_mask(get_runs_Zn)#&(np.degrees(angles)>15)
# n_protons_Zn = len(tpc_energy_Zn[proton_mask_Zn])
# print('total protons in 59Zn runs: ', n_protons_Zn)
# proton_spectrum_Zn = ROOT.TH1D('proton_spectrum_59Zn', 'Proton Spectrum from 59Zn Runs', 1000, 0.5, 3.5)#3000
# proton_spectrum_Zn.FillN(n_protons_Zn, tpc_energy_Zn[proton_mask_Zn], np.ones(n_protons_Zn, dtype='float64'))

# alpha_mask_Zn = e23035_runs.get_alpha_mask(get_runs_Zn)#&(np.degrees(angles)>15)
# n_alphas_Zn = len(tpc_energy_Zn[alpha_mask_Zn])
# print('total alphas in 59Zn runs: ', n_alphas_Zn)
# alpha_spectrum_Zn = ROOT.TH1D('alpha_spectrum_59Zn', 'Alpha Spectrum from 59Zn Runs', 350, 2, 9)
# alpha_spectrum_Zn.FillN(n_alphas_Zn, tpc_energy_Zn[alpha_mask_Zn], np.ones(n_alphas_Zn, dtype='float64'))

# print('loading 60Ga data')
# get_run_canidates_Ga = np.array(e23035_runs.run_df['GET'][(e23035_runs.run_df['Run Type']=='60Ga')])#np.array(range(275,279))#
# get_runs_Ga = []
# for run in get_run_canidates_Ga:
#     if not np.isnan(run):
#         if os.path.exists(process_runs.get_h5_path('e23035', run)):
#             get_runs_Ga.append(run)
# get_runs_Ga = np.array(get_runs_Ga)
# get_runs_Ga = np.sort(get_runs_Ga)
# tpc_energy_Ga  = e23035_runs.get_energy_MeV(get_runs_Ga )
# angles_Ga = process_runs.get_angle('e23035', get_runs_Ga)

# proton_mask_Ga = e23035_runs.get_proton_mask(get_runs_Ga)#&(np.degrees(angles)>15)
# n_protons_Ga = len(tpc_energy_Ga[proton_mask_Ga])
# print('total protons in 60Ga runs: ', n_protons_Ga)
# proton_spectrum_Ga = ROOT.TH1D('proton_spectrum_60Ga', 'Proton Spectrum from 60Ga Runs', 1000, 0.5, 3.5)#3000
# proton_spectrum_Ga.FillN(n_protons_Ga, tpc_energy_Ga[proton_mask_Ga], np.ones(n_protons_Ga, dtype='float64'))

# alpha_mask_Ga = e23035_runs.get_alpha_mask(get_runs_Ga)#&(np.degrees(angles)>15)
# n_alphas_Ga = len(tpc_energy_Ga[alpha_mask_Ga])
# print('total alphas in 60Ga runs: ', n_alphas_Ga)
# alpha_spectrum_Ga = ROOT.TH1D('alpha_spectrum_60Ga', 'Alpha Spectrum from 60Ga Runs', 350, 2, 9)
# alpha_spectrum_Ga.FillN(n_alphas_Ga, tpc_energy_Ga[alpha_mask_Ga], np.ones(n_alphas_Ga, dtype='float64'))


# if False:
#     zn_run_cross_scint_counts = 0
#     for ddas_run in np.array(e23035_runs.run_df['DDAS'][e23035_runs.run_df['GET'].isin(get_runs_Zn)]):
#         zn_run_cross_scint_counts += ddas_interface.get_cross_scint_counts(ddas_run) #TODO: use only cross scintilator counts during this get run
#     zn_pid_run_cross_scint_counts = ddas_interface.get_cross_scint_counts(262)
#     ga_in_zn_runs = ddas_interface.get_counts_in_pid_cut(262, '60Ga')/zn_pid_run_cross_scint_counts*zn_run_cross_scint_counts*Ga60_cycle_efficiency
#     zn_in_zn_runs = ddas_interface.get_counts_in_pid_cut(262, '59Zn')/zn_pid_run_cross_scint_counts*zn_run_cross_scint_counts*Zn59_cycle_efficiency

#     ga_run_cross_scint_counts = 0
#     for ddas_run in np.array(e23035_runs.run_df['DDAS'][e23035_runs.run_df['GET'].isin(get_runs_Ga)]):
#         ga_run_cross_scint_counts += ddas_interface.get_cross_scint_counts(ddas_run) #TODO: use only cross scintilator counts during this get run
#     ga_pid_run_cross_scint_counts = ddas_interface.get_cross_scint_counts(240)
#     ga_in_ga_runs = ddas_interface.get_counts_in_pid_cut(240, '60Ga')/ga_pid_run_cross_scint_counts*ga_run_cross_scint_counts*Ga60_cycle_efficiency
#     zn_in_ga_runs = ddas_interface.get_counts_in_pid_cut(240, '59Zn')/ga_pid_run_cross_scint_counts*ga_run_cross_scint_counts*Zn59_cycle_efficiency

# peaks_to_fit = [[0.913], [1.063], [0.913, 1.063, 1.1, 1.264, 1.331,1.376]]
# peaks_60Ga_to_fit = [0.72, 1.11, 1.2]
# # fit_res, background, peaks, rp, canvas, spectrum_to_plot, f_to_fit, h_fit = fitting_tools.fit_peaks(proton_spectrum_Ga,[1.063,1.11,1.2,1.25,1.3,1.35, 1.4,1.5,1.62,1.77,1.85, 1.95,2.04],0.05,(1.05,2.16))

# f_ga_protons = spectrum_fitter.spectrum_fitter(proton_spectrum_Ga, 'gaus')