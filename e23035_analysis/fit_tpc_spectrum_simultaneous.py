import os
import csv
from pathlib import Path

import ROOT
import numpy as np

from raw_viewer import ddas_interface, process_runs, degai
from e23035_analysis import e23035_runs, fitting_tools, spectrum_fitter, root_vis_tools

experiment = 'e23035'
tpc_config = 'smart2_rpr.csv'
num_workers = 200
force_refit=False

# efficiencies with 0.100000 s implant time and 0.100000 s decay time
# Assumes 12 ms dead time at start of decay window + 2 ms at end
# These efficiencies are defined in terms of fractions of implanted nuclie which decay during the measurement window
# 61Ge efficiency =  0.3230255772737927
Zn59_cycle_efficiency =  0.41616841590773374
Ga60_cycle_efficiency =  0.37410064021102757

proton_binning = (4000//5, 0, 4000)

ddas_runs_protons_all_energies_60Ga = e23035_runs.get_ddas_60_Ga_runs(good_gamma=False, final_beam_settings=True, good_low_energy_tpc=True, good_long_tracks_tpc=True)
pspec_all_energy_60Ga = ddas_interface.get_histogram(experiment, ddas_runs_protons_all_energies_60Ga, proton_binning, "proton_spectrum_60Ga", "60Ga proton_spectrum", "tpc_energy", "tpc_particle_id==1", num_workers=num_workers, tpc_ini_filename=tpc_config)

proton_energy_v_tsbo_60Ga = ddas_interface.get_histogram(experiment, ddas_runs_protons_all_energies_60Ga, (100, 0, 0.100, 400, 0, 4000), "proton_energy_v_tsbo_60Ga", "60Ga proton energy vs time since beam off", "tpc_energy:time_since_beam_off", "tpc_particle_id==1", num_workers=num_workers, tpc_ini_filename=tpc_config)


ddas_runs_protons_59Zn = e23035_runs.get_ddas_59_Zn_runs(good_gamma=False, final_beam_settings=True, good_low_energy_tpc=True, good_long_tracks_tpc=True)
pspec_59Zn = ddas_interface.get_histogram(experiment, ddas_runs_protons_59Zn, proton_binning, "proton_spectrum_59Zn", "59Zn proton_spectrum", "tpc_energy", "tpc_particle_id==1", num_workers=num_workers, tpc_ini_filename=tpc_config)

if True: #show charged particle spectra
    _1=root_vis_tools.draw_overlaid_histograms({'60Ga':pspec_all_energy_60Ga, '59Zn':pspec_59Zn}, 'proton spectra')

fit_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tpc_spectrum_fitting')
def load_peaks_from_csv(filename):
    all_peaks = []
    with open(os.path.join(fit_path, filename), 'r') as f:
        reader = csv.reader(f)
        current_group = []
        fit_window=(0,0)
        for i, row in enumerate(reader):
            if i == 0 or not row:
                continue
            if row[0]=='STOP':
                break
            try:
                if len(row[0]) > 0:
                    if len(current_group) > 0:
                        all_peaks.append((current_group, *fit_window))
                        current_group = []
                    start, stop = row[0].split('-')
                    fit_window = (float(start), float(stop))
                current_group.append(float(row[1]))
            except Exception as e:
                print(f'failed to load peak from row {row}: {e}')
        
        if len(current_group) > 0:
            all_peaks.append((current_group, *fit_window))
    return all_peaks

def get_save_path(save_name):
    return os.path.join(fit_path,save_name)

def fit_multi_peaks(spectra, peaks, save_name, zero_bg_shift=False, likelihood=True, force_refit=False, additional_param_bounds={}, loc_wiggle=10):
    root_filepath = save_name if save_name.endswith('.root') else save_name + '.root'
    if os.path.exists(root_filepath) and not force_refit:
        print(f"Loading previous multi-spectrum fit from {root_filepath}")
        return spectrum_fitter.load_spectrum_fitter_from_file(root_filepath)

    f = spectrum_fitter.multi_spectrum_fitter(spectra, 'bg_shift_gaus')
    f.parameterizations = {
        'sigma': {
            'formula': '[sigma_c] + [sigma_m]*({mu})',
            'params': ['sigma_c', 'sigma_m'],
            'guesses': [26, 0.01],
            'bounds': [(8, 30), (0.0001, 0.03)]
        }
    }
    for spec in f.spectra:
        spec.GetXaxis().UnZoom()
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
    if save_name:
        f.save(save_name)
    return f

proton_peak_guesses = load_peaks_from_csv('proton_peaks.csv')

# Simultaneous Fit!
import os
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tpc_spectrum_fitting', '60Ga_59Zn_simultaneous_protons')

f_proton_simultaneous = fit_multi_peaks(
    [pspec_all_energy_60Ga, pspec_59Zn], 
    proton_peak_guesses,
    save_path, force_refit=force_refit,
    additional_param_bounds={'bg_slope':lambda E: (0,0) if E > 1000 else (-1,1), 'amplitude': lambda E:(1, 1e6)}, 
    loc_wiggle=10
)
ROOT.gROOT.SetBatch(False)
# Display multi-spectrum fit for the first peak
f_proton_simultaneous.show_fit_results(0)


zn_ga_comparison_overlay = root_vis_tools.draw_overlaid_histograms({'60Ga':pspec_all_energy_60Ga, '59Zn':pspec_59Zn}, 'proton spectra')
