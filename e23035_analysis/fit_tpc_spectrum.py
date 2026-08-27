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
alpha_binning = (7000//10, 2000, 9000)

ddas_runs_protons_all_energies_60Ga = e23035_runs.get_ddas_60_Ga_runs(good_gamma=False, final_beam_settings=True, good_low_energy_tpc=True, good_long_tracks_tpc=True)
pspec_all_energy_60Ga = ddas_interface.get_histogram(experiment, ddas_runs_protons_all_energies_60Ga, proton_binning, "proton_spectrum_60Ga", "60Ga proton_spectrum", "tpc_energy", "tpc_particle_id==1", num_workers=num_workers, tpc_ini_filename=tpc_config)
ddas_runs_low_energy_protons = e23035_runs.get_ddas_60_Ga_runs(good_gamma=False, final_beam_settings=True, good_low_energy_tpc=True, good_long_tracks_tpc=False)
pspec_low_energy_60Ga = ddas_interface.get_histogram(experiment, ddas_runs_low_energy_protons, proton_binning, "proton_spectrum_low_energy_60Ga", "60Ga proton_spectrum_low_energy", "tpc_energy", "tpc_particle_id==1", num_workers=num_workers, tpc_ini_filename=tpc_config)
ddas_runs_alphas_60Ga = e23035_runs.get_ddas_60_Ga_runs(good_gamma=False, final_beam_settings=True, good_long_tracks_tpc=False, good_low_energy_tpc=False)
aspec_60Ga = ddas_interface.get_histogram(experiment, ddas_runs_alphas_60Ga, alpha_binning, 'alpha_spectrum_60Ga', '60Ga run alpha spectrum', 'tpc_energy', 'tpc_particle_id==2', num_workers=num_workers, tpc_ini_filename=tpc_config)
ddas_runs_high_energy_protons_60Ga = e23035_runs.get_ddas_60_Ga_runs(good_gamma=False, final_beam_settings=True, good_low_energy_tpc=False, good_long_tracks_tpc=True)
pspec_high_energy_60Ga = ddas_interface.get_histogram(experiment, ddas_runs_high_energy_protons_60Ga, proton_binning, "proton_spectrum_high_energy_60Ga", "60Ga proton_spectrum_high_energy", "tpc_energy", "tpc_particle_id==1", num_workers=num_workers, tpc_ini_filename=tpc_config)

alpha_energy_v_tsbo_60Ga = ddas_interface.get_histogram(experiment, ddas_runs_alphas_60Ga, (100, 0, 0.100, 140, 2000, 9000), "alpha_energy_v_tsbo_60Ga", "60Ga alpha energy vs time since beam off", "tpc_energy:time_since_beam_off", "tpc_particle_id==2", num_workers=num_workers, tpc_ini_filename=tpc_config)
proton_energy_v_tsbo_60Ga = ddas_interface.get_histogram(experiment, ddas_runs_protons_all_energies_60Ga, (100, 0, 0.100, 400, 0, 4000), "proton_energy_v_tsbo_60Ga", "60Ga proton energy vs time since beam off", "tpc_energy:time_since_beam_off", "tpc_particle_id==1", num_workers=num_workers, tpc_ini_filename=tpc_config)

sigma_tpc = lambda E: (0.011107*E/1e3 + 0.008813049)*1e3

ddas_runs_protons_59Zn = e23035_runs.get_ddas_59_Zn_runs(good_gamma=False, final_beam_settings=True, good_low_energy_tpc=True, good_long_tracks_tpc=True)
pspec_59Zn = ddas_interface.get_histogram(experiment, ddas_runs_protons_59Zn, proton_binning, "proton_spectrum_59Zn", "59Zn proton_spectrum", "tpc_energy", "tpc_particle_id==1", num_workers=num_workers, tpc_ini_filename=tpc_config)
aspec_59Zn = ddas_interface.get_histogram(experiment, ddas_runs_protons_59Zn, alpha_binning, 'alpha_spectrum_59Zn', '59Zn alpha spectrum', 'tpc_energy', 'tpc_particle_id==2', num_workers=num_workers, tpc_ini_filename=tpc_config)

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
proton_peak_guesses = load_peaks_from_csv('proton_peaks.csv')
f_all_proton_60Ga = fit_peaks(pspec_all_energy_60Ga, 
                         proton_peak_guesses,
                         '60Ga_all_proton_energies', force_refit=force_refit
                        ,additional_param_bounds={'bg_slope':lambda E: (0,0) if E > 2000 else (-1,1), 'amplitude': lambda E:(1, 1e6)}, loc_wiggle=20)
f_proton_low_energy_60Ga = fit_peaks(pspec_low_energy_60Ga, 
                         proton_peak_guesses,
                         '60Ga_low_energy_protons', force_refit=force_refit
                        ,additional_param_bounds={'bg_slope':lambda E: (0,0) if E > 2000 else (-1,1)})
f_high_energy_proton_60Ga = fit_peaks(pspec_high_energy_60Ga, 
                         proton_peak_guesses,
                         '60Ga_high_energy_protons', force_refit=force_refit
                        ,additional_param_bounds={'bg_slope':lambda E: (0,0) if E > 2000 else (-1,1)})

alpha_guess = load_peaks_from_csv('alpha_peaks.csv')
f_alpha_60Ga = fit_peaks(aspec_60Ga, alpha_guess,
                    '60Ga_alpha', force_refit=force_refit,
                    additional_param_bounds={'bg_slope':lambda E: (0,0), 'amplitude': lambda E:(1, 1e6)}, loc_wiggle=50)#3356

f_proton_59Zn = fit_peaks(pspec_59Zn, proton_peak_guesses, '59Zn_protons', force_refit=force_refit,
                            additional_param_bounds={'bg_slope':lambda E: (0,0) if E > 1000 else (-1,1), 'amplitude': lambda E:(1, 1e6)}, loc_wiggle=20)
ROOT.Math.MinimizerOptions.SetDefaultErrorDef(1)
# Ensure batch mode is off so any interactive plots will display
ROOT.gROOT.SetBatch(False)
#show zn/ga comparison
zn_ga_comparison_overlay = root_vis_tools.draw_overlaid_histograms({'60Ga':pspec_all_energy_60Ga, '59Zn':pspec_59Zn}, 'proton spectra')

def make_energy_calibration(fit_name, peaks_csv, show_fit_result=True, force_0_offset=False):
    #use a TGraph to fit the peaks specified in the csv file.
    #Return the slope, offset, and paramter covariances so fit uncertainty can be propataged.
    fitter = get_fitter(fit_name)
    x_vals = []
    x_errs = []
    y_vals = []
    y_errs = []
    
    csv_path = os.path.join(fit_path, peaks_csv)
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row or len(row) < 6:
                continue
            use_calib = row[5].strip().lower()
            if use_calib in ['yes', 'true', '1', 'y']:
                try:
                    guess_E = float(row[1])
                    known_E = float(row[3])
                    known_E_err = float(row[4]) if row[4].strip() else 0.0
                except ValueError:
                    continue
                
                fitted_mu, fitted_mu_err = fitter.get_param_for_guess('mu', guess_E)
                if fitted_mu is not None:
                    x_vals.append(fitted_mu)
                    x_errs.append(fitted_mu_err)
                    y_vals.append(known_E)
                    y_errs.append(known_E_err)
                else:
                    print(f"Warning: Could not find fitted mu for guess {guess_E}")

    if len(x_vals) < 2:
        raise ValueError("Not enough points for calibration")
        
    n = len(x_vals)
    graph = ROOT.TGraphErrors(n, np.array(x_vals, dtype='float64'), np.array(y_vals, dtype='float64'),
                              np.array(x_errs, dtype='float64'), np.array(y_errs, dtype='float64'))
    graph.SetTitle(f"{fit_name} Energy Calibration")
    graph.GetXaxis().SetTitle("Fitted #mu (raw)")
    graph.GetYaxis().SetTitle("Known Energy (keV)")
    graph.SetMarkerStyle(20)
    
    fit_func = ROOT.TF1(f"calib_fit_{fit_name}", "pol1", min(x_vals)*0.9, max(x_vals)*1.1)
    if force_0_offset:
        fit_func.FixParameter(0, 0)
    fit_res = graph.Fit(fit_func, "SQ")
    
    offset = fit_func.GetParameter(0)
    slope = fit_func.GetParameter(1)
    
    cov_matrix = fit_res.GetCovarianceMatrix()
    cov = np.zeros((2,2))
    if cov_matrix.GetNrows() == 2:
        cov[0,0] = cov_matrix(0,0)
        cov[0,1] = cov_matrix(0,1)
        cov[1,0] = cov_matrix(1,0)
        cov[1,1] = cov_matrix(1,1)
    
    if show_fit_result:
        canvas = ROOT.TCanvas(f"c_calib_{fit_name}", f"{fit_name} Calibration", 800, 800)
        
        pad1 = ROOT.TPad("pad1", "pad1", 0, 0.3, 1, 1.0)
        pad1.SetBottomMargin(0.02)
        pad1.Draw()
        pad1.cd()
        
        graph.GetXaxis().SetLabelSize(0)
        graph.GetXaxis().SetTitleSize(0)
        graph.Draw("AP")
        
        canvas.cd()
        pad2 = ROOT.TPad("pad2", "pad2", 0, 0, 1, 0.3)
        pad2.SetTopMargin(0.02)
        pad2.SetBottomMargin(0.3)
        pad2.Draw()
        pad2.cd()
        
        res_vals = []
        res_errs = []
        for i in range(n):
            expected_y = slope * x_vals[i] + offset
            res_vals.append(y_vals[i] - expected_y)
            calib_var = cov[0,0] + (x_vals[i]**2) * cov[1,1] + 2 * x_vals[i] * cov[0,1]
            total_var = y_errs[i]**2 + (slope * x_errs[i])**2 + calib_var
            err = np.sqrt(total_var) if total_var > 0 else 0.0
            res_errs.append(err)
            
        res_graph = ROOT.TGraphErrors(n, np.array(x_vals, dtype='float64'), np.array(res_vals, dtype='float64'),
                                      np.array(x_errs, dtype='float64'), np.array(res_errs, dtype='float64'))
        res_graph.SetTitle("")
        res_graph.GetXaxis().SetTitle("Fitted #mu (raw)")
        res_graph.GetYaxis().SetTitle("Residual (keV)")
        res_graph.SetMarkerStyle(20)
        
        res_graph.GetYaxis().SetTitleSize(0.1)
        res_graph.GetYaxis().SetTitleOffset(0.5)
        res_graph.GetYaxis().SetLabelSize(0.08)
        res_graph.GetXaxis().SetTitleSize(0.12)
        res_graph.GetXaxis().SetTitleOffset(0.9)
        res_graph.GetXaxis().SetLabelSize(0.1)
        
        res_graph.Draw("AP")
        
        line = ROOT.TLine(min(x_vals)*0.9, 0, max(x_vals)*1.1, 0)
        line.SetLineStyle(2)
        line.Draw("SAME")
        
        canvas.Update()
        ROOT.SetOwnership(canvas, False)
        ROOT.SetOwnership(graph, False)
        ROOT.SetOwnership(res_graph, False)
        ROOT.SetOwnership(line, False)
        ROOT.SetOwnership(pad1, False)
        ROOT.SetOwnership(pad2, False)
        
    return slope, offset, cov

def apply_fit_to_point(fit_to_apply, mu, mu_err=0.0):
    slope, offset, cov = fit_to_apply
    
    calib_var = cov[0,0] + (mu**2) * cov[1,1] + 2 * mu * cov[0,1]
    total_var = calib_var + (slope * mu_err)**2
    
    new_mu = slope * mu + offset
    new_mu_err = np.sqrt(total_var) if total_var > 0 else 0.0
    
    return new_mu, new_mu_err

def apply_fit_to_csv(fit_to_aply, apply_to, cal_name='calibrated'):
    #make  a copy of the csv file, with the mu values scaled by the fit, and with uncertainties propaged to mu_err
    
    input_csv = os.path.join(fit_path, apply_to + '.csv')
    output_csv = os.path.join(fit_path, apply_to + '_' + cal_name + '.csv')
    
    with open(input_csv, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        header = next(reader)
        writer.writerow(header)
        
        try:
            mu_val_idx = header.index('mu_val')
            mu_err_idx = header.index('mu_err')
        except ValueError:
            print("Error: 'mu_val' or 'mu_err' column not found in CSV.")
            return
            
        for row in reader:
            if not row:
                writer.writerow(row)
                continue
                
            try:
                mu = float(row[mu_val_idx])
                mu_err = float(row[mu_err_idx])
                
                new_mu, new_mu_err = apply_fit_to_point(fit_to_aply, mu, mu_err)
                
                row[mu_val_idx] = f"{new_mu:.6g}"
                row[mu_err_idx] = f"{new_mu_err:.6g}"
            except ValueError:
                pass
                
            writer.writerow(row)

ecal_60Ga = make_energy_calibration('60Ga_alpha', 'alpha_peaks.csv', show_fit_result=True, force_0_offset=False)
apply_fit_to_csv(ecal_60Ga, '60Ga_alpha', 'alpha_cal')
apply_fit_to_csv(ecal_60Ga, '59Zn_protons', 'alpha_cal')
apply_fit_to_csv(ecal_60Ga, '60Ga_low_energy_protons', 'alpha_cal')

###################################
# TPC - gamma coincidence fitting #
###################################
#TODO: load histograms for 511, 491, etc coincidences just as is done in tp_gamma_coincidence.py


#mesh_spectrum = ddas_interface.get_histogram(ddas_runs, (1000,0,10000), "mesh_spectrum", "mesh_spectrum", 'mesh_pre_amp_cr',"tpc_particle_id==1", num_workers=200)

coinc_runs = e23035_runs.get_ddas_60_Ga_runs(good_gamma=True, good_long_tracks_tpc=False, good_low_energy_tpc=False, final_beam_settings=True)
adj_dict = degai.get_adjacency_dict(30)
cal_name = 'gm_511and2614_1'
nlc_name = 'c1'
addback_ethresh = 150
event_build_window = 500 #ns
tstart, tstop = 0, 7.6e-6
time_gate_str = f'(mesh_pre_amp_t - time)>{tstart} && (mesh_pre_amp_t - time)<{tstop}'
t_accidental_start, t_accidental_stop = -15e-6, -1e-6
accidental_time_gate_str = f'(mesh_pre_amp_t - time)>{t_accidental_start} && (mesh_pre_amp_t - time)<{t_accidental_stop}'
if False:
    gammaE_v_protonE = degai.get_histogram(experiment, runs, adj_dict, cal_name, (150, 0, 3000, 7000-150, 150, 7000), "gamma_v_proton_energy_time_gate", "gamma energy (keV) vs proton energy (keV) w/ expected (mesh time - gamma time)",
                                            "addback_energy:tpc_energy", 
                                        selection='tpc_particle_id==1 &&'+time_gate_str,
                                            dt_window_ns=event_build_window, e_thresh=addback_ethresh, nonlinearity_correction_name=nlc_name, tpc_ini_filename=tpc_config)
    gammaE_v_protonE_accidental = degai.get_histogram(experiment, runs, adj_dict, cal_name, (150, 0, 3000, 7000-150, 150, 7000), "gamma_v_proton_energy_accidental_gate", 
                                                    "gamma energy (keV) vs proton energy (keV) for accidental coincidences",
                                            "addback_energy:tpc_energy", 
                                        selection='tpc_particle_id==1 &&'+accidental_time_gate_str,
                                            dt_window_ns=event_build_window, e_thresh=addback_ethresh, nonlinearity_correction_name=nlc_name, tpc_ini_filename=tpc_config)
    gammaE_v_protonE.Sumw2()
    gammaE_v_protonE_accidental.Sumw2()
    gammaE_v_protonE_bg_subtracted = gammaE_v_protonE.Clone()
    gammaE_v_protonE_bg_subtracted.SetName('gammaE_v_protonE_bg_subtracted')
    gammaE_v_protonE_bg_subtracted.SetTitle('gamma energy (keV) vs proton energy (keV) with accidental coincidences subtracted')
    gammaE_v_protonE_bg_subtracted.Add(gammaE_v_protonE_accidental, -(tstop-tstart)/(t_accidental_stop-t_accidental_start))

    h491 = degai.get_bg_subtracted_projection(gammaE_v_protonE_bg_subtracted,(488, 494), (494, 501))
    h914 = degai.get_bg_subtracted_projection(gammaE_v_protonE_bg_subtracted, (911, 917), (918,927))
    h1398 = degai.get_bg_subtracted_projection(gammaE_v_protonE_bg_subtracted, (1395, 1401), (1420,1430))
    h511 = degai.get_bg_subtracted_projection(gammaE_v_protonE_bg_subtracted, (508, 513), (518,529))

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