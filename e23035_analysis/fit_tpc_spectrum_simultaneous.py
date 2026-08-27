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
force_refit=True

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
    if True: #linear energy dependence
        f.parameterizations = {
            'sigma': {
                'formula': '[sigma_c] + [sigma_m]*({mu})',
                'params': ['sigma_c', 'sigma_m'],
                'guesses': [26, 0.01],
                'bounds': [(-40, 40), (0.0001, 0.1)]
            }
        }
    else: #sqrt depenence
        f.parameterizations = {
            'sigma': {
                'formula': 'std::sqrt([sigma_c] + [sigma_m]*({mu}))',
                'params': ['sigma_c', 'sigma_m'],
                'guesses': [26**2, 0.0],
                'bounds': [(-100, 40**2), (0, 1)]
            }
        }
    for spec in f.spectra:
        spec.GetXaxis().UnZoom()
    f.peaks_to_fit = peaks
    f.location_wiggle = loc_wiggle
    f.shared_sigma = False
    if zero_bg_shift:
        f.param_bound_functions['bg_shift'] = lambda E: (0, 0)
    else:
        f.shared_bg_shift = False
    for p in additional_param_bounds:
        f.param_bound_functions[p] = additional_param_bounds[p]
    if not likelihood:
        f.fit_options = f.fit_options.replace('L','')
    f.fit_peaks()
    
    failed_fits = []
    for i, res in enumerate(f.fit_results):
        if res is None or 'fit_res' not in res:
            failed_fits.append((i, f.peaks_to_fit[i], "Missing result"))
            continue
        
        fit_res = res['fit_res']
        if not fit_res.IsValid():
            status = int(fit_res)
            failed_fits.append((i, f.peaks_to_fit[i], f"Status {status}"))
            
    if failed_fits:
        print(f"Summary of {len(failed_fits)} failed fits:")
        for i, peak_info, reason in failed_fits:
            print(f"  Index {i}: Peaks {peak_info[0]} in window ({peak_info[1]:.1f}, {peak_info[2]:.1f}) -> {reason}")
    else:
        print(f"All {len(f.fit_results)} fits successful.")
        
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
    additional_param_bounds={'bg_slope':lambda E: (-1,1), #if E < 1000 else (0,0),
                             'amplitude': lambda E:(1e-3, 1e6)}, 
    loc_wiggle=15
)
ROOT.gROOT.SetBatch(False)
# Display multi-spectrum fit for the first peak
#f_proton_simultaneous.show_fit_results(4, False, True)


zn_ga_comparison_overlay = root_vis_tools.draw_overlaid_histograms({'60Ga':pspec_all_energy_60Ga, '59Zn':pspec_59Zn}, 'proton spectra')

def make_energy_calibration(fitter, fit_name, peaks_csv, show_fit_result=True, force_0_offset=False):
    #use a TGraph to fit the peaks specified in the csv file.
    #Return the slope, offset, and paramter covariances so fit uncertainty can be propataged.
    x_vals = []
    x_errs = []
    y_vals = []
    y_errs = []
    
    x_vals_unused = []
    x_errs_unused = []
    y_vals_unused = []
    y_errs_unused = []
    
    csv_path = os.path.join(fit_path, peaks_csv)
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row or len(row) < 6:
                continue
            use_calib = row[5].strip().lower()
            try:
                guess_E = float(row[1])
                known_E_str = row[3].strip()
                if not known_E_str:
                    continue
                known_E = float(known_E_str)
                known_E_err = float(row[4]) if row[4].strip() else 0.0
            except ValueError:
                continue
            
            fitted_mu, fitted_mu_err = fitter.get_param_for_guess('mu', guess_E)
            if fitted_mu is not None:
                if use_calib in ['yes', 'true', '1', 'y']:
                    x_vals.append(fitted_mu)
                    x_errs.append(fitted_mu_err)
                    y_vals.append(known_E)
                    y_errs.append(known_E_err)
                else:
                    x_vals_unused.append(fitted_mu)
                    x_errs_unused.append(fitted_mu_err)
                    y_vals_unused.append(known_E)
                    y_errs_unused.append(known_E_err)
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
        
        mg = ROOT.TMultiGraph()
        mg.SetTitle(f"{fit_name} Energy Calibration;Fitted #mu (raw);Known Energy (keV)")
        mg.Add(graph)
        
        if len(x_vals_unused) > 0:
            graph_unused = ROOT.TGraphErrors(len(x_vals_unused), np.array(x_vals_unused, dtype='float64'), np.array(y_vals_unused, dtype='float64'),
                                      np.array(x_errs_unused, dtype='float64'), np.array(y_errs_unused, dtype='float64'))
            graph_unused.SetMarkerStyle(24)
            graph_unused.SetMarkerColor(ROOT.kRed)
            graph_unused.SetLineColor(ROOT.kRed)
            mg.Add(graph_unused)
        
        mg.Draw("AP")
        mg.GetXaxis().SetLabelSize(0)
        mg.GetXaxis().SetTitleSize(0)
        
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
            
        res_graph = ROOT.TGraphErrors(n, np.array(y_vals, dtype='float64'), np.array(res_vals, dtype='float64'),
                                      np.array(y_errs, dtype='float64'), np.array(res_errs, dtype='float64'))
        res_graph.SetMarkerStyle(20)
        
        res_mg = ROOT.TMultiGraph()
        res_mg.SetTitle(";Known Energy (keV);Residual (Known - Fit) (keV)")
        res_mg.Add(res_graph)
        
        if len(x_vals_unused) > 0:
            res_vals_unused = []
            res_errs_unused = []
            for i in range(len(x_vals_unused)):
                expected_y = slope * x_vals_unused[i] + offset
                res_vals_unused.append(y_vals_unused[i] - expected_y)
                calib_var = cov[0,0] + (x_vals_unused[i]**2) * cov[1,1] + 2 * x_vals_unused[i] * cov[0,1]
                total_var = y_errs_unused[i]**2 + (slope * x_errs_unused[i])**2 + calib_var
                err = np.sqrt(total_var) if total_var > 0 else 0.0
                res_errs_unused.append(err)
            res_graph_unused = ROOT.TGraphErrors(len(x_vals_unused), np.array(y_vals_unused, dtype='float64'), np.array(res_vals_unused, dtype='float64'),
                                      np.array(y_errs_unused, dtype='float64'), np.array(res_errs_unused, dtype='float64'))
            res_graph_unused.SetMarkerStyle(24)
            res_graph_unused.SetMarkerColor(ROOT.kRed)
            res_graph_unused.SetLineColor(ROOT.kRed)
            res_mg.Add(res_graph_unused)

        res_mg.Draw("AP")
        res_mg.GetYaxis().SetTitleSize(0.1)
        res_mg.GetYaxis().SetTitleOffset(0.5)
        res_mg.GetYaxis().SetLabelSize(0.08)
        res_mg.GetXaxis().SetTitleSize(0.12)
        res_mg.GetXaxis().SetTitleOffset(0.9)
        res_mg.GetXaxis().SetLabelSize(0.1)
        
        line = ROOT.TLine(res_mg.GetXaxis().GetXmin(), 0, res_mg.GetXaxis().GetXmax(), 0)
        line.SetLineStyle(2)
        line.Draw("SAME")
        
        canvas.Update()
        ROOT.SetOwnership(canvas, False)
        ROOT.SetOwnership(graph, False)
        ROOT.SetOwnership(res_graph, False)
        ROOT.SetOwnership(mg, False)
        ROOT.SetOwnership(res_mg, False)
        ROOT.SetOwnership(line, False)
        ROOT.SetOwnership(pad1, False)
        ROOT.SetOwnership(pad2, False)
        if len(x_vals_unused) > 0:
            ROOT.SetOwnership(graph_unused, False)
            ROOT.SetOwnership(res_graph_unused, False)
            
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
            print(f"Error: 'mu_val' or 'mu_err' column not found in CSV {input_csv}.")
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

def show_detector_energy_resolution(fit_save_name):
    #extract detector energy resolution each of the fit windows in the specified fit file.
    #Make a plot where the y axis is energy resolutuion and the x axis is energy.
    #Show the energy resolution over each fit window, and include a shaded 1 sigma uncertainty
    #in energy resolution calculated from the covariance matrix for sigma_c and sigma_m
    root_filepath = fit_save_name if fit_save_name.endswith('.root') else fit_save_name + '.root'
    if not os.path.isabs(root_filepath):
        root_filepath = os.path.join(fit_path, root_filepath)
        
    fitter = spectrum_fitter.load_spectrum_fitter_from_file(root_filepath)
    
    canvas = ROOT.TCanvas(f"c_res_{fit_save_name}", f"Detector Energy Resolution", 800, 600)
    mg = ROOT.TMultiGraph()
    mg.SetTitle("Detector Energy Resolution;Energy (keV);Energy Resolution #sigma (keV)")
    
    graphs = []
    
    for i, res in enumerate(fitter.fit_results):
        if res is None or 'fit_res' not in res:
            continue
            
        fit_res = res['fit_res']
        f_to_fit = res.get('f_to_fit') or res.get('f_to_fit_2d')
        if not f_to_fit:
            continue
            
        idx_c = f_to_fit.GetParNumber("sigma_c")
        idx_m = f_to_fit.GetParNumber("sigma_m")
        
        if idx_c < 0 or idx_m < 0:
            continue
            
        sigma_c = f_to_fit.GetParameter(idx_c)
        sigma_m = f_to_fit.GetParameter(idx_m)
        
        cov_matrix = fit_res.GetCovarianceMatrix()
        if not cov_matrix or cov_matrix.GetNrows() <= max(idx_c, idx_m):
            continue
            
        var_c = cov_matrix(idx_c, idx_c)
        var_m = cov_matrix(idx_m, idx_m)
        cov_cm = cov_matrix(idx_c, idx_m)
        
        window_start = fitter.peaks_to_fit[i][1]
        window_end = fitter.peaks_to_fit[i][2]
        
        n_pts = 100
        e_vals = np.linspace(window_start, window_end, n_pts)
        res_vals = np.zeros(n_pts)
        res_errs = np.zeros(n_pts)
        e_errs = np.zeros(n_pts)
        
        for j, E in enumerate(e_vals):
            sigma = sigma_c + sigma_m * E
            var_sigma = var_c + (E**2)*var_m + 2*E*cov_cm
            
            res_vals[j] = sigma
            res_errs[j] = np.sqrt(max(0, var_sigma))
            
        gr = ROOT.TGraphErrors(n_pts, np.array(e_vals, dtype='float64'), np.array(res_vals, dtype='float64'), np.array(e_errs, dtype='float64'), np.array(res_errs, dtype='float64'))
        
        color = ROOT.kBlue + (i % 4)
        gr.SetLineColor(color)
        gr.SetFillColorAlpha(color, 0.3)
        gr.SetFillStyle(1001)
        
        mg.Add(gr, "3") # shaded band
        
        gr_line = ROOT.TGraph(n_pts, np.array(e_vals, dtype='float64'), np.array(res_vals, dtype='float64'))
        gr_line.SetLineColor(color)
        gr_line.SetLineWidth(2)
        mg.Add(gr_line, "L")
        
        graphs.extend([gr, gr_line])
        
    if len(graphs) > 0:
        mg.Draw("A")
        canvas.Update()
        
    ROOT.SetOwnership(canvas, False)
    ROOT.SetOwnership(mg, False)
    for gr in graphs:
        ROOT.SetOwnership(gr, False)
        
    return canvas, mg, graphs

ecal_simul = make_energy_calibration(f_proton_simultaneous, '60Ga_59Zn_simultaneous_protons', 'proton_peaks.csv', show_fit_result=True, force_0_offset=False)
apply_fit_to_csv(ecal_simul, '60Ga_59Zn_simultaneous_protons', 'proton_cal')
print(apply_fit_to_point(ecal_simul, 8522.04, 9.35))
show_detector_energy_resolution('60Ga_59Zn_simultaneous_protons')