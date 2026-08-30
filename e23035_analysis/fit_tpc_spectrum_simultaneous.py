import os
import csv
from pathlib import Path

import ROOT
import numpy as np

from raw_viewer import ddas_interface, process_runs, degai
from e23035_analysis import e23035_runs, fitting_tools, spectrum_fitter, root_vis_tools

fit_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tpc_spectrum_fitting')
def load_peaks_from_csv(filename):
    all_peaks = []
    all_isotopes = []
    with open(os.path.join(fit_path, filename), 'r') as f:
        reader = csv.reader(f)
        current_group = []
        current_iso_group = []
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
                        all_isotopes.append(current_iso_group)
                        current_group = []
                        current_iso_group = []
                    start, stop = row[0].split('-')
                    fit_window = (float(start), float(stop))
                current_group.append(float(row[1]))
                current_iso_group.append(row[2].strip() if len(row) > 2 and row[2].strip() else 'unknown')
            except Exception as e:
                print(f"Error parsing row: {row}, error: {e}")
                
        if len(current_group) > 0:
            all_peaks.append((current_group, *fit_window))
            all_isotopes.append(current_iso_group)
            
    return all_peaks, all_isotopes


def get_save_path(save_name):
    return os.path.join(fit_path,save_name)

def fit_multi_peaks(spectra, peaks, save_name, likelihood=True, force_refit=False, additional_param_bounds={}, 
                    loc_wiggle=10, bg_model='linear', bg_order=1, sigma_poly_order=None, sigma_bernstein_order=None, sigma_min=18.0, sigma_max=200.0,
                    sigma_coef_bounds=(-1000, 1000), fraction_bernstein_order=None, peak_isotopes=None):
    root_filepath = save_name if save_name.endswith('.root') else save_name + '.root'
    loaded_from_file = False
    if os.path.exists(root_filepath) and not force_refit:
        print(f"Loading previous multi-spectrum fit from {root_filepath}")
        f = spectrum_fitter.load_spectrum_fitter_from_file(root_filepath)
        loaded_from_file = True
        for p in additional_param_bounds:
            f.param_bound_functions[p] = additional_param_bounds[p]
    else:
        f = spectrum_fitter.multi_spectrum_fitter(spectra, 'bg_shift_gaus', bg_model=bg_model, bg_order=bg_order)
        if sigma_bernstein_order is not None:
            import math
            e_low_global = spectra[0].GetXaxis().GetXmin()
            e_high_global = spectra[0].GetXaxis().GetXmax()
            X_str = f"(({{mu}} - ({e_low_global}))/(({e_high_global}) - ({e_low_global})))"
            
            param_names = [f"sigma_b{i}" for i in range(sigma_bernstein_order + 1)]
            guesses = [20.0] * (sigma_bernstein_order + 1)
            
            terms = []
            n = sigma_bernstein_order
            for k in range(n + 1):
                coef = math.comb(n, k)
                term = f"({coef} * TMath::Power({X_str}, {k}) * TMath::Power(1.0 - {X_str}, {n - k}))"
                terms.append(f"[{param_names[k]}]*{term}")
                
            formula = "(" + " + ".join(terms) + ")"
            
            lower_bound = sigma_min if sigma_min is not None else sigma_coef_bounds[0]
            upper_bound = sigma_max if sigma_max is not None else sigma_coef_bounds[1]
            
            f.parameterizations = {
                'sigma': {
                    'formula': formula,
                    'params': param_names,
                    'guesses': guesses,
                    'bounds': [(lower_bound, upper_bound)] * len(param_names)
                }
            }
        elif sigma_poly_order is not None:
            e_low_global = spectra[0].GetXaxis().GetXmin()
            e_high_global = spectra[0].GetXaxis().GetXmax()
            X_str = f"(2.0*({{mu}} - ({e_low_global}))/(({e_high_global}) - ({e_low_global})) - 1.0)"
            
            param_names = [f"sigma_p{i}" for i in range(sigma_poly_order + 1)]
            
            p0 = 18.8 + 0.01 * 0.5 * (e_high_global + e_low_global)
            p1 = 0.01 * 0.5 * (e_high_global - e_low_global)
            
            guesses = [p0]
            if sigma_poly_order >= 1:
                guesses.append(p1)
            for i in range(2, sigma_poly_order + 1):
                guesses.append(0.0)
                
            if sigma_poly_order == 0:
                formula = f"[{param_names[0]}]"
            elif sigma_poly_order == 1:
                formula = f"([{param_names[0]}] + [{param_names[1]}]*{X_str})"
            else:
                terms = [f"[{param_names[0]}]", f"[{param_names[1]}]*{X_str}"]
                T_n2 = "1.0"
                T_n1 = X_str
                for i in range(2, sigma_poly_order + 1):
                    T_n = f"(2.0*{X_str}*{T_n1} - {T_n2})"
                    terms.append(f"[{param_names[i]}]*{T_n}")
                    T_n2 = T_n1
                    T_n1 = T_n
                formula = "(" + " + ".join(terms) + ")"
                
            if sigma_min is not None and sigma_max is not None:
                formula = f"max((double){sigma_min}, min((double){sigma_max}, (double){formula}))"
            elif sigma_min is not None:
                formula = f"max((double){sigma_min}, (double){formula})"
            elif sigma_max is not None:
                formula = f"min((double){sigma_max}, (double){formula})"
                
            f.parameterizations = {
                'sigma': {
                    'formula': formula,
                    'params': param_names,
                    'guesses': guesses,
                    'bounds': [sigma_coef_bounds] * len(param_names)
                }
            }
        elif True: #linear energy dependence
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
            
        if fraction_bernstein_order is not None and len(spectra) == 2:
            import math
            e_low_global = spectra[0].GetXaxis().GetXmin()
            e_high_global = spectra[0].GetXaxis().GetXmax()
            X_str = f"(({{mu}} - ({e_low_global}))/(({e_high_global}) - ({e_low_global})))"
            
            # Actually, to apply different formulas to different peaks, we must map peak_idx to iso
            # But wait, peak_idx is the local index within a window. If different windows have different isotopes at the same peak_idx, this breaks!
            # Let's check if there is only 1 window.
            if len(peaks) > 1:
                print("WARNING: fraction_bernstein_order with multiple isotopes per peak index across windows is not fully supported in this script. Assuming 1 window or consistent ordering.")
                
            # Let's map each peak_idx to its formula based on the FIRST window's isotopes
            max_peaks = max(len(grp[0]) for grp in peaks) if peaks else 0
            for peak_idx in range(max_peaks):
                iso = peak_isotopes[0][peak_idx] if peak_isotopes and len(peak_isotopes[0]) > peak_idx else 'all'
                iso_suffix = f"_{iso}" if iso != 'all' else ""
                
                if isinstance(fraction_bernstein_order, dict):
                    order = fraction_bernstein_order.get(iso, fraction_bernstein_order.get('default', fraction_bernstein_order.get('all', 1)))
                else:
                    order = fraction_bernstein_order
                
                param_names = [f"amp_frac_b{i}{iso_suffix}" for i in range(order + 1)]
                guesses = [0.5] * (order + 1)
                
                terms = []
                n = order
                for k in range(n + 1):
                    coef = math.comb(n, k)
                    term = f"({coef} * TMath::Power({X_str}, {k}) * TMath::Power(1.0 - {X_str}, {n - k}))"
                    terms.append(f"[{param_names[k]}]*{term}")
                    
                frac_str = "(" + " + ".join(terms) + ")"
                
                f.parameterizations[f'amplitude_{peak_idx}_0'] = {
                    'formula': f"[total_amp_{peak_idx}]*{frac_str}",
                    'params': [f"total_amp_{peak_idx}"] + param_names,
                    'guesses': [200.0] + guesses,
                    'bounds': [(1e-3, 1e6)] + [(0, 1)] * len(param_names)
                }
                f.parameterizations[f'amplitude_{peak_idx}_1'] = {
                    'formula': f"[total_amp_{peak_idx}]*(1.0 - {frac_str})",
                    'params': [f"total_amp_{peak_idx}"] + param_names,
                    'guesses': [200.0] + guesses,
                    'bounds': [(1e-3, 1e6)] + [(0, 1)] * len(param_names)
                }

        for spec in f.spectra:
            spec.GetXaxis().UnZoom()
        f.peaks_to_fit = peaks
        f.location_wiggle = loc_wiggle
        f.shared_sigma = False
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
        elif fit_res.CovMatrixStatus() != 3:
            cov_status = fit_res.CovMatrixStatus()
            failed_fits.append((i, f.peaks_to_fit[i], f"Covariance Matrix Status {cov_status} (expected 3)"))
            
    if failed_fits:
        print(f"Summary of {len(failed_fits)} failed fits:")
        for i, peak_info, reason in failed_fits:
            print(f"  Index {i}: Peaks {peak_info[0]} in window ({peak_info[1]:.1f}, {peak_info[2]:.1f}) -> {reason}")
    else:
        print(f"All {len(f.fit_results)} fits successful.")
        
    for i, res in enumerate(f.fit_results):
        if res is None or 'fit_res' not in res:
            continue
        
        f_to_fit = res.get('f_to_fit') or res.get('f_to_fit_2d')
        if not f_to_fit:
            continue
            
        for j in range(f_to_fit.GetNpar()):
            val = f_to_fit.GetParameter(j)
            name = f_to_fit.GetParName(j)
            
            try:
                import ctypes
                low = ctypes.c_double(0)
                high = ctypes.c_double(0)
                f_to_fit.GetParLimits(j, low, high)
                low_val, high_val = low.value, high.value
            except TypeError:
                low = ROOT.Double(0)
                high = ROOT.Double(0)
                f_to_fit.GetParLimits(j, low, high)
                low_val, high_val = float(low), float(high)
                
            if low_val < high_val:
                range_width = high_val - low_val
                if abs(val - low_val) < 1e-4 * range_width or abs(high_val - val) < 1e-4 * range_width:
                    print(f"Warning (Fit index {i}): Parameter '{name}' is pinned at limit {val:.4g} (bounds: [{low_val:.4g}, {high_val:.4g}])")
        
    if not loaded_from_file and save_name:
        f.save(save_name) 
        
        try:
            import csv
            import math
            import re
            
            eval_csv_path = save_name + "_evaluated.csv"
            if eval_csv_path.endswith('.root_evaluated.csv'):
                eval_csv_path = eval_csv_path.replace('.root_evaluated.csv', '_evaluated.csv')
                
            with open(eval_csv_path, 'w', newline='') as f_csv:
                writer = csv.writer(f_csv)
                header = ['window_idx', 'peak_idx', 'mu', 'mu_err', 'sigma', 'sigma_err']
                if len(spectra) == 2:
                    header.extend(['total_amp', 'total_amp_err', 'amplitude_0', 'amplitude_0_err', 'amplitude_1', 'amplitude_1_err', 'amplitude_fraction_0'])
                else:
                    header.extend(['amplitude', 'amplitude_err'])
                writer.writerow(header)
                
                e_low_global = spectra[0].GetXaxis().GetXmin()
                e_high_global = spectra[0].GetXaxis().GetXmax()
                
                for i, res in enumerate(f.fit_results):
                    if res is None or 'fit_res' not in res: continue
                    f_to_fit = res.get('f_to_fit_2d') or res.get('f_to_fit')
                    if not f_to_fit: continue
                    
                    mu_params = {}
                    for j in range(f_to_fit.GetNpar()):
                        name = f_to_fit.GetParName(j)
                        if name.startswith('mu'):
                            idx = 0 if name == 'mu' else int(name.split('_')[1])
                            mu_params[idx] = (f_to_fit.GetParameter(j), f_to_fit.GetParError(j))
                            
                    for idx, (mu_val, mu_err) in mu_params.items():
                        row = [str(i), str(idx), f"{mu_val:.6g}", f"{mu_err:.6g}"]
                        
                        # Evaluate sigma
                        sigma = 0
                        sigma_err = 0
                        if hasattr(f, 'parameterizations') and 'sigma' in f.parameterizations:
                            param_names = f.parameterizations['sigma']['params']
                            formula_str = f.parameterizations['sigma']['formula']
                            p_indices = [f_to_fit.GetParNumber(n) for n in param_names]
                            if not any(pi < 0 for pi in p_indices):
                                p_vals = [f_to_fit.GetParameter(pi) for pi in p_indices]
                                E = mu_val
                                import numpy as np
                                grad = np.zeros(len(param_names))
                                if param_names == ['sigma_c', 'sigma_m']:
                                    if 'sqrt' in formula_str:
                                        inner = p_vals[0] + p_vals[1] * E
                                        sigma = np.sqrt(inner) if inner > 0 else 0
                                        if inner > 0: grad = np.array([1.0 / (2*sigma), E / (2*sigma)])
                                    else:
                                        sigma = p_vals[0] + p_vals[1] * E
                                        grad = np.array([1.0, E])
                                elif param_names[0].startswith('sigma_p'):
                                    X = 2.0 * (E - e_low_global) / (e_high_global - e_low_global) - 1.0
                                    if len(param_names) > 0: grad[0] = 1.0
                                    if len(param_names) > 1: grad[1] = X
                                    for k in range(2, len(param_names)):
                                        grad[k] = 2.0 * X * grad[k-1] - grad[k-2]
                                    sigma = np.dot(p_vals, grad)
                                elif param_names[0].startswith('sigma_b'):
                                    import math
                                    X = (E - e_low_global) / (e_high_global - e_low_global)
                                    n = len(param_names) - 1
                                    for k in range(n + 1):
                                        grad[k] = math.comb(n, k) * (X**k) * ((1.0 - X)**(n - k))
                                    sigma = np.dot(p_vals, grad)
                                    
                                cov_matrix = res['fit_res'].GetCovarianceMatrix()
                                if cov_matrix and cov_matrix.GetNrows() > max(p_indices):
                                    cov_sub = np.zeros((len(param_names), len(param_names)))
                                    for r in range(len(param_names)):
                                        for c in range(len(param_names)):
                                            cov_sub[r,c] = cov_matrix(p_indices[r], p_indices[c])
                                    var_sigma = grad.T @ cov_sub @ grad
                                    sigma_err = np.sqrt(max(0, var_sigma))
                                    
                                import re
                                if "max" in formula_str:
                                    match = re.search(r'max\(\(double\)([\d.]+),', formula_str)
                                    if match: sigma = max(float(match.group(1)), sigma)
                                if "min" in formula_str:
                                    match = re.search(r'min\(\(double\)([\d.]+),', formula_str)
                                    if match: sigma = min(float(match.group(1)), sigma)
                        else:
                            sig_idx = f_to_fit.GetParNumber(f'sigma_{idx}' if f"sigma_{idx}" in [f_to_fit.GetParName(k) for k in range(f_to_fit.GetNpar())] else 'sigma')
                            if sig_idx >= 0:
                                sigma = f_to_fit.GetParameter(sig_idx)
                                sigma_err = f_to_fit.GetParError(sig_idx)
                                
                        row.extend([f"{sigma:.6g}", f"{sigma_err:.6g}"])
                        
                        # Evaluate amplitudes
                        if len(spectra) == 2:
                            tot_idx = f_to_fit.GetParNumber(f"total_amp_{idx}")
                            if tot_idx >= 0:
                                tot_amp = f_to_fit.GetParameter(tot_idx)
                                tot_amp_err = f_to_fit.GetParError(tot_idx)
                                iso = peak_isotopes[0][idx] if peak_isotopes and len(peak_isotopes[0]) > idx else 'all'
                                iso_suffix = f"_{iso}" if iso != 'all' else ""
                                
                                if isinstance(fraction_bernstein_order, dict):
                                    order = fraction_bernstein_order.get(iso, fraction_bernstein_order.get('default', fraction_bernstein_order.get('all', 1)))
                                else:
                                    order = fraction_bernstein_order
                                    
                                frac_param_names = [f"amp_frac_b{k}{iso_suffix}" for k in range(order + 1)] if order is not None else []
                                p_indices = [f_to_fit.GetParNumber(n) for n in frac_param_names]
                                
                                if len(p_indices) > 0 and not any(pi < 0 for pi in p_indices):
                                    p_vals = [f_to_fit.GetParameter(pi) for pi in p_indices]
                                    import math
                                    X = (mu_val - e_low_global) / (e_high_global - e_low_global)
                                    n = len(frac_param_names) - 1
                                    frac = 0
                                    frac_grad = np.zeros(len(frac_param_names))
                                    for k in range(n + 1):
                                        basis = math.comb(n, k) * (X**k) * ((1.0 - X)**(n - k))
                                        frac += p_vals[k] * basis
                                        frac_grad[k] = basis
                                        
                                    a0 = tot_amp * frac
                                    a1 = tot_amp * (1.0 - frac)
                                    
                                    grad_a0 = np.zeros(1 + len(frac_param_names))
                                    grad_a0[0] = frac
                                    grad_a0[1:] = tot_amp * frac_grad
                                    
                                    grad_a1 = np.zeros(1 + len(frac_param_names))
                                    grad_a1[0] = (1.0 - frac)
                                    grad_a1[1:] = -tot_amp * frac_grad
                                    
                                    all_indices = [tot_idx] + p_indices
                                    cov_matrix = res['fit_res'].GetCovarianceMatrix()
                                    a0_err, a1_err = 0, 0
                                    if cov_matrix and cov_matrix.GetNrows() > max(all_indices):
                                        cov_sub = np.zeros((len(all_indices), len(all_indices)))
                                        for r in range(len(all_indices)):
                                            for c in range(len(all_indices)):
                                                cov_sub[r,c] = cov_matrix(all_indices[r], all_indices[c])
                                        var_a0 = grad_a0.T @ cov_sub @ grad_a0
                                        var_a1 = grad_a1.T @ cov_sub @ grad_a1
                                        a0_err = np.sqrt(max(0, var_a0))
                                        a1_err = np.sqrt(max(0, var_a1))
                                        
                                    row.extend([f"{tot_amp:.6g}", f"{tot_amp_err:.6g}", f"{a0:.6g}", f"{a0_err:.6g}", f"{a1:.6g}", f"{a1_err:.6g}", f"{frac:.6g}"])
                                else:
                                    row.extend([f"{tot_amp:.6g}", f"{tot_amp_err:.6g}", "", "", "", "", ""])
                            else:
                                amp0_idx = f_to_fit.GetParNumber(f"amplitude_{idx}_0")
                                amp1_idx = f_to_fit.GetParNumber(f"amplitude_{idx}_1")
                                if amp0_idx >= 0 and amp1_idx >= 0:
                                    a0 = f_to_fit.GetParameter(amp0_idx)
                                    a0_err = f_to_fit.GetParError(amp0_idx)
                                    a1 = f_to_fit.GetParameter(amp1_idx)
                                    a1_err = f_to_fit.GetParError(amp1_idx)
                                    tot = a0 + a1
                                    
                                    grad_tot = np.array([1.0, 1.0])
                                    cov_matrix = res['fit_res'].GetCovarianceMatrix()
                                    tot_err = 0
                                    if cov_matrix and cov_matrix.GetNrows() > max(amp0_idx, amp1_idx):
                                        cov_sub = np.array([[cov_matrix(amp0_idx, amp0_idx), cov_matrix(amp0_idx, amp1_idx)],
                                                            [cov_matrix(amp1_idx, amp0_idx), cov_matrix(amp1_idx, amp1_idx)]])
                                        tot_err = np.sqrt(max(0, grad_tot.T @ cov_sub @ grad_tot))
                                        
                                    frac = a0 / tot if tot > 0 else 0
                                    row.extend([f"{tot:.6g}", f"{tot_err:.6g}", f"{a0:.6g}", f"{a0_err:.6g}", f"{a1:.6g}", f"{a1_err:.6g}", f"{frac:.6g}"])
                                else:
                                    row.extend(["", "", "", "", "", "", ""])
                        else:
                            amp_idx = f_to_fit.GetParNumber(f"amplitude_{idx}")
                            if amp_idx >= 0:
                                row.extend([f"{f_to_fit.GetParameter(amp_idx):.6g}", f"{f_to_fit.GetParError(amp_idx):.6g}"])
                            else:
                                row.extend(["", ""])
                                
                        writer.writerow(row)
        except Exception as e:
            print(f"Failed to generate evaluated csv: {e}")
            
    return f

def make_merged_fit(source_fitter, save_name, force_refit=False, fit_windows_to_include=None, bg_model='chebyshev', bg_order=4, sigma_poly_order=None, sigma_min=18.0, sigma_max=200.0, sigma_coef_bounds=(-1000, 1000), loc_wiggle=10, additional_peaks=None):
    """
    Creates a merged fit from multiple limited-window fits in the source_fitter.
    
    Args:
        source_fitter: The multi_spectrum_fitter instance containing the limited window fits.
        save_name: Base path to save the merged fit ROOT file.
        force_refit: Whether to force refitting if the file already exists.
        fit_windows_to_include: List of indices for the fit windows to merge. Defaults to all.
        bg_model: The background model to use for the merged fit.
        bg_order: The polynomial order for the background.
        sigma_poly_order: Polynomial order for sigma.
        sigma_min: Minimum value for sigma.
        sigma_max: Maximum value for sigma.
        sigma_coef_bounds: Bounds for sigma polynomial coefficients.
        loc_wiggle: The window wiggle range for parameter peak locations.
    """
    merged_peaks = []
    global_start = float('inf')
    global_end = float('-inf')

    fitted_mus = {}
    fitted_amps = {}

    if fit_windows_to_include is None:
        fit_windows_to_include = list(range(len(source_fitter.peaks_to_fit)))

    # Extract parameters from the previous limited window fits
    for i in fit_windows_to_include:
        res = source_fitter.fit_results[i]
        if not res or 'fit_res' not in res:
            continue
        f_to_fit = res.get('f_to_fit_2d') or res.get('f_to_fit')
        if not f_to_fit:
            continue
            
        for j in range(f_to_fit.GetNpar()):
            name = f_to_fit.GetParName(j)
            val = f_to_fit.GetParameter(j)
            if name.startswith('mu'):
                idx = 0 if name == 'mu' else int(name.split('_')[1])
                loc_guess = source_fitter.peaks_to_fit[i][0][idx]
                fitted_mus[loc_guess] = val
            elif name.startswith('amplitude'):
                parts = name.split('_')
                if len(parts) == 3: # amplitude_i_j
                    peak_idx = int(parts[1])
                    spec_idx = int(parts[2])
                else: 
                    peak_idx = 0 if len(parts) == 2 else int(parts[1])
                    spec_idx = int(parts[-1])
                
                loc_guess = source_fitter.peaks_to_fit[i][0][peak_idx]
                if loc_guess not in fitted_amps:
                    fitted_amps[loc_guess] = {}
                fitted_amps[loc_guess][spec_idx] = val

        peaks, w_start, w_end = source_fitter.peaks_to_fit[i]
        global_start = min(global_start, w_start)
        global_end = max(global_end, w_end)
        for p in peaks:
            if p not in merged_peaks:
                merged_peaks.append(p)
                
    if additional_peaks:
        for p in additional_peaks:
            if p not in merged_peaks:
                merged_peaks.append(p)
            global_start = min(global_start, p - 100)
            global_end = max(global_end, p + 100)
                
    merged_peaks.sort()
    merged_proton_guesses = [(merged_peaks, global_start, global_end)]

    # Inherit parameter bounds from the source fitter (like bg_shift, bg_slope)
    merged_param_bounds = {}
    for k, v in source_fitter.param_bound_functions.items():
        if not k.startswith('mu') and not k.startswith('amplitude') and callable(v):
            merged_param_bounds[k] = v

    # Add back the initial guesses we extracted
    for i, p in enumerate(merged_peaks):
        if p in fitted_mus:
            merged_param_bounds[f'mu_{i}'] = lambda E, p=p, val=fitted_mus[p], w=loc_wiggle: (val, p - w, p + w)
        if p in fitted_amps:
            for spec_idx, amp_val in fitted_amps[p].items():
                merged_param_bounds[f'amplitude_{i}_{spec_idx}'] = lambda E, val=amp_val: (val, 1e-3, 1e6)

    return fit_multi_peaks(
        source_fitter.spectra, 
        merged_proton_guesses,
        save_name, force_refit=force_refit,
        additional_param_bounds=merged_param_bounds, 
        loc_wiggle=loc_wiggle,
        bg_model=bg_model,
        bg_order=bg_order,
        sigma_poly_order=sigma_poly_order,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sigma_coef_bounds=sigma_coef_bounds
    )

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
        
        equation_text = ROOT.TLatex()
        equation_text.SetNDC()
        equation_text.SetTextSize(0.04)
        equation_text.SetTextAlign(13)
        slope_err = np.sqrt(cov[1,1]) if cov[1,1] > 0 else 0.0
        offset_err = np.sqrt(cov[0,0]) if cov[0,0] > 0 else 0.0
        equation_str = f"E = ({slope:.4g} #pm {slope_err:.4g}) #mu + ({offset:.4g} #pm {offset_err:.4g})"
        equation_text.DrawLatex(0.15, 0.85, equation_str)
        ROOT.SetOwnership(equation_text, False)
        
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

def show_detector_energy_resolution(fitter_or_filename):
    #extract detector energy resolution each of the fit windows in the specified fit file.
    #Make a plot where the y axis is energy resolutuion and the x axis is energy.
    #Show the energy resolution over each fit window, and include a shaded 1 sigma uncertainty
    #in energy resolution calculated from the covariance matrix for sigma_c and sigma_m
    if isinstance(fitter_or_filename, str):
        root_filepath = fitter_or_filename if fitter_or_filename.endswith('.root') else fitter_or_filename + '.root'
        if not os.path.isabs(root_filepath):
            root_filepath = os.path.join(fit_path, root_filepath)
        fitter = spectrum_fitter.load_spectrum_fitter_from_file(root_filepath)
        name = fitter_or_filename
    else:
        fitter = fitter_or_filename
        name = "simultaneous"
        
    canvas = ROOT.TCanvas(f"c_res_{name}", f"Detector Energy Resolution", 800, 600)
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
            
        if fitter.parameterizations and 'sigma' in fitter.parameterizations:
            param_names = fitter.parameterizations['sigma']['params']
            formula_str = fitter.parameterizations['sigma']['formula']
        else:
            param_names = ['sigma_c', 'sigma_m']
            formula_str = ""
            
        p_indices = [f_to_fit.GetParNumber(n) for n in param_names]
        if any(idx < 0 for idx in p_indices):
            continue
            
        p_vals = [f_to_fit.GetParameter(idx) for idx in p_indices]
            
        cov_matrix = fit_res.GetCovarianceMatrix()
        if not cov_matrix or cov_matrix.GetNrows() <= max(p_indices):
            continue
            
        cov_sub = np.zeros((len(param_names), len(param_names)))
        for r in range(len(param_names)):
            for c in range(len(param_names)):
                cov_sub[r,c] = cov_matrix(p_indices[r], p_indices[c])
                
        e_low_global = fitter.spectra[0].GetXaxis().GetXmin()
        e_high_global = fitter.spectra[0].GetXaxis().GetXmax()
        
        window_start = fitter.peaks_to_fit[i][1]
        window_end = fitter.peaks_to_fit[i][2]
        
        n_pts = 100
        e_vals = np.linspace(window_start, window_end, n_pts)
        res_vals = np.zeros(n_pts)
        res_errs = np.zeros(n_pts)
        e_errs = np.zeros(n_pts)
        
        for j, E in enumerate(e_vals):
            if param_names == ['sigma_c', 'sigma_m']:
                if 'sqrt' in formula_str:
                    inner = p_vals[0] + p_vals[1] * E
                    sigma = np.sqrt(inner) if inner > 0 else 0
                    if inner > 0:
                        grad = np.array([1.0 / (2*sigma), E / (2*sigma)])
                    else:
                        grad = np.array([0.0, 0.0])
                else:
                    grad = np.array([1.0, E])
                    sigma = p_vals[0] + p_vals[1] * E
            elif param_names[0].startswith('sigma_p'):
                X = 2.0 * (E - e_low_global) / (e_high_global - e_low_global) - 1.0
                grad = np.zeros(len(param_names))
                if len(param_names) > 0: grad[0] = 1.0
                if len(param_names) > 1: grad[1] = X
                for k in range(2, len(param_names)):
                    grad[k] = 2.0 * X * grad[k-1] - grad[k-2]
                sigma = np.dot(p_vals, grad)
                
                # Apply bounds if present in the formula
                import re
                if "max" in formula_str:
                    match = re.search(r'max\(\(double\)([\d.]+),', formula_str)
                    if match: sigma = max(float(match.group(1)), sigma)
                if "min" in formula_str:
                    match = re.search(r'min\(\(double\)([\d.]+),', formula_str)
                    if match: sigma = min(float(match.group(1)), sigma)
            elif param_names[0].startswith('sigma_b'):
                import math
                X = (E - e_low_global) / (e_high_global - e_low_global)
                n = len(param_names) - 1
                grad = np.zeros(len(param_names))
                for k in range(n + 1):
                    grad[k] = math.comb(n, k) * (X**k) * ((1.0 - X)**(n - k))
                sigma = np.dot(p_vals, grad)
                
                # Apply bounds if present in the formula
                import re
                if "max" in formula_str:
                    match = re.search(r'max\(\(double\)([\d.]+),', formula_str)
                    if match: sigma = max(float(match.group(1)), sigma)
                if "min" in formula_str:
                    match = re.search(r'min\(\(double\)([\d.]+),', formula_str)
                    if match: sigma = min(float(match.group(1)), sigma)
            else:
                continue
                
            var_sigma = grad.T @ cov_sub @ grad
            
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

####################################################################
experiment = 'e23035'
tpc_config = 'smart2_rpr.csv'
num_workers = 200

# efficiencies with 0.100000 s implant time and 0.100000 s decay time
# Assumes 12 ms dead time at start of decay window + 2 ms at end
# These efficiencies are defined in terms of fractions of implanted nuclie which decay during the measurement window
# 61Ge efficiency =  0.3230255772737927
Zn59_cycle_efficiency =  0.41616841590773374
Ga60_cycle_efficiency =  0.37410064021102757

proton_binning = (4000//5, 0, 4000)
ddas_runs_protons_59Zn = e23035_runs.get_ddas_59_Zn_runs(good_gamma=False, final_beam_settings=True, good_low_energy_tpc=True, good_long_tracks_tpc=True)
pspec_59Zn = ddas_interface.get_histogram(experiment, ddas_runs_protons_59Zn, proton_binning, "proton_spectrum_59Zn", "59Zn proton_spectrum", "tpc_energy", "tpc_particle_id==1", num_workers=num_workers, tpc_ini_filename=tpc_config)

#############################################################################
# Fit including runs where high energy protons may not be recorded correctly.
#############################################################################
force_refit=True
ddas_runs_protons_low_energies_60Ga = e23035_runs.get_ddas_60_Ga_runs(good_gamma=False, final_beam_settings=True, good_low_energy_tpc=True, good_long_tracks_tpc=False)
pspec_low_energy_60Ga = ddas_interface.get_histogram(experiment, ddas_runs_protons_low_energies_60Ga, proton_binning, "proton_spectrum_low_energy_60Ga", "60Ga proton_spectrum low energy", "tpc_energy", "tpc_particle_id==1", num_workers=num_workers, tpc_ini_filename=tpc_config)

proton_peak_guesses, peak_isotopes = load_peaks_from_csv('proton_peaks.csv')

save_path_low = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tpc_spectrum_fitting', '60Ga_59Zn_simultaneous_protons_low_energy')
bg_shift_upper_bound = 0
f_proton_simultaneous_low = fit_multi_peaks(
    [pspec_low_energy_60Ga, pspec_59Zn], 
    proton_peak_guesses,
    save_path_low, force_refit=force_refit,
    additional_param_bounds={'total_amp': lambda E:(1e-3, 1e6),
                            'bg_shift': lambda E: (0, bg_shift_upper_bound)}, 
    loc_wiggle=10,
    bg_model='chebyshev',
    bg_order=4,
    fraction_bernstein_order={'61Ge': 1, 'default': 4},
    sigma_bernstein_order=4,
    sigma_min=10,
    sigma_max=200,
    peak_isotopes=peak_isotopes
)
ROOT.gROOT.SetBatch(False)
f_proton_simultaneous_low.show_fit_result(0, False, True)
ecal_simul_low = make_energy_calibration(f_proton_simultaneous_low, '60Ga_59Zn_simultaneous_protons_low_energy', 'proton_peaks.csv', show_fit_result=True, force_0_offset=False)
apply_fit_to_csv(ecal_simul_low, '60Ga_59Zn_simultaneous_protons_low_energy', 'proton_cal_low_energy')
print(apply_fit_to_point(ecal_simul_low, 8522.04, 9.35))
show_detector_energy_resolution(f_proton_simultaneous_low)

# save_path_merged_low = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tpc_spectrum_fitting', '60Ga_59Zn_simultaneous_protons_low_energy_merged_cheb')
# f_proton_simultaneous_merged_low = make_merged_fit(
#     source_fitter=f_proton_simultaneous_low,
#     save_name=save_path_merged_low,
#     force_refit=force_refit,
#     fit_windows_to_include=None,
#     bg_model='chebyshev',
#     bg_order=4,
#     sigma_poly_order=2,
#     sigma_min=10.0,
#     sigma_max=200.0,
#     sigma_coef_bounds=(-1000, 1000),
#     loc_wiggle=15
# )
# ROOT.gROOT.SetBatch(False)
# ecal_simul_low = make_energy_calibration(f_proton_simultaneous_merged_low, '60Ga_59Zn_simultaneous_protons_low_energy_merged_cheb', 'proton_peaks.csv', show_fit_result=True, force_0_offset=False)
# apply_fit_to_csv(ecal_simul_low, '60Ga_59Zn_simultaneous_protons_low_energy_merged_cheb', 'proton_cal_low_energy')
# print(apply_fit_to_point(ecal_simul_low, 8522.04, 9.35))
# show_detector_energy_resolution(f_proton_simultaneous_merged_low)

if False:
    #############################################################
    # Fit using just runs where all proton energies are valid 1 #
    #############################################################
    ddas_runs_protons_all_energies_60Ga = e23035_runs.get_ddas_60_Ga_runs(good_gamma=False, final_beam_settings=True, good_low_energy_tpc=True, good_long_tracks_tpc=True)
    pspec_all_energy_60Ga = ddas_interface.get_histogram(experiment, ddas_runs_protons_all_energies_60Ga, proton_binning, "proton_spectrum_60Ga", "60Ga proton_spectrum", "tpc_energy", "tpc_particle_id==1", num_workers=num_workers, tpc_ini_filename=tpc_config)


    zn_ga_comparison_overlay = root_vis_tools.draw_overlaid_histograms({'60Ga':pspec_all_energy_60Ga, '59Zn':pspec_59Zn}, 'proton spectra')
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tpc_spectrum_fitting', '60Ga_59Zn_simultaneous_protons')
    #no more than 50% of events should be wall effect below 2 MeV b/c range <<200 mm
    #w/ 5 keV bins, this would put 50%/(2000 keV / (5 keV/bin)) counts per bin wall effect
    #let's let it go up to 2X this in case my estimate is off
    proton_peak_guesses, peak_isotopes = load_peaks_from_csv('proton_peaks.csv')
    bg_shift_upper_bound = 0#2*0.5/(2000/5) 
    force_refit=False
    f_proton_simultaneous = fit_multi_peaks(
        [pspec_all_energy_60Ga, pspec_59Zn], 
        proton_peak_guesses,
        save_path, force_refit=force_refit,
        additional_param_bounds={'bg_slope':lambda E: (-1,1), #if E < 1000 else (0,0),
                                'amplitude': lambda E:(1e-3, 1e6),
                                'bg_shift': lambda E: (0, bg_shift_upper_bound),
                                'sigma_c': lambda E: (0, 10) if E < 1000 else (0, 100)}, 
        loc_wiggle=15,
        peak_isotopes=peak_isotopes
    )

    save_path_merged = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tpc_spectrum_fitting', '60Ga_59Zn_simultaneous_protons_merged_cheb')
    f_proton_simultaneous_merged = make_merged_fit(
        source_fitter=f_proton_simultaneous,
        save_name=save_path_merged,
        force_refit=force_refit,
        fit_windows_to_include=None, # defaults to all
        bg_model='chebyshev',
        bg_order=4,
        sigma_poly_order=2, # You can change this order as needed
        sigma_min=10.0,
        sigma_max=200.0,
        sigma_coef_bounds=(-1000, 1000),
        loc_wiggle=15
    )
    # Display multi-spectrum fit for the first peak
    #f_proton_simultaneous.show_fit_results(4, False, True)
    ROOT.gROOT.SetBatch(False)

    ecal_simul = make_energy_calibration(f_proton_simultaneous_merged, '60Ga_59Zn_simultaneous_protons_merged_cheb', 'proton_peaks.csv', show_fit_result=True, force_0_offset=False)
    apply_fit_to_csv(ecal_simul, '60Ga_59Zn_simultaneous_protons_merged_cheb', 'proton_cal')
    print(apply_fit_to_point(ecal_simul, 8522.04, 9.35))
    show_detector_energy_resolution(f_proton_simultaneous_merged)

    #############################################################################
    # Fit using all datasets over the entire fit range (3 datasets)
    #############################################################################
    save_path_all = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tpc_spectrum_fitting', '60Ga_59Zn_simultaneous_protons_all_3')

    force_refit = False
    f_proton_simultaneous_all = fit_multi_peaks(
        [pspec_all_energy_60Ga, pspec_low_energy_60Ga, pspec_59Zn], 
        proton_peak_guesses,
        save_path_all, force_refit=force_refit,
        additional_param_bounds={'bg_slope':lambda E: (-1,1),
                                'amplitude': lambda E:(1e-3, 1e6),
                                'bg_shift': lambda E: (0, bg_shift_upper_bound),
                                'sigma_c': lambda E: (0, 10) if E < 1000 else (0, 100)}, 
        loc_wiggle=15,
        peak_isotopes=peak_isotopes
    )

    save_path_merged_all = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tpc_spectrum_fitting', '60Ga_59Zn_simultaneous_protons_all_3_merged_cheb')
    f_proton_simultaneous_merged_all = make_merged_fit(
        source_fitter=f_proton_simultaneous_all,
        save_name=save_path_merged_all,
        force_refit=force_refit,
        fit_windows_to_include=None,
        bg_model='chebyshev',
        bg_order=4,
        sigma_poly_order=2,
        sigma_min=10.0,
        sigma_max=200.0,
        sigma_coef_bounds=(-1000, 1000),
        loc_wiggle=15
    )

    additional_peaks = [1000,1950]

    save_path_merged_all_additional = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tpc_spectrum_fitting', '60Ga_59Zn_simultaneous_protons_all_3_merged_cheb_additional')
    f_proton_simultaneous_merged_all_additional = make_merged_fit(
        source_fitter=f_proton_simultaneous_merged_all,
        save_name=save_path_merged_all_additional,
        force_refit=force_refit,
        fit_windows_to_include=None,
        bg_model='chebyshev',
        bg_order=4,
        sigma_poly_order=2,
        sigma_min=10.0,
        sigma_max=200.0,
        sigma_coef_bounds=(-1000, 1000),
        loc_wiggle=15,
        additional_peaks=additional_peaks
    )

    ecal_simul_all = make_energy_calibration(f_proton_simultaneous_merged_all_additional, '60Ga_59Zn_simultaneous_protons_all_3_merged_cheb_additional', 'proton_peaks.csv', show_fit_result=True, force_0_offset=False)
    apply_fit_to_csv(ecal_simul_all, '60Ga_59Zn_simultaneous_protons_all_3_merged_cheb_additional', 'proton_cal_all_3')
    print(apply_fit_to_point(ecal_simul_all, 8522.04, 9.35))
    show_detector_energy_resolution(f_proton_simultaneous_merged_all_additional)

