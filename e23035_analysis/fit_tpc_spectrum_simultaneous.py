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
                
        if len(current_group) > 0:
            all_peaks.append((current_group, *fit_window))
            all_isotopes.append(current_iso_group)
            
    return all_peaks, all_isotopes


def get_save_path(save_name):
    return os.path.join(fit_path,save_name)

def find_de_guesses(spectra, fit_window, isotopes_list, save_csv_name, **kwargs):
    """
    Runs Differential Evolution to find optimal starting locations for the peaks,
    then saves the results in the CSV format expected by load_peaks_from_csv.
    """
    window_start, window_end = fit_window
    n_peaks = len(isotopes_list)
    
    # Generate evenly spaced dummy locations across the window
    spacing = (window_end - window_start) / (n_peaks + 1)
    dummy_guesses = [window_start + (i + 1) * spacing for i in range(n_peaks)]
    
    peaks_arg = [(dummy_guesses, window_start, window_end)]
    peak_isotopes = [isotopes_list]
    
    # We pass de_only=True so it stops before the Minuit fit and returns the f_to_fit populated with DE results
    workers = kwargs.pop('workers', 1)
    fs = fit_multi_peaks(spectra, peaks_arg, save_name=save_csv_name.replace('.csv', ''), 
                         peak_isotopes=peak_isotopes, use_de=True, de_only=True, force_refit=True, workers=workers, **kwargs)
    
    # Extract the optimized mu values from the fit results
    # For a single window fit, fs.fit_results[0] holds the results dictionary
    f_to_fit = fs.fit_results[0]['f_to_fit_2d']
    
    csv_path = get_save_path(save_csv_name)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Window', 'Location', 'Isotope'])
        for i in range(n_peaks):
            par_name = f"mu_{i}" if n_peaks > 1 else "mu"
            par_idx = f_to_fit.GetParNumber(par_name)
            mu_val = f_to_fit.GetParameter(par_idx)
            iso = isotopes_list[i]
            if i == 0:
                writer.writerow([f"{window_start}-{window_end}", f"{mu_val:.3f}", iso])
            else:
                writer.writerow(["", f"{mu_val:.3f}", iso])
                
    print(f"DE guesses successfully saved to {csv_path}")
    return csv_path

def fit_multi_peaks(spectra, peaks, save_name, likelihood=True, force_refit=False, additional_param_bounds={}, 
                    loc_wiggle=10, bg_model='linear', bg_order=1, sigma_poly_order=None, sigma_bernstein_order=None, sigma_monotonic_bernstein_order=None, sigma_min=18.0, sigma_max=200.0,
                    sigma_coef_bounds=(-1000, 1000), fraction_bernstein_order=None, bg_shift_bernstein_order=2, bg_shift_monotonic_bernstein_order=None, bg_shift_upper_bound=1.0, peak_isotopes=None,
                    use_de=False, de_only=False, workers=1):
    
    def _pow_str(base, exp):
        if exp == 0: return "1.0"
        if exp == 1: return f"({base})"
        return "(" + "*".join([f"({base})"] * exp) + ")"

    root_filepath = save_name if save_name.endswith('.root') else save_name + '.root'
    loaded_from_file = False
    if os.path.exists(root_filepath) and not force_refit:
        print(f"Loading previous multi-spectrum fit from {root_filepath}")
        f = spectrum_fitter.load_spectrum_fitter_from_file(root_filepath)
        loaded_from_file = True
        for p in additional_param_bounds:
            f.param_bound_functions[p] = additional_param_bounds[p]
    else:
        f = spectrum_fitter.multi_spectrum_fitter(spectra, 'bg_shift_gaus', bg_model=bg_model, bg_order=bg_order, use_de=use_de, de_only=de_only, workers=workers)
        if sigma_monotonic_bernstein_order is not None:
            import math
            e_low_global = min(p[1] for p in peaks) if peaks else spectra[0].GetXaxis().GetXmin()
            e_high_global = max(p[2] for p in peaks) if peaks else spectra[0].GetXaxis().GetXmax()
            X_str = f"(({{mu}} - ({e_low_global}))/(({e_high_global}) - ({e_low_global})))"
            
            N = sigma_monotonic_bernstein_order
            param_names = [f"sigma_mono_p_{k}" for k in range(N + 1)]
            
            lower_bound = float(sigma_min if sigma_min is not None else sigma_coef_bounds[0])
            upper_bound = float(sigma_max if sigma_max is not None else sigma_coef_bounds[1])
            
            c_exprs = []
            prod = f"({upper_bound} - {lower_bound})"
            for k in range(N + 1):
                prod = f"{prod}*(1.0 - [{param_names[k]}])"
                c_exprs.append(f"({upper_bound} - {prod})")
            
            shape_terms = []
            for k in range(N + 1):
                coef = math.comb(N, k)
                basis = f"({coef} * {_pow_str(X_str, k)} * {_pow_str(f'1.0 - {X_str}', N - k)})"
                shape_terms.append(f"({c_exprs[k]}) * {basis}")
                
            formula = "(" + " + ".join(shape_terms) + ")"
            
            guesses = [0.5] * len(param_names)
            bounds = [(0.0, 1.0)] * len(param_names)
            
            f.parameterizations = {
                'sigma': {
                    'formula': formula,
                    'params': param_names,
                    'guesses': guesses,
                    'bounds': bounds
                }
            }
        elif sigma_bernstein_order is not None:
            import math
            e_low_global = min(p[1] for p in peaks) if peaks else spectra[0].GetXaxis().GetXmin()
            e_high_global = max(p[2] for p in peaks) if peaks else spectra[0].GetXaxis().GetXmax()
            X_str = f"(({{mu}} - ({e_low_global}))/(({e_high_global}) - ({e_low_global})))"
            
            param_names = [f"sigma_b{i}" for i in range(sigma_bernstein_order + 1)]
            guesses = [20.0] * (sigma_bernstein_order + 1)
            
            terms = []
            n = sigma_bernstein_order
            for k in range(n + 1):
                coef = math.comb(n, k)
                term = f"({coef} * {_pow_str(X_str, k)} * {_pow_str(f'1.0 - {X_str}', n - k)})"
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
            e_low_global = min(p[1] for p in peaks) if peaks else spectra[0].GetXaxis().GetXmin()
            e_high_global = max(p[2] for p in peaks) if peaks else spectra[0].GetXaxis().GetXmax()
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
            e_low_global = min(p[1] for p in peaks) if peaks else spectra[0].GetXaxis().GetXmin()
            e_high_global = max(p[2] for p in peaks) if peaks else spectra[0].GetXaxis().GetXmax()
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
                    term = f"({coef} * {_pow_str(X_str, k)} * {_pow_str(f'1.0 - {X_str}', n - k)})"
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

        if bg_shift_monotonic_bernstein_order is not None:
            f.shared_bg_shift = False
            import math
            e_low_global = min(p[1] for p in peaks) if peaks else spectra[0].GetXaxis().GetXmin()
            e_high_global = max(p[2] for p in peaks) if peaks else spectra[0].GetXaxis().GetXmax()
            X_str = f"(({{mu}} - ({e_low_global}))/(({e_high_global}) - ({e_low_global})))"
            
            max_peaks = max(len(grp[0]) for grp in peaks) if peaks else 0
            for j in range(len(spectra)):
                for peak_idx in range(max_peaks):
                    iso = peak_isotopes[0][peak_idx] if peak_isotopes and len(peak_isotopes[0]) > peak_idx else 'all'
                    iso_suffix = f"_{iso}" if iso != 'all' else ""
                    
                    if isinstance(bg_shift_monotonic_bernstein_order, dict):
                        order = bg_shift_monotonic_bernstein_order.get(iso, bg_shift_monotonic_bernstein_order.get('default', bg_shift_monotonic_bernstein_order.get('all', 1)))
                    else:
                        order = bg_shift_monotonic_bernstein_order
                        
                    N = order
                    param_names = [f"bg_shift_mono_p_{k}_{j}{iso_suffix}" for k in range(N + 1)]
                    
                    L = 0.0
                    U = float(bg_shift_upper_bound)
                    
                    c_exprs = []
                    prod = f"({U} - {L})"
                    for k in range(N + 1):
                        prod = f"{prod}*(1.0 - [{param_names[k]}])"
                        c_exprs.append(f"({U} - {prod})")
                    
                    shape_terms = []
                    for k in range(N + 1):
                        coef = math.comb(N, k)
                        basis = f"({coef} * {_pow_str(X_str, k)} * {_pow_str(f'1.0 - {X_str}', N - k)})"
                        shape_terms.append(f"({c_exprs[k]}) * {basis}")
                        
                    formula = "(" + " + ".join(shape_terms) + ")"
                    
                    guesses = [0.5] * len(param_names)
                    bounds = [(0.0, 1.0)] * len(param_names)
                    
                    f.parameterizations[f'bg_shift_{peak_idx}_{j}'] = {
                        'formula': formula,
                        'params': param_names,
                        'guesses': guesses,
                        'bounds': bounds
                    }

        elif bg_shift_bernstein_order is not None:
            f.shared_bg_shift = False
            import math
            e_low_global = min(p[1] for p in peaks) if peaks else spectra[0].GetXaxis().GetXmin()
            e_high_global = max(p[2] for p in peaks) if peaks else spectra[0].GetXaxis().GetXmax()
            X_str = f"(({{mu}} - ({e_low_global}))/(({e_high_global}) - ({e_low_global})))"
            
            max_peaks = max(len(grp[0]) for grp in peaks) if peaks else 0
            for j in range(len(spectra)):
                for peak_idx in range(max_peaks):
                    iso = peak_isotopes[0][peak_idx] if peak_isotopes and len(peak_isotopes[0]) > peak_idx else 'all'
                    iso_suffix = f"_{iso}" if iso != 'all' else ""
                    
                    if isinstance(bg_shift_bernstein_order, dict):
                        order = bg_shift_bernstein_order.get(iso, bg_shift_bernstein_order.get('default', bg_shift_bernstein_order.get('all', 1)))
                    else:
                        order = bg_shift_bernstein_order
                    
                    param_names = [f"bg_shift_b{k}_{j}{iso_suffix}" for k in range(order + 1)]
                    guesses = [0.002] * (order + 1)
                    
                    terms = []
                    n = order
                    for k in range(n + 1):
                        coef = math.comb(n, k)
                        term = f"({coef} * {_pow_str(X_str, k)} * {_pow_str(f'1.0 - {X_str}', n - k)})"
                        terms.append(f"[{param_names[k]}]*{term}")
                        
                    formula = "(" + " + ".join(terms) + ")"
                    
                    f.parameterizations[f'bg_shift_{peak_idx}_{j}'] = {
                        'formula': formula,
                        'params': param_names,
                        'guesses': guesses,
                        'bounds': [(0, bg_shift_upper_bound)] * len(param_names)
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
        if de_only:
            return f
    
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
        
        if True: # Removed try/except to prevent silent failures
            import csv
            import math
            import re
            
            eval_csv_path = save_name + "_evaluated.csv"
            if eval_csv_path.endswith('.root_evaluated.csv'):
                eval_csv_path = eval_csv_path.replace('.root_evaluated.csv', '_evaluated.csv')
                
            with open(eval_csv_path, 'w', newline='') as f_csv:
                writer = csv.writer(f_csv)
                header = ['window_idx', 'peak_idx', 'isotope', 'loc_guess', 'mu', 'mu_err', 'sigma', 'sigma_err']
                if len(spectra) == 2:
                    header.extend(['total_amp', 'total_amp_err', 'amplitude_0', 'amplitude_0_err', 'amplitude_1', 'amplitude_1_err', 'amplitude_fraction_0'])
                else:
                    header.extend(['amplitude', 'amplitude_err'])
                writer.writerow(header)
                
                e_low_global = min(p[1] for p in peaks) if peaks else spectra[0].GetXaxis().GetXmin()
                e_high_global = max(p[2] for p in peaks) if peaks else spectra[0].GetXaxis().GetXmax()
                
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
                        iso = peak_isotopes[i][idx] if peak_isotopes and len(peak_isotopes) > i and len(peak_isotopes[i]) > idx else 'unknown'
                        loc_guess = peaks[i][0][idx] if peaks and len(peaks) > i and len(peaks[i][0]) > idx else 0.0
                        row = [str(i), str(idx), iso, f"{loc_guess:.6g}", f"{mu_val:.6g}", f"{mu_err:.6g}"]
                        
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
                                elif param_names[0] == 'sigma_mono_a0' or param_names[0].startswith('sigma_mono_p'):
                                    import math
                                    X = (E - e_low_global) / (e_high_global - e_low_global)
                                    N = len(param_names) - 1
                                    
                                    L_val = float(sigma_min if sigma_min is not None else sigma_coef_bounds[0])
                                    U_val = float(sigma_max if sigma_max is not None else sigma_coef_bounds[1])
                                    
                                    basis_vals = [math.comb(N, k) * (X**k) * ((1.0 - X)**(N - k)) for k in range(N + 1)]
                                    
                                    def eval_sigma(vals):
                                        if param_names[0] == 'sigma_mono_a0':
                                            t_a0, t_a1 = vals[0], vals[1]
                                            t_ps = vals[2:]
                                            t_P_start = L_val + (U_val - L_val) * t_a0
                                            t_P_stop = t_P_start + (U_val - t_P_start) * t_a1
                                            t_c_vals = [0.0]
                                            t_prod = 1.0
                                            for p in t_ps:
                                                t_prod *= (1.0 - p)
                                                t_c_vals.append(1.0 - t_prod)
                                            t_c_vals.append(1.0)
                                            t_shape = sum(t_c_vals[k] * basis_vals[k] for k in range(N + 1))
                                            return t_P_start + (t_P_stop - t_P_start) * t_shape
                                        else:
                                            t_c_vals = []
                                            t_c = L_val + (U_val - L_val) * vals[0]
                                            t_c_vals.append(t_c)
                                            for k in range(1, N + 1):
                                                t_c = t_c + (U_val - t_c) * vals[k]
                                                t_c_vals.append(t_c)
                                            t_shape = sum(t_c_vals[k] * basis_vals[k] for k in range(N + 1))
                                            return t_shape
                                        
                                    sigma = eval_sigma(p_vals)
                                    grad = np.zeros(len(param_names))
                                    eps = 1e-6
                                    for p_idx in range(len(param_names)):
                                        vals_plus = list(p_vals)
                                        vals_minus = list(p_vals)
                                        vals_plus[p_idx] += eps
                                        vals_minus[p_idx] -= eps
                                        grad[p_idx] = (eval_sigma(vals_plus) - eval_sigma(vals_minus)) / (2 * eps)
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
                                iso = peak_isotopes[i][idx] if peak_isotopes and len(peak_isotopes) > i and len(peak_isotopes[i]) > idx else 'all'
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
        # Removed except Exception block to prevent silent failures
            
    f.save_name = save_name
    f.fit_multi_peaks_kwargs = {
        'likelihood': likelihood,
        'bg_model': bg_model,
        'bg_order': bg_order,
        'sigma_poly_order': sigma_poly_order,
        'sigma_bernstein_order': sigma_bernstein_order,
        'sigma_monotonic_bernstein_order': sigma_monotonic_bernstein_order,
        'sigma_min': sigma_min,
        'sigma_max': sigma_max,
        'sigma_coef_bounds': sigma_coef_bounds,
        'fraction_bernstein_order': fraction_bernstein_order,
        'bg_shift_bernstein_order': bg_shift_bernstein_order,
        'bg_shift_monotonic_bernstein_order': bg_shift_monotonic_bernstein_order,
        'peak_isotopes': peak_isotopes
    }
            
    return f

def make_merged_fit(source_fitter, save_name, force_refit=False, fit_windows_to_include=None, bg_model='chebyshev', bg_order=4, sigma_poly_order=None, sigma_bernstein_order=None, sigma_monotonic_bernstein_order=None, sigma_min=18.0, sigma_max=200.0, sigma_coef_bounds=(-1000, 1000), fraction_bernstein_order=None, bg_shift_bernstein_order=None, bg_shift_monotonic_bernstein_order=None, bg_shift_upper_bound=1.0, loc_wiggle=10, additional_peaks=None):
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
        sigma_bernstein_order=sigma_bernstein_order,
        sigma_monotonic_bernstein_order=sigma_monotonic_bernstein_order,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sigma_coef_bounds=sigma_coef_bounds,
        fraction_bernstein_order=fraction_bernstein_order,
        bg_shift_bernstein_order=bg_shift_bernstein_order,
        bg_shift_monotonic_bernstein_order=bg_shift_monotonic_bernstein_order,
        bg_shift_upper_bound=bg_shift_upper_bound
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
                
        e_low_global = min(p[1] for p in fitter.peaks_to_fit) if fitter.peaks_to_fit else fitter.spectra[0].GetXaxis().GetXmin()
        e_high_global = max(p[2] for p in fitter.peaks_to_fit) if fitter.peaks_to_fit else fitter.spectra[0].GetXaxis().GetXmax()
        
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
            elif param_names[0] == 'sigma_mono_a0' or param_names[0].startswith('sigma_mono_p'):
                import math
                X = (E - e_low_global) / (e_high_global - e_low_global)
                N = len(param_names) - 1
                L_val = 18.0
                U_val = 200.0
                import re
                match_old = re.search(r'\(([\d.]+)\s*\+\s*\(([\d.]+)\s*\-\s*[\d.]+\)\*\[sigma_mono_[ap]', formula_str)
                match_new = re.search(r'\(([\d.]+)\s*\-\s*\(\1\s*\-\s*([\d.]+)\)\*\(1\.0\s*\-\s*\[sigma_mono_p', formula_str)
                if match_old:
                    L_val = float(match_old.group(1))
                    U_val = float(match_old.group(2))
                elif match_new:
                    U_val = float(match_new.group(1))
                    L_val = float(match_new.group(2))
                else:
                    raise RuntimeError(f"Failed to parse sigma_mono bounds from formula: {formula_str}")
                basis_vals = [math.comb(N, k) * (X**k) * ((1.0 - X)**(N - k)) for k in range(N + 1)]
                def eval_sigma(vals):
                    if param_names[0] == 'sigma_mono_a0':
                        t_a0, t_a1 = vals[0], vals[1]
                        t_ps = vals[2:]
                        t_P_start = L_val + (U_val - L_val) * t_a0
                        t_P_stop = t_P_start + (U_val - t_P_start) * t_a1
                        t_c_vals = [0.0]
                        t_prod = 1.0
                        for p in t_ps:
                            t_prod *= (1.0 - p)
                            t_c_vals.append(1.0 - t_prod)
                        t_c_vals.append(1.0)
                        t_shape = sum(t_c_vals[k] * basis_vals[k] for k in range(N + 1))
                        return t_P_start + (t_P_stop - t_P_start) * t_shape
                    else:
                        t_c_vals = []
                        t_c = L_val + (U_val - L_val) * vals[0]
                        t_c_vals.append(t_c)
                        for k in range(1, N + 1):
                            t_c = t_c + (U_val - t_c) * vals[k]
                            t_c_vals.append(t_c)
                        t_shape = sum(t_c_vals[k] * basis_vals[k] for k in range(N + 1))
                        return t_shape
                sigma = eval_sigma(p_vals)
                grad = np.zeros(len(param_names))
                eps = 1e-6
                for p_idx in range(len(param_names)):
                    vals_plus = list(p_vals)
                    vals_minus = list(p_vals)
                    vals_plus[p_idx] += eps
                    vals_minus[p_idx] -= eps
                    grad[p_idx] = (eval_sigma(vals_plus) - eval_sigma(vals_minus)) / (2 * eps)
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

def show_peak_fractions(fitter_or_filename):
    if isinstance(fitter_or_filename, str):
        root_filepath = fitter_or_filename if fitter_or_filename.endswith('.root') else fitter_or_filename + '.root'
        if not os.path.isabs(root_filepath):
            root_filepath = os.path.join(fit_path, root_filepath)
        fitter = spectrum_fitter.load_spectrum_fitter_from_file(root_filepath)
        name = fitter_or_filename
    else:
        fitter = fitter_or_filename
        name = "simultaneous"

    canvas = ROOT.TCanvas(f"c_frac_{name}", "Peak Fractions", 800, 600)
    mg = ROOT.TMultiGraph()
    mg.SetTitle("Peak Fraction;Energy (keV);Fraction")
    
    graphs = []
    
    species_colors = {'61Ge': ROOT.kRed, '60Ga': ROOT.kBlue, 'default': ROOT.kBlack, 'all': ROOT.kBlack}
    color_idx = 1
    
    import math
    for i, res in enumerate(fitter.fit_results):
        if res is None or 'fit_res' not in res:
            continue
            
        fit_res = res['fit_res']
        f_to_fit = res.get('f_to_fit_2d') or res.get('f_to_fit')
        if not f_to_fit:
            continue
            
        e_low_global = min(p[1] for p in fitter.peaks_to_fit) if fitter.peaks_to_fit else fitter.spectra[0].GetXaxis().GetXmin()
        e_high_global = max(p[2] for p in fitter.peaks_to_fit) if fitter.peaks_to_fit else fitter.spectra[0].GetXaxis().GetXmax()
        window_start = fitter.peaks_to_fit[i][1]
        window_end = fitter.peaks_to_fit[i][2]
        
        loc_guesses = fitter.peaks_to_fit[i][0]
        num_peaks_in_window = len(loc_guesses) if isinstance(loc_guesses, (list, tuple, np.ndarray)) else 1
        
        plotted_species_in_window = set()
        
        for peak_idx in range(num_peaks_in_window):
            param_key = f'amplitude_{peak_idx}_0'
            if fitter.parameterizations and param_key in fitter.parameterizations:
                param_names = fitter.parameterizations[param_key]['params'][1:]
            else:
                continue 
                
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
            
            species = 'default'
            if '_' in param_names[0]:
                parts = param_names[0].split('_')
                if len(parts) > 2 and parts[-1] not in ['b0', 'b1', 'b2', 'b3', 'b4', 'b5']:
                    species = parts[-1]
            
            if species in plotted_species_in_window:
                continue
            plotted_species_in_window.add(species)
            
            if species not in species_colors:
                species_colors[species] = color_idx
                color_idx += 1
                
            n_pts = 100
            e_vals = np.linspace(window_start, window_end, n_pts)
            frac_vals = np.zeros(n_pts)
            frac_errs = np.zeros(n_pts)
            e_errs = np.zeros(n_pts)
            
            for j, E in enumerate(e_vals):
                X = (E - e_low_global) / (e_high_global - e_low_global)
                n_b = len(param_names) - 1
                grad = np.zeros(len(param_names))
                for k in range(n_b + 1):
                    grad[k] = math.comb(n_b, k) * (X**k) * ((1.0 - X)**(n_b - k))
                frac = np.dot(p_vals, grad)
                err = np.sqrt(max(0, np.dot(grad.T, np.dot(cov_sub, grad))))
                
                frac_vals[j] = frac
                frac_errs[j] = err
                
            gr = ROOT.TGraphErrors(n_pts, np.array(e_vals, dtype='float64'), np.array(frac_vals, dtype='float64'), np.array(e_errs, dtype='float64'), np.array(frac_errs, dtype='float64'))
            color = species_colors[species]
            gr.SetLineColor(color)
            gr.SetFillColorAlpha(color, 0.3)
            gr.SetFillStyle(1001)
            gr.SetTitle(species)
            
            mg.Add(gr, "3") # shaded band
            
            gr_line = ROOT.TGraph(n_pts, np.array(e_vals, dtype='float64'), np.array(frac_vals, dtype='float64'))
            gr_line.SetLineColor(color)
            gr_line.SetLineWidth(2)
            gr_line.SetTitle(species)
            mg.Add(gr_line, "L")
            
            graphs.extend([gr, gr_line])
            
    if len(graphs) > 0:
        mg.Draw("A")
        canvas.Update()
    
    legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    added_species = set()
    for g in graphs:
        if g.GetTitle() not in added_species:
            legend.AddEntry(g, g.GetTitle(), "lf")
            added_species.add(g.GetTitle())
    legend.Draw()
    
    canvas.Update()
    
    ROOT.SetOwnership(canvas, False)
    ROOT.SetOwnership(mg, False)
    ROOT.SetOwnership(legend, False)
    for gr in graphs:
        ROOT.SetOwnership(gr, False)
    
    return canvas, mg, graphs, legend

def show_bg_shifts(fitter_or_filename):
    if isinstance(fitter_or_filename, str):
        root_filepath = fitter_or_filename if fitter_or_filename.endswith('.root') else fitter_or_filename + '.root'
        if not os.path.isabs(root_filepath):
            root_filepath = os.path.join(fit_path, root_filepath)
        fitter = spectrum_fitter.load_spectrum_fitter_from_file(root_filepath)
        name = fitter_or_filename
    else:
        fitter = fitter_or_filename
        name = "simultaneous"

    canvas = ROOT.TCanvas(f"c_bg_shift_{name}", "Background Shifts", 800, 600)
    mg = ROOT.TMultiGraph()
    mg.SetTitle("Background Shift;Energy (keV);Shift")
    
    graphs = []
    
    species_colors = {'61Ge': ROOT.kRed, '60Ga': ROOT.kBlue, 'default': ROOT.kBlack, 'all': ROOT.kBlack}
    color_idx = 1
    
    import math
    for i, res in enumerate(fitter.fit_results):
        if res is None or 'fit_res' not in res:
            continue
            
        fit_res = res['fit_res']
        f_to_fit = res.get('f_to_fit_2d') or res.get('f_to_fit')
        if not f_to_fit:
            continue
            
        e_low_global = min(p[1] for p in fitter.peaks_to_fit) if fitter.peaks_to_fit else fitter.spectra[0].GetXaxis().GetXmin()
        e_high_global = max(p[2] for p in fitter.peaks_to_fit) if fitter.peaks_to_fit else fitter.spectra[0].GetXaxis().GetXmax()
        window_start = fitter.peaks_to_fit[i][1]
        window_end = fitter.peaks_to_fit[i][2]
        
        loc_guesses = fitter.peaks_to_fit[i][0]
        num_peaks_in_window = len(loc_guesses) if isinstance(loc_guesses, (list, tuple, np.ndarray)) else 1
        
        plotted_species_in_window = set()
        
        for peak_idx in range(num_peaks_in_window):
            param_key = f'bg_shift_{peak_idx}_0'
            if fitter.parameterizations and param_key in fitter.parameterizations:
                param_names = fitter.parameterizations[param_key]['params']
                formula_str = fitter.parameterizations[param_key]['formula']
            else:
                param_key = 'bg_shift_0'
                if fitter.parameterizations and param_key in fitter.parameterizations:
                    param_names = fitter.parameterizations[param_key]['params']
                    formula_str = fitter.parameterizations[param_key]['formula']
                else:
                    continue 
                
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
            
            species = 'default'
            if '_' in param_names[0]:
                parts = param_names[0].split('_')
                if len(parts) > 2 and parts[-1] not in ['b0', 'b1', 'b2', 'b3', 'b4', 'b5']:
                    species = parts[-1]
            
            if species in plotted_species_in_window:
                continue
            plotted_species_in_window.add(species)
            
            if species not in species_colors:
                species_colors[species] = color_idx
                color_idx += 1
                
            n_pts = 100
            e_vals = np.linspace(window_start, window_end, n_pts)
            shift_vals = np.zeros(n_pts)
            shift_errs = np.zeros(n_pts)
            e_errs = np.zeros(n_pts)
            
            for j, E in enumerate(e_vals):
                if param_names[0].startswith('bg_shift_mono_a0') or param_names[0].startswith('bg_shift_mono_p'):
                    import math
                    X = (E - e_low_global) / (e_high_global - e_low_global)
                    N = len(param_names) - 1
                    L_val = 0.0
                    U_val = 1.0
                    import re
                    match_old = re.search(r'\(([\d.]+)\s*\+\s*\(([\d.]+)\s*\-\s*[\d.]+\)\*\[bg_shift_mono_[ap]', formula_str)
                    match_new = re.search(r'\(([\d.]+)\s*\-\s*\(\1\s*\-\s*([\d.]+)\)\*\(1\.0\s*\-\s*\[bg_shift_mono_p', formula_str)
                    if match_old:
                        L_val = float(match_old.group(1))
                        U_val = float(match_old.group(2))
                    elif match_new:
                        U_val = float(match_new.group(1))
                        L_val = float(match_new.group(2))
                    else:
                        raise RuntimeError(f"Failed to parse bg_shift_mono bounds from formula: {formula_str}")
                    basis_vals = [math.comb(N, k) * (X**k) * ((1.0 - X)**(N - k)) for k in range(N + 1)]
                    def eval_shift(vals):
                        if param_names[0].startswith('bg_shift_mono_a0'):
                            t_a0, t_a1 = vals[0], vals[1]
                            t_ps = vals[2:]
                            t_P_start = L_val + (U_val - L_val) * t_a0
                            t_P_stop = t_P_start + (U_val - t_P_start) * t_a1
                            t_c_vals = [0.0]
                            t_prod = 1.0
                            for p in t_ps:
                                t_prod *= (1.0 - p)
                                t_c_vals.append(1.0 - t_prod)
                            t_c_vals.append(1.0)
                            t_shape = sum(t_c_vals[k] * basis_vals[k] for k in range(N + 1))
                            return t_P_start + (t_P_stop - t_P_start) * t_shape
                        else:
                            t_c_vals = []
                            t_c = L_val + (U_val - L_val) * vals[0]
                            t_c_vals.append(t_c)
                            for k in range(1, N + 1):
                                t_c = t_c + (U_val - t_c) * vals[k]
                                t_c_vals.append(t_c)
                            t_shape = sum(t_c_vals[k] * basis_vals[k] for k in range(N + 1))
                            return t_shape
                    shift = eval_shift(p_vals)
                    grad = np.zeros(len(param_names))
                    eps = 1e-6
                    for p_idx in range(len(param_names)):
                        vals_plus = list(p_vals)
                        vals_minus = list(p_vals)
                        vals_plus[p_idx] += eps
                        vals_minus[p_idx] -= eps
                        grad[p_idx] = (eval_shift(vals_plus) - eval_shift(vals_minus)) / (2 * eps)
                else:
                    X = (E - e_low_global) / (e_high_global - e_low_global)
                    n_b = len(param_names) - 1
                    grad = np.zeros(len(param_names))
                    for k in range(n_b + 1):
                        grad[k] = math.comb(n_b, k) * (X**k) * ((1.0 - X)**(n_b - k))
                    shift = np.dot(p_vals, grad)
                err = np.sqrt(max(0, np.dot(grad.T, np.dot(cov_sub, grad))))
                
                shift_vals[j] = shift
                shift_errs[j] = err
                
            gr = ROOT.TGraphErrors(n_pts, np.array(e_vals, dtype='float64'), np.array(shift_vals, dtype='float64'), np.array(e_errs, dtype='float64'), np.array(shift_errs, dtype='float64'))
            color = species_colors[species]
            gr.SetLineColor(color)
            gr.SetFillColorAlpha(color, 0.3)
            gr.SetFillStyle(1001)
            gr.SetTitle(species)
            
            mg.Add(gr, "3") # shaded band
            
            gr_line = ROOT.TGraph(n_pts, np.array(e_vals, dtype='float64'), np.array(shift_vals, dtype='float64'))
            gr_line.SetLineColor(color)
            gr_line.SetLineWidth(2)
            gr_line.SetTitle(species)
            mg.Add(gr_line, "L")
            
            graphs.extend([gr, gr_line])
            
    if len(graphs) > 0:
        mg.Draw("A")
        canvas.Update()
    
    legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    added_species = set()
    for g in graphs:
        if g.GetTitle() not in added_species:
            legend.AddEntry(g, g.GetTitle(), "lf")
            added_species.add(g.GetTitle())
    legend.Draw()
    
    canvas.Update()
    
    ROOT.SetOwnership(canvas, False)
    ROOT.SetOwnership(mg, False)
    ROOT.SetOwnership(legend, False)
    for gr in graphs:
        ROOT.SetOwnership(gr, False)
    
    return canvas, mg, graphs, legend

def show_backgrounds(fitter_or_filename):
    if isinstance(fitter_or_filename, str):
        root_filepath = fitter_or_filename if fitter_or_filename.endswith('.root') else fitter_or_filename + '.root'
        if not os.path.isabs(root_filepath):
            root_filepath = os.path.join(fit_path, root_filepath)
        fitter = spectrum_fitter.load_spectrum_fitter_from_file(root_filepath)
        name = fitter_or_filename
    else:
        fitter = fitter_or_filename
        name = "simultaneous"

    canvas = ROOT.TCanvas(f"c_bg_{name}", "Backgrounds", 800, 600)
    mg = ROOT.TMultiGraph()
    mg.SetTitle("Backgrounds;Energy (keV);Counts")
    
    graphs = []
    colors = [ROOT.kBlack, ROOT.kBlue, ROOT.kRed, ROOT.kGreen+2, ROOT.kOrange, ROOT.kMagenta, ROOT.kCyan]
    
    for i, res in enumerate(fitter.fit_results):
        if res is None or 'fit_res' not in res:
            continue
            
        fit_res = res['fit_res']
        f_to_fit = res.get('f_to_fit_2d') or res.get('f_to_fit')
        pm = res.get('pm')
        if not f_to_fit or not pm:
            continue
            
        bg_func_name = getattr(pm, 'bg_func_name', None)
        if not bg_func_name:
            continue
            
        bg_eval_func = getattr(ROOT, bg_func_name, None)
        if not bg_eval_func:
            continue

        window_start = fitter.peaks_to_fit[i][1]
        window_end = fitter.peaks_to_fit[i][2]
        
        n_params = f_to_fit.GetNpar()
        params = np.array([f_to_fit.GetParameter(idx) for idx in range(n_params)], dtype=np.float64)
        
        cov_matrix = fit_res.GetCovarianceMatrix()
        if not cov_matrix or cov_matrix.GetNrows() <= max(range(n_params)):
            continue
            
        n_spectra = len(fitter.spectra) if hasattr(fitter, 'spectra') else 1
        
        for j in range(n_spectra):
            n_pts = 100
            e_vals = np.linspace(window_start, window_end, n_pts)
            bg_vals = np.zeros(n_pts)
            bg_errs = np.zeros(n_pts)
            e_errs = np.zeros(n_pts)
            
            eps = 1e-5
            for pt_idx, E in enumerate(e_vals):
                x_arr = np.array([E, j], dtype=np.float64)
                val = bg_eval_func(x_arr, params)
                
                grad = np.zeros(n_params)
                for k in range(n_params):
                    if cov_matrix(k, k) == 0:
                        continue
                    p_plus = np.copy(params)
                    p_plus[k] += eps
                    p_minus = np.copy(params)
                    p_minus[k] -= eps
                    grad[k] = (bg_eval_func(x_arr, p_plus) - bg_eval_func(x_arr, p_minus)) / (2 * eps)
                    
                var = 0.0
                for r in range(n_params):
                    for c in range(n_params):
                        if grad[r] != 0 and grad[c] != 0:
                            var += grad[r] * grad[c] * cov_matrix(r, c)
                            
                bg_vals[pt_idx] = val
                bg_errs[pt_idx] = np.sqrt(max(0, var))
                
            gr = ROOT.TGraphErrors(n_pts, np.array(e_vals, dtype='float64'), np.array(bg_vals, dtype='float64'), np.array(e_errs, dtype='float64'), np.array(bg_errs, dtype='float64'))
            color = colors[j % len(colors)]
            gr.SetLineColor(color)
            gr.SetFillColorAlpha(color, 0.3)
            gr.SetFillStyle(1001)
            gr.SetTitle(f"Spectrum {j}")
            mg.Add(gr, "3")
            
            gr_line = ROOT.TGraph(n_pts, np.array(e_vals, dtype='float64'), np.array(bg_vals, dtype='float64'))
            gr_line.SetLineColor(color)
            gr_line.SetLineWidth(2)
            gr_line.SetTitle(f"Spectrum {j}")
            mg.Add(gr_line, "L")
            
            graphs.extend([gr, gr_line])
            
    if len(graphs) > 0:
        mg.Draw("A")
        canvas.Update()
    
    legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    added_species = set()
    for g in graphs:
        if g.GetTitle() not in added_species:
            legend.AddEntry(g, g.GetTitle(), "lf")
            added_species.add(g.GetTitle())
    legend.Draw()
    
    canvas.Update()
    
    ROOT.SetOwnership(canvas, False)
    ROOT.SetOwnership(mg, False)
    ROOT.SetOwnership(legend, False)
    for gr in graphs:
        ROOT.SetOwnership(gr, False)
    
    return canvas, mg, graphs, legend

####################################################################

#############################################################################
# Fit helper functions to add/remove peaks and fix previous parameters
#############################################################################
def _get_fitter_iteration_name(fitter):
    import re
    save_name = getattr(fitter, 'save_name', 'modified_fit')
    if not save_name:
        save_name = 'modified_fit'
    match = re.search(r'_(\d+)$', save_name)
    if match:
        base_name = save_name[:match.start()]
        iteration = int(match.group(1)) + 1
    else:
        base_name = save_name
        iteration = 1
    return f"{base_name}_{iteration}"

def _extract_fitter_bounds(fitter):
    import ctypes
    old_window_bounds = {}
    for i, res in enumerate(fitter.fit_results):
        if res is None or 'fit_res' not in res: continue
        f_to_fit = res.get('f_to_fit_2d') or res.get('f_to_fit')
        if not f_to_fit: continue
        old_window_bounds[i] = {}
        for j in range(f_to_fit.GetNpar()):
            name = f_to_fit.GetParName(j)
            val = f_to_fit.GetParameter(j)
            try:
                low = ctypes.c_double(0)
                high = ctypes.c_double(0)
                f_to_fit.GetParLimits(j, low, high)
                low_val, high_val = low.value, high.value
            except TypeError:
                low = ROOT.Double(0)
                high = ROOT.Double(0)
                f_to_fit.GetParLimits(j, low, high)
                low_val, high_val = float(low), float(high)
            old_window_bounds[i][name] = (val, low_val, high_val)
    return old_window_bounds

def _build_param_bounds(old_window_bounds, window_mapping, new_peaks, fix_params, fitter):
    merged_param_bounds = {}
    
    def make_peak_dependent_bound_func(base_name):
        def bound_func(E):
            if E not in window_mapping: return (E, E-fitter.location_wiggle, E+fitter.location_wiggle)
            old_i, old_to_new = window_mapping[E]
            new_to_old = {v: k for k, v in old_to_new.items()}
            
            parts = base_name.split('_')
            if base_name.startswith('mu_'):
                new_idx = int(parts[1])
                old_idx = new_to_old.get(new_idx)
                old_name = f'mu_{old_idx}'
            elif base_name.startswith('amplitude_'):
                new_idx = int(parts[1])
                spec = parts[-1]
                old_idx = new_to_old.get(new_idx)
                old_name = f'amplitude_{old_idx}_{spec}'
            elif base_name.startswith('total_amp_'):
                new_idx = int(parts[2])
                old_idx = new_to_old.get(new_idx)
                old_name = f'total_amp_{old_idx}'
            elif base_name == 'mu':
                old_idx = new_to_old.get(0)
                old_name = f'mu_{old_idx}' if f'mu_{old_idx}' in old_window_bounds.get(old_i, {}) else 'mu'
            else:
                old_idx = None
                old_name = base_name
                
            if old_idx is None:
                # It's a new peak
                if base_name.startswith('mu'): 
                    new_i = next((idx for idx, (locs, _, _) in enumerate(new_peaks) if abs(locs[0] - E) < 1e-2), None)
                    if new_i is not None and new_idx < len(new_peaks[new_i][0]):
                        actual_loc = new_peaks[new_i][0][new_idx]
                    else:
                        actual_loc = E
                    return (actual_loc, actual_loc - fitter.location_wiggle, actual_loc + fitter.location_wiggle)
                elif base_name.startswith('amplitude') or base_name.startswith('total_amp'): return (200, 1e-3, 1e6)
                else: return (0, -1e6, 1e6)
                
            bounds = old_window_bounds.get(old_i, {}).get(old_name)
            if bounds is None:
                if base_name.startswith('mu'): 
                    new_i = next((idx for idx, (locs, _, _) in enumerate(new_peaks) if abs(locs[0] - E) < 1e-2), None)
                    if new_i is not None and new_idx < len(new_peaks[new_i][0]):
                        actual_loc = new_peaks[new_i][0][new_idx]
                    else:
                        actual_loc = E
                    return (actual_loc, actual_loc - fitter.location_wiggle, actual_loc + fitter.location_wiggle)
                elif base_name.startswith('amplitude') or base_name.startswith('total_amp'): return (200, 1e-3, 1e6)
                else: return (0, -1e6, 1e6)
            val, low, high = bounds
            if not fix_params and low == 0 and high == 0:
                if base_name.startswith('amplitude') or base_name.startswith('total_amp'):
                    low, high = 1e-3, 1e6
                elif base_name.startswith('sigma'):
                    low, high = 0, 1000
                elif base_name.startswith('mu'):
                    low, high = val - fitter.location_wiggle, val + fitter.location_wiggle
                else:
                    low, high = -1e6, 1e6
            return (val, val, val) if fix_params else (val, low, high)
        return bound_func
        
    def make_global_bound_func(name):
        def bound_func(E):
            if E not in window_mapping: return (0,0,0)
            old_i, _ = window_mapping[E]
            bounds = old_window_bounds.get(old_i, {}).get(name)
            if bounds is None: return (0, -1e6, 1e6)
            val, low, high = bounds
            if not fix_params and low == 0 and high == 0:
                low, high = -1e6, 1e6
            return (val, val, val) if fix_params else (val, low, high)
        return bound_func

    max_peaks = max((len(locs) for locs, _, _ in new_peaks), default=1)
    
    for i in range(max_peaks):
        merged_param_bounds['mu'] = make_peak_dependent_bound_func('mu')
        merged_param_bounds[f'mu_{i}'] = make_peak_dependent_bound_func(f'mu_{i}')
        merged_param_bounds[f'total_amp_{i}'] = make_peak_dependent_bound_func(f'total_amp_{i}')
        for spec_idx in range(len(fitter.spectra)):
            merged_param_bounds[f'amplitude_{i}_{spec_idx}'] = make_peak_dependent_bound_func(f'amplitude_{i}_{spec_idx}')
            
    global_names = set()
    for i, params in old_window_bounds.items():
        for name in params:
            if not name.startswith('mu') and not name.startswith('amplitude') and not name.startswith('total_amp'):
                global_names.add(name)
                
    for name in global_names:
        merged_param_bounds[name] = make_global_bound_func(name)
        
    return merged_param_bounds

def remove_peak_from_fit(fitter, peaks_to_remove, fix_params=False):
    '''
    Removes one or more peaks from an existing fit.
    
    Parameters:
    fitter : SpectrumFitter
        The fitter object containing the existing fit to modify.
    peaks_to_remove : tuple or list of tuples
        Each tuple should be (window_index, peak_index) specifying which peak to remove.
    fix_params : bool
        Determines how the previously fitted parameters (peak locations, amplitudes, 
        background coefficients, sigma parameters, etc.) are handled in the new fit.
        If True, all previously fitted parameters are strictly fixed to their exact prior values.
        If False, the previously fitted values are used as the starting initial guesses for 
        the new fit, but are allowed to float and adjust.
    '''
    if isinstance(peaks_to_remove, tuple) and len(peaks_to_remove) == 2 and isinstance(peaks_to_remove[0], int):
        peaks_to_remove = [peaks_to_remove]
        
    new_save_name = _get_fitter_iteration_name(fitter)
    old_window_bounds = _extract_fitter_bounds(fitter)
    
    new_peaks = []
    new_isotopes = []
    original_isotopes = getattr(fitter, 'peak_isotopes', getattr(fitter, 'fit_multi_peaks_kwargs', {}).get('peak_isotopes'))
    window_mapping = {}

    for i, (locs, w_start, w_end) in enumerate(fitter.peaks_to_fit):
        new_locs = []
        new_isos = []
        old_to_new_idx = {}
        new_idx = 0
        for j, loc in enumerate(locs):
            if (i, j) not in peaks_to_remove:
                old_mu_name = 'mu' if len(locs) == 1 else f'mu_{j}'
                fitted_mu = old_window_bounds.get(i, {}).get(old_mu_name, (loc, 0, 0))[0]
                new_locs.append(fitted_mu)
                if original_isotopes and i < len(original_isotopes) and j < len(original_isotopes[i]):
                    new_isos.append(original_isotopes[i][j])
                old_to_new_idx[j] = new_idx
                new_idx += 1
                
        if new_locs:
            new_peaks.append((new_locs, w_start, w_end))
            if new_isos:
                new_isotopes.append(new_isos)
            window_mapping[new_locs[0]] = (i, old_to_new_idx)

    merged_param_bounds = _build_param_bounds(old_window_bounds, window_mapping, new_peaks, fix_params, fitter)

    kwargs = getattr(fitter, 'fit_multi_peaks_kwargs', {}).copy()
    if new_isotopes:
        kwargs['peak_isotopes'] = new_isotopes
        
    new_fitter = fit_multi_peaks(
        fitter.spectra, 
        new_peaks,
        new_save_name,
        force_refit=True,
        additional_param_bounds=merged_param_bounds,
        loc_wiggle=fitter.location_wiggle,
        **kwargs
    )
    new_fitter.save_name = new_save_name
    new_fitter.fit_multi_peaks_kwargs = kwargs
    return new_fitter

def add_peak_to_fit(fitter, new_peak_loc, new_peak_iso='unknown', fix_params=False):
    '''
    Adds one or more new peaks to an existing fit.
    
    Parameters:
    fitter : SpectrumFitter
        The fitter object containing the existing fit to modify.
    new_peak_loc : float or list of floats
        The initial guess location(s) (e.g., energy in keV) of the new peak(s) to add. 
    new_peak_iso : str or list of strs, optional
        The isotope label(s) for the new peak(s). Defaults to 'unknown'.
    fix_params : bool, optional
        Determines how the previously fitted parameters (peak locations, amplitudes, 
        background coefficients, sigma parameters, etc.) are handled in the new fit.
        If True, all previously fitted parameters are strictly fixed to their exact prior values.
        If False, the previously fitted values are used as the starting initial guesses for 
        the new fit, but are allowed to float and adjust.
        
    Returns:
    SpectrumFitter
        A new fitter object with the newly added peak(s) fit alongside the existing ones.
    '''
    if not isinstance(new_peak_loc, (list, tuple)):
        new_peak_loc = [new_peak_loc]
    if not isinstance(new_peak_iso, (list, tuple)):
        new_peak_iso = [new_peak_iso] * len(new_peak_loc)
        
    new_save_name = _get_fitter_iteration_name(fitter)
    old_window_bounds = _extract_fitter_bounds(fitter)
    
    new_peaks = []
    new_isotopes = []
    original_isotopes = getattr(fitter, 'peak_isotopes', getattr(fitter, 'fit_multi_peaks_kwargs', {}).get('peak_isotopes'))
    window_mapping = {}

    closest_window_indices = []
    for loc in new_peak_loc:
        closest_window_idx = 0
        min_dist = float('inf')
        for i, (locs, w_start, w_end) in enumerate(fitter.peaks_to_fit):
            dist = min(abs(loc - w_start), abs(loc - w_end))
            if w_start <= loc <= w_end:
                closest_window_idx = i
                break
            if dist < min_dist:
                min_dist = dist
                closest_window_idx = i
        closest_window_indices.append(closest_window_idx)

    for i, (locs, w_start, w_end) in enumerate(fitter.peaks_to_fit):
        new_locs = []
        new_isos = []
        source_idx = []
        
        for j, loc in enumerate(locs):
            old_mu_name = 'mu' if len(locs) == 1 else f'mu_{j}'
            fitted_mu = old_window_bounds.get(i, {}).get(old_mu_name, (loc, 0, 0))[0]
            new_locs.append(fitted_mu)
            source_idx.append(j)
            if original_isotopes and i < len(original_isotopes) and j < len(original_isotopes[i]):
                new_isos.append(original_isotopes[i][j])
                
        for k, p_loc in enumerate(new_peak_loc):
            if closest_window_indices[k] == i:
                new_locs.append(p_loc)
                source_idx.append(-1)
                new_isos.append(new_peak_iso[k])
                
        sorted_indices = sorted(range(len(new_locs)), key=lambda idx: new_locs[idx])
        sorted_new_locs = [new_locs[idx] for idx in sorted_indices]
        sorted_new_isos = [new_isos[idx] for idx in sorted_indices]
        sorted_source_idx = [source_idx[idx] for idx in sorted_indices]
        
        old_to_new_idx = {}
        for new_idx, s_idx in enumerate(sorted_source_idx):
            if s_idx != -1:
                old_to_new_idx[s_idx] = new_idx

        if sorted_new_locs:
            for k, p_loc in enumerate(new_peak_loc):
                if closest_window_indices[k] == i:
                    w_start = min(w_start, p_loc - 100)
                    w_end = max(w_end, p_loc + 100)
                    
            new_peaks.append((sorted_new_locs, w_start, w_end))
            if sorted_new_isos:
                new_isotopes.append(sorted_new_isos)
            window_mapping[sorted_new_locs[0]] = (i, old_to_new_idx)

    merged_param_bounds = _build_param_bounds(old_window_bounds, window_mapping, new_peaks, fix_params, fitter)

    kwargs = getattr(fitter, 'fit_multi_peaks_kwargs', {}).copy()
    if new_isotopes:
        kwargs['peak_isotopes'] = new_isotopes
        
    new_fitter = fit_multi_peaks(
        fitter.spectra, 
        new_peaks,
        new_save_name,
        force_refit=True,
        additional_param_bounds=merged_param_bounds,
        loc_wiggle=fitter.location_wiggle,
        **kwargs
    )
    new_fitter.save_name = new_save_name
    new_fitter.fit_multi_peaks_kwargs = kwargs
    return new_fitter

#############################################################################
# Fit including runs where high energy protons may not be recorded correctly.
#############################################################################
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

force_refit=True
ddas_runs_protons_low_energies_60Ga = e23035_runs.get_ddas_60_Ga_runs(good_gamma=False, final_beam_settings=True, good_low_energy_tpc=True, good_long_tracks_tpc=False)
pspec_low_energy_60Ga = ddas_interface.get_histogram(experiment, ddas_runs_protons_low_energies_60Ga, proton_binning, "proton_spectrum_low_energy_60Ga", "60Ga proton_spectrum low energy", "tpc_energy", "tpc_particle_id==1", num_workers=num_workers, tpc_ini_filename=tpc_config)
loc_wiggle = 15
#initial fitter with no peaks, and a fit window of 600 to 2900 keV
save_path_initial = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tpc_spectrum_fitting/protons_le', 'protons_le')
bg_shift_upper_bound = 10*0.5/(2000/5) 
isotopes_list = ['60Ga'] * 35 + ['59Zn'] * 20

# 1. Run Differential Evolution to find starting locations and save to CSV
find_de_guesses(
    [pspec_low_energy_60Ga, pspec_59Zn], 
    fit_window=(600.0, 2900.0), 
    isotopes_list=isotopes_list,
    save_csv_name='de_proton_peaks.csv',
    additional_param_bounds={'total_amp': lambda E:(1e-3, 1e6)}, 
    loc_wiggle=loc_wiggle,
    bg_model='chebyshev',
    bg_order=5,
    fraction_bernstein_order={'61Ge': 1, 'default': 2},
    sigma_monotonic_bernstein_order=4,
    bg_shift_monotonic_bernstein_order=4,
    bg_shift_upper_bound=bg_shift_upper_bound,
    sigma_min=10,
    sigma_max=200,
    workers=num_workers
)

# 2. Load the perfectly optimized DE guesses
proton_peak_guesses, peak_isotopes = load_peaks_from_csv('de_proton_peaks.csv')

# 3. Run the final Minuit fit
fs = [fit_multi_peaks(
        [pspec_low_energy_60Ga, pspec_59Zn], 
        proton_peak_guesses,
        save_path_initial, force_refit=force_refit,
        additional_param_bounds={'total_amp': lambda E:(1e-3, 1e6)}, 
        loc_wiggle=loc_wiggle,
        bg_model='chebyshev',
        bg_order=5,
        fraction_bernstein_order={'61Ge': 1, 'default': 2},
        # sigma_bernstein_order=2,
        # bg_shift_bernstein_order=2,
        sigma_monotonic_bernstein_order=4,
        bg_shift_monotonic_bernstein_order=4,
        bg_shift_upper_bound=bg_shift_upper_bound,
        sigma_min=10,
        sigma_max=200,
        peak_isotopes=peak_isotopes,
        use_de=False
    )]
ROOT.gROOT.SetBatch(False)
fs[0].show_fit_results(0, False, True)

#added 2210, 2275, 2380, 2600
# add peaks
# peaks_to_add = [2211, 2273]
# fs.append(add_peak_to_fit(fs[-1], peaks_to_add, len(peaks_to_add)*['60Ga']))
# peaks_to_add = [2380]
# fs.append(add_peak_to_fit(fs[-1], peaks_to_add, len(peaks_to_add)*['60Ga']))
#fs[-1].show_fit_results(0, False, True)
show_detector_energy_resolution(fs[-1])
show_peak_fractions(fs[-1])
show_bg_shifts(fs[-1])
show_backgrounds(fs[-1])


#fs.append(add_peak_to_fit(fs[0], 716, '60Ga')) 
#show_detector_energy_resolution(f_proton_initial)






#######################################################################
#old code
#######################################################################
if False:
    proton_peak_guesses, peak_isotopes = load_peaks_from_csv('proton_peaks.csv')
    new_peak_guesses, new_peak_isotopes = [],[]
    loc_wiggle = 10
    #fill new peak guesses and isotopes with just those from 59Zn peaks
    #then add a 60Ga peak every 2*loc_wiggle between 600 and 2850 keV
    for peaks_group, iso_group in zip(proton_peak_guesses, peak_isotopes):
        locations, window_start, window_end = peaks_group
        
        new_locations = [loc for loc, iso in zip(locations, iso_group) if iso == '59Zn']
        new_isos = ['59Zn'] * len(new_locations)
        
        for loc in np.arange(600, 2850 + 1e-5, 2 * loc_wiggle):
            new_locations.append(float(loc))
            new_isos.append('60Ga')
            
        sorted_pairs = sorted(zip(new_locations, new_isos))
        new_locations = [p[0] for p in sorted_pairs]
        new_isos = [p[1] for p in sorted_pairs]
        
        new_peak_guesses.append((new_locations, window_start, window_end))
        new_peak_isotopes.append(new_isos)

    save_path_low = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tpc_spectrum_fitting', '60Ga_59Zn_simultaneous_protons_low_energy')
    bg_shift_upper_bound = 0
    f_proton_simultaneous_low = fit_multi_peaks(
        [pspec_low_energy_60Ga, pspec_59Zn], 
        new_peak_guesses,
        save_path_low, force_refit=force_refit,
        additional_param_bounds={'total_amp': lambda E:(1e-3, 1e6),
                                'bg_shift': lambda E: (0, bg_shift_upper_bound)}, 
        loc_wiggle=loc_wiggle,
        bg_model='chebyshev',
        bg_order=5,
        fraction_bernstein_order={'61Ge': 1, 'default': 2},
        # sigma_bernstein_order=2,
        sigma_monotonic_bernstein_order=2,
        sigma_min=10,
        sigma_max=200,
        peak_isotopes=new_peak_isotopes
    )
    ROOT.gROOT.SetBatch(False)
    f_proton_simultaneous_low.show_fit_results(0, False, True)
    if False:
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

