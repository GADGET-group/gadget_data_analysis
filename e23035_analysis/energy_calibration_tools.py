import re
import os
import pickle
from pathlib import Path
import concurrent.futures

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import ROOT
import numpy as np
import pandas as pd

from e23035_analysis import fitting_tools
from raw_viewer import ddas_interface

def get_calibration_directory(ddas_run, calibration_name, branch_name):
    ddas_run = int(ddas_run)
    return f"e23035_analysis/calibrations/{ddas_run}/{calibration_name}/{branch_name}"

def get_calibration_result(ddas_run, calibration_name, branch_name):
    cal_dir = Path(get_calibration_directory(ddas_run, calibration_name, branch_name))
    pkl_path = cal_dir / f"{calibration_name}.pkl"
    with open(pkl_path, 'rb') as f:
        calibration_results = pickle.load(f)
    return calibration_results

def get_calibrated_energy_string(ddas_run, calibration_name, branch_name):
    calibration_results = get_calibration_result(ddas_run, calibration_name, branch_name)
    
    # New format with polynomial parameters
    if 'poly_params' in calibration_results:
        params = calibration_results['poly_params']
        cal_str = ""
        for i, p in enumerate(params):
            if i == 0:
                cal_str += f"({p})"
            else:
                term = f"{p}"
                for _ in range(i):
                    term += f"*{branch_name}"
                cal_str += f" + ({term})"
        return f"({cal_str})"
    # Old format for backward compatibility
    elif 'slope' in calibration_results:
        slope, offset = calibration_results['slope'], calibration_results['offset']
        return f'({slope}*{branch_name} + {offset})'
    else:
        raise ValueError(f"Calibration file for run {ddas_run}, cal '{calibration_name}', branch '{branch_name}' has an unknown format.")

def make_energy_calibration(ddas_run, calibration_name:str, branch_name:str, binning_for_fit:tuple, peaks:list, 
                            selection_string='', data_source='gamma_adc', time_branch='', time_bin_size=1800,
                            normalization_dict=None, min_counts=30, peak_model='gaus', poly_degree=1):
    '''
    Fit peaks to get energy calibration

    ddas_run: single run or iterable
    calibration_name: name by which the generated slope/offset will be saved and may be retrieved
    branch_name: name of the branch to retrieve data from (eg clover_1a_e, tpc_energy, etc)
    binning_for_fit: tuple specifying TH1D binning
    peaks: List of peaks to use to specify the calibration. Each list entry should contain tuples of 
        (true energy, true energy_uncertainty, (search window start, stop), (fit window +, -), rebin_factor). 
        Will assume offset = 0 if length is 1. The largest bin in the specified range will be used as a starting guess for the peak location.
    time_branch: will default to changing "_c" ending of branch name to _t
    time_bin_size: bin size to use for testing if gain is stable over time, in seconds. Defaults to 10 minutes.
    normalization_dict: Used to choose if time dependent histogram should be normalized on a per slice basis, for beam products or daughters ith
                        short decay timescales. Use "slice" for this case, or "total" for room background lines which should have truely time
                        independent rates.
    poly_degree: The degree of the polynomial to use for the energy calibration fit. Defaults to 1 (linear).
    peak_model: gaus for gaussian, or emg for exponentially modified gaussian, bg_shift_gaus for gaussian with different background to left and right
    '''
    if normalization_dict is None:
        normalization_dict = {}
    # turn off plotting so there aren't a bunch of pop ups
    original_batch_state = ROOT.gROOT.IsBatch()
    ROOT.gROOT.SetBatch(True)
    
    # Safely check if ddas_run is an iterable of runs or a single run
    try:
        num_workers = min(200, len(ddas_run))
    except TypeError:
        num_workers = 1

    print(f"[{branch_name}] Building 1D Histogram...", flush=True)    
    hist_to_fit = ddas_interface.get_histogram(ddas_run, binning_for_fit, branch_name, branch_name, branch_name, selection_string, num_workers)

    # Time-dependent histogram setup
    if time_branch == '':
        time_branch = branch_name[:-2] + '_t'
        
    start_time, stop_time = ddas_interface.get_first_and_last_ddas_time(ddas_run)
    
    # Ensure num_bins is an integer for ROOT
    num_time_bins = int(np.ceil((stop_time - start_time) / time_bin_size))
    if num_time_bins < 1: num_time_bins = 1 
    time_bins = (num_time_bins, start_time, stop_time)
    
    time_dependent_binning = (*binning_for_fit, *time_bins)
    time_dep_exp = '%s:%s' % (time_branch, branch_name)
    
    # Pass time_dep_exp explicitly to var_exp
    print(f"[{branch_name}] Building 2D Time Histogram...", flush=True)

    time_dependent_hist = ddas_interface.get_histogram(
        ddas_run=ddas_run, 
        binning=time_dependent_binning, 
        hist_name=f"{branch_name}_time_dep", 
        hist_title=f"{branch_name} Time Stability", 
        var_exp=time_dep_exp, 
        selection=selection_string, 
        num_workers=num_workers
    )

    # Make folder using pathlib for clean cross-platform path handling
    cal_dir = Path(get_calibration_directory(ddas_run, calibration_name, branch_name))
    cal_dir.mkdir(parents=True, exist_ok=True)

    # Open ROOT file to save the individual peak fit results safely
    results_file = ROOT.TFile(str(cal_dir / f"{calibration_name}_results.root"), "RECREATE")

    # Initialize lists
    true_energies, true_energy_uncertainties, peak_locations, peak_location_uncertainties, amplitudes, amplitude_uncertainties = [], [], [], [], [], []
    probs = []
    emg_fit_parameters = {} # Dictionary to store all fit parameters & p-values

    # Fit peaks
    for true_energy, true_energy_err, search_window, fit_window, rebin in peaks:
        
        hist_for_this_peak = hist_to_fit.Clone(f"hist_temp_{true_energy}")
        
        if rebin != 1:
            hist_for_this_peak.Rebin(rebin)

        hist_for_this_peak.GetXaxis().SetRangeUser(*search_window)
        
        start_bin = hist_for_this_peak.GetXaxis().FindBin(search_window[0])
        stop_bin = hist_for_this_peak.GetXaxis().FindBin(search_window[1])
        counts_in_window = hist_for_this_peak.Integral(start_bin, stop_bin)
        
        if counts_in_window < min_counts:
            print(f"[{branch_name}] WARNING: Skipping {true_energy} keV peak. Only {counts_in_window} counts.")
            hist_for_this_peak.SetDirectory(0) # Memory cleanup
            continue 
            
        max_bin_in_range = hist_for_this_peak.GetMaximumBin()        
        location_guess = hist_for_this_peak.GetXaxis().GetBinCenter(max_bin_in_range)
        hist_for_this_peak.GetXaxis().UnZoom()
        
        location_wiggle = np.max(np.abs(fit_window)) / 2
        fit_range = (location_guess + fit_window[0], location_guess + fit_window[1])

        # --- NEW: Route to the correct fitting engine ---
        if peak_model.lower() == 'gaus':
            fit_res, background, peak_func, rp, canvas, spectrum_to_plot, f_to_fit, h_fit = fitting_tools.fit_gaussian_peak(
                hist_for_this_peak, data_source, location_guess, fit_range, param_bounds={'mu': (location_guess - location_wiggle, location_guess + location_wiggle)}
            )
        elif peak_model.lower() == 'emg':
            fit_res, background, peak_func, rp, canvas, spectrum_to_plot, f_to_fit, h_fit = fitting_tools.fit_emg_peak(
                hist_for_this_peak, data_source, location_guess, fit_range, param_bounds={'mu': (location_guess - location_wiggle, location_guess + location_wiggle)}
            )
        elif peak_model.lower() == 'bg_shift_gaus':
            fit_res, background, peak_func, rp, canvas, spectrum_to_plot, f_to_fit, h_fit = fitting_tools.fit_gaussian_w_bg_shift(
                hist_for_this_peak, location_guess, fit_range,data_source, param_bounds={'mu': (location_guess - location_wiggle, location_guess + location_wiggle)})
        else:
            raise ValueError(f'unknown peak model {peak_model}')
        
        # Now it is safe to append the true energies!
        true_energies.append(true_energy)
        true_energy_uncertainties.append(true_energy_err)

        # Write result to file
        results_file.WriteObject(fit_res, f'peak_fit_res_{true_energy}')
          
        # Extract ALL parameters to save to pickle
        peak_params = {}
        for i in range(f_to_fit.GetNpar()):
            p_name = f_to_fit.GetParName(i)
            peak_params[p_name] = {
                'value': fit_res.Parameter(i),
                'error': fit_res.ParError(i)
            }
        
        # Adding Chi2, NDF, and 1D p-value to the dict
        peak_params['chi2'] = fit_res.Chi2()
        peak_params['ndf'] = fit_res.Ndf()
        peak_params['1d_p_value'] = fit_res.Prob()
        
        emg_fit_parameters[true_energy] = peak_params
          
        mu_val = fit_res.Parameter(2)
        mu_err = fit_res.ParError(2)
        peak_locations.append(mu_val)
        peak_location_uncertainties.append(mu_err)
        amplitudes.append(fit_res.Parameter(1))
        amplitude_uncertainties.append(fit_res.ParError(1))
        probs.append(fit_res.Prob())
        
        # 1D plot handling
        canvas.cd()
        upper_pad = rp.GetUpperPad()
        upper_pad.cd()
        ROOT.gStyle.SetOptFit(1111)
        canvas.Update()
        
        plot_path = str(cal_dir / f"fit_{true_energy}keV.pdf")
        canvas.SaveAs(plot_path + "(") 

        # --- FIX 4: Removed duplicate blocks below ---
        raw_start_x = time_dependent_hist.GetXaxis().FindBin(location_guess + fit_window[0])
        raw_stop_x  = time_dependent_hist.GetXaxis().FindBin(location_guess + fit_window[1])
        
        bin_start_x = min(raw_start_x, raw_stop_x)
        bin_stop_x  = max(raw_start_x, raw_stop_x)
        
        time_dependent_hist.GetXaxis().SetRange(bin_start_x, bin_stop_x)
        
        hist_residuals = time_dependent_hist.Clone(f"residuals_{true_energy}")
        hist_residuals.Reset()
        hist_residuals.SetTitle(f"Residuals (Data - Fit) {true_energy} keV;ADC Channel;Time [s]")
        
        norm_method = normalization_dict.get(true_energy, 'slice')
        
        total_chi2 = 0.0
        total_ndf = 0
        actual_entries = 0
        max_res = 0.0 
        
        n_time_bins = time_dependent_hist.GetNbinsY()
        x_axis = time_dependent_hist.GetXaxis()
        
        hist_integral = hist_to_fit.Integral(bin_start_x, bin_stop_x)

        for iy in range(1, n_time_bins + 1):
            slice_ndf = 0
            
            # --- NORMALIZATION LOGIC ---
            if norm_method == 'total':
                scale_factor = 1.0 / n_time_bins
            else:
                slice_integral = time_dependent_hist.Integral(bin_start_x, bin_stop_x, iy, iy)
                scale_factor = slice_integral / hist_integral if hist_integral > 0 else 1.0
            
            for ix in range(bin_start_x, bin_stop_x + 1):
                obs = time_dependent_hist.GetBinContent(ix, iy)
                
                # --- FIX: Read from the un-rebinned hist_to_fit ---
                base_exp = hist_to_fit.GetBinContent(ix)
                exp = base_exp * scale_factor
                
                residual = obs - exp
                
                # Plotting: Fill the visual histogram for ALL bins in the window
                hist_residuals.SetBinContent(ix, iy, residual)
                actual_entries += 1
                
                if abs(residual) > max_res:
                    max_res = abs(residual)
                
                # --- THRESHOLD STATISTICAL GATE ---
                if exp > 10 or obs > 10:
                    if obs > 0:
                        # --- FIX: Safety net to prevent ZeroDivisionError ---
                        safe_exp = max(exp, 1e-9) 
                        bc_term = 2.0 * (safe_exp - obs + obs * np.log(obs / safe_exp))
                    else:
                        bc_term = 2.0 * exp 
                    
                    total_chi2 += bc_term
                    slice_ndf += 1
                    
            # --- NDF LOGIC ---
            if slice_ndf > 0:
                if norm_method == 'total':
                    total_ndf += slice_ndf      
                else:
                    total_ndf += (slice_ndf - 1)

        total_ndf = max(1, total_ndf)

        # Update ROOT internals
        hist_residuals.SetEntries(actual_entries)
        hist_residuals.ResetStats() 

        time_indep_p_value = ROOT.TMath.Prob(total_chi2, total_ndf) if total_ndf > 0 else 0.0
        emg_fit_parameters[true_energy]['t_indep_p_value'] = time_indep_p_value
        
        # Memory cleanup for the temporary 1D clone
        hist_for_this_peak.SetDirectory(0)

        # Save 2D Data to middle page
        c_time = ROOT.TCanvas(f"c_time_{true_energy}", "2D Time vs Energy", 800, 600)
        time_dependent_hist.SetStats(0)
        time_dependent_hist.Draw("COLZ")
        c_time.SaveAs(plot_path)

        # Save 2D Residuals to last page and close PDF
        c_res = ROOT.TCanvas(f"c_res_{true_energy}", "Time Residuals", 800, 600)
        
        # Center the color scale on 0 for symmetric visualization of the raw residuals
        if max_res > 0:
            hist_residuals.SetMinimum(-max_res)
            hist_residuals.SetMaximum(max_res)
        
        hist_residuals.SetStats(0)
        hist_residuals.Draw("COLZ")

        # Create a text box to overlay the global P-Value and total chi2
        pave = ROOT.TPaveText(0.12, 0.75, 0.38, 0.88, "NDC")
        pave.SetFillColor(ROOT.kWhite)
        pave.SetBorderSize(1)
        pave.SetTextAlign(12)
        pave.AddText(f"Overall p-value = {time_indep_p_value:.3e}")
        pave.AddText(f"Total #chi^{{2}} / ndf = {total_chi2:.1f} / {total_ndf}")
        pave.Draw()
        
        c_res.Update() # Force ROOT to render the canvas before saving
        c_res.SaveAs(plot_path + ")") # Append ')' to close PDF
        
        # Memory cleanup
        c_time.Close()
        c_res.Close()

    # Close the ROOT file where peak fits are stored
    results_file.Close()

    # --- NEW: Filter out any fits that returned NaN or <= 0 uncertainties ---
    valid_indices = [
        i for i, err in enumerate(peak_location_uncertainties) 
        if not np.isnan(err) and not np.isnan(peak_locations[i]) and err > 0.01 and amplitudes[i]/amplitude_uncertainties[i]>3 and probs[i]>0.01
    ]

    valid_peak_locs = [peak_locations[i] for i in valid_indices]
    valid_peak_errs = [peak_location_uncertainties[i] for i in valid_indices]
    valid_true_Es = [true_energies[i] for i in valid_indices]
    valid_true_E_errs = [true_energy_uncertainties[i] for i in valid_indices]

    calibration_results = {}
    
    # Check how many valid peaks survived the filtering
    num_valid_peaks = len(valid_peak_locs)

    if num_valid_peaks == 0:
        print(f"[{calibration_name}] ERROR: All peak fits failed or returned NaN. Calibration aborted.")
        # Return a dummy calibration so the pipeline doesn't crash
        calibration_results = {
            'slope': 1.0, 
            'offset': 0.0,
            'cov_matrix': {'var_offset': 0.0, 'var_slope': 0.0, 'cov_offset_slope': 0.0},
            'cal_p_value': 0.0,
            'emg_fit_parameters': emg_fit_parameters
        }

    elif num_valid_peaks == 1:
        # Offset = 0, E = m * ADC
        slope = valid_true_Es[0] / valid_peak_locs[0]
        offset = 0.0
        
        rel_err_E = valid_true_E_errs[0] / valid_true_Es[0] if valid_true_Es[0] != 0 else 0
        rel_err_mu = valid_peak_errs[0] / valid_peak_locs[0]
        sigma_slope = slope * np.sqrt(rel_err_E**2 + rel_err_mu**2)
        
        calibration_results = {
            'slope': slope, 
            'offset': offset,
            'cov_matrix': {
                'var_offset': 0.0,
                'var_slope': sigma_slope**2,
                'cov_offset_slope': 0.0
            },
            'cal_p_value': 1.0,
            'emg_fit_parameters': emg_fit_parameters
        }
        print(f"[{calibration_name}] 1-Point Cal (Fallback): slope={slope:.4f} ± {sigma_slope:.4f}, offset=0")
        
    else:
        # Enforce a minimum error of 0.001 keV to prevent weight overflow!
        safe_true_E_errs = np.maximum(valid_true_E_errs, 0.001)

        graph = ROOT.TGraphErrors(
            num_valid_peaks, 
            np.array(valid_peak_locs, dtype=np.float64), 
            np.array(valid_true_Es, dtype=np.float64), 
            np.array(valid_peak_errs, dtype=np.float64), 
            safe_true_E_errs
        )
        
        # Remove the X-axis title here so it doesn't get squashed between the pads
        graph.SetTitle(f"Energy Calibration: {calibration_name};;True Energy (keV)")
        graph.SetMarkerStyle(20)
        
        # --- NEW: Make canvas slightly taller to fit the dual pads cleanly ---
        cal_canvas = ROOT.TCanvas(f"c_cal_{calibration_name}", "Calibration Curve", 800, 800)
        
        # --- NEW: Setup the Upper Pad (70% height) ---
        pad1 = ROOT.TPad("pad1", "pad1", 0.0, 0.3, 1.0, 1.0)
        pad1.SetBottomMargin(0.02) # Leave a tiny gap between plots
        pad1.Draw()
        pad1.cd()
        
        graph.Draw("AP")
        
        # Style the main graph's Y-axis to look proportional in the split pad
        graph.GetYaxis().SetTitleSize(0.045)
        graph.GetYaxis().SetLabelSize(0.04)
        graph.GetYaxis().SetTitleOffset(0.9)
        
        cal_fit = ROOT.TF1("cal_fit", f"pol{poly_degree}", min(valid_peak_locs)*0.8, max(valid_peak_locs)*1.2)
        
        if poly_degree == 1 and num_valid_peaks > 1:
            # Give MINUIT a basic slope guess to prevent Effective Variance 0-division
            guess_slope = (valid_true_Es[-1] - valid_true_Es[0]) / (valid_peak_locs[-1] - valid_peak_locs[0])
            cal_fit.SetParameters(0.0, guess_slope)

        cal_fit_res = graph.Fit(cal_fit, "SQ")
        
        poly_params = [cal_fit.GetParameter(i) for i in range(poly_degree + 1)]
        
        ROOT.gStyle.SetOptFit(1111)
        pad1.Update() # Force the stat box to draw on pad1
        
        # --- NEW: Setup the Lower Pad (30% height) for Residuals ---
        cal_canvas.cd() # Return to main canvas before drawing pad2
        pad2 = ROOT.TPad("pad2", "pad2", 0.0, 0.0, 1.0, 0.3)
        pad2.SetTopMargin(0.02)
        pad2.SetBottomMargin(0.3) # Need extra room on the bottom for the X-axis label
        pad2.Draw()
        pad2.cd()
        
        # Calculate residuals: True Energy - Fit(ADC)
        res_y = [valid_true_Es[i] - cal_fit.Eval(valid_peak_locs[i]) for i in range(num_valid_peaks)]
        
        # Propagate error from ADC uncertainty through the polynomial fit for residuals plot
        derivs = np.array([cal_fit.Derivative(x) for x in valid_peak_locs])
        effective_err = np.sqrt(safe_true_E_errs**2 + (derivs * np.array(valid_peak_errs))**2)
        print('effective chi^2 ', np.sum((np.array(res_y)/effective_err)**2))
        graph_res = ROOT.TGraphErrors(
            num_valid_peaks,
            np.array(valid_peak_locs, dtype=np.float64),
            np.array(res_y, dtype=np.float64),
            np.zeros(len(valid_peak_errs)),
            effective_err
        )
        
        graph_res.SetTitle("")
        graph_res.SetMarkerStyle(20)
        graph_res.Draw("AP")
        
        # Because this pad is physically shorter, we have to scale up the text size 
        # so it matches the font size of the upper pad!
        graph_res.GetXaxis().SetTitle("ADC Channel (#mu)")
        graph_res.GetXaxis().SetTitleSize(0.12)
        graph_res.GetXaxis().SetLabelSize(0.10)
        
        graph_res.GetYaxis().SetTitle("Resid (keV)")
        graph_res.GetYaxis().SetTitleSize(0.12)
        graph_res.GetYaxis().SetLabelSize(0.10)
        graph_res.GetYaxis().SetTitleOffset(0.4)
        graph_res.GetYaxis().SetNdivisions(505) # Prevent crowded Y-axis labels
        
        # Draw a dashed reference line at Y=0
        zero_line = ROOT.TLine(graph_res.GetXaxis().GetXmin(), 0, graph_res.GetXaxis().GetXmax(), 0)
        zero_line.SetLineStyle(2)
        zero_line.SetLineColor(ROOT.kBlack)
        zero_line.Draw()
        
        cal_canvas.Update()
        cal_canvas.SaveAs(str(cal_dir / "calibration_curve.pdf"))
        
        p_value = cal_fit_res.Prob() if num_valid_peaks > (poly_degree + 1) else 1.0

        cov_matrix_dict = {}
        for i in range(poly_degree + 1):
            for j in range(i, poly_degree + 1):
                cov_matrix_dict[f'cov_p{i}_p{j}'] = cal_fit_res.CovMatrix(i, j)

        calibration_results = {
            'poly_params': poly_params,
            'poly_degree': poly_degree,
            'cov_matrix': cov_matrix_dict,
            'cal_p_value': p_value,
            'emg_fit_parameters': emg_fit_parameters,
            'num_peaks_used': num_valid_peaks
        }
        
        # For backward compatibility of NEWLY created linear files
        if poly_degree == 1:
            calibration_results['offset'] = poly_params[0]
            calibration_results['slope'] = poly_params[1]

        print(f"[{calibration_name}] {poly_degree}-degree Cal: params={poly_params}")
        
    # Save parameters to pickle file
    pkl_path = cal_dir / f"{calibration_name}.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(calibration_results, f)
    
    ROOT.gROOT.SetBatch(original_batch_state)

    return calibration_results

def get_run_dataframe(ddas_run, tree_name='merged_data'):
    """
    Retrieves the ROOT file for a given run and initializes an RDataFrame.
    """
    # TODO: Adapt this line to however your ddas_interface locates ROOT files
    file_path = ddas_interface.get_merged_root_file_path(ddas_run)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find ROOT file for run {ddas_run} at {file_path}")
        
    df = ROOT.RDataFrame(tree_name, file_path)
    return df

def apply_calibration(df, ddas_run, branch_list, calibration_name):
    """
    Reads calibration parameters and defines new calibrated columns in the RDataFrame.
    Sets the calibrated energy to 0.0 if the corresponding multiplicity is 0.
    Returns the updated RDataFrame node.
    """
    for branch in branch_list:
        new_branch_name = f"{branch}_{calibration_name}"

        # Only define the column if it hasn't been defined already in this graph
        if not df.HasColumn(new_branch_name):
            
            # Dynamically determine the corresponding multiplicity branch
            # e.g., 'clover_1a_c' -> 'clover_1a_m'
            if branch.endswith('_c') or branch.endswith('_e'):
                mult_branch = branch[:-2] + '_m'
            else:
                assert False #not implemented yet
            
            # Locate and load the pickle file
            cal_dir = Path(get_calibration_directory(ddas_run, calibration_name, branch))
            pkl_path = cal_dir / f"{calibration_name}.pkl"
                
            with open(pkl_path, 'rb') as f:
                cal_data = pickle.load(f)
                
            m = cal_data['slope']
            b = cal_data['offset']

            # Create the C++ logic string
            # If multiplicity > 0, calculate energy. Otherwise, return 0.0.
            cpp_logic = f"({mult_branch} > 0) ? ({m} * {branch} + {b}) : 0.0"

            # Create the new node and update our df reference
            df = df.Define(new_branch_name, cpp_logic)

    return df


def get_df_histogram(df, ddas_run, column_list, binning, cache_tag="calibrated", force_recreate=False):
    """
    Takes an RDataFrame, books 1D histograms for the requested columns, 
    executes the event loop if needed, and caches the results to a .root file.
    """
    cache_dir = Path(f"e23035_analysis/cache/run_{ddas_run}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"hists_{cache_tag}.root"

    hist_dict = {}

    # 1. Check Cache
    if cache_file.exists() and not force_recreate:
        f = ROOT.TFile(str(cache_file), "READ")
        all_cached = True
        
        for col in column_list:
            h = f.Get(f"h_{col}")
            if h:
                h.SetDirectory(0) # Detach from file
                hist_dict[col] = h
            else:
                all_cached = False
                break 
                
        f.Close()
        if all_cached:
            print(f"Loaded {len(column_list)} histograms from cache: {cache_file.name}")
            return hist_dict

    # 2. Book Histograms (Lazy)
    hist_ptrs = {}
    n_bins, x_min, x_max = binning

    for col in column_list:
        h_model = (f"h_{col}", f"{col};Energy (keV);Counts", n_bins, x_min, x_max)
        hist_ptrs[col] = df.Histo1D(h_model, col)

    # 3. Trigger Event Loop & Save to Cache
    if hist_ptrs:
        print(f"Executing RDataFrame loop to generate {len(hist_ptrs)} histograms...")
        f = ROOT.TFile(str(cache_file), "RECREATE")
        
        for col, ptr in hist_ptrs.items():
            # ptr.GetValue() triggers the loop for ALL booked pointers at once
            h = ptr.GetValue() 
            h.Write()
            h.SetDirectory(0)
            hist_dict[col] = h
            
        f.Close()

    return hist_dict


def create_calibration_summary(cal_name, pvalue_threshold_dict, run_list=None):
    '''
    Create a PDF file summarizing the calibration for all runs that have it.
    
    cal_name: the name of the calibration to summarize
    pvalue_threshold_dict: Specifies threshold at which p-values should be flagged. 
        May specify thresholds with the following keys:
        - ['cal'] : threshold for the overall linear fit p-value
        - [peak_energy]['1d'] : threshold for the 1D EMG fit p-value
        - [peak_energy]['t_indep'] : threshold for the 2D time-independence p-value
        Not all possible entries need to be included.
    run_list: Optional list of integer run numbers to include. If provided, 
        only these runs will be processed. Missing runs are safely ignored.
    '''
    
    # 1. Automatically scoop up all fit results matching the calibration name
    cal_data = {}
    
    # Convert run_list to a set for faster lookups (if it was provided)
    valid_runs = set(run_list) if run_list is not None else None
    
    # Search specifically inside the known analysis directory structure
    pkl_files = list(Path('.').rglob(f"e23035_analysis/calibrations/*/{cal_name}/*/{cal_name}.pkl"))
    
    # Fallback just in case the script is run from inside the calibrations folder itself
    if not pkl_files:
        pkl_files = list(Path('.').rglob(f"{cal_name}.pkl"))

    if not pkl_files:
        print(f"No calibration files found for '{cal_name}'.")
        return

    for pkl_path in pkl_files:
        # Path structure: .../{ddas_run}/{cal_name}/{branch_name}/{cal_name}.pkl
        branch_name = pkl_path.parent.name
        
        try:
            # The run number is exactly 3 folders up from the .pkl file
            run_num = int(pkl_path.parent.parent.parent.name)
        except ValueError:
            print(f"Warning: Could not parse run number from path {pkl_path}. Skipping.")
            continue
            
        # If the user provided a list of runs, skip any run not in that list
        if valid_runs is not None and run_num not in valid_runs:
            continue
        
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            
        if run_num not in cal_data:
            cal_data[run_num] = {}
        cal_data[run_num][branch_name] = data

    # Check if we actually found data for the requested runs
    if not cal_data:
        print("No valid calibration data found for the requested runs.")
        return

    runs = sorted(cal_data.keys())
    all_branches = set(b for r in runs for b in cal_data[r].keys())

    # 2. Check thresholds and prepare the flagged summary
    flagged_issues = []
    
    for run in runs:
        for branch, data in cal_data[run].items():
            
            # --- NEW: Check for failed/fallback calibrations ---
            # Default to 2 for older pickle files that might not have this key yet
            num_peaks = data.get('num_peaks_used', 2) 
            
            if num_peaks == 0:
                flagged_issues.append(f"CRITICAL: Run {run} [{branch}] - 0 peaks fit (Calibration FAILED)")
            elif num_peaks == 1:
                flagged_issues.append(f"WARNING: Run {run} [{branch}] - Only 1 peak fit (Assumed offset=0)")

            # Check Overall Cal P-value
            if 'cal' in pvalue_threshold_dict:
                p_val = data.get('cal_p_value', 1.0)
                if p_val < pvalue_threshold_dict['cal']:
                    flagged_issues.append(f"Run {run} [{branch}] - Overall Cal p-value: {p_val:.2e}")
            
            # Check individual peak P-values
            emg_params = data.get('emg_fit_parameters', {})
            for peak_energy, params in emg_params.items():
                if peak_energy in pvalue_threshold_dict:
                    
                    if '1d' in pvalue_threshold_dict[peak_energy]:
                        p_1d = params.get('1d_p_value', 1.0)
                        if p_1d < pvalue_threshold_dict[peak_energy]['1d']:
                            flagged_issues.append(f"Run {run} [{branch}] - {peak_energy} keV 1D p-value: {p_1d:.2e}")
                            
                    if 't_indep' in pvalue_threshold_dict[peak_energy]:
                        p_t = params.get('t_indep_p_value', 1.0)
                        if p_t < pvalue_threshold_dict[peak_energy]['t_indep']:
                            flagged_issues.append(f"Run {run} [{branch}] - {peak_energy} keV Time-Indep p-value: {p_t:.2e}")

    # 3. Generate the PDF in the specified analysis directory
    output_dir = Path("e23035_analysis/calibrations")
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_filename = output_dir / f"{cal_name}_summary.pdf"
    
    with PdfPages(pdf_filename) as pdf:
        
        # --- Page 1: Text Summary of Flags ---
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        y_pos = 0.95
        ax.text(0.05, y_pos, f"Calibration Summary: {cal_name}", fontsize=16, fontweight='bold')
        y_pos -= 0.05
        
        if not flagged_issues:
            ax.text(0.05, y_pos, "All p-values are above the specified thresholds.", fontsize=12)
        else:
            ax.text(0.05, y_pos, "Flagged Fits (Below Threshold):", fontsize=12, fontweight='bold')
            y_pos -= 0.03
            for issue in flagged_issues:
                if y_pos < 0.05: # Create a new page if we run out of vertical space
                    pdf.savefig(fig)
                    plt.close(fig)
                    fig, ax = plt.subplots(figsize=(8.5, 11))
                    ax.axis('off')
                    y_pos = 0.95
                ax.text(0.05, y_pos, issue, fontsize=10)
                y_pos -= 0.015
                
        pdf.savefig(fig)
        plt.close(fig)

        # Helper function for scatter plots
        def make_scatter_page(title, y_label, extract_func, is_log=False):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title(title)
            ax.set_xlabel("Run Number")
            ax.set_ylabel(y_label)
            if is_log:
                ax.set_yscale('log')
                
            plotted_anything = False
            for branch in all_branches:
                x_vals, y_vals = [], []
                for run in runs:
                    if branch in cal_data[run]:
                        val = extract_func(cal_data[run][branch])
                        if val is not None:
                            x_vals.append(run)
                            y_vals.append(val)
                if x_vals:
                    ax.plot(x_vals, y_vals, label=branch, alpha=0.7)
                    plotted_anything = True
                    
            if plotted_anything:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                fig.tight_layout()
                ax.grid(True, linestyle='--', alpha=0.6)
                pdf.savefig(fig)
            plt.close(fig)

        # --- Plot: Overall Slope/Offset Cal P-Value ---
        make_scatter_page(
            "Overall Slope & Offset Fit P-Value", "p-value",
            lambda d: d.get('cal_p_value'), is_log=True
        )

        # --- Get list of all peak energies found across all files ---
        all_peaks = set()
        for r in runs:
            for b, d in cal_data[r].items():
                all_peaks.update(d.get('emg_fit_parameters', {}).keys())
        all_peaks = sorted(list(all_peaks))

        # --- Plots per Peak ---
        for peak in all_peaks:
            
            # 1D P-values
            make_scatter_page(
                f"{peak} keV - 1D Fit P-Values", "p-value",
                lambda d: d.get('emg_fit_parameters', {}).get(peak, {}).get('1d_p_value'), is_log=True
            )
            
            # Time-Indep P-values
            make_scatter_page(
                f"{peak} keV - Time Independence P-Values", "p-value",
                lambda d: d.get('emg_fit_parameters', {}).get(peak, {}).get('t_indep_p_value'), is_log=True
            )
            
            # Fit Parameters
            sample_params = None
            for r in runs:
                for b, d in cal_data[r].items():
                    if peak in d.get('emg_fit_parameters', {}):
                        sample_params = d['emg_fit_parameters'][peak]
                        break
                if sample_params: break
            
            if sample_params:
                # Exclude the metrics we already plotted
                param_names = [k for k in sample_params.keys() if k not in ('1d_p_value', 't_indep_p_value', 'chi2', 'ndf')]
                
                for param in param_names:
                    make_scatter_page(
                        f"{peak} keV Fit Parameter: {param}", param,
                        lambda d, p=param: d.get('emg_fit_parameters', {}).get(peak, {}).get(p, {}).get('value')
                    )

    print(f"Summary saved to {pdf_filename}")



def _fetch_calibrated_run_hist(run, cal_name, ch, binning):
    """
    Worker function to fetch a single histogram in a parallel process.
    Must be at the top level of the module to be picklable.
    """
    import ROOT # Ensure the worker process has ROOT loaded
    
    cal_exp = get_calibrated_energy_string(run, cal_name, ch)
    h_run = ddas_interface.get_histogram(
        run, binning, f'{ch}_{run}', f'{ch} Run {run}', cal_exp
    )
    
    if h_run:
        # CRITICAL: Detach the C++ object from the worker's memory 
        # so it survives being serialized and sent back to the main process!
        h_run.SetDirectory(0) 
        
    return run, h_run

def create_stability_summary(cal_name, binning, pvalue_threshold, energy_threshold, run_list=None):
    '''
    Generates a PDF summary of gain stability across runs for a given calibration.
    Builds a calibrated sum of all runs, compares each run to the sum, and calculates a p-value.
    '''
    original_batch_state = ROOT.gROOT.IsBatch()
    ROOT.gROOT.SetBatch(True)

    # 1. Scope out all valid runs and branches from the file system
    pkl_files = list(Path('.').rglob(f"e23035_analysis/calibrations/*/{cal_name}/*/{cal_name}.pkl"))
    if not pkl_files:
        pkl_files = list(Path('.').rglob(f"{cal_name}.pkl"))
        
    branch_run_map = {}
    for p in pkl_files:
        branch = p.parent.name
        try:
            run = int(p.parent.parent.parent.name)
        except ValueError:
            continue
            
        if run_list is not None and run not in run_list:
            continue
            
        if branch not in branch_run_map:
            branch_run_map[branch] = []
        branch_run_map[branch].append(run)

    if not branch_run_map:
        print(f"No valid runs or branches found for calibration '{cal_name}'.")
        return

    p_value_matrix = {}
    # 2. Iterate through each crystal/branch
    for ch, runs_for_ch in branch_run_map.items():
        runs_for_ch = sorted(runs_for_ch)
        print(f"[{ch}] Processing stability over {len(runs_for_ch)} runs...")
        
        # Create a blank master histogram for the sum
        n_bins, x_min, x_max = binning
        summed_hist = ROOT.TH1D(f'{ch}_all_run', f'Sum of {ch} for all runs', n_bins, x_min, x_max)
        summed_hist.SetDirectory(0) # Keep alive in memory
        
        # --- PARALLELIZED Loop 1: Build calibrated runs and add to master sum ---
        run_hists = []
        
        # Safely determine the number of cores to use (cap at 64 to avoid thrashing)
        max_workers = min(os.cpu_count() or 4, len(runs_for_ch), 64)
        print(f"[{ch}] Fetching histograms using {max_workers} parallel workers...")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Dispatch all the runs to the worker pool
            future_to_run = {
                executor.submit(_fetch_calibrated_run_hist, run, cal_name, ch, binning): run 
                for run in runs_for_ch
            }
            
            # Collect them as they finish
            for future in concurrent.futures.as_completed(future_to_run):
                run = future_to_run[future]
                try:
                    ret_run, h_run = future.result()
                    if h_run:
                        h_run.SetDirectory(0) # Re-detach in the main process just to be safe
                        run_hists.append((ret_run, h_run))
                        summed_hist.Add(h_run)
                except Exception as exc:
                    print(f"[{ch}] ERROR: Run {run} generated an exception during fetch: {exc}")

        # Because parallel workers finish out of order, we MUST sort the 
        # list back into chronological run order so your PDF pages make sense!
        run_hists.sort(key=lambda x: x[0])

        # Restrict the physical axis range to ignore low-energy noise
        summed_hist.GetXaxis().SetRangeUser(energy_threshold, x_max)
        sum_integral = summed_hist.Integral()
        
        if sum_integral == 0:
            print(f"[{ch}] WARNING: Summed histogram is entirely empty above {energy_threshold} keV. Skipping.")
            continue

        run_data = []
        flagged_runs = []
        p_value_matrix[ch] = {}
        
        # --- Loop 2: Fetch individual runs and compute p-values ---
        for run, h_run in run_hists:
            
            # Apply the exact same threshold to the individual run
            h_run.GetXaxis().SetRangeUser(energy_threshold, x_max)
            
            run_integral = h_run.Integral()
            h_sum_scaled = summed_hist.Clone(f"{ch}_sum_scaled_{run}")
            h_sum_scaled.SetDirectory(0)
            
            if run_integral > 10: 
                # The Chi2Test automatically respects the X-axis bounds and handles empty bins
                p_val = h_run.Chi2Test(summed_hist, "UU NORM")
                
                scale_factor = run_integral / sum_integral
                h_sum_scaled.Scale(scale_factor)
            else:
                p_val = 0.0 
                h_sum_scaled.Scale(0)
                
            if p_val < pvalue_threshold:
                flagged_runs.append((run, p_val))
                
            p_value_matrix[ch][run] = p_val
            run_data.append((run, p_val, h_run, h_sum_scaled))

        # 4. Create the PDF Document using ROOT's TPDF Engine
        out_dir = Path(f"e23035_analysis/calibrations/{cal_name}")
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = str(out_dir / f"{ch}_stability.pdf")
        
        c = ROOT.TCanvas(f"c_stab_{ch}", "Stability Summary", 1000, 800)
        c.Print(pdf_path + "[") 
        
        # --- Draw Summary Page(s) ---
        text = ROOT.TLatex()
        text.SetNDC()
        text.SetTextFont(42)
        
        def draw_header():
            c.Clear()
            text.SetTextSize(0.05)
            text.SetTextColor(ROOT.kBlack)
            text.DrawLatex(0.1, 0.9, f"Gain Stability Summary: {ch} (> {energy_threshold} keV)")
            text.SetTextSize(0.035)
            
        draw_header()
        y_pos = 0.82
        
        if not flagged_runs:
            text.SetTextColor(ROOT.kGreen + 2)
            text.DrawLatex(0.1, y_pos, f"All runs stable! (p > {pvalue_threshold})")
            c.Print(pdf_path) 
        else:
            text.SetTextColor(ROOT.kRed)
            text.DrawLatex(0.1, y_pos, f"Flagged Runs (p < {pvalue_threshold}):")
            y_pos -= 0.05
            text.SetTextColor(ROOT.kBlack)
            
            for run, pval in flagged_runs:
                text.DrawLatex(0.15, y_pos, f"Run {run}: p-value = {pval:.2e}")
                y_pos -= 0.04
                
                if y_pos < 0.05: 
                    c.Print(pdf_path)
                    draw_header()
                    y_pos = 0.82
            c.Print(pdf_path) 
            
        # --- Draw Individual Run Plots vs Sum ---
        for run, pval, h_run, h_sum_scaled in run_data:
            c.Clear()
            
            # Style the Run Data (Black Points)
            h_run.SetLineColor(ROOT.kBlack)
            h_run.SetMarkerStyle(20)
            h_run.SetMarkerSize(0.8)
            h_run.SetStats(0)
            
            # Prevent log(0) errors on empty background bins
            h_run.SetMinimum(0.5) 
            
            # Style the Scaled Sum (Red Line / Shaded)
            h_sum_scaled.SetLineColor(ROOT.kRed)
            h_sum_scaled.SetLineWidth(2)
            h_sum_scaled.SetFillColorAlpha(ROOT.kRed, 0.2)
            h_sum_scaled.SetStats(0)
            
            # Use TRatioPlot for the Residuals ("diff" mode means Data - Fit)
            rp = ROOT.TRatioPlot(h_run, h_sum_scaled, "diff")
            rp.SetH1DrawOpt("E")
            rp.SetH2DrawOpt("HIST")
            rp.Draw()
            
            # --- NEW: Force the X-axis to obey the threshold ---
            rp.GetXaxis().SetRangeUser(energy_threshold, binning[2])
            
            # --- NEW: Dynamically zoom the Y-axis to ignore bin 0 anomalies ---
            max_res = 0.0
            start_bin = h_run.GetXaxis().FindBin(energy_threshold)
            stop_bin = h_run.GetXaxis().FindBin(binning[2])

            for i in range(start_bin, stop_bin + 1):
                res = abs(h_run.GetBinContent(i) - h_sum_scaled.GetBinContent(i))
                if res > max_res:
                    max_res = res

            # Add a 20% visual buffer to the top and bottom
            y_limit = max_res * 1.2 if max_res > 0 else 10.0
            rp.GetLowerRefYaxis().SetRangeUser(-y_limit, y_limit)
            
            # Set the upper pad to a Log-Y scale
            rp.GetUpperPad().SetLogy()
            
            rp.GetLowerRefYaxis().SetTitle("Resid (Run - Sum)")
            rp.GetLowerPad().SetGridy()
            
            rp.GetUpperPad().cd()
            pave = ROOT.TPaveText(0.70, 0.75, 0.88, 0.88, "NDC")
            pave.SetFillColor(ROOT.kWhite)
            pave.SetBorderSize(1)
            pave.AddText(f"Run: {run}")
            
            pval_text = pave.AddText(f"p-value: {pval:.2e}")
            if pval < pvalue_threshold:
                pval_text.SetTextColor(ROOT.kRed)
                
            pave.Draw()
            
            c.Update()
            c.Print(pdf_path) 
            
        c.Print(pdf_path + "]") 
        
    # 5. Save P-Value Matrix to CSV
    out_dir = Path(f"e23035_analysis/calibrations/{cal_name}")
    csv_path = out_dir / f"stability_p_values.csv"
    df_pvals = pd.DataFrame(p_value_matrix)
    df_pvals.index.name = 'run'
    df_pvals.to_csv(csv_path)
    print(f"P-value matrix saved to {csv_path}")

    ROOT.gROOT.SetBatch(original_batch_state)
    print("Stability generation complete!")