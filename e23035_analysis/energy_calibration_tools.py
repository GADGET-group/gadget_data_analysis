import re
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import ROOT
import numpy as np

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
    slope, offset = calibration_results['slope'], calibration_results['offset']
    return f'({slope}*{branch_name} + {offset})'

def make_energy_calibration(ddas_run, calibration_name:str, branch_name:str, binning_for_fit:tuple, peaks:list, 
                            selection_string='', data_source='gamma_adc', time_branch='', time_bin_size=1800,
                            normalization_dict=None, min_counts=30):
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
    true_energies, true_energy_uncertainties, peak_locations, peak_location_uncertainties = [], [], [], []
    emg_fit_parameters = {} # Dictionary to store all fit parameters & p-values

    # Fit peaks
    for true_energy, true_energy_err, search_window, fit_window, rebin in peaks:
        
        # --- FIX 3: Clone the histogram so rebinning doesn't ruin the next peak! ---
        hist_for_this_peak = hist_to_fit.Clone(f"hist_temp_{true_energy}")
        
        if rebin != 1:
            hist_for_this_peak.Rebin(rebin)

        hist_for_this_peak.GetXaxis().SetRangeUser(*search_window)
        
        # --- FIX 2: Calculate counts strictly inside the zoomed window ---
        start_bin = hist_for_this_peak.GetXaxis().FindBin(search_window[0])
        stop_bin = hist_for_this_peak.GetXaxis().FindBin(search_window[1])
        counts_in_window = hist_for_this_peak.Integral(start_bin, stop_bin)
        
        # --- FIX 1: Use continue instead of break ---
        if counts_in_window < min_counts:
            print(f"[{branch_name}] WARNING: Skipping {true_energy} keV peak. Only {counts_in_window} counts.")
            hist_for_this_peak.SetDirectory(0) # Memory cleanup
            continue 
            
        max_bin_in_range = hist_for_this_peak.GetMaximumBin()        
        location_guess = hist_for_this_peak.GetXaxis().GetBinCenter(max_bin_in_range)
        hist_for_this_peak.GetXaxis().UnZoom()
        
        location_wiggle = np.max(np.abs(fit_window)) / 2

        # Call to fitting tools (using the cloned, safely-rebinned histogram)
        fit_res, background, peak_func, rp, canvas, spectrum_to_plot, f_to_fit, h_fit = fitting_tools.fit_emg_peak(
            hist_for_this_peak, data_source, location_guess, location_wiggle, (location_guess + fit_window[0], location_guess + fit_window[1])
        )
        
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
        if not np.isnan(err) and not np.isnan(peak_locations[i]) and err > 0.01
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
        safe_true_E_errs = np.maximum(valid_true_E_errs, 0.001)

        graph = ROOT.TGraphErrors(
            num_valid_peaks, 
            np.array(valid_peak_locs, dtype=np.float64), 
            np.array(valid_true_Es, dtype=np.float64), 
            np.array(valid_peak_errs, dtype=np.float64), 
            safe_true_E_errs # <-- Use the safe array here!
        )
        
        graph.SetTitle(f"Energy Calibration: {calibration_name};ADC Channel (#mu);True Energy (keV)")
        graph.SetMarkerStyle(20)
        
        cal_canvas = ROOT.TCanvas(f"c_cal_{calibration_name}", "Calibration Curve", 800, 600)
        graph.Draw("AP")
        
        cal_fit = ROOT.TF1("cal_fit", "pol1", min(valid_peak_locs)*0.8, max(valid_peak_locs)*1.2)
        cal_fit_res = graph.Fit(cal_fit, "SQ")
        
        offset = cal_fit.GetParameter(0)
        slope = cal_fit.GetParameter(1)
        
        var_offset = cal_fit_res.CovMatrix(0, 0)
        var_slope = cal_fit_res.CovMatrix(1, 1)
        cov_offset_slope = cal_fit_res.CovMatrix(0, 1) 
        
        ROOT.gStyle.SetOptFit(1111)
        cal_canvas.Update()
        cal_canvas.SaveAs(str(cal_dir / "calibration_curve.pdf"))
        
        p_value = cal_fit_res.Prob() if num_valid_peaks > 2 else 1.0

        calibration_results = {
            'slope': slope, 
            'offset': offset,
            'cov_matrix': {
                'var_offset': var_offset,
                'var_slope': var_slope,
                'cov_offset_slope': cov_offset_slope
            },
            'cal_p_value': p_value,
            'emg_fit_parameters': emg_fit_parameters,
            'num_peaks_used': num_valid_peaks
        }
        print(f"[{calibration_name}] Linear Cal: slope={slope:.4f}, offset={offset:.4f}")
        
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