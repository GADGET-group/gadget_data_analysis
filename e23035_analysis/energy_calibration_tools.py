import os
import pickle
import ROOT
import numpy as np
from pathlib import Path

from e23035_analysis import fitting_tools
from raw_viewer import ddas_interface

def get_calibration_directory(ddas_run, calibration_name, branch_name):
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

def make_energy_calibration(ddas_run, calibration_name:str, branch_name:str, binning_for_fit:tuple, peaks:list, selection_string='', data_source='gamma_adc'):
    '''
    Fit peaks to get energy calibration

    ddas_run: single run or iterable
    calibration_name: name by which the generated slope/offset will be saved and may be retrieved
    branch_name: name of the branch to retrieve data from (eg clover_1a_e, tpc_energy, etc)
    binning_for_fit: tuple specifying TH1D binning
    peaks: List of peaks to use to specify the calibration. Each list entry should contain tuples of 
        (true energy, true energy_uncertainty, (search window start, stop), (fit window +, -). 
        Will assume offset = 0 if length is 1. The largest bin in the specified range will be used as a starting guess for the peak location.
    '''
    #turn off plotting so there aren't a bunch of pop ups
    original_batch_state = ROOT.gROOT.IsBatch()
    ROOT.gROOT.SetBatch(True)
    # Safely check if ddas_run is an iterable of runs or a single run
    try:
        num_workers = min(200, len(ddas_run))
    except TypeError:
        num_workers = 1
        
    hist_to_fit = ddas_interface.get_histogram(ddas_run, binning_for_fit, branch_name, branch_name, branch_name, selection_string, num_workers)

    #  1: Make folder using pathlib for clean cross-platform path handling
    cal_dir = Path(get_calibration_directory(ddas_run, calibration_name, branch_name))
    cal_dir.mkdir(parents=True, exist_ok=True)

    # Initialize lists
    true_energies, true_energy_uncertainties, peak_locations, peak_location_uncertainties = [], [], [], []
    
    # Fit peaks
    for true_energy, true_energy_err, search_window, fit_window in peaks:
        true_energies.append(true_energy)
        true_energy_uncertainties.append(true_energy_err)
        
        location_wiggle = np.max(np.abs(fit_window))/2

        hist_to_fit.GetXaxis().SetRangeUser(*search_window)
        max_bin_in_range = hist_to_fit.GetMaximumBin()        
        location_guess = hist_to_fit.GetXaxis().GetBinCenter(max_bin_in_range)
        hist_to_fit.GetXaxis().UnZoom()
        
        # Assuming fitting_tools is your module, or just call fit_emg_peak directly
        fit_res, background, peak_func, rp, canvas, spectrum_to_plot, f_to_fit, h_fit = fitting_tools.fit_emg_peak(
            hist_to_fit, data_source, location_guess, location_wiggle, (location_guess + fit_window[0], location_guess + fit_window[1])
        )
          
        #  2: Save the fit plot to a pdf, including parameter fit results and p-value
        # ... inside the for loop ...
        mu_val = fit_res.Parameter(2)
        mu_err = fit_res.ParError(2)
        peak_locations.append(mu_val)
        peak_location_uncertainties.append(mu_err)
        p_value = fit_res.Prob()
        
        # Calculate reduced chi-squared for goodness of fit
        chi2_ndf = fit_res.Chi2() / fit_res.Ndf() if fit_res.Ndf() > 0 else 0
        

        # 2: Save the fit plot to a pdf, including parameter fit results and p-value
        canvas.cd()
        upper_pad = rp.GetUpperPad()
        upper_pad.cd()
        
        ROOT.gStyle.SetOptFit(1111)
        
        canvas.Update()
        plot_path = cal_dir / f"fit_{true_energy}keV.pdf"
        canvas.SaveAs(str(plot_path))
    
    #  4: Fit line to peak locations and save to a pkl file
    calibration_results = {}
    
    if len(peaks) == 1:
        # Offset = 0, E = m * ADC
        slope = true_energies[0] / peak_locations[0]
        offset = 0.0
        
        # Error propagation for 1-point slope: sigma_m = m * sqrt((sigma_E/E)^2 + (sigma_mu/mu)^2)
        rel_err_E = true_energy_uncertainties[0] / true_energies[0] if true_energies[0] != 0 else 0
        rel_err_mu = peak_location_uncertainties[0] / peak_locations[0]
        sigma_slope = slope * np.sqrt(rel_err_E**2 + rel_err_mu**2)
        
        calibration_results = {
            'slope': slope, 
            'offset': offset,
            'cov_matrix': {
                'var_offset': 0.0,
                'var_slope': sigma_slope**2,
                'cov_offset_slope': 0.0
            }
        }
        print(f"[{calibration_name}] 1-Point Cal: slope={slope:.4f} ± {sigma_slope:.4f}, offset=0")
        
    else:
        print('fit peak locations: ', peak_locations)
        # Linear fit: y = mx + b  ->  Energy = slope * ADC + offset
        graph = ROOT.TGraphErrors(
            len(peaks), 
            np.array(peak_locations, dtype=np.float64), 
            np.array(true_energies, dtype=np.float64), 
            np.array(peak_location_uncertainties, dtype=np.float64), 
            np.array(true_energy_uncertainties, dtype=np.float64)
        )
        
        graph.SetTitle(f"Energy Calibration: {calibration_name};ADC Channel (#mu);True Energy (keV)")
        graph.SetMarkerStyle(20)
        
        cal_canvas = ROOT.TCanvas(f"c_cal_{calibration_name}", "Calibration Curve", 800, 600)
        graph.Draw("AP")
        
        # Fit a 1st degree polynomial
        cal_fit = ROOT.TF1("cal_fit", "pol1", min(peak_locations)*0.8, max(peak_locations)*1.2)
        
        # "S": Return TFitResultPtr, "Q": Quiet
        cal_fit_res = graph.Fit(cal_fit, "SQ")
        
        offset = cal_fit.GetParameter(0)
        slope = cal_fit.GetParameter(1)
        
        # Extract Covariance Matrix elements
        # Index 0 is parameter 0 (offset), Index 1 is parameter 1 (slope)
        var_offset = cal_fit_res.CovMatrix(0, 0)
        var_slope = cal_fit_res.CovMatrix(1, 1)
        cov_offset_slope = cal_fit_res.CovMatrix(0, 1) # Same as CovMatrix(1, 0)
        
        # Display fit parameters on the plot
        ROOT.gStyle.SetOptFit(1111)
        cal_canvas.Update()
        cal_canvas.SaveAs(str(cal_dir / "calibration_curve.pdf"))
        
        calibration_results = {
            'slope': slope, 
            'offset': offset,
            'cov_matrix': {
                'var_offset': var_offset,
                'var_slope': var_slope,
                'cov_offset_slope': cov_offset_slope
            }
        }
        print(f"[{calibration_name}] Linear Cal: slope={slope:.4f}, offset={offset:.4f}")
        
    # Save parameters to pickle file
    pkl_path = cal_dir / f"{calibration_name}.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(calibration_results, f)
    
    ROOT.gROOT.SetBatch(original_batch_state)

    #save a root file with the following:
    # TTree peak_fit_results which holds fit results for each peak
    # TFitResult for energy calibration ecal_result
    root_path = cal_dir / 'results.root'

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