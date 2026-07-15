import numpy as np
import ROOT
from e23035_analysis import e23035_runs, root_vis_tools, spectrum_fitter
from raw_viewer import ddas_interface

experiment = 'e23035'

def show_stability_hist(runs, selection_str, scale_factors=None, binning=(1000, 0, 10000), title_prefix='Particle'):
    hist_dict = {}
    
    print(f"Found {len(runs)} runs to process for {title_prefix}.")
    
    if scale_factors is not None and len(scale_factors) != len(runs):
        print("Warning: length of scale_factors does not match length of runs")
    
    # Pre-caching only works easily if the expression is the same for all. 
    if scale_factors is None:
        print("Pre-caching histograms in parallel...")
        try:
            ddas_interface.get_histogram(
                experiment, 
                runs, 
                binning, 
                "dummy_name", 
                "dummy_title", 
                "tpc_energy", 
                selection=selection_str,
                num_workers=100
            )
        except Exception as e:
            print(f"Pre-caching encountered an issue: {e}")

    for i, run in enumerate(runs):
        hist_name = f'{title_prefix.lower()}_hist_{run}'
        hist_title = f'Run {run} {title_prefix} Energy'
        
        expr = "tpc_energy"
        if scale_factors is not None:
            expr = f"tpc_energy*{scale_factors[i]}"
            
        try:
            hist = ddas_interface.get_histogram(
                experiment, 
                run, 
                binning, 
                hist_name, 
                hist_title, 
                expr, 
                selection=selection_str
            )
            if hist:
                hist_dict[run] = hist
        except Exception as e:
            print(f"Failed to process run {run}: {e}")
    
    canvas, th2 = root_vis_tools.create_2d_hist_from_dict(hist_dict, title=f"{title_prefix} Energies by Run", y_label="Run Number")
    
    if canvas:
        save_name = f"{title_prefix.lower()}_energy_spectrum.png"
        canvas.SaveAs(save_name)
        print(f"Saved {save_name}")
        
        return canvas, th2, hist_dict

def _get_scale_factor_worker(args):
    import numpy as np
    from scipy.interpolate import interp1d
    from scipy.optimize import minimize_scalar
    
    energies, fit_edges, ref_bin_centers, reference_spectrum = args
    if len(energies) == 0:
        return 1.0
        
    ref_sum = np.sum(reference_spectrum)
    if ref_sum == 0:
        return 1.0
        
    fit_bin_centers = (fit_edges[:-1] + fit_edges[1:]) / 2
    
    # Histogram the unscaled energies ONCE to fix the counts S in the fiducial fit window.
    # This avoids discrete step-functions in the optimizer when s changes.
    S, _ = np.histogram(energies, bins=fit_edges)
    S_sum = np.sum(S)
    if S_sum == 0:
        return 1.0
        
    # Create a smooth continuous function for the reference spectrum (using wide binning)
    reference_interp = interp1d(ref_bin_centers, reference_spectrum, kind='linear', bounds_error=False, fill_value=0)
    
    # Pre-mask to only compute log on bins that have observed counts
    mask = S > 0
    S_masked = S[mask]
    fit_bin_centers_masked = fit_bin_centers[mask]
    
    def obj_func(s):
        # To align scaled run energies (E_raw * s) to E_ref, we evaluate the reference
        # shape at s * E_raw. The expected shape density scales by s (Jacobian).
        shape_all = s * reference_interp(fit_bin_centers * s) + 1e-9
        
        # Normalize so the expected total counts matches the observed total counts
        norm_factor = S_sum / np.sum(shape_all)
        
        # Evaluate expected counts (lambda) just for the masked bins
        shape_masked = s * reference_interp(fit_bin_centers_masked * s) + 1e-9
        lam_masked = shape_masked * norm_factor
        
        # Poisson log-likelihood: sum(S * log(lam) - lam - log(S!))
        # Since S is fixed, log(S!) is constant.
        # Since sum(lam) is normalized to S_sum, it is also constant.
        # We only need to maximize sum(S * log(lam)).
        logL = np.sum(S_masked * np.log(lam_masked))
        
        return -logL
    
    res = minimize_scalar(obj_func, bounds=(0.8, 1.2), method='bounded')
    return res.x

def _load_run_data(args):
    experiment, run, event_selection_mask = args
    import os
    import numpy as np
    import ROOT
    from raw_viewer import ddas_interface
    
    root_file_path = ddas_interface.get_merged_root_file_path(experiment, run)
    if not os.path.exists(root_file_path):
        return run, np.array([])
        
    df = ROOT.RDataFrame('merged_data', root_file_path)
    if event_selection_mask:
        df = df.Filter(event_selection_mask)
        
    energies = df.AsNumpy(['tpc_energy'])['tpc_energy']
    return run, energies

def align_runs(runs, alignment_iterations, event_selection_mask='', fit_binning=(3500, 500, 4000), reference_binning=(10000, 0, 10000)):
    import tqdm
    import concurrent.futures
    import multiprocessing
    
    raw_tpc_energies_dict = {}
    print(f"Loading TPC energies for {len(runs)} runs...")
    
    args_list = [(experiment, run, event_selection_mask) for run in runs]
    
    ctx = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(32, len(runs)), mp_context=ctx) as executor:
        for run, energies in tqdm.tqdm(executor.map(_load_run_data, args_list), total=len(runs)):
            if len(energies) == 0:
                print(f"File not found or empty for run {run}, skipping...")
            raw_tpc_energies_dict[run] = energies
            
    # Ensure they are in the same order as the input 'runs'
    raw_tpc_energies = [raw_tpc_energies_dict[run] for run in runs]
        
    return align_tpc_energies(raw_tpc_energies, alignment_iterations, reference_spectrum=None, fit_binning=fit_binning, reference_binning=reference_binning)

def align_tpc_energies(raw_tpc_energies, alignment_iterations, reference_spectrum=None, fit_binning=(3500, 500, 4000), reference_binning=(10000, 0, 10000)):
    '''
    Maximize probability individual spectra are drawn from the same probability distribution as the reference spectrum by 
    scaling tpc_energy.
    If alignment_iterations>1, reperform this process and update reference_spectrum to the sum of scaled_spectra.
    We assume scale_factors will all be close to 1.
    '''
    import concurrent.futures
    import tqdm
    
    ref_edges = np.linspace(reference_binning[1], reference_binning[2], reference_binning[0] + 1)
    ref_bin_centers = (ref_edges[:-1] + ref_edges[1:]) / 2
    
    fit_edges = np.linspace(fit_binning[1], fit_binning[2], fit_binning[0] + 1)
    
    current_reference = reference_spectrum
    if current_reference is None: 
        current_reference = np.zeros(reference_binning[0])
        for energies in raw_tpc_energies:
            hist, _ = np.histogram(energies, bins=ref_edges)
            current_reference += hist
            
    scale_factors = np.ones(len(raw_tpc_energies))
    
    for iteration in range(alignment_iterations):
        print(f"--- Alignment Iteration {iteration + 1}/{alignment_iterations} ---")
        
        args_list = [(energies, fit_edges, ref_bin_centers, current_reference) for energies in raw_tpc_energies]
        scale_factors = []
        
        print(f"Finding scale factors sequentially...")
        for args in tqdm.tqdm(args_list):
            scale_factors.append(_get_scale_factor_worker(args))
                
        scale_factors = np.array(scale_factors)
        print(scale_factors)
        
        if iteration < alignment_iterations - 1:
            current_reference = np.zeros(reference_binning[0])
            for sf, energies in zip(scale_factors, raw_tpc_energies):
                hist, _ = np.histogram(energies * sf, bins=ref_edges)
                current_reference += hist
                
    return scale_factors

def get_aligned_spectrum(runs, binning, scale_factors, event_selection_mask='', name='aligned_spectrum', title='Aligned Spectrum'):
    '''
    Takes a list of runs, histogram binning (bins, min, max), and scale factors,
    and returns the corresponding aligned energy spectrum as a ROOT.TH1D.
    '''
    import os
    import ROOT
    import numpy as np

    edges = np.linspace(binning[1], binning[2], binning[0] + 1)
    spectrum = np.zeros(binning[0])
    
    for run, sf in zip(runs, scale_factors):
        root_file_path = ddas_interface.get_merged_root_file_path(experiment, run)
        if not os.path.exists(root_file_path):
            print(f"File not found for run {run}, skipping...")
            continue
            
        df = ROOT.RDataFrame('merged_data', root_file_path)
        if event_selection_mask:
            df = df.Filter(event_selection_mask)
            
        energies = df.AsNumpy(['tpc_energy'])['tpc_energy']
        hist, _ = np.histogram(energies * sf, bins=edges)
        spectrum += hist
        
    th1 = ROOT.TH1D(name, title, binning[0], binning[1], binning[2])
    for i, count in enumerate(spectrum):
        th1.SetBinContent(i + 1, count)
        th1.SetBinError(i + 1, np.sqrt(count))
        
    return th1


if False:
    cp, tp, hp = show_stability_hist(1)
    ca, ta, ha = show_stability_hist(2)

    # runs = np.array(e23035_runs.get_ddas_60_Ga_runs(False, False, False, False, tpc_data_valid=True))
    # runs = runs[runs<150]
    fs = []
    runs=[132, 133, 134]
    for run in runs:
        h = ddas_interface.get_histogram(experiment, run, (350,500,4000), 'Proton Energy Spectrum for 60Ga',
                                        'Run # 60Ga Proton Energy Spectrum', 'tpc_energy',
                                        selection='tpc_particle_id==1', num_workers=100)
        f = spectrum_fitter.spectrum_fitter(h,'bg_shift_gaus')
        sigma_tpc = lambda E: (0.011107*E/1e3 + 0.008813049)*1e3
        f.param_bound_functions['sigma'] = lambda E: (sigma_tpc(E), sigma_tpc(E))
        f.location_wiggle = 10
        f.peaks_to_fit=[(1100, 1030, 1150),(2020, 1920, 2120)]
        f.shared_sigma = False
        f.fit_peaks()
        f.show_fit_results(0)
        fs.append(f)

    c = ROOT.TCanvas()
    h = ddas_interface.get_histogram(experiment, [134], (10, 0, 6000, 350,500,4000), 'proton_energy_vs_time',
                                    'Proton Energy vs Time',
                                    'tpc_energy:mesh_pre_amp_t',
                                    selection='tpc_particle_id==1', num_workers=100)
    h.Draw('colz')

if __name__ == '__main__':
    runs = e23035_runs.get_ddas_60_Ga_runs(good_gamma=False, good_low_energy_tpc=False, good_long_tracks_tpc=False, final_beam_settings=True, tpc_data_valid=True)
    sfs = align_runs(runs, 3, '!tpc_should_veto', fit_binning=(1000,1100,1800), reference_binning=(10000,0,10000))
    c_aligned,t_aligned,h_aligned = show_stability_hist(runs, '!tpc_should_veto', sfs, binning=(900,500,9500))
    c_aligned.SetTitle('aligned')
    c_unaligned,t_unaligned,h_unaligned = show_stability_hist(runs, '!tpc_should_veto', binning=(900,500,9500))
    c_unaligned.SetTitle('unaligned')