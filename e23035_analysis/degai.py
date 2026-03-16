from pathlib import Path
import os
import hashlib
import concurrent.futures
import uuid

import pandas as pd
import numpy as np
import ROOT
import tqdm

from raw_viewer import ddas_interface
from track_fitting import srim_interface, build_sim
from e23035_analysis import energy_calibration_tools

def is_iterable_runs(obj):
    if isinstance(obj, (str, bytes)):
        return False
    try:
        iter(obj)
        return True
    except TypeError:
        return False

current_dir = Path(__file__).parent.resolve()
det_loc_table = pd.read_csv(os.path.join(current_dir, 'clarion_det_locations.csv'))

#include clover's used for the experiment here
clover_str_list = []
clover_list = []
for num in range(1, 12):
    if num == 4 or num == 8:
        continue
    for letter in ['a', 'b', 'c', 'd']:
        clover_str_list.append(f'clover_{num}{letter}')
    for i in range(1,5):
        clover_list.append((num, i))
clover_to_index = {clover_list[i]:i for i in range(len(clover_list))}

def get_clover_strings(hist_to_get='e'):
    #get list of strings as they appear in the merged root file
    to_return = []
    for num in [1,2,3,5,6,7,9,10,11]:
        for letter in ('a','b','c','d'):
            to_return.append('clover_%d%s_%s'%(num, letter, hist_to_get))
    return to_return

def get_angle_between_crystals(clover1, crystal1, clover2, crystal2):
    theta1 = np.radians(det_loc_table['thetai'][(det_loc_table['det']==clover1)&(det_loc_table['deti']==crystal1)].iloc[0])
    theta2 = np.radians(det_loc_table['thetai'][(det_loc_table['det']==clover2)&(det_loc_table['deti']==crystal2)].iloc[0])
    phi1 = np.radians(det_loc_table['phii'][(det_loc_table['det']==clover1)&(det_loc_table['deti']==crystal1)].iloc[0])
    phi2 = np.radians(det_loc_table['phii'][(det_loc_table['det']==clover2)&(det_loc_table['deti']==crystal2)].iloc[0])
    rhat1 = [np.sin(theta1)*np.cos(phi1), np.sin(theta1)*np.sin(phi1), np.cos(theta1)]
    rhat2 = [np.sin(theta2)*np.cos(phi2), np.sin(theta2)*np.sin(phi2), np.cos(theta2)]
    return np.degrees(np.arccos(np.clip(np.dot(rhat1, rhat2), -1.0, 1.0)))

def get_adjacency_dict(max_angle):
    to_return = {}
    for clover, crystal  in clover_list:
        to_return[(clover, crystal)] = []
        for clover2, crystal2 in clover_list:
            angle = get_angle_between_crystals(clover, crystal, clover2, crystal2)
            if angle > 0 and angle < max_angle:
                to_return[(clover, crystal)].append((clover2, crystal2))
    return to_return

#adjacency dictionairy which treats crystals as adjacent iff they are in the same clover
clover_adj_dict = {}
for clover, crystal in clover_list:
    clover_adj_dict[(clover, crystal)] = []
    for crystal2 in range(1,5):
        if crystal2 != crystal:
            clover_adj_dict[(clover, crystal)].append((clover, crystal2))

def _worker_cache_crystal_run(run, binning, hist_to_get, cal_name):
    """
    Helper function for the parallel worker. 
    It calls get_crystal_histograms for a single run just to force the 
    underlying get_histogram to write the results to the safe .root cache files.
    It returns only the run number (an integer) to avoid PyROOT pickling segfaults.
    """
    _ = get_crystal_histograms(run, binning, hist_to_get, cal_name, num_workers=1)
    return run

def get_crystal_histograms(ddas_run, binning, hist_to_get, cal_name='', num_workers=None):
    '''
    hist_to_get: c, t, m, e, or cal
    If cal, cal_name must be given, and corresponding cal generated with energy_calibration_tools will be applied.
    '''
    
    # --- MULTIPLE RUNS LOGIC ---
    if is_iterable_runs(ddas_run):
        run_list = list(ddas_run)
        summed_hists = {}
    
        original_batch_mode = ROOT.gROOT.IsBatch()
        ROOT.gROOT.SetBatch(True)
        
        if num_workers is None or num_workers > 1:
            # 1. PARALLEL CACHING: Spawn workers to process the TTrees and populate the disk caches.
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(_worker_cache_crystal_run, run, binning, hist_to_get, cal_name)
                    for run in run_list
                ]
                
                for _ in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(run_list), desc=f"Caching {hist_to_get} (Parallel)"):
                    pass
                    
        # 2. SEQUENTIAL SUMMING: All runs are safely cached on disk now.
        # Reading from the ROOT cache is nearly instantaneous, so we sum them safely in the main thread.
        for run in tqdm.tqdm(run_list, desc=f"Summing {hist_to_get}"):
            # Grab the dict of histograms for this specific run
            run_hists = get_crystal_histograms(run, binning, hist_to_get, cal_name, num_workers=1)
            
            if not summed_hists:
                # For the first run, clone the dictionaries to establish the baseline sum
                for name, hist in run_hists.items():
                    summed_hists[name] = hist.Clone(name)
                    summed_hists[name].SetDirectory(0)
            else:
                # Add subsequent runs
                for name, hist in run_hists.items():
                    summed_hists[name].Add(hist)
        
        ROOT.gROOT.SetBatch(original_batch_mode)
        return summed_hists

    # --- SINGLE RUN LOGIC ---
    to_return = {}
    
    if hist_to_get == 'cal':
        clover_strings = [energy_calibration_tools.get_calibrated_energy_string(ddas_run, cal_name, s) for s in get_clover_strings('c')]
        names = get_clover_strings(cal_name)
    else:
        clover_strings = get_clover_strings(hist_to_get)
        names = clover_strings
        
    for clover_string, name in zip(clover_strings, names):
        # We pass num_workers=1 because the parallelization is now handled by the outer loop above
        to_return[name] = ddas_interface.get_histogram(ddas_run, binning, name, name, clover_string, selection="", force_recreate=False, num_workers=1)
        
    return to_return

def get_summed_gamma_spectrum(ddas_run, binning, cal_name='init'):
    '''
    Sum histograms of individual crystals
    '''
    # 1. Let our newly parallelized function do all the heavy lifting
    if cal_name == 'init': #use calibration applied for merging process
        crystal_hists = get_crystal_histograms(ddas_run, binning, 'e')
    else:
        crystal_hists = get_crystal_histograms(ddas_run, binning, 'cal', cal_name)
        
    # 2. Create a ROOT-safe name for the histogram
    if is_iterable_runs(ddas_run):
        run_list = list(ddas_run)
        hist_name = f'summed_gammas_{run_list[0]}_to_{run_list[-1]}'
    else:
        hist_name = f'summed_gammas_{ddas_run}'
        
    # 3. Create the empty histogram and detach it from memory management
    to_return = ROOT.TH1D(hist_name, 'summed gamma spectrum', *binning)
    to_return.SetDirectory(0)
    
    # 4. Add all the crystal histograms together
    for crystal in crystal_hists:
        to_return.Add(crystal_hists[crystal])
        
    return to_return


def get_addback_tree(ddas_run, adj_dict, cal_name, dt_window_ns):
    '''
    Make a ttree with the gamma ray add back, split into sub-events by time. 
    '''
    cache_dir = os.path.join('e23035_analysis', 'clarion_cache', 'add_back_tree')
    os.makedirs(cache_dir, exist_ok=True)
    
    # --- CRITICAL: Add dt_window_ns to the hash so it generates a new cache! ---
    adj_hash = hashlib.md5((str(adj_dict) + str(dt_window_ns)).encode()).hexdigest()
    cache_file_path = os.path.join(cache_dir, f"{ddas_run}_{adj_hash}_{cal_name}_dt{int(dt_window_ns)}.root")
    
    # Check if root file exists. Load it and return if it does.
    if os.path.exists(cache_file_path):
        try:
            read_file = ROOT.TFile.Open(cache_file_path, 'READ')
            if not read_file or read_file.IsZombie():
                print(f"Warning: Cache file for run {ddas_run} is corrupted. Deleting and recreating...")
                if read_file: read_file.Close()
                os.remove(cache_file_path) 
            else:
                out_tree = read_file.Get('add_back')
                if out_tree: 
                    out_tree._keepalive_file = read_file
                    return out_tree
                else:
                    print(f"Warning: Tree missing in cache for run {ddas_run}. Deleting and recreating...")
                    read_file.Close()
                    os.remove(cache_file_path)
        except OSError:
            print(f"Warning: PyROOT failed to open {cache_file_path}. Deleting and recreating...")
            if os.path.exists(cache_file_path): os.remove(cache_file_path)

    # Load energy calibrations
    slopes, offsets = [], []
    for s in clover_str_list:
        res = energy_calibration_tools.get_calibration_result(ddas_run, cal_name, s+'_c')
        slopes.append(res['slope'])
        offsets.append(res['offset'])
    slopes, offsets = np.array(slopes), np.array(offsets)

    # Set up input file for reading
    infile = ROOT.TFile.Open(ddas_interface.get_merged_root_file_path(ddas_run))
    intree = infile.Get('merged_data')
    
    invals = []
    tvals = [] # NEW: Array to hold timestamps
    for s in clover_str_list:
        invals.append(np.zeros(1, dtype=np.int32))
        tvals.append(np.zeros(1, dtype=np.float64))
        intree.SetBranchAddress(s+'_c', invals[-1])
        intree.SetBranchAddress(s+'_t', tvals[-1]) # Load the timestamp
    
    # Build the add back tree
    cf = ROOT.TFile.Open(cache_file_path, 'RECREATE')
    out_tree = ROOT.TTree('add_back', 'add_back')
    gamma_vec = ROOT.std.vector('double')()
    out_tree.Branch('energy', gamma_vec)

    dt_sec = dt_window_ns * 1e-9 # Convert ns to seconds for comparison

    for ddas_index in tqdm.tqdm(range(intree.GetEntries())):
        intree.GetEntry(ddas_index)
        
        counts = np.array(invals, copy=True).flatten()
        times = np.array(tvals, copy=True).flatten()
        
        energies = np.zeros(len(counts))
        nonzero_mask = counts > 0
        energies[nonzero_mask] = counts[nonzero_mask]*slopes[nonzero_mask] + offsets[nonzero_mask]
        
        fired_indexes = np.where(nonzero_mask)[0].tolist()
        
        if len(fired_indexes) == 0:
            continue

        # --- NEW TIMING CLUSTERING LOGIC ---
        # 1. Pair each fired index with its timestamp and sort them chronologically
        hit_data = [(idx, times[idx]) for idx in fired_indexes]
        hit_data.sort(key=lambda x: x[1])
        
        # 2. Slice into time clusters
        clusters = []
        curr_cluster = [hit_data[0][0]]
        cluster_start_t = hit_data[0][1] # Anchor the window to the first hit in the cluster
        
        for idx, t in hit_data[1:]:
            if (t - cluster_start_t) <= dt_sec:
                curr_cluster.append(idx)
            else:
                clusters.append(curr_cluster)   # Save the completed cluster
                curr_cluster = [idx]            # Start a new cluster
                cluster_start_t = t             # Reset the anchor time
        clusters.append(curr_cluster) # Don't forget the last one!

        # 3. Process Add-back for EACH time cluster independently
        for cluster_idx_list in clusters:
            gamma_vec.clear()
            unprocessed = cluster_idx_list.copy()
            
            while len(unprocessed) > 0:
                indexes_to_add_to_this_event = [unprocessed.pop()]
                indexes_in_this_event = []
                
                while len(indexes_to_add_to_this_event) > 0:
                    i = indexes_to_add_to_this_event.pop()
                    indexes_in_this_event.append(i)
                    adj_clovers = adj_dict[clover_list[i]]
                    
                    for clover in adj_clovers:
                        c_idx = clover_to_index[clover]
                        if c_idx in unprocessed:
                            indexes_to_add_to_this_event.append(c_idx)
                            unprocessed.remove(c_idx)
                            
                gamma_vec.push_back(np.sum(energies[indexes_in_this_event]))
            
            # Fill the tree ONCE PER TIME CLUSTER (Sub-event)
            if gamma_vec.size() > 0:
                out_tree.Fill()

    cf.Write()
    cf.Close()

    read_file = ROOT.TFile.Open(cache_file_path, 'READ')
    out_tree = read_file.Get('add_back')
    out_tree._keepalive_file = read_file
    return out_tree

def get_addback_spectrum(ddas_run, adj_dict, cal_name, binning, dt_window_ns, max_workers=None):
    """
    Generates a 1D add-back energy spectrum for a single run or list of runs.
    Utilizes multiprocessing for multiple runs, caching both individual and combined results.
    Includes time-clustering to prevent accidental coincidences.
    """
    # ---------------------------------------------------------
    # 1. PARALLEL RUN PROCESSING & COMBINED CACHING
    # ---------------------------------------------------------
    if not isinstance(ddas_run, str):
        try:
            _ = iter(ddas_run)
            
            # --- Setup Combined Hash and Paths ---
            sorted_runs = sorted(list(ddas_run))
            # CRITICAL: Include dt_window_ns in the hash!
            combined_hash_str = hashlib.md5(
                (str(sorted_runs) + cal_name + str(binning) + str(adj_dict) + str(dt_window_ns)).encode()
            ).hexdigest()
            
            cache_dir = os.path.join('e23035_analysis', 'clarion_cache', 'histograms_1d')
            os.makedirs(cache_dir, exist_ok=True)
            
            combined_hist_name = f"h1_combined_{combined_hash_str}"
            combined_cache_file = os.path.join(cache_dir, f"h1_combined_{combined_hash_str}.root")
            
            # --- Check the Combined Cache ---
            if os.path.exists(combined_cache_file):
                try:
                    cf = ROOT.TFile.Open(combined_cache_file, 'READ')
                    if not cf or cf.IsZombie():
                        print("Warning: Combined 1D cache is corrupted. Recreating...")
                        if cf: cf.Close()
                        os.remove(combined_cache_file)
                    else:
                        hist = cf.Get(combined_hist_name)
                        # Ensure it's a true 1D histogram
                        if isinstance(hist, ROOT.TH1) and hist.GetDimension() == 1:
                            hist.SetDirectory(0)
                            cf.Close()
                            print('loaded from cache:', combined_cache_file)
                            return hist
                        cf.Close()
                        os.remove(combined_cache_file)
                except OSError:
                    print("Warning: Failed to open combined 1D cache. Recreating...")
                    if os.path.exists(combined_cache_file):
                        os.remove(combined_cache_file)

            # --- Not in Combined Cache: Farm out to Multiprocessing ---
            total_hist = None
            
            # Save the current state and force Batch Mode to prevent X11 crashes
            original_batch_mode = ROOT.gROOT.IsBatch()
            ROOT.gROOT.SetBatch(True)
            
            try:
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        # Pass dt_window_ns down to the workers
                        executor.submit(get_addback_spectrum, run, adj_dict, cal_name, binning, dt_window_ns) 
                        for run in ddas_run
                    ]
                    
                    for future in concurrent.futures.as_completed(futures):
                        h = future.result() 
                        
                        if total_hist is None:
                            total_hist = h.Clone(combined_hist_name)
                            total_hist.SetTitle(f"Addback Energy (Runs: {sorted_runs[0]}...{sorted_runs[-1]}, dt={dt_window_ns}ns);Energy (keV);Counts")
                            total_hist.SetDirectory(0)
                        else:
                            total_hist.Add(h)
            finally:
                # ALWAYS restore the original graphics state
                ROOT.gROOT.SetBatch(original_batch_mode)
            
            # --- Save the Combined Cache ---
            if total_hist:
                cf = ROOT.TFile.Open(combined_cache_file, 'RECREATE')
                total_hist.Write()
                cf.Close()
                
            return total_hist
                
        except TypeError:
            pass # It's a single run, proceed below

    # ---------------------------------------------------------
    # 2. SINGLE RUN PROCESSING
    # ---------------------------------------------------------
    ddas_run = int(ddas_run)
    # CRITICAL: Include dt_window_ns in the single-run hash!
    hash_str = hashlib.md5((str(ddas_run) + cal_name + str(binning) + str(adj_dict) + str(dt_window_ns)).encode()).hexdigest()
    cache_dir = os.path.join('e23035_analysis', 'clarion_cache', 'histograms_1d')
    os.makedirs(cache_dir, exist_ok=True)
    
    hist_name = f"h1_{hash_str}"
    cache_file_path = os.path.join(cache_dir, f"h1_{hash_str}.root")
    
    # --- Check Cache ---
    if os.path.exists(cache_file_path):
        try:
            read_file = ROOT.TFile.Open(cache_file_path, 'READ')
            
            if not read_file or read_file.IsZombie():
                if read_file: read_file.Close()
                os.remove(cache_file_path)
            else:
                hist = read_file.Get(hist_name)
                if isinstance(hist, ROOT.TH1) and hist.GetDimension() == 1: 
                    hist.SetDirectory(0)
                    read_file.Close()
                    return hist
                read_file.Close()
                os.remove(cache_file_path)
        except OSError:
            if os.path.exists(cache_file_path):
                os.remove(cache_file_path)

    # --- Generate Histogram ---
    # Pass dt_window_ns down to the tree builder!
    tree = get_addback_tree(ddas_run, adj_dict, cal_name, dt_window_ns=dt_window_ns)
    df = ROOT.RDataFrame(tree)
    
    h1_ptr = df.Histo1D(
        (hist_name, f"Addback Energy (Run {ddas_run}, dt={dt_window_ns}ns);Energy (keV);Counts", *binning), 
        "energy"
    )
    
    hist = h1_ptr.GetValue()
    hist.SetDirectory(0)
    
    # --- Write Cache ---
    cf = ROOT.TFile.Open(cache_file_path, 'RECREATE')
    hist.Write()
    cf.Close()

    return hist

ROOT.gInterpreter.Declare("""
#include <ROOT/RVec.hxx>

struct CoincPairs {
    std::vector<double> x;
    std::vector<double> y;
};

CoincPairs get_symmetric_pairs(const ROOT::RVec<double>& energies) {
    CoincPairs pairs;
    
    // Only process events with at least 2 gammas
    if (energies.size() < 2) return pairs; 
    
    // Create all symmetric combinations (i != j)
    for (size_t i = 0; i < energies.size(); ++i) {
        for (size_t j = 0; j < energies.size(); ++j) {
            if (i == j) continue; // Don't pair a gamma with itself
            pairs.x.push_back(energies[i]);
            pairs.y.push_back(energies[j]);
        }
    }
    return pairs;
}
""")

def get_addback_coincidence_spectrum(ddas_run, adj_dict, cal_name, binning, dt_window_ns, max_workers=None):
    # ---------------------------------------------------------
    # 1. PARALLEL RUN PROCESSING & COMBINED CACHING
    # ---------------------------------------------------------
    if not isinstance(ddas_run, str):
        try:
            _ = iter(ddas_run)
            
            sorted_runs = sorted(list(ddas_run))
            # CRITICAL: Include dt_window_ns in the hash!
            combined_hash_str = hashlib.md5(
                (str(sorted_runs) + cal_name + str(binning) + str(adj_dict) + str(dt_window_ns)).encode()
            ).hexdigest()
            
            cache_dir = os.path.join('e23035_analysis', 'clarion_cache', 'histograms')
            os.makedirs(cache_dir, exist_ok=True)
            
            combined_hist_name = f"gg_combined_{combined_hash_str}"
            combined_cache_file = os.path.join(cache_dir, f"gg_combined_{combined_hash_str}.root")
            
            if os.path.exists(combined_cache_file):
                try:
                    cf = ROOT.TFile.Open(combined_cache_file, 'READ')
                    if not cf or cf.IsZombie():
                        if cf: cf.Close()
                        os.remove(combined_cache_file)
                    else:
                        hist = cf.Get(combined_hist_name)
                        if isinstance(hist, ROOT.TH2): 
                            hist.SetDirectory(0)
                            cf.Close()
                            print('loaded from cache:', combined_cache_file)
                            return hist
                        cf.Close()
                        os.remove(combined_cache_file)
                except OSError:
                    if os.path.exists(combined_cache_file): os.remove(combined_cache_file)

            total_hist = None
            
            original_batch_mode = ROOT.gROOT.IsBatch()
            ROOT.gROOT.SetBatch(True)
            try:
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        # Pass dt_window_ns down to the workers
                        executor.submit(get_addback_coincidence_spectrum, run, adj_dict, cal_name, binning, dt_window_ns) 
                        for run in ddas_run
                    ]
                    
                    for future in concurrent.futures.as_completed(futures):
                        h = future.result() 
                        if total_hist is None:
                            total_hist = h.Clone(combined_hist_name)
                            total_hist.SetTitle(f"Gamma-Gamma Coincidence (Runs: {sorted_runs[0]}...{sorted_runs[-1]}, dt={dt_window_ns}ns)")
                            total_hist.SetDirectory(0)
                        else:
                            total_hist.Add(h)
            finally:
                ROOT.gROOT.SetBatch(original_batch_mode)
                        
            if total_hist:
                cf = ROOT.TFile.Open(combined_cache_file, 'RECREATE')
                total_hist.Write()
                cf.Close()
                        
            return total_hist
            
        except TypeError:
            pass 

    # ---------------------------------------------------------
    # 2. SINGLE RUN PROCESSING
    # ---------------------------------------------------------
    ddas_run = int(ddas_run)
    hash_str = hashlib.md5((str(ddas_run) + cal_name + str(binning) + str(adj_dict) + str(dt_window_ns)).encode()).hexdigest()
    cache_dir = os.path.join('e23035_analysis', 'clarion_cache', 'histograms')
    os.makedirs(cache_dir, exist_ok=True)
    
    hist_name = f"gg_{hash_str}"
    cache_file_path = os.path.join(cache_dir, f"gg_{hash_str}.root")
    
    if os.path.exists(cache_file_path):
        try:
            cf = ROOT.TFile.Open(cache_file_path, 'READ')
            if not cf or cf.IsZombie():
                if cf: cf.Close()
                os.remove(cache_file_path)
            else:
                hist = cf.Get(hist_name)
                if isinstance(hist, ROOT.TH2): 
                    hist.SetDirectory(0)
                    cf.Close()
                    return hist
                cf.Close()
                os.remove(cache_file_path)
        except OSError:
            if os.path.exists(cache_file_path): os.remove(cache_file_path)

    # Pass the window down to the tree builder!
    tree = get_addback_tree(ddas_run, adj_dict, cal_name, dt_window_ns=dt_window_ns)
    df = ROOT.RDataFrame(tree)
    df = df.Define("pairs", "get_symmetric_pairs(energy)")
    df = df.Define("energy_x", "pairs.x")
    df = df.Define("energy_y", "pairs.y")

    h2_matrix = df.Histo2D(
        (hist_name, f"Gamma-Gamma Coincidence (Run {ddas_run});Energy 1 (keV);Energy 2 (keV)", 
        *binning, *binning), 
        "energy_x", 
        "energy_y"
    )
    
    hist = h2_matrix.GetValue()
    hist.SetDirectory(0) 
    
    cf = ROOT.TFile.Open(cache_file_path, 'RECREATE')
    hist.Write() 
    cf.Close()

    return hist

def get_gated_projection(h2_matrix, gate_energy, gate_width):
    """
    Slices a 2D gamma-gamma matrix along the Y-axis at a specific energy
    and projects it onto the X-axis to create a 1D coincidence spectrum.
    
    Parameters:
    - h2_matrix: The TH2D ROOT object to slice.
    - gate_energy: The center of the peak to gate on (in keV).
    - gate_width: The +/- range around the peak to include (in keV).
    """
    h2_matrix.GetXaxis().UnZoom()
    h2_matrix.GetYaxis().UnZoom()

    # 1. Define the physical energy boundaries of the gate
    energy_min = gate_energy - gate_width
    energy_max = gate_energy + gate_width
    
    # 2. Convert physical energies to ROOT bin numbers
    y_axis = h2_matrix.GetYaxis()
    bin_min = y_axis.FindBin(energy_min)
    bin_max = y_axis.FindBin(energy_max)
    
    # 3. Create a strictly unique name for ROOT's internal memory registry
    # uuid.uuid4().hex[:8] generates a random 8-character string 
    unique_name = f"proj_{gate_energy}keV_{uuid.uuid4().hex[:8]}"
    
    # 4. Project the slice onto the X-axis
    h1_proj = h2_matrix.ProjectionX(unique_name, bin_min, bin_max)
    
    # 5. Make it look nice
    h1_proj.SetTitle(f"Coincidence Spectrum gated on {gate_energy} #pm {gate_width} keV;Energy (keV);Counts / Bin")
    h1_proj.SetLineColor(ROOT.kBlue + 1)
    h1_proj.SetLineWidth(2)
    
    # 6. Detach from ROOT's garbage collector so it safely survives the return
    h1_proj.SetDirectory(0)
    
    return h1_proj

def get_bg_subtracted_projection(h2_matrix, peak_energy, peak_width, bg_energy, bg_width):
    """
    Slices a 2D matrix at a peak, slices it again at a background region, 
    scales the background, and subtracts it to return a clean 1D spectrum.
    """
    h2_matrix.GetXaxis().UnZoom()
    h2_matrix.GetYaxis().UnZoom()

    y_axis = h2_matrix.GetYaxis()
    
    # 1. Project the Peak + Background
    p_min = y_axis.FindBin(peak_energy - peak_width)
    p_max = y_axis.FindBin(peak_energy + peak_width)
    name_peak = f"proj_{peak_energy}keV_{uuid.uuid4().hex[:8]}"
    h1_peak = h2_matrix.ProjectionX(name_peak, p_min, p_max)
    
    # 2. Project the pure Background
    b_min = y_axis.FindBin(bg_energy - bg_width)
    b_max = y_axis.FindBin(bg_energy + bg_width)
    name_bg = f"bg_{bg_energy}keV_{uuid.uuid4().hex[:8]}"
    h1_bg = h2_matrix.ProjectionX(name_bg, b_min, b_max)
    
    # 3. Calculate the scaling factor
    # We use the number of bins spanned by the gates to be perfectly exact
    n_bins_peak = (p_max - p_min) + 1
    n_bins_bg = (b_max - b_min) + 1
    scale_factor = n_bins_peak / n_bins_bg
    
    # Scale the background projection
    h1_bg.Scale(scale_factor)
    
    # 4. Perform the subtraction: h1_peak = h1_peak + (-1 * h1_bg)
    h1_peak.Add(h1_bg, -1)
    
    # 5. Clean up the styling
    h1_peak.SetTitle(f"Clean Coincidence (Gated {peak_energy} keV, BG Subtracted);Energy (keV);Counts / Bin")
    h1_peak.SetLineColor(ROOT.kRed + 1)
    h1_peak.SetLineWidth(2)
    
    # Detach from ROOT directory to prevent garbage collection
    h1_peak.SetDirectory(0)
    
    return h1_peak

import os
import hashlib
import numpy as np
import ROOT

def get_adjacent_timing_spectrum(ddas_run, adj_dict, binning):
    """
    Creates a histogram of time differences (in nanoseconds) between 
    adjacent crystals that fired in the same event, with custom binning.
    """
    # ---------------------------------------------------------
    # 1. SETUP & CACHE CHECKING
    # ---------------------------------------------------------
    ddas_run = int(ddas_run)
    # CRITICAL: Add str(binning) to the hash so different binnings don't overwrite each other!
    hash_str = hashlib.md5((str(ddas_run) + str(binning) + str(adj_dict)).encode()).hexdigest()
    
    cache_dir = os.path.join('e23035_analysis', 'clarion_cache', 'timing')
    os.makedirs(cache_dir, exist_ok=True)
    
    hist_name = f"dt_{hash_str}"
    cache_file_path = os.path.join(cache_dir, f"dt_{hash_str}.root")
    
    if os.path.exists(cache_file_path):
        try:
            cf = ROOT.TFile.Open(cache_file_path, 'READ')
            if not cf or cf.IsZombie():
                if cf: cf.Close()
                os.remove(cache_file_path)
            else:
                hist = cf.Get(hist_name)
                if isinstance(hist, ROOT.TH1) and hist.GetDimension() == 1: 
                    hist.SetDirectory(0)
                    cf.Close()
                    return hist
                cf.Close()
                os.remove(cache_file_path)
        except OSError:
            if os.path.exists(cache_file_path): os.remove(cache_file_path)

    # ---------------------------------------------------------
    # 2. READ THE RAW TREE
    # ---------------------------------------------------------
    in_filepath = ddas_interface.get_merged_root_file_path(ddas_run)
    infile = ROOT.TFile.Open(in_filepath, 'READ')
    if not infile or infile.IsZombie():
        raise RuntimeError(f"Cannot open {in_filepath}")
        
    intree = infile.Get('merged_data')
    
    counts_vals, time_vals = [], []
    for s in clover_str_list:
        c_val = np.zeros(1, dtype=np.int32)
        t_val = np.zeros(1, dtype=np.float64) 
        
        intree.SetBranchAddress(s + '_c', c_val)
        intree.SetBranchAddress(s + '_t', t_val)
        
        counts_vals.append(c_val)
        time_vals.append(t_val)

    # Unpack the custom binning using *binning
    hist = ROOT.TH1D(
        hist_name, 
        f"Adjacent Crystal Time Difference (Run {ddas_run});#Delta t (ns);Counts / bin", 
        *binning
    )

    # ---------------------------------------------------------
    # 3. PROCESS EVENTS
    # ---------------------------------------------------------
    entries = intree.GetEntries()
    for evt in tqdm.tqdm(range(entries)):
        intree.GetEntry(evt)
        
        c_arr = np.array(counts_vals, copy=True).flatten()
        t_arr = np.array(time_vals, copy=True).flatten()
        
        fired_indexes = np.where(c_arr > 0)[0]
        
        if len(fired_indexes) < 2:
            continue
            
        # ITERATE WITHOUT DOUBLE COUNTING (Combinations, not Permutations)
        for i in range(len(fired_indexes)):
            idx1 = fired_indexes[i]
            
            # Start the inner loop one index past the current outer loop index
            for j in range(i + 1, len(fired_indexes)):
                idx2 = fired_indexes[j]
                
                name1 = clover_list[idx1]
                name2 = clover_list[idx2]
                
                if name2 in adj_dict[name1]:
                    # Take the absolute time difference
                    dt_ns = abs(t_arr[idx1] - t_arr[idx2]) * 1e9
                    hist.Fill(dt_ns)

    hist.SetDirectory(0)
    infile.Close()

    # ---------------------------------------------------------
    # 4. SAVE CACHE
    # ---------------------------------------------------------
    cf = ROOT.TFile.Open(cache_file_path, 'RECREATE')
    hist.Write()
    cf.Close()

    return hist