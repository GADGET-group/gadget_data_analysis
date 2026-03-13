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


def get_addback_tree(ddas_run, adj_dict, cal_name):
    '''
    Make a ttree with the gamma ray add back. 
    '''
    #make cache directory
    cache_dir = os.path.join('e23035_analysis', 'clarion_cache', 'add_back_tree')
    os.makedirs(cache_dir, exist_ok=True)
    
    #check if root file exists. Load it and return if it does.
    adj_hash = hashlib.md5(str(adj_dict).encode()).hexdigest()
    cache_file_path = os.path.join(cache_dir, f"{ddas_run}_{adj_hash}_{cal_name}.root")
    # Check if root file exists. Load it and return if it does.
    # Check if root file exists. Load it and return if it does.
    if os.path.exists(cache_file_path):
        try:
            read_file = ROOT.TFile.Open(cache_file_path, 'READ')
            
            if not read_file or read_file.IsZombie():
                print(f"Warning: Cache file for run {ddas_run} is corrupted (Zombie). Deleting and recreating...")
                if read_file: 
                    read_file.Close()
                os.remove(cache_file_path)  # Actively delete the bad file
                
            else:
                out_tree = read_file.Get('add_back')
                if out_tree: 
                    out_tree._keepalive_file = read_file
                    return out_tree
                else:
                    print(f"Warning: Tree missing in cache for run {ddas_run}. Deleting and recreating...")
                    read_file.Close()
                    os.remove(cache_file_path)  # Actively delete the bad file
                    
        except OSError:
            # PyROOT raises this if the file is 0-bytes or completely unreadable C-side
            print(f"Warning: PyROOT failed to open {cache_file_path}. It is likely empty. Deleting and recreating...")
            os.remove(cache_file_path)
            # Code naturally falls through to recreate the file

    #load energy calibrations
    slopes, offsets = [], []
    for s in clover_str_list:
        res = energy_calibration_tools.get_calibration_result(ddas_run, cal_name, s+'_c')
        slopes.append(res['slope'])
        offsets.append(res['offset'])
    slopes, offsets = np.array(slopes), np.array(offsets)

    #cached result doens't exist. Set up input file for reading
    infile = ROOT.TFile.Open(ddas_interface.get_merged_root_file_path(ddas_run))
    intree = infile.Get('merged_data')
    invals = []
    for s in clover_str_list:
        invals.append(np.zeros(1, dtype=np.int32))
        intree.SetBranchAddress(s+'_c', invals[-1])
    
    # Build the add back tree
    cf = ROOT.TFile.Open(cache_file_path, 'RECREATE')
    out_tree = ROOT.TTree('add_back', 'add_back')
    gamma_vec = ROOT.std.vector('double')()
    out_tree.Branch('energy', gamma_vec)

    debug = False
    for ddas_index in tqdm.tqdm(range(intree.GetEntries())):
        if debug:
            print('processing event %d'%ddas_index)
        #load gamma ray energies for event and apply energy calibraiton 
        intree.GetEntry(ddas_index)
        counts = np.array(invals, copy=True).flatten()
        energies = np.zeros(len(counts))
        nonzero_mask = counts>0
        energies[nonzero_mask] = counts[nonzero_mask]*slopes[nonzero_mask] + offsets[nonzero_mask]
        gamma_vec.clear()

        #choose which gammas are from the same original photon based on adj_dict
        fired_indexes = np.where(nonzero_mask)[0].tolist()
        if debug:
            print('the following crystals fired:', [clover_list[i] for i in fired_indexes], '\n')
        while len(fired_indexes) >0:
            indexes_to_add_to_this_event = [fired_indexes.pop()]
            indexes_in_this_event = []
            while len(indexes_to_add_to_this_event) > 0:
                i = indexes_to_add_to_this_event.pop()
                indexes_in_this_event.append(i)
                adj_clovers = adj_dict[clover_list[i]]
                for clover in adj_clovers:
                    if clover_to_index[clover] in fired_indexes:
                        indexes_to_add_to_this_event.append(clover_to_index[clover])
                        fired_indexes.remove(clover_to_index[clover])
            if debug:
                print('summing gammas from the following crystals: ', [clover_list[i] for i in indexes_in_this_event], '\n')
            gamma_vec.push_back(np.sum(energies[indexes_in_this_event]))
        out_tree.Fill()

    #save the tree, close the file, and return the tree
    # 1. Write and close the RECREATE file to safely flush all buffers
    cf.Write()
    cf.Close()

    # 2. Re-open the file in READ mode
    read_file = ROOT.TFile.Open(cache_file_path, 'READ')
    out_tree = read_file.Get('add_back')

    # 3. Attach the file to the tree so it doesn't get garbage collected!
    out_tree._keepalive_file = read_file
    
    return out_tree

def get_addback_spectrum(ddas_run, adj_dict, cal_name, binning, max_workers=None):
    """
    Generates a 1D add-back energy spectrum for a single run or list of runs.
    Utilizes multiprocessing for multiple runs and caches individual results.
    """
    # ---------------------------------------------------------
    # 1. PARALLEL RUN PROCESSING
    # ---------------------------------------------------------
    if not isinstance(ddas_run, str):
        try:
            _ = iter(ddas_run)
            total_hist = None
            
            # 1. Save the current state and force Batch Mode
            original_batch_mode = ROOT.gROOT.IsBatch()
            ROOT.gROOT.SetBatch(True)
            
            try:
                # 2. Run the multiprocessing safely without X11 crashes
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(get_addback_coincidence_spectrum, run, adj_dict, cal_name, binning) 
                        for run in ddas_run
                    ]
                    
                    for future in concurrent.futures.as_completed(futures):
                        h = future.result() 
                        
                        if total_hist is None:
                            hash_str = hashlib.md5(str(ddas_run).encode()).hexdigest()
                            total_hist = h.Clone(f"gg_combined_{hash_str}")
                            total_hist.SetDirectory(0)
                        else:
                            total_hist.Add(h)
                            
                return total_hist
                
            finally:
                # 3. ALWAYS restore the original graphics state, even if an error occurs above
                ROOT.gROOT.SetBatch(original_batch_mode)
                
        except TypeError:
            pass # It's a single run, proceed below
    # ---------------------------------------------------------
    # 2. SINGLE RUN CACHE CHECKING
    # ---------------------------------------------------------
    # I've put this in a separate 'histograms_1d' folder to keep it 
    # cleanly separated from your 2D matrices.
    hash_str = hashlib.md5((str(ddas_run) + cal_name + str(binning) + str(adj_dict)).encode()).hexdigest()
    cache_dir = os.path.join('e23035_analysis', 'clarion_cache', 'histograms_1d')
    os.makedirs(cache_dir, exist_ok=True)
    
    hist_name = f"h1_addback_{hash_str}"
    cache_file_path = os.path.join(cache_dir, f"{hash_str}.root")
    
    if os.path.exists(cache_file_path):
        try:
            read_file = ROOT.TFile.Open(cache_file_path, 'READ')
            
            if not read_file or read_file.IsZombie():
                print(f"Warning: 1D Cache for run {ddas_run} is corrupted. Recreating...")
                if read_file: 
                    read_file.Close()
                os.remove(cache_file_path)
            else:
                hist = read_file.Get(hist_name)
                if hist: 
                    hist.SetDirectory(0)
                    read_file.Close()
                    return hist
                else:
                    read_file.Close()
                    os.remove(cache_file_path)
                    
        except OSError:
            print(f"Warning: PyROOT failed to open 1D cache {cache_file_path}. Recreating...")
            os.remove(cache_file_path)

    # ---------------------------------------------------------
    # 3. GENERATE HISTOGRAM
    # ---------------------------------------------------------
    # Note: ensure get_addback_tree is accessible in this scope 
    # (e.g., clarion.get_addback_tree if imported)
    tree = get_addback_tree(ddas_run, adj_dict, cal_name)
    df = ROOT.RDataFrame(tree)
    
    # RDataFrame automatically flattens the std::vector 'energy' 
    h1_ptr = df.Histo1D(
        (hist_name, f"Addback Energy (Run {ddas_run});Energy (keV);Counts", *binning), 
        "energy"
    )
    
    # Force evaluation
    hist = h1_ptr.GetValue()
    hist.SetDirectory(0)
    
    # ---------------------------------------------------------
    # 4. WRITE CACHE
    # ---------------------------------------------------------
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

def get_addback_coincidence_spectrum(ddas_run, adj_dict, cal_name, binning, max_workers=None):
    # ---------------------------------------------------------
    # 1. PARALLEL RUN PROCESSING (If an iterable of runs is passed)
    # ---------------------------------------------------------
    if not isinstance(ddas_run, str):
        try:
            _ = iter(ddas_run)
            total_hist = None
            
            # Farm out the individual runs to multiple CPU processes
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all runs to the executor pool
                futures = [
                    executor.submit(get_addback_coincidence_spectrum, run, adj_dict, cal_name, binning) 
                    for run in ddas_run
                ]
                
                # As each process finishes its run, grab the histogram and add it
                for future in concurrent.futures.as_completed(futures):
                    h = future.result() # Grabs the TH2D from the worker process
                    
                    if total_hist is None:
                        # Clone the first completed histogram to establish the base
                        hash_str = hashlib.md5(str(ddas_run).encode()).hexdigest()
                        total_hist = h.Clone(f"gg_combined_{hash_str}")
                        total_hist.SetDirectory(0)
                    else:
                        total_hist.Add(h)
                        
            return total_hist
            
        except TypeError:
            pass # It's a single run (e.g., an int), drop down to the main logic

    # ---------------------------------------------------------
    # 2. SINGLE RUN PROCESSING (The main logic)
    # ---------------------------------------------------------
    ddas_run = int(ddas_run)
    hash_str = hashlib.md5((str(ddas_run) + cal_name + str(binning) + str(adj_dict)).encode()).hexdigest()
    cache_dir = os.path.join('e23035_analysis', 'clarion_cache', 'histograms')
    os.makedirs(cache_dir, exist_ok=True)
    
    hist_name = f"gg_{hash_str}"
    cache_file_path = os.path.join(cache_dir, f"{hash_str}.root")
    
    # Check cache first
    if os.path.exists(cache_file_path):
        cf = ROOT.TFile.Open(cache_file_path, 'READ')
        hist = cf.Get(hist_name)
        if hist: 
            hist.SetDirectory(0)
            cf.Close()
            return hist
        cf.Close() 

    # Not in cache: build the DataFrame
    df = ROOT.RDataFrame(get_addback_tree(ddas_run, adj_dict, cal_name))
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
    
    # Write to cache
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

