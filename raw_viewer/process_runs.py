import os
import uproot
import awkward as ak
import hashlib
import multiprocessing as mp
import subprocess
import socket

from tqdm import tqdm
import numpy as np

from raw_viewer import raw_h5_file

OUTER_RING_PADS = [762,761,760,1015,1016,1017,759,758,757,756,1011,1012,1013,1014,752,751,743,742,732,720,707,693,678,663,
                   647,631,614,597,580,563,545,527,272,290,308,325,342,359,376,392,408,423,438,452,465,477,488,487,497,496,
                   504,503,502,501,507,506,505,250,251,252,246,247,248,249,241,242,232,233,222,210,197,183,168,153,137,121,
                   104,87,70,53,35,17,782,800,818,835,852,869,886,902,918,933,948,962,975,987,998,997]

TPC_PROCESSING_GPUS = [0, 2,3] if socket.gethostname() == 'tpcgpu' else [0]
TPC_MAX_WORKERS_PER_GPU = 2

def get_save_path(experiment):
    if socket.gethostname() == 'tpcgpu':
        if experiment == 'e21072':
            save_path = '/egr/research-tpc/shared/Run_Data/proc_pkl'
        elif experiment == 'e23035_prep_2cobo':
            save_path = '/egr/research-tpc/shared/e23035_prep/2cobo/proc_pkl'
        elif experiment == 'e23035_prep_4cobo':
            save_path = '/egr/research-tpc/shared/e23035_prep/4cobo/proc_pkl'
        elif experiment == 'e23035_prep_vault':
            save_path = '/egr/research-tpc/shared/e23035_prep/vault/proc_pkl'
        else:
            save_path = '/egr/research-tpc/shared/proc_runs/%s/proc_pkl'%experiment
        return save_path
    elif 'gadget' in socket.gethostname().lower():
        if experiment == 'e23035_prep_vault':
            return '/Volumes/Extreme SSD/e23035prepvault/'
        if experiment == 'e25058':
            return '/Volumes/e25058_v1/e25058/proc_pkl'

def get_h5_path(experiment, run_number):
    run_number = int(run_number)
    h5_base_path = ''
    if socket.gethostname() == 'tpcgpu':
        h5_base_path = '/egr/research-tpc/shared/experiments'
    elif 'gadget' in socket.gethostname().lower():
        if experiment == '23035':
            h5_base_path = '/Volumes/Extreme SSD'
        elif experiment == 'e25058':
            h5_base_path = '/Volumes/e25058_v1'
    if experiment == 'e21072':
        return '/egr/research-tpc/shared/Run_Data/run_%04d.h5'%run_number
    elif experiment == 'e23035_prep_2cobo':
        return '/egr/research-tpc/shared/e23035_prep/2Cobo/run_%04d.h5'%run_number
    elif experiment == 'e23035_prep_4cobo':
        return '/egr/research-tpc/shared/e23035_prep/4cobo/run_%04d.h5'%run_number
    elif experiment == 'e23035_prep_vault':
        return '/egr/research-tpc/shared/e23035_prep/vault/run_%04d.h5'%run_number
    else:
        return'%s/%s/h5/run_%04d.h5'%(h5_base_path, experiment, run_number)

def get_h5_file(experiment, run_number, config_filename=""):
    run_number = int(run_number)
    raw_h5_path = get_h5_path(experiment, run_number)
    
    # Defaults in case not all are in config
    h5file = raw_h5_file.raw_h5_file(raw_h5_path, zscale=1.088, flat_lookup_csv='raw_viewer/channel_mappings/flatlookup4cobos.csv')
    
    if config_filename and config_filename != 'none':
        config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'tpc_processing_configs', config_filename)
        if os.path.exists(config_path):
            h5file.load_config(config_path)
        else:
            raise FileNotFoundError(f"Config file '{config_filename}' not found at: {config_path}")
    
    return h5file

_settings_hash_cache = {}
def get_experiment_settings_hash(experiment, example_run, config_filename=""):
    key = (experiment, config_filename)
    if key not in _settings_hash_cache:
        h5file = get_h5_file(experiment, example_run, config_filename)
        _settings_hash_cache[key] = h5file.get_settings_hash()
        h5file.close()
    return _settings_hash_cache[key]

#coppied from field distortions folder in track fitting branch
#and modified to configure h5 file differently
def process_tpc_run(experiment, run_number, force_reprocess=False, config_filename="", gpu_to_use=0):
    '''
    Get information about track direction, width, and charge per pad, which isn't normally stored when processing runs.
    Only redoes processing if a ROOT version of this information isn't available.
    '''
    run_number = int(run_number)
    try:
        import cupy as cp
        cp.cuda.runtime.setDevice(gpu_to_use)
    except ImportError:
        pass
    
    #save_path = os.path.dirname(os.path.abspath(__file__))
    h5file = get_h5_file(experiment, run_number, config_filename)
    settings_hash = h5file.get_settings_hash()
    config_name = os.path.splitext(config_filename)[0]
    fname = os.path.join(get_save_path(experiment), f'{experiment}_run{int(run_number)}_{config_name}.root')
    
    #git info to save
    git_version = subprocess.run(['git', 'rev-parse', '--verify', 'HEAD'], capture_output=True, text=True, check=True).stdout
    git_status = subprocess.run(['git', 'status'], capture_output=True, text=True, check=True).stdout
    git_diff = subprocess.run(['git', 'diff'], capture_output=True, text=True, check=True, errors='replace').stdout

    if force_reprocess:
        import glob
        save_path = get_save_path(experiment)
        cache_patterns = [
            f'gm_ic_{experiment}_run{int(run_number)}_*.npy',
            f'veto_counts_{experiment}_run{int(run_number)}_*.npy',
            f'max_veto_counts_{experiment}_run{int(run_number)}_*.npy',
            f'outer_ring_counts_{experiment}_run{int(run_number)}_*.npy',
            f'max_outer_ring_counts_{experiment}_run{int(run_number)}_*.npy',
            f'veto_mask_{experiment}_run{int(run_number)}_*.npy'
        ]
        for pattern in cache_patterns:
            for cache_file in glob.glob(os.path.join(save_path, pattern)):
                try:
                    os.remove(cache_file)
                    print(f"Cleared cache file: {cache_file}")
                except OSError as e:
                    print(f"Error removing {cache_file}: {e}")
    
    #fname += '.no_neg'
    if os.path.exists(fname) and not force_reprocess:
        print('run %d previously processed, ROOT file exists'%run_number)
        return
    else:
        # h5file = build_sim.get_rawh5_object(experiment, run_number)
        print('processing run %d'%run_number)
        first_event, last_event = h5file.get_event_num_bounds()
        track_centers, principle_axes,variances_along_axes, pad_charges, track_endpoints, charge_widths, width_above_thresholds = [],[],[],[],[],[], []
        pad_maxs, railed_pads = [], []
        for evt in tqdm(range(first_event, last_event + 1)):
            railed_pads.append(h5file.get_railed_pads(evt))
            center, dd,vv = h5file.get_track_axis(evt, return_all_svd_results=True, threshold=h5file.length_counts_threshold)
            xs, ys, zs, es = h5file.get_xyze(evt, threshold=h5file.length_counts_threshold, include_veto_pads=False)
            # pad vv to (3, 3) and dd to (3,) if needed
            if vv.shape[0] < 3:
                padded_vv = np.zeros((3, 3))
                padded_vv[:vv.shape[0], :] = vv
                vv = padded_vv
            if dd.shape[0] < 3:
                padded_dd = np.zeros(3)
                padded_dd[:dd.shape[0]] = dd
                dd = padded_dd
                
            principle_axes.append(vv)
            if len(xs) > 1:
                variances_along_axes.append(dd**2/(len(xs)-1))
            else:
                variances_along_axes.append(np.zeros(3))
            track_centers.append(center)
            pad_counts = np.zeros(1024)
            pad_maxs.append(np.zeros(1024))
            for pad, trace in zip(*h5file.get_pad_traces(evt)):
                pad_counts[pad] = np.sum(trace)
                pad_maxs[-1][pad] = np.max(trace)
            pad_charges.append(pad_counts)


            #get track end points
            if len(xs) > 1:
                points = np.concatenate((xs[:, np.newaxis], 
                        ys[:, np.newaxis], 
                        zs[:, np.newaxis]), 
                        axis=1)
                rbar = points - track_centers[-1]
                track_direction = principle_axes[-1][0]
                rdotv = np.dot(rbar, track_direction)
                #project endpoints onto track axis
                first_point = np.min(rdotv)*track_direction + track_centers[-1]
                last_point = np.max(rdotv)*track_direction + track_centers[-1]
                track_endpoints.append([first_point, last_point])
                #above variance is just variance in postiion of points above some threshold
                #instead calcualte variance along 2nd axis of charge
                width_axis = principle_axes[-1][1]
                total_charge = np.sum(es)
                center_of_charge = np.einsum('i,ij->j',es, points)/total_charge
                displacement_from_center = points - center_of_charge
                displacement_dot_width_axis_squared = np.einsum('ij, j', displacement_from_center, width_axis)**2
                charge_widths.append((np.einsum('i,i', displacement_dot_width_axis_squared, es)/total_charge)**0.5)
                #calculate width in the same way we do length
                rdotv = np.dot(rbar, width_axis)
                width_above_thresholds.append(np.max(rdotv) - np.min(rdotv))

            else:
                track_endpoints.append([(0,0,0), (0,0,0)])
                charge_widths.append(0)
                width_above_thresholds.append(0)
        track_centers = np.array(track_centers).reshape(-1, 3)
        pad_charges = np.array(pad_charges).reshape(-1, 1024)
        pad_maxs = np.array(pad_maxs).reshape(-1, 1024)
        
        ts = h5file.get_timestamps_array()



        res_centers = []
        for c in track_centers:
            try:
                if len(c) == 3:
                    res_centers.append(list(c))
                else:
                    res_centers.append([0.0, 0.0, 0.0])
            except TypeError:
                res_centers.append([0.0, 0.0, 0.0])
                
        res_endpoints = []
        for ep in track_endpoints:
            try:
                if len(ep) == 2 and len(ep[0]) == 3 and len(ep[1]) == 3:
                    res_endpoints.append([list(ep[0]), list(ep[1])])
                else:
                    res_endpoints.append([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
            except TypeError:
                res_endpoints.append([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
                
        events_data = {'track_center':np.array(res_centers, dtype=np.float64).reshape(-1, 3), 'principle_axes':np.array(principle_axes, dtype=np.float64).reshape(-1, 3, 3), 'variance_along_axes': np.array(variances_along_axes, dtype=np.float64).reshape(-1, 3),
                   'pad_charge': pad_charges, 'endpoints':np.array(res_endpoints, dtype=np.float64).reshape(-1, 2, 3), 'charge_width':np.array(charge_widths, dtype=np.float64),
                   'width_above_threshold':np.array(width_above_thresholds, dtype=np.float64), 'pad_max':pad_maxs, 'timestamps':ts}
                   
        counts = np.array([len(x) for x in railed_pads], dtype=np.int64)
        if sum(counts) == 0:
            events_data['railed_pads'] = ak.unflatten(np.array([], dtype=np.int64), counts)
        else:
            events_data['railed_pads'] = ak.from_iter(railed_pads)
        settings_str = h5file.get_settings_str()
        metadata = {'git_version':[git_version], 'git_status':[git_status], 'git_diff':[git_diff], 'settings_hash':[settings_hash], 'settings_json':[settings_str]}
        print('saving to ROOT file')
        with uproot.recreate(fname) as file:
            lengths = [len(v) for v in events_data.values()]
            if not lengths or min(lengths) == 0:
                file['events'] = events_data
            else:
                total_events = min(lengths)
                chunk_size = 1000
                for i in range(0, total_events, chunk_size):
                    chunk = {}
                    for k, v in events_data.items():
                        chunk[k] = v[i:i+chunk_size]
                    
                    if i == 0:
                        file['events'] = chunk
                    else:
                        file['events'].extend(chunk)
            file['metadata'] = metadata

def _load_run_quantities(args):
    if len(args) == 6:
        experiment, run, qnames, settings_hash, config_filename, gpus_to_use = args
        import multiprocessing
        identity = multiprocessing.current_process()._identity
        if identity:
            idx = (identity[0] - 1) % len(gpus_to_use)
            gpu_to_use = gpus_to_use[idx]
        else:
            gpu_to_use = gpus_to_use[0]
    else:
        experiment, run, qnames, settings_hash, config_filename = args
        gpu_to_use = 0
    config_name = os.path.splitext(config_filename)[0]
    fname = os.path.join(get_save_path(experiment), f'{experiment}_run{int(run)}_{config_name}.root')
    if not os.path.exists(fname):
        import multiprocessing
        if multiprocessing.current_process().name != 'MainProcess':
            print(f"WARNING: Processing TPC run {run} directly inside a worker process ({multiprocessing.current_process().name}). This can cause memory issues if too many workers run concurrently. Please call process_runs.ensure_processed() before launching parallel workers.")
        process_tpc_run(experiment, run, config_filename=config_filename, gpu_to_use=gpu_to_use)
        
    result = {}
    with uproot.open(fname) as file:
        file_hash = file['metadata']['settings_hash'].array(library='np')[0]
        if file_hash != settings_hash:
            error_msg = f"Settings hash in file {fname} ({file_hash}) does not match current settings hash ({settings_hash}) for config {config_filename}."
            if 'settings_json' in file['metadata']:
                import json
                file_settings_str = file['metadata']['settings_json'].array(library='np')[0]
                file_settings = json.loads(file_settings_str)
                
                h5file_current = get_h5_file(experiment, run, config_filename)
                current_settings = json.loads(h5file_current.get_settings_str())
                h5file_current.close()
                
                disagreements = []
                for k in current_settings:
                    if k not in file_settings:
                        disagreements.append(f"  {k}: missing in file, current={current_settings[k]}")
                    elif file_settings[k] != current_settings[k]:
                        disagreements.append(f"  {k}: file={file_settings[k]}, current={current_settings[k]}")
                for k in file_settings:
                    if k not in current_settings:
                        disagreements.append(f"  {k}: missing in current, file={file_settings[k]}")
                
                if disagreements:
                    error_msg += "\nDisagreements:\n" + "\n".join(disagreements)
            else:
                error_msg += "\n(Settings details are not stored in this older root file to see the disagreement)."
            raise ValueError(error_msg)
        
        for qname in qnames:
            if qname in file['events']:
                if qname == 'railed_pads':
                    arr = file['events'][qname].array(library='ak')
                    val = ak.to_list(arr)
                    result[qname] = val
                else:
                    arr = file['events'][qname].array(library='np')
                    val = arr
                    result[qname] = val
            elif qname in file['metadata']:
                arr = file['metadata'][qname].array(library='np')
                val = arr[0]
                result[qname] = val
            else:
                raise ValueError(f"Quantity {qname} not found in ROOT file")
    return result

def _worker_ensure_processed(args):
    experiment, run, config_filename, gpus_to_use = args
    import multiprocessing
    identity = multiprocessing.current_process()._identity
    if identity:
        idx = (identity[0] - 1) % len(gpus_to_use)
        gpu_to_use = gpus_to_use[idx]
    else:
        gpu_to_use = gpus_to_use[0]
        
    config_name = os.path.splitext(config_filename)[0]
    fname = os.path.join(get_save_path(experiment), f'{experiment}_run{int(run)}_{config_name}.root')
    if not os.path.exists(fname):
        process_tpc_run(experiment, run, config_filename=config_filename, gpu_to_use=gpu_to_use)

def ensure_processed(experiment, runs, config_filename="", show_progress=True):
    runs = [int(r) for r in runs]
    config_name = os.path.splitext(config_filename)[0]
    
    # Filter for runs that actually need processing
    runs_to_process = []
    for run in runs:
        fname = os.path.join(get_save_path(experiment), f'{experiment}_run{int(run)}_{config_name}.root')
        if not os.path.exists(fname):
            runs_to_process.append(run)
            
    if not runs_to_process:
        return
        
    gpus_to_use = TPC_PROCESSING_GPUS
    max_workers = len(gpus_to_use) * TPC_MAX_WORKERS_PER_GPU
    
    if show_progress:
        print(f"Pre-processing {len(runs_to_process)} TPC runs with {max_workers} workers...")
        
    args_list = [(experiment, run, config_filename, gpus_to_use) for run in runs_to_process]
    
    if max_workers > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=multiprocessing.get_context('spawn')) as executor:
            if show_progress:
                list(tqdm(executor.map(_worker_ensure_processed, args_list), total=len(args_list)))
            else:
                list(executor.map(_worker_ensure_processed, args_list))
    else:
        iterable = tqdm(args_list) if show_progress else args_list
        for args in iterable:
            _worker_ensure_processed(args)

def get_quantity(qname, experiment, runs, show_load_progress=False, num_workers=1, config_filename="", gpus_to_use=None):
    is_single = isinstance(qname, str)
    qnames = [qname] if is_single else qname
    runs = [int(r) for r in runs]
    to_return = {q: [] for q in qnames}
    
    if show_load_progress:
        print(f'loading {qnames} for {runs}')
        
    settings_hash = get_experiment_settings_hash(experiment, runs[0] if runs else 0, config_filename)
    if gpus_to_use is None:
        gpus_to_use = TPC_PROCESSING_GPUS
    args_list = [(experiment, run, qnames, settings_hash, config_filename, gpus_to_use) for i, run in enumerate(runs)]
    
    if num_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=multiprocessing.get_context('spawn')) as executor:
            if show_load_progress:
                results = list(tqdm(executor.map(_load_run_quantities, args_list), total=len(runs)))
            else:
                results = list(executor.map(_load_run_quantities, args_list))
    else:
        results = []
        iterable = tqdm(args_list) if show_load_progress else args_list
        for args in iterable:
            results.append(_load_run_quantities(args))
            
    for res in results:
        for q in qnames:
            if q == 'railed_pads':
                to_return[q].extend(res[q])
            else:
                to_return[q].append(res[q])
                
    final_returns = []
    for q in qnames:
        if q == 'railed_pads':
            final_returns.append(to_return[q])
        elif q in ['git_version', 'git_status', 'git_diff']:
            final_returns.append(to_return[q])
        else:
            final_returns.append(np.concatenate(to_return[q], axis=0))
            
    if is_single:
        return final_returns[0]
    else:
        return final_returns
    
def _parallel_cache_loop(runs, cache_fname_fn, compute_fn, num_workers=1):
    def process_run(run):
        cache_fname = cache_fname_fn(run)
        if os.path.exists(cache_fname):
            return np.load(cache_fname)
        else:
            res = compute_fn(run)
            np.save(cache_fname, res)
            return res
            
    if num_workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            to_return = list(executor.map(process_run, runs))
    else:
        to_return = [process_run(r) for r in runs]
    return np.concatenate(to_return, axis=0)
    

def get_lengths(experiment_or_endpoints, runs=None, num_workers=1, config_filename=""):
    if runs is None:
        endpoints = np.array(experiment_or_endpoints)
    else:
        endpoints = np.array(get_quantity('endpoints', experiment_or_endpoints, runs, num_workers=num_workers, config_filename=config_filename))
    dr = endpoints[:, 0] - endpoints[:, 1]
    return np.sqrt(np.sum(dr*dr, axis=1))

def get_veto_counts(experiment, runs, num_workers=1, config_filename=""):
    runs = [int(r) for r in runs]
    def cache_fname_fn(run):
        settings_hash = get_experiment_settings_hash(experiment, run, config_filename)
        return os.path.join(get_save_path(experiment), f'veto_counts_{experiment}_run{int(run)}_{settings_hash}.npy')
        
    def compute_fn(run):
        veto_pad_mask = np.zeros(1024)
        for i in raw_h5_file.VETO_PADS:
            veto_pad_mask[i] = 1
        return np.einsum('ij, j', get_quantity('pad_charge', experiment, [run], config_filename=config_filename), veto_pad_mask)
        
    return _parallel_cache_loop(runs, cache_fname_fn, compute_fn, num_workers=num_workers)

def get_veto_mask(experiment, runs, veto_thresholds, num_workers=1, config_filename=""):
    runs = [int(r) for r in runs]
    thresh_hash = hashlib.sha256(veto_thresholds.tobytes()).hexdigest()[:16]
    
    def cache_fname_fn(run):
        settings_hash = get_experiment_settings_hash(experiment, run, config_filename)
        return os.path.join(get_save_path(experiment), f'veto_mask_{experiment}_run{int(run)}_{thresh_hash}_{settings_hash}.npy')
        
    def compute_fn(run):
        pad_maxs = get_quantity('pad_max', experiment, [run], config_filename=config_filename)
        return np.all(pad_maxs < veto_thresholds, axis=1)
        
    return _parallel_cache_loop(runs, cache_fname_fn, compute_fn, num_workers=num_workers)

def get_max_veto_counts(experiment, runs, num_workers=1, config_filename=""):
    '''
    gets array of max counts on any individual veto pad
    '''
    runs = [int(r) for r in runs]
    
    def cache_fname_fn(run):
        settings_hash = get_experiment_settings_hash(experiment, run, config_filename)
        return os.path.join(get_save_path(experiment), f'max_veto_counts_{experiment}_run{int(run)}_{settings_hash}.npy')
        
    def compute_fn(run):
        pad_maxs = get_quantity('pad_max', experiment, [run], config_filename=config_filename)
        veto_pad_mask = np.zeros(1024)
        for i in raw_h5_file.VETO_PADS:
            veto_pad_mask[i] = 1
        return np.max(pad_maxs[:,veto_pad_mask==1], axis=1)
        
    return _parallel_cache_loop(runs, cache_fname_fn, compute_fn, num_workers=num_workers)

def get_outer_ring_counts(experiment, runs, num_workers=1, config_filename=""):
    runs = [int(r) for r in runs]
    
    def cache_fname_fn(run):
        settings_hash = get_experiment_settings_hash(experiment, run, config_filename)
        return os.path.join(get_save_path(experiment), f'outer_ring_counts_{experiment}_run{int(run)}_{settings_hash}.npy')
        
    def compute_fn(run):
        outer_ring_mask = np.zeros(1024)
        for i in OUTER_RING_PADS:
            outer_ring_mask[i] = 1
        return np.einsum('ij, j', get_quantity('pad_charge', experiment, [run], config_filename=config_filename), outer_ring_mask)
        
    return _parallel_cache_loop(runs, cache_fname_fn, compute_fn, num_workers=num_workers)

def get_outer_ring_max_counts(experiment, runs, num_workers=1, config_filename=""):
    runs = [int(r) for r in runs]
    
    def cache_fname_fn(run):
        settings_hash = get_experiment_settings_hash(experiment, run, config_filename)
        return os.path.join(get_save_path(experiment), f'max_outer_ring_counts_{experiment}_run{int(run)}_{settings_hash}.npy')
        
    def compute_fn(run):
        pad_maxs = get_quantity('pad_max', experiment, [run], config_filename=config_filename)
        outer_ring_mask = np.zeros(1024)
        for i in OUTER_RING_PADS:
            outer_ring_mask[i] = 1
        return np.max(pad_maxs[:,outer_ring_mask==1], axis=1)
        
    return _parallel_cache_loop(runs, cache_fname_fn, compute_fn, num_workers=num_workers)
    
def get_gm_ic(experiment, runs, gains, num_workers=1, config_filename=""):
    runs = [int(r) for r in runs]
    gains_hash = hashlib.sha256(gains.tobytes()).hexdigest()[:16]
    
    def cache_fname_fn(run):
        settings_hash = get_experiment_settings_hash(experiment, run, config_filename)
        return os.path.join(get_save_path(experiment), f'gm_ic_{experiment}_run{int(run)}_{gains_hash}_{settings_hash}.npy')
        
    def compute_fn(run):
        counts_per_pad = get_quantity('pad_charge', experiment, [run], config_filename=config_filename)
        return np.einsum('ij, j', counts_per_pad, gains)
        
    return _parallel_cache_loop(runs, cache_fname_fn, compute_fn, num_workers=num_workers)

def get_angle(experiment_or_endpoints, runs=None, num_workers=1, config_filename=""):
    if runs is None:
        endpoints = np.array(experiment_or_endpoints)
    else:
        endpoints = np.array(get_quantity('endpoints', experiment_or_endpoints, runs, num_workers=num_workers, config_filename=config_filename))
    dr = endpoints[:, 0] - endpoints[:, 1]
    return np.arctan2(np.sqrt(dr[:,0]**2 + dr[:,1]**2), np.abs(dr[:,2]))

def get_time_since_beam_off(experiment, runs, config_filename=""):
    to_return = []
    for run in runs:
        times_since_start_of_window = []
        ts = get_quantity('timestamps', experiment, [run], config_filename=config_filename)
        time_since_last_event = ts - np.roll(ts, 1)
        start_of_current_window = -np.inf
        time_since_last_event[0] = np.inf
        for t, dt in zip(ts, time_since_last_event):
            if dt > 0.1:
                start_of_current_window = t
            times_since_start_of_window.append(t - start_of_current_window)
        times_since_start_of_window = np.array(times_since_start_of_window)
        to_return.append(times_since_start_of_window)
    return np.concatenate(to_return, axis=0)

def get_run_and_event_numbers(experiment, runs, config_filename=""):
    runs = [int(r) for r in runs]
    run_numbers = []
    event_numbers = []
    for run in runs:
        h5 = get_h5_file(experiment, run, config_filename)
        first, last  = h5.get_event_num_bounds()
        if last < first: #empty file (attpc merger sets first to a large number until it reads the first event, and last will be 0)
            event_numbers.append(np.array([], dtype=int))
            run_numbers.append(np.array([], dtype=int))
        else:
            event_numbers.append(np.arange(first, last+1))
            run_numbers.append(np.ones(last - first + 1)*run)
    return np.concatenate(run_numbers, axis=0), np.concatenate(event_numbers, axis=0)
