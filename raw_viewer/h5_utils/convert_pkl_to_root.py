import os
import glob
import gzip
import pickle
import uproot
import awkward as ak
import numpy as np
import argparse
import sys

# Add the root directory to path to import process_runs
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from raw_viewer import process_runs

def convert_pkl_to_root(experiment, run=None, force=False):
    save_path = process_runs.get_save_path(experiment)
    if run is not None:
        run_str = str(run).zfill(4)
        pkl_files = glob.glob(os.path.join(save_path, f'{experiment}_run{run_str}.pkl.gz'))
        if not pkl_files:
            pkl_files = glob.glob(os.path.join(save_path, f'{experiment}_run{run}.pkl.gz'))
    else:
        pkl_files = glob.glob(os.path.join(save_path, f'{experiment}_run*.pkl.gz'))
    print(f"Found {len(pkl_files)} pkl.gz files for experiment {experiment} in {save_path}")
    
    for pkl_fname in pkl_files:
        run_str = pkl_fname.split('run')[-1].split('.pkl.gz')[0]
        root_fname = os.path.join(save_path, f'{experiment}_run{run_str}.root')
        
        if os.path.exists(root_fname) and not force:
            is_valid = False
            try:
                with uproot.open(root_fname) as f:
                    if 'events' in f:
                        is_valid = True
            except Exception:
                pass
            
            if is_valid:
                print(f"Skipping {pkl_fname}, {root_fname} already exists and is valid.")
                continue
            else:
                print(f"Overwriting corrupted/incomplete {root_fname}")
        elif os.path.exists(root_fname) and force:
            print(f"Force reconvert: Overwriting {root_fname}")
            
            
        print(f"Converting {pkl_fname} to {root_fname}")
        
        try:
            with gzip.open(pkl_fname, 'rb') as f:
                data = pickle.load(f)
                
            expected_keys = {'track_center', 'endpoints', 'pad_charge', 'charge_width', 
                             'width_above_threshold', 'pad_max', 'timestamps', 
                             'principle_axes', 'variance_along_axes', 'railed_pads'}
            if not any(k in data for k in expected_keys):
                print(f"Skipping {pkl_fname}: No recognized keys found in dictionary.")
                continue
                
            with uproot.recreate(root_fname) as root_file:
                events_data = {}
                def sanitize_jagged(lst):
                    def _walk(obj):
                        if isinstance(obj, tuple):
                            return [_walk(x) for x in obj]
                        elif isinstance(obj, list):
                            return [_walk(x) for x in obj]
                        elif isinstance(obj, np.ndarray):
                            if obj.dtype == object:
                                return [_walk(x) for x in obj]
                            return obj.tolist()
                        return obj
                    
                    cleaned = [_walk(x) for x in lst]
                    if len(cleaned) == 0:
                        return np.empty(0, dtype=np.float64)
                    try:
                        counts = [len(x) for x in cleaned]
                        if sum(counts) == 0:
                            return ak.unflatten(np.array([], dtype=np.float64), counts)
                    except TypeError:
                        pass
                    return ak.from_iter(cleaned)

                if 'track_center' in data:
                    res = []
                    for c in data['track_center']:
                        try:
                            if len(c) == 3:
                                res.append(list(c))
                            else:
                                res.append([0.0, 0.0, 0.0])
                        except TypeError:
                            res.append([0.0, 0.0, 0.0])
                    events_data['track_center'] = np.array(res, dtype=np.float64).reshape(-1, 3)
                    
                if 'endpoints' in data:
                    res = []
                    for ep in data['endpoints']:
                        try:
                            if len(ep) == 2 and len(ep[0]) == 3 and len(ep[1]) == 3:
                                res.append([list(ep[0]), list(ep[1])])
                            else:
                                res.append([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
                        except TypeError:
                            res.append([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
                    events_data['endpoints'] = np.array(res, dtype=np.float64).reshape(-1, 2, 3)

                for k in ['pad_charge', 'charge_width', 
                          'width_above_threshold', 'pad_max', 'timestamps']:
                    if k in data:
                        events_data[k] = np.array(data[k])
                        
                for k in ['principle_axes', 'variance_along_axes']:
                    if k in data:
                        events_data[k] = sanitize_jagged(data[k])
                    
                if 'railed_pads' in data:
                    counts = [len(x) for x in data['railed_pads']]
                    if sum(counts) == 0:
                        events_data['railed_pads'] = ak.unflatten(np.array([], dtype=np.int64), counts)
                    else:
                        events_data['railed_pads'] = ak.from_iter(data['railed_pads'])
                
                if not events_data:
                    print(f"Skipping {pkl_fname}: no valid events_data arrays could be extracted.")
                    continue
                    
                lengths = [len(v) for v in events_data.values()]
                if len(set(lengths)) > 1:
                    print(f"Warning: Arrays in {pkl_fname} have different lengths: { {k: len(v) for k, v in events_data.items()} }")
                    min_len = min(lengths)
                    print(f"Trimming all arrays to minimum length {min_len}")
                    for k in events_data:
                        events_data[k] = events_data[k][:min_len]

                if min(lengths) == 0:
                    print(f"Warning: TTree for {pkl_fname} will have 0 events.")
                    root_file['events'] = events_data
                else:
                    total_events = min(lengths)
                    chunk_size = 50000
                    for i in range(0, total_events, chunk_size):
                        chunk = {}
                        for k, v in events_data.items():
                            chunk[k] = v[i:i+chunk_size]
                        
                        if i == 0:
                            root_file['events'] = chunk
                        else:
                            root_file['events'].extend(chunk)
                
                metadata = {
                    'git_version': [data.get('git_version', '')],
                    'git_status': [data.get('git_status', '')],
                    'git_diff': [data.get('git_diff', '')]
                }
                root_file['metadata'] = metadata
        except Exception as e:
            print(f"Failed to convert {pkl_fname}: {e}")
            if 'events_data' in locals():
                print("Array shapes/types for debugging:")
                for k, v in events_data.items():
                    if isinstance(v, ak.Array):
                        print(f"  {k}: awkward array, type={ak.type(v)}, len={len(v)}")
                    elif isinstance(v, np.ndarray):
                        print(f"  {k}: numpy array, shape={v.shape}, dtype={v.dtype}")
                    else:
                        print(f"  {k}: {type(v)}, len={len(v)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert pkl.gz processed runs to root format.")
    parser.add_argument("experiment", help="Experiment name (e.g. e25058)")
    parser.add_argument("--run", type=str, help="Specific run number to convert (e.g. 0134 or 134)", default=None)
    parser.add_argument("--force", action="store_true", help="Force reconvert and overwrite existing root files")
    args = parser.parse_args()
    convert_pkl_to_root(args.experiment, args.run, args.force)
