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

def convert_pkl_to_root(experiment):
    save_path = process_runs.get_save_path(experiment)
    pkl_files = glob.glob(os.path.join(save_path, f'{experiment}_run*.pkl.gz'))
    print(f"Found {len(pkl_files)} pkl.gz files for experiment {experiment} in {save_path}")
    
    for pkl_fname in pkl_files:
        run_str = pkl_fname.split('run')[-1].split('.pkl.gz')[0]
        root_fname = os.path.join(save_path, f'{experiment}_run{run_str}.root')
        
        if os.path.exists(root_fname):
            print(f"Skipping {pkl_fname}, {root_fname} already exists.")
            continue
            
        print(f"Converting {pkl_fname} to {root_fname}")
        
        try:
            with gzip.open(pkl_fname, 'rb') as f:
                data = pickle.load(f)
                
            with uproot.recreate(root_fname) as root_file:
                events_data = {}
                for k in ['track_center', 'principle_axes', 'variance_along_axes', 
                          'pad_charge', 'endpoints', 'charge_width', 
                          'width_above_threshold', 'pad_max', 'timestamps']:
                    if k in data:
                        events_data[k] = np.array(data[k])
                    
                if 'railed_pads' in data:
                    events_data['railed_pads'] = ak.from_iter(data['railed_pads'])
                    
                root_file['events'] = events_data
                
                metadata = {
                    'git_version': [data.get('git_version', '')],
                    'git_status': [data.get('git_status', '')],
                    'git_diff': [data.get('git_diff', '')]
                }
                root_file['metadata'] = metadata
        except Exception as e:
            print(f"Failed to convert {pkl_fname}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert pkl.gz processed runs to root format.")
    parser.add_argument("experiment", help="Experiment name (e.g. e25058)")
    args = parser.parse_args()
    convert_pkl_to_root(args.experiment)
