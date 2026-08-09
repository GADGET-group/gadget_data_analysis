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

def get_h5_file(experiment, run_number):
    raw_h5_path = get_h5_path(experiment, run_number)
    if experiment == 'e21072':
        h5file = raw_h5_file.raw_h5_file(raw_h5_path, zscale=0.92, flat_lookup_csv='raw_viewer/channel_mappings/flatlookup4cobos.csv')
        h5file.length_counts_threshold = 100
        h5file.ic_counts_threshold = 0
        h5file.background_subtract_mode = 'smart'
        h5file.smart_bins_away_to_check = 10
        h5file.num_smart_background_ave_bins = 10
        h5file.cache_enable = True
    elif experiment == 'e23035_prep_2cobo':
        h5file = raw_h5_file.raw_h5_file(raw_h5_path, zscale=1.088, flat_lookup_csv='raw_viewer/channel_mappings/flatlookup2cobos.csv')
        h5file.length_counts_threshold = 25
        h5file.ic_counts_threshold = 0
        h5file.background_subtract_mode = 'smart'
        h5file.smart_bins_away_to_check = 25
        h5file.num_smart_background_ave_bins = 10
        h5file.cache_enable = True
    elif experiment == 'e23035_prep_4cobo':
        h5file = raw_h5_file.raw_h5_file(raw_h5_path, zscale=0.544, flat_lookup_csv='raw_viewer/channel_mappings/flatlookup4cobos.csv')
        h5file.length_counts_threshold = 100
        h5file.ic_counts_threshold = 0
        h5file.background_subtract_mode = 'smart'
        h5file.smart_bins_away_to_check = 25
        h5file.num_smart_background_ave_bins = 10
        h5file.cache_enable = True
    elif experiment == 'e23035_prep_vault' or experiment == 'e23035':
        h5file = raw_h5_file.raw_h5_file(raw_h5_path, zscale=1.088, flat_lookup_csv='raw_viewer/channel_mappings/flatlookup4cobos.csv')
        h5file.length_counts_threshold = 100
        h5file.ic_counts_threshold = 0
        h5file.background_subtract_mode = 'smart2'
        h5file.smart_bins_away_to_check = 3
        h5file.num_smart_background_ave_bins = 20
        h5file.smart2_min_bins_in_peak = 5
        h5file.smart2_min_sigma = 2
        h5file.cache_enable = True
    elif experiment == 'e25058':
        h5file = raw_h5_file.raw_h5_file(raw_h5_path, zscale=1.088, flat_lookup_csv='raw_viewer/channel_mappings/flatlookup4cobos.csv')
        h5file.length_counts_threshold = 100
        h5file.ic_counts_threshold = 0
        h5file.background_subtract_mode = 'smart'
        h5file.smart_bins_away_to_check = 25
        h5file.num_smart_background_ave_bins = 10
        h5file.cache_enable = True
    else:
        raise ValueError
    return h5file

#coppied from field distortions folder in track fitting branch
#and modified to configure h5 file differently
def process_tpc_run(experiment, run_number, force_reprocess=False):
    '''
    Get information about track direction, width, and charge per pad, which isn't normally stored when processing runs.
    Only redoes processing if a ROOT version of this information isn't available.
    '''
    #save_path = os.path.dirname(os.path.abspath(__file__))
    fname = os.path.join(get_save_path(experiment), f'{experiment}_run{run_number}.root')
    
    #git info to save
    git_version = subprocess.run(['git', 'rev-parse', '--verify', 'HEAD'], capture_output=True, text=True, check=True).stdout
    git_status = subprocess.run(['git', 'status'], capture_output=True, text=True, check=True).stdout
    git_diff = subprocess.run(['git', 'diff'], capture_output=True, text=True, check=True, error='replace').stdout

    if force_reprocess:
        import glob
        save_path = get_save_path(experiment)
        cache_patterns = [
            f'gm_ic_{experiment}_run{run_number}_*.npy',
            f'veto_counts_{experiment}_run{run_number}.npy',
            f'max_veto_counts_{experiment}_run{run_number}.npy',
            f'outer_ring_counts_{experiment}_run{run_number}.npy',
            f'max_outer_ring_counts_{experiment}_run{run_number}.npy',
            f'veto_mask_{experiment}_run{run_number}_*.npy'
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
        h5file = get_h5_file(experiment, run_number)
        print('processing run %d'%run_number)
        first_event, last_event = h5file.get_event_num_bounds()
        track_centers, principle_axes,variances_along_axes, pad_charges, track_endpoints, charge_widths, width_above_thresholds = [],[],[],[],[],[], []
        pad_maxs, railed_pads = [], []
        for evt in tqdm(range(first_event, last_event + 1)):
            railed_pads.append(h5file.get_railed_pads(evt))
            center, dd,vv = h5file.get_track_axis(evt, return_all_svd_results=True, threshold=h5file.length_counts_threshold)
            xs, ys, zs, es = h5file.get_xyze(evt, threshold=h5file.length_counts_threshold, include_veto_pads=False)
            principle_axes.append(vv)
            variances_along_axes.append(dd**2/(len(xs)-1))
            track_centers.append(center)
            pad_counts = np.zeros(1024)
            pad_maxs.append(np.zeros(1024))
            for pad, trace in zip(*h5file.get_pad_traces(evt)):
                pad_counts[pad] = np.sum(trace)
                pad_maxs[-1][pad] = np.max(trace)
            pad_charges.append(pad_counts)


            #get track end points
            if len(variances_along_axes[-1])==3:
                points = np.concatenate((xs[:, np.newaxis], 
                        ys[:, np.newaxis], 
                        zs[:, np.newaxis]), 
                        axis=1)
                rbar = points - center
                track_direction = vv[0]/np.sqrt(np.sum(vv[0]*vv[0]))
                rdotv = np.dot(rbar, track_direction)
                #project endpoints onto track axis
                first_point = np.min(rdotv)*track_direction + center#points[np.argmin(rdotv)]
                last_point = np.max(rdotv)*track_direction + center#points[np.argmax(rdotv)]
                track_endpoints.append([first_point, last_point])
                #above variance is just variance in postiion of points above some threshold
                #instead calcualte variance along 2nd axis of charge
                width_axis = vv[1]/np.sqrt(np.sum(vv[1]*vv[1]))
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
        track_centers = np.array(track_centers)
        pad_charges = np.array(pad_charges)
        pad_maxs = np.array(pad_maxs)
        
        ts = h5file.get_timestamps_array()

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
                
        events_data = {'track_center':np.array(res_centers, dtype=np.float64), 'principle_axes':sanitize_jagged(principle_axes), 'variance_along_axes': sanitize_jagged(variances_along_axes),
                   'pad_charge': pad_charges, 'endpoints':np.array(res_endpoints, dtype=np.float64), 'charge_width':np.array(charge_widths),
                   'width_above_threshold':np.array(width_above_thresholds), 'pad_max':pad_maxs, 'timestamps':ts}
                   
        counts = [len(x) for x in railed_pads]
        if sum(counts) == 0:
            events_data['railed_pads'] = ak.unflatten(np.array([], dtype=np.int64), counts)
        else:
            events_data['railed_pads'] = ak.from_iter(railed_pads)
        metadata = {'git_version':[git_version], 'git_status':[git_status], 'git_diff':[git_diff]}
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

def get_quantity(qname, experiment, runs):
    to_return = []
    for run in runs:
        fname = os.path.join(get_save_path(experiment), f'{experiment}_run{run}.root')
        if not os.path.exists(fname):
            process_tpc_run(experiment, run)
            
        with uproot.open(fname) as file:
            if qname in file['events']:
                if qname == 'railed_pads':
                    arr = file['events'][qname].array(library='ak')
                    to_return.extend(ak.to_list(arr))
                else:
                    arr = file['events'][qname].array(library='np')
                    to_return.append(arr)
            elif qname in file['metadata']:
                arr = file['metadata'][qname].array(library='np')
                to_return.append(arr[0])
            else:
                raise ValueError(f"Quantity {qname} not found in ROOT file")
                
    if qname == 'railed_pads':
        return to_return
    elif qname in ['git_version', 'git_status', 'git_diff']:
        return to_return
    else:
        return np.concatenate(to_return, axis=0)
    

def get_lengths(experiment, runs):
    endpoints = np.array(get_quantity('endpoints', experiment, runs))
    dr = endpoints[:, 0] - endpoints[:, 1]
    return np.sqrt(np.sum(dr*dr, axis=1))

def get_veto_counts(experiment, runs):
    to_return = []
    for run in runs:
        cache_fname = os.path.join(get_save_path(experiment), f'veto_counts_{experiment}_run{run}.npy')
        if os.path.exists(cache_fname):
            veto_counts = np.load(cache_fname)
        else:
            veto_pad_mask = np.zeros(1024)
            for i in raw_h5_file.VETO_PADS:
                veto_pad_mask[i] = 1
            veto_counts = np.einsum('ij, j', get_quantity('pad_charge', experiment, [run]), veto_pad_mask)
            np.save(cache_fname, veto_counts)
        to_return.append(veto_counts)
    return np.concatenate(to_return, axis=0)

def get_veto_mask(experiment, runs, veto_thresholds):
    thresh_hash = hashlib.sha256(veto_thresholds.tobytes()).hexdigest()[:16]
    to_return = []
    for run in runs:
        cache_fname = os.path.join(get_save_path(experiment), f'veto_mask_{experiment}_run{run}_{thresh_hash}.npy')
        if os.path.exists(cache_fname):
            veto_mask = np.load(cache_fname)
        else:
            pad_maxs = get_quantity('pad_max', experiment, [run])
            veto_mask = np.all(pad_maxs < veto_thresholds, axis=1)
            np.save(cache_fname, veto_mask)
        to_return.append(veto_mask)
    return np.concatenate(to_return, axis=0)

def get_max_veto_counts(experiment, runs):
    '''
    gets array of max counts on any individual veto pad
    '''
    to_return = []
    for run in runs:
        cache_fname = os.path.join(get_save_path(experiment), f'max_veto_counts_{experiment}_run{run}.npy')
        if os.path.exists(cache_fname):
            max_pad_counts = np.load(cache_fname)
        else:
            pad_maxs = get_quantity('pad_max', experiment, [run])
            veto_pad_mask = np.zeros(1024)
            for i in raw_h5_file.VETO_PADS:
                veto_pad_mask[i] = 1
            max_pad_counts = np.max(pad_maxs[:,veto_pad_mask==1], axis=1)
            np.save(cache_fname, max_pad_counts)
        to_return.append(max_pad_counts)
    return np.concatenate(to_return, axis=0)

def get_outer_ring_counts(experiment, runs):
    to_return = []
    for run in runs:
        cache_fname = os.path.join(get_save_path(experiment), f'outer_ring_counts_{experiment}_run{run}.npy')
        if os.path.exists(cache_fname):
            counts = np.load(cache_fname)
        else:
            outer_ring_mask = np.zeros(1024)
            for i in OUTER_RING_PADS:
                outer_ring_mask[i] = 1
            counts = np.einsum('ij, j', get_quantity('pad_charge', experiment, [run]), outer_ring_mask)
            np.save(cache_fname, counts)
        to_return.append(counts)
    return np.concatenate(to_return, axis=0)

def get_outer_ring_max_counts(experiment, runs):
    to_return = []
    for run in runs:
        cache_fname = os.path.join(get_save_path(experiment), f'max_outer_ring_counts_{experiment}_run{run}.npy')
        if os.path.exists(cache_fname):
            max_pad_counts = np.load(cache_fname)
        else:
            pad_maxs = get_quantity('pad_max', experiment, [run])
            outer_ring_mask = np.zeros(1024)
            for i in OUTER_RING_PADS:
                outer_ring_mask[i] = 1
            max_pad_counts = np.max(pad_maxs[:,outer_ring_mask==1], axis=1)
            np.save(cache_fname, max_pad_counts)
        to_return.append(max_pad_counts)
    return np.concatenate(to_return, axis=0)
    
def get_gm_ic(experiment, runs, gains):
    gains_hash = hashlib.sha256(gains.tobytes()).hexdigest()[:16]
    to_return = []
    for run in runs:
        cache_fname = os.path.join(get_save_path(experiment), f'gm_ic_{experiment}_run{run}_{gains_hash}.npy')
        if os.path.exists(cache_fname):
            gm_ic = np.load(cache_fname)
        else:
            counts_per_pad = get_quantity('pad_charge', experiment, [run])
            #counts per pad needs to already be on the gpu
            gm_ic = np.einsum('ij, j', counts_per_pad, gains)
            np.save(cache_fname, gm_ic)
        to_return.append(gm_ic)
    return np.concatenate(to_return, axis=0)

def get_angle(experiment, runs):
    endpoints = np.array(get_quantity('endpoints', experiment, runs))
    dr = endpoints[:, 0] - endpoints[:, 1]
    return np.arctan2(np.sqrt(dr[:,0]**2 + dr[:,1]**2), np.abs(dr[:,2]))

def get_time_since_beam_off(experiment, runs):
    to_return = []
    for run in runs:
        times_since_start_of_window = []
        ts = get_quantity('timestamps', experiment, [run])
        time_since_last_event = ts - np.roll(ts, 1)
        start_of_current_window = -np.inf
        time_since_last_event[0] = np.inf
        for t, dt in zip(ts, time_since_last_event):
            if dt > 0.1:
                start_of_current_winow = t
            times_since_start_of_window.append(t - start_of_current_winow)
        times_since_start_of_window = np.array(times_since_start_of_window)
        to_return.append(times_since_start_of_window)
    return np.concatenate(to_return, axis=0)

def get_run_and_event_numbers(experiment, runs):
    run_numbers = []
    event_numbers = []
    for run in runs:
        h5 = get_h5_file(experiment, run)
        first, last  = h5.get_event_num_bounds()
        event_numbers.append(np.arange(first, last+1))
        run_numbers.append(np.ones(last - first + 1)*run)
    return np.concatenate(run_numbers, axis=0), np.concatenate(event_numbers, axis=0)
