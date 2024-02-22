import numpy as np
from scipy.signal import find_peaks
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import time
from multiprocessing import Pool, cpu_count
import multiprocessing
import multiprocessing.pool
from tqdm import tqdm
import sys
sys.path.insert(0, 'TPC-utils')
from tpc_utils import background, search_high_res
#from tspectrum import background, search_high_res
from TPCH5_utils import get_first_last_event_num, load_trace
import h5py
from multiprocessing import TimeoutError
import concurrent.futures
from sklearn.cluster import DBSCAN


# PHASE 1 (Constructing point clouds from trace data)

# Daemon workaround found from: https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic

class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, val):
        pass


class NoDaemonProcessPool(multiprocessing.pool.Pool):

    def Process(self, *args, **kwds):
        proc = super(NoDaemonProcessPool, self).Process(*args, **kwds)
        proc.__class__ = NoDaemonProcess

        return proc

def deconv(traces):
    return search_high_res(traces, sigma = 4, threshold = 60, remove_bkg = True, number_it = 200, markov = True, aver_window = 5)[0]

def remove_outliers(xset, yset, zset, eset, pads):
	"""
	Uses DBSCAN to find and remove outliers in 3D data
	"""
	skip = False

	data = np.array([xset.T, yset.T, zset.T]).T

	# Run DBSCAN algorithm
	DBSCAN_cluster = DBSCAN(eps=6, min_samples=8).fit(data)

	# Identify largest cluster
	labels = DBSCAN_cluster.labels_
	unique_labels = set(labels)
	if -1 in unique_labels:
		unique_labels.remove(-1)

	if len(unique_labels) > 0:
		largest_cluster_label = max(unique_labels, key=lambda x: np.sum(labels == x))

		# Relabel non-main-cluster points as outliers
		labels[labels != largest_cluster_label] = -1

		# Remove outlier points
		out_of_cluster_index = np.where(labels == -1)
		del data
		rev = out_of_cluster_index[0][::-1]
		for i in rev:
			xset = np.delete(xset, i)
			yset = np.delete(yset, i)
			zset = np.delete(zset, i)
			eset = np.delete(eset, i)
			pads = np.delete(pads, i)
	else:
		skip = True

	return xset, yset, zset, eset, pads, skip 

"""
def make_pc(event_num_array):
    all_clouds_seg = []
    for event_num_i in tqdm(range(len(event_num_array))):
    #for event_num_i in range(len(event_num_array)):

        event_ind = event_num_array[event_num_i]

        # Picks out the traces from the first event.
        meta, all_traces = load_trace(PATH, event_ind)
        all_traces = all_traces.astype(np.float64)

        # Replaces the first and last values in each trace.
        all_traces[:, 0] = all_traces[:, 1]
        all_traces[:, -1] = all_traces[:, -2]

        all_peaks = np.array([])
        all_energies = np.array([])
        all_x = np.array([])
        all_y = np.array([])
        all_pad_nums = np.array([])

        all_traces_parts = np.array_split(all_traces, deconv_cores, axis = 0)
        
        with Pool(deconv_cores) as deconv_p:
            deconv_parts = deconv_p.map(deconv, all_traces_parts)
            response = np.vstack(deconv_parts)
        

        for trace_num in range(len(all_traces)):
                # 65535 is -1 in 16 bit unsigned format. This corresponds to “LEMO” channels that have no pad assignments 
                if meta[:,4][trace_num] == 65535:
                    continue 

                sig = 4

                peaks, _ = find_peaks(response[trace_num], height = 70)
                peaks = peaks[np.argsort(response[trace_num][peaks])[::-1]][:]

                if len(peaks) != 0:
                    num_pts = int(np.round(sig))

                    energies = np.array([])

                    # Loop calculates the energy for each peak
                    for peak in peaks:
                            if (((peak+num_pts) < len(response[0])) and ((peak-num_pts) > 0)):
                                extra_pts = np.arange(peak-num_pts, peak+num_pts, dtype = int)

                            energies = np.append(energies, trapezoid(response[trace_num][extra_pts], extra_pts))
                            #all_x = np.append(all_x, padxy['x'][meta[:,4][trace_num]])
                            all_x = np.append(all_x, padxy[:,0][meta[:,4][trace_num]])
                            #all_y = np.append(all_y, padxy['y'][meta[:,4][trace_num]])
                            all_y = np.append(all_y, padxy[:,1][meta[:,4][trace_num]])
                            all_pad_nums = np.append(all_pad_nums, meta[:,4][trace_num])

                    all_peaks = np.append(all_peaks, peaks)
                    all_energies = np.append(all_energies, energies)

        #all_z = all_peaks / np.shape(all_traces)[1] * 1000
        all_z = all_peaks

        pc = np.stack((all_x, all_y, all_z, all_energies, all_pad_nums)).T

        all_clouds_seg.append([event_ind, pc])

    return all_clouds_seg
"""
def make_pc(args):
    event_num_array, padxy, PATH, deconv_cores = args
    skip_pads = [253, 254, 508, 509, 763, 764, 1018, 1019, 65535]
    all_clouds_seg = []
    for event_num_i in tqdm(range(len(event_num_array))):
        event_ind = event_num_array[event_num_i]
        meta, all_traces = load_trace(PATH, event_ind)

        all_traces = all_traces.astype(np.float64)
        all_traces[:, 0] = all_traces[:, 1]
        all_traces[:, -1] = all_traces[:, -2]

        all_peaks = np.array([])
        all_energies = np.array([])
        all_x = np.array([])
        all_y = np.array([])
        all_pad_nums = np.array([])

        all_traces_parts = np.array_split(all_traces, deconv_cores, axis=0)
        #print(all_traces_parts[9])
        #print(len(all_traces_parts[9]))
        #print(len(all_traces_parts[9][0]))
        if len(all_traces_parts[0]) < 1:
                continue
        if len(all_traces_parts[1]) < 1:
                continue
        if len(all_traces_parts[2]) < 1:
                continue
        if len(all_traces_parts[3]) < 1:
                continue
        if len(all_traces_parts[4]) < 1:
                continue
        if len(all_traces_parts[5]) < 1:
                continue
        if len(all_traces_parts[6]) < 1:
                continue
        if len(all_traces_parts[7]) < 1:
                continue
        if len(all_traces_parts[8]) < 1:
                continue
        if len(all_traces_parts[9]) < 1:
                continue

        with Pool(deconv_cores) as deconv_p:
            deconv_parts = deconv_p.map(deconv, all_traces_parts)
            response = np.vstack(deconv_parts)

        for trace_num in range(len(all_traces)):
            if meta[:, 4][trace_num] in skip_pads:
                continue
            sig = 4

            peaks, _ = find_peaks(response[trace_num], height=70)
            peaks = peaks[np.argsort(response[trace_num][peaks])[::-1]][:]
            if len(peaks) != 0:
                num_pts = int(np.round(sig))
                energies = np.array([])

                for peak in peaks:
                    if ((peak + num_pts) < len(response[0])) and ((peak - num_pts) > 0):
                        extra_pts = np.arange(peak - num_pts, peak + num_pts, dtype=int)

                    energies = np.append(energies, trapezoid(response[trace_num][extra_pts], extra_pts))
                    all_x = np.append(all_x, padxy[:, 0][meta[:, 4][trace_num]])
                    all_y = np.append(all_y, padxy[:, 1][meta[:, 4][trace_num]])
                    all_pad_nums = np.append(all_pad_nums, meta[:, 4][trace_num])

                all_peaks = np.append(all_peaks, peaks)
                all_energies = np.append(all_energies, energies)

        all_z = all_peaks
        pc = np.stack((all_x, all_y, all_z, all_energies, all_pad_nums)).T
        if len(pc) == 0:
            continue
        """
        new_pc = []
        radius = 3
        for i in range(len(pc)):
            p = pc[i]
            x, y, z = p[0], p[1], p[2]
            energy = p[3]
            pad_num = p[4]
            new_pc.append([x, y, z, energy, pad_num])
            for j in range(len(pc)):
                if i == j:
                    continue
                p2 = pc[j]
                x2, y2, z2 = p2[0], p2[1], p2[2]
                dist = np.sqrt((x2-x)**2 + (y2-y)**2 + (z2-z)**2)
                if dist <= radius:
                    new_pc.append([(x+x2)/2, (y+y2)/2, (z+z2)/2, (energy+p2[3])/2, pad_num])

        new_pc = np.array(new_pc)
        """
        ############## New
        min_z_spacing = 8  # or any other desired minimum z-spacing
        new_pc = []

        for idx, point in enumerate(pc[:-1]):
            x, y, z, energy, pad_num = point
            next_point = pc[idx + 1]
            _, _, next_z, next_energy, _ = next_point

            new_pc.append([x, y, z, energy, pad_num])

            z_diff = next_z - z
            num_new_points = int(z_diff / min_z_spacing)

            if num_new_points > 0:
                z_interval = z_diff / (num_new_points + 1)
                energy_interval = (energy + next_energy) / 2 / (num_new_points + 1)

                for i in range(num_new_points):
                    new_z = z + (i + 1) * z_interval
                    new_energy = energy_interval
                    new_pc.append([x, y, new_z, new_energy, pad_num])

        # Add the last point of the original point cloud
        new_pc.append(pc[-1])
        new_pc = np.array(new_pc)
   
        # Apply outlier removal to the new point cloud
        xset, yset, zset, eset, pads = new_pc[:, 0], new_pc[:, 1], new_pc[:, 2], new_pc[:, 3], new_pc[:, 4]
        xset, yset, zset, eset, pads, skip = remove_outliers(xset, yset, zset, eset, pads)
        if skip == True:
            continue
      
        cleaned_pc = np.stack((xset, yset, zset, eset, pads)).T


        all_clouds_seg.append([event_ind, cleaned_pc])

        #all_clouds_seg.append([event_ind, new_pc])
        #all_clouds_seg.append([event_ind, pc])

    return all_clouds_seg

        
if __name__ == '__main__':
    start = time.time()

    all_cores = cpu_count()
    deconv_cores = 10

    PATH = '/mnt/analysis/e21072/h5test/run_9999.h5'#'/mnt/analysis/e21072/h5test/gas_test_run_99.h5'

    first_event_num, last_event_num = get_first_last_event_num(PATH)

    padxy = np.loadtxt('padxy.csv', delimiter = ',', skiprows = 1)

    evt_cores = 3
    
    print('1st Event:', first_event_num, '| Last Event:', last_event_num, '| Number of Cores:', evt_cores*10)
    print(PATH)

    total_events = last_event_num - first_event_num + 1
    num_per_core = total_events // evt_cores
    extra_events = total_events % evt_cores

    evt_parts = []
    start_num = first_event_num
    for i in range(evt_cores):
        end_num = start_num + num_per_core - 1
        if i < extra_events:
            end_num += 1
        evt_parts.append((list(range(start_num, end_num + 1)), padxy, PATH, deconv_cores))
        start_num = end_num + 1

    with NoDaemonProcessPool(evt_cores) as evt_p:
        run_parts = evt_p.map(make_pc, evt_parts)

    # ... [rest of your code] ...


    #with concurrent.futures.ThreadPoolExecutor(max_workers=evt_cores) as executor:
    #    run_parts = list(tqdm(executor.map(make_pc, evt_parts), total=len(evt_parts), desc='Processing events'))

    #print(sys.getsizeof(run_parts))
        
    print('It takes', time.time()-start, 'seconds to process all', last_event_num-first_event_num, 'events.')

    #print(len(run_parts[0][0]))

    #f = h5py.File('TestClouds.h5', 'r+')
    f = h5py.File(PATH, 'r+')
    try:
        clouds = f.create_group('clouds')
    except ValueError:
        print('Cloud group already exists')
        clouds = f['clouds']

    for part in run_parts:
        for evt in part:
            try:
                evt_name = 'evt' + str(evt[0]) + '_cloud'
                if evt_name in clouds:
                    del clouds[evt_name]
                clouds.create_dataset(evt_name, data=evt[1])
            except OSError:
                print('Error creating dataset for event', evt[0])
    f.close()

  

    

