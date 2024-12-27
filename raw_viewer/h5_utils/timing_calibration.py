import pickle

import numpy as np
import scipy.optimize as opt
import matplotlib.pylab as plt
import tqdm

from raw_viewer.raw_h5_file import raw_h5_file
from track_fitting import build_sim

run = 124

if False:
    h5file = raw_h5_file('/egr/research-tpc/shared/gastest_2024/run_0089.h5', 
                 flat_lookup_csv='raw_viewer/channel_mappings/flatlookup2cobos.csv')
else:
    h5file = build_sim.get_rawh5_object('e21072', run)

def f(x, A, mu, sigma, c):
    if sigma <= 0:
        return 1e100    
    return A*np.e**(-(x-mu)**2/2/sigma) + c

first_event, last_event = h5file.get_event_num_bounds()

peak_locations = {}#list of peak locations indexed by pad number

events_to_process = 1000000
for event_num in tqdm.tqdm(range(first_event, np.min((last_event+1, first_event + events_to_process)))):
    pads, traces = h5file.get_pad_traces(event_num, False)
    for p, t in zip(pads, traces):
        peak_loc = np.argmax(t)
        if t[peak_loc]>200:
            if p not in peak_locations:
                peak_locations[p] = []
            peak_locations[p].append(peak_loc)
        
        
        
if False:
    for i in range(events_to_process):
        plt.figure()
        image = np.zeros(np.shape(h5file.pad_plane))
        for p in peak_locations:
            image[h5file.pad_to_xy_index[p]] = peak_locations[p][i] - peak_locations[1][i]
        plt.imshow(image)
        plt.colorbar()
        plt.title('peak location in event %d'%(i+first_event))

plt.figure()
image = np.zeros(np.shape(h5file.pad_plane))
for p in peak_locations:
    image[h5file.pad_to_xy_index[p]] = np.median(peak_locations[p])
plt.imshow(image, vmin=60, vmax=80)
plt.colorbar()
plt.title('median peak location')

plt.show(block=False)
if True:
    with open('./raw_viewer/h5_utils/timing_offsets_e21072_run%d.pkl'%run, 'wb') as f:
        pickle.dump({p:np.median(peak_locations[p]) - np.median(peak_locations[1])for p in peak_locations}, f)
    with open('./raw_viewer/h5_utils/peak_positions_e21072_run%d.pkl'%run, 'wb') as f:
        pickle.dump(peak_locations, f)