import os
import pickle

import numpy as np
from tqdm import tqdm
import matplotlib.pylab as plt
import matplotlib.colors
import scipy.optimize as opt

from track_fitting.field_distortion import extract_track_axis_info
from track_fitting import build_sim

do_simple_linear_correction=False
do_rmap = True

experiment, run = 'e21072', 124


track_info_dict = extract_track_axis_info.get_track_info(experiment, run)
processed_directory = '/egr/research-tpc/shared/Run_Data/run_%04d_raw_viewer/run_%04dsmart'%(run, run)

#load histogram arrays
counts = np.load(os.path.join(processed_directory, 'counts.npy'))
dxys = np.load(os.path.join(processed_directory, 'dxy.npy'))
dts = np.load(os.path.join(processed_directory, 'dt.npy'))
max_veto_counts = np.load(os.path.join(processed_directory, 'veto.npy'))
timestamps = np.load(os.path.join(processed_directory, 'timestamps.npy'))

h5file = build_sim.get_rawh5_object(experiment, run)
dzs = dts*h5file.zscale
ranges = np.sqrt(dzs**2 + dxys**2)

#use track angle from pca rather than that exported by raw event viewer
angles = []
for axes in track_info_dict['principle_axes']:
    dir = axes[0]
    angles.append(np.arctan2(np.sqrt(dir[0]**2 + dir[1]**2), np.abs(dir[2])))
angles = np.array(angles)

#estimate time since start of decay window to be time since first event in the window
time_since_last_event = timestamps - np.roll(timestamps, 1)
time_since_last_event[0] = .15 #we don't actuallly know what this is for the first event, so just putting a typical value for start of window
start_of_current_winow = 0
times_since_start_of_window = []

track_widths = []
for event_index in range(len(timestamps)):
    if len(track_info_dict['variance_along_axes'][event_index]) == 3:
        track_widths.append(track_info_dict['variance_along_axes'][event_index][1]**0.5)
    else:
        track_widths.append(0)
track_widths = np.array(track_widths)

print('calculating event times in decay window')
for t, dt in tqdm(zip(timestamps, time_since_last_event)):
    if dt > 0.1:
        start_of_current_winow = t
    times_since_start_of_window.append(t - start_of_current_winow)
times_since_start_of_window = np.array(times_since_start_of_window)

min_width, max_width = 2,5
track_width_mask = (track_widths>=min_width) & (track_widths <= max_width)
if run==124:
    mask_1500keV_protons = (ranges > 31) & (ranges < 65) & (counts > 1.64e5) & (counts < 2.15e5)
    mask_750keV_protons = (ranges>20) & (ranges<30) & (counts>8.67e4) & (counts<9.5e4)& (track_widths>=min_width)
elif run==212:
    mask_1500keV_protons = (ranges > 40) & (ranges < 65) & (counts > 3e5) & (counts < 3.5e5) & (track_widths>=min_width) 
    mask_750keV_protons = (ranges>24) & (ranges<30) & (counts>1.5e5) & (counts<1.64e5) & (track_widths>=min_width) 
    pass
true_range_1500keV_proton = 46.7
true_range_750keV_protons = 16.1

plt.figure()
plt.title('1500 keV protons')
plt.scatter(ranges[mask_1500keV_protons], track_widths[mask_1500keV_protons], c=times_since_start_of_window[mask_1500keV_protons])
plt.xlabel('track width (mm)')
plt.ylabel('range (mm)')
plt.colorbar()

plt.figure()
plt.title('750 keV protons')
plt.scatter(ranges[mask_750keV_protons], track_widths[mask_750keV_protons], c=times_since_start_of_window[mask_750keV_protons])
plt.xlabel('track width (mm)')
plt.ylabel('range (mm)')
plt.colorbar()

endpoints = np.array(track_info_dict['endpoints'])
rscale, wscale, tscale = 25, 4, 0.05
#try mapping r->r + r'(r, t, w)= r + sum_{i,j,k s.t i+j+k < N} a_ijk r^i t^j w^k
N = 2
ijk_array = []
for n in range(N+1):
    for i in range(n+1):
        for j in range(n - i +1):
            k = n - i - j
            ijk_array.append((i,j,k))
ijk_array = np.array(ijk_array)

def map_r(a_ijk, r, t, w):
    new_r = np.copy(r)
    for ijk, a in zip(ijk_array, a_ijk):
        i,j,k = ijk
        new_r += a*((r/rscale)**i)*((t/tscale)**j)*((w/wscale)**k)
    if type(new_r) == np.ndarray:
        new_r[new_r < 0] = 0
    elif new_r < 0:
        new_r = 0
    return new_r

def map_endpoints(a_ijk, event_select_mask):
    to_return = []
    for event_index, pair in enumerate(endpoints):
        if not event_select_mask[event_index]:
            continue
        new_pair = []
        t = times_since_start_of_window[event_index]
        w = track_widths[event_index]
        new_pair = []
        for point in pair:
            r = np.sqrt(point[0]**2 + point[1]**2)
            if r == 0:
                new_pair.append(point) #points at the origin get mapped to points at the origin
                continue
            new_r = map_r(a_ijk, r, t, w)
            new_point = np.array(point, copy=True)
            new_point[0:2] *= new_r/r
            new_pair.append(new_point)
        to_return.append(new_pair)
    return np.array(to_return)

def map_ranges(a_ijk, event_select_mask):
    new_endpoints = map_endpoints(a_ijk, event_select_mask)
    return np.linalg.norm(new_endpoints[:,0,:] - new_endpoints[:, 1,:], axis=1)


def to_minimize(a_ijk):
    #try to minimize spread  in proton ranges within each peak, while preserving the distance between the two peaks
    p1500_ranges = map_ranges(a_ijk, mask_1500keV_protons&track_width_mask)
    p750_ranges = map_ranges(a_ijk, mask_750keV_protons&track_width_mask)
    to_return = np.std(p1500_ranges) + np.std(p750_ranges) + ((np.mean(p1500_ranges) - np.mean(p750_ranges)) - (true_range_1500keV_proton - true_range_750keV_protons))**2
    print(to_return, np.std(p1500_ranges), np.std(p750_ranges), np.mean(p1500_ranges), np.mean(p750_ranges))
    return to_return

package_directory = os.path.dirname(os.path.abspath(__file__))
fname = os.path.join(package_directory, '%s_run%d_rmap_order%d.pkl'%(experiment, run, N))
if os.path.exists(fname):
    print('optimizer previously run, loading saved result')
    with open(fname, 'rb') as file:
        res =  pickle.load(file)
else:
    print('optimizing a_ijk parameters')
    previous_fname = os.path.join(package_directory, '%s_run%d_rmap_order%d.pkl'%(experiment, run, N-1))
    #if a solution for N-1 exists, use this as starting guess. Otherwise guess r->r.
    a_ijk_guess = np.zeros(len(ijk_array))
    if os.path.exists(previous_fname):
        print('rmap exists for N-1, using as intial guess')
        with open(previous_fname, 'rb') as file:
            prev_res = pickle.load(file)
            prev_ijk_array = []
            for n in range(N):
                for i in range(n+1):
                    for j in range(n - i +1):
                        k = n - i - j
                        prev_ijk_array.append((i,j,k))
            prev_ijk_array = np.array(prev_ijk_array)
            for prev_a_ijk, prev_ijk in zip(prev_res.x, prev_ijk_array):
                for new_index, ijk in enumerate(ijk_array):
                    if np.all(ijk == prev_ijk):
                        a_ijk_guess[new_index] = prev_a_ijk
        
    res = opt.minimize(to_minimize, a_ijk_guess)
    with open(fname, 'wb') as file:
        pickle.dump(res, file)
print(res)
a_ijk_best = res.x
        

plt.figure()
plt.title('uncorrected RvE')
plt_mask = (ranges>0)&(ranges<150)&(counts>0)
plt.hist2d(counts[plt_mask], ranges[plt_mask], 200, norm=matplotlib.colors.LogNorm())

mapped_ranges = map_ranges(a_ijk_best, ranges==ranges)
plt.figure()
plt.title('RvE corrected using r-map')
plt_mask = (mapped_ranges>0)&(mapped_ranges<150)&(counts>0)
plt.hist2d(counts[plt_mask], mapped_ranges[plt_mask], 200, norm=matplotlib.colors.LogNorm())

plt.figure()
range_hist_bins = np.linspace(30, 70, 80)
plt.hist(ranges[mask_1500keV_protons], bins=range_hist_bins, alpha=0.6, label='uncorrected range')
plt.hist(mapped_ranges[mask_1500keV_protons], bins=range_hist_bins, alpha=0.6, label='corrected range')
plt.legend()

plt.figure()
wsquared = 16
r_obs = np.linspace(0, 50, 100)#radius at which charge was observed
plt.title('r map for track with %f mm width'%wsquared**0.5)
for t in [0, 0.02, 0.05, 0.1]:
    plt.plot(r_obs, map_r(a_ijk_best, r_obs, t, wsquared) - r_obs, label='%f s'%t)
plt.xlabel('position charge was observed')
plt.ylabel('r_dep - r_obs')
plt.legend()

plt.figure()
wsquared = 9
r_obs = np.linspace(0, 50, 100)#radius at which charge was observed
plt.title('r map for track with %f mm width'%wsquared**0.5)
for t in [0, 0.02, 0.05, 0.1]:
    plt.plot(r_obs, map_r(a_ijk_best, r_obs, t, wsquared) - r_obs, label='%f s'%t)
plt.xlabel('position charge was observed')
plt.ylabel('r_dep - r_obs')
plt.legend()


plt.show(block=False)

print('standard deviation of 1500 keV proton ranges: %f mm'%np.std(mapped_ranges[mask_1500keV_protons]))