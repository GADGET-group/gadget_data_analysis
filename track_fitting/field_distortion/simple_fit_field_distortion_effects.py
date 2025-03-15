import os
import pickle

import numpy as np
from tqdm import tqdm
import matplotlib.pylab as plt
import matplotlib.colors
import scipy.optimize as opt

from track_fitting.field_distortion import extract_track_axis_info
from track_fitting import build_sim

do_simple_correction=False
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
print('calculating event times in decay window')
for t, dt in tqdm(zip(timestamps, time_since_last_event)):
    if dt > 0.1:
        start_of_current_winow = t
    times_since_start_of_window.append(t - start_of_current_winow)
times_since_start_of_window = np.array(times_since_start_of_window)

mask_1500keV_protons = (ranges > 31) & (ranges < 65) & (counts > 1.64e5) & (counts < 2.15e5)
true_range_1500keV_proton = 46.7
mask_750keV_protons = (ranges>20) & (ranges<30) & (counts>8.67e4) & (counts<9.5e4)
true_range_750keV_protons = 16.1

if do_simple_correction:
    selected_ranges = ranges[mask_1500keV_protons]
    selected_times_into_decay_window = times_since_start_of_window[mask_1500keV_protons]
    selected_angles = angles[mask_1500keV_protons]
    selected_track_widths = [] #contains square of variance form pca
    for i in range(len(ranges)):
        if mask_1500keV_protons[i]:
            selected_track_widths.append(track_info_dict['variance_along_axes'][i][1])
    selected_track_widths = np.array(selected_track_widths)

    theta_mask = selected_angles>np.radians(70)
    plt.figure()
    plt.title('selected events to fit range to within 20 degrees of the pad plane')
    plt.scatter(np.sqrt(selected_track_widths), selected_ranges, c=selected_times_into_decay_window, marker='.')
    plt.colorbar()
    plt.xlabel('width from pca')
    plt.ylabel('track range (mm)')

    plt.figure()
    plt.title('selected events to fit range to within 20 degrees of the pad plane')
    plt.scatter(np.sqrt(selected_track_widths[theta_mask]), selected_ranges[theta_mask], c=selected_times_into_decay_window[theta_mask], marker='.')
    plt.colorbar()
    plt.xlabel('width from pca')
    plt.ylabel('track range (mm)')

    def to_minimize(constants):
        r_corrects = r =  (1 + constants[2]*selected_times_into_decay_window + constants[3]*selected_angles + constants[4]*selected_track_widths)*selected_ranges
        return np.std(r_corrects)

    print('minimizing parameters')
    res = opt.minimize(to_minimize, (0,0,0,0,0))

    selected_ranges_corrected =  (1 + res.x[[2]]*selected_times_into_decay_window + res.x[3]*selected_angles + res.x[4]*selected_track_widths)*selected_ranges

    corrected_ranges = []
    good_counts = []
    for i in range(len(ranges)):
        if len(track_info_dict['variance_along_axes'][i]) ==3:
            good_counts.append(counts[i])
            corrected_ranges.append( (1 + res.x[2]*times_since_start_of_window[i] + res.x[3]*angles[i] + res.x[4]*track_info_dict['variance_along_axes'][i][1])*ranges[i] )

    good_counts = np.array(good_counts)
    corrected_ranges = np.array(corrected_ranges)
    plt.figure()
    plt.title('RvE with corrected ranges')
    plt_mask = (good_counts > 0) & (corrected_ranges < 150) & (corrected_ranges > 0)
    plt.hist2d(good_counts[plt_mask], corrected_ranges[plt_mask], 200, norm=matplotlib.colors.LogNorm())

    plt.figure()
    plt.title('uncorrected RvE')
    plt_mask = (ranges>0)&(ranges<150)&(counts>0)
    plt.hist2d(counts[plt_mask], ranges[plt_mask], 200, norm=matplotlib.colors.LogNorm())

    plt.figure()
    range_hist_bins = np.linspace(30, 65, 70)
    plt.hist(selected_ranges, bins=range_hist_bins, alpha=0.6, label='uncorrected range')
    plt.hist(selected_ranges_corrected, bins=range_hist_bins, alpha=0.6, label='corrected range')
    plt.legend()


if do_rmap:
    endpoints = np.array(track_info_dict['endpoints'])
    #try mapping r->r'(r, t, w)=sum_{i,j,k s.t i+j+k < N} a_ijk r^i t^j w&k
    N = 5
    ijk_array = []
    for n in range(N+1):
        for i in range(n+1):
            for j in range(n - i +1):
                k = n - i - j
                ijk_array.append((i,j,k))
    ijk_array = np.array(ijk_array)

    #guess a_ijk = 0 except a_100 = 1
    a_ijk_guess = np.zeros(len(ijk_array))
    a_ijk_guess[np.all(ijk_array==(1,0,0), axis=1)] = 1

    def map_r(a_ijk, r, t, w):
        new_r = 0
        for ijk, a in zip(ijk_array, a_ijk):
            i,j,k = ijk
            new_r += a*(r**i)*(t**j)*(w**k)
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
            if len(track_info_dict['variance_along_axes'][event_index]) == 3:
                w = track_info_dict['variance_along_axes'][event_index][1]
            else:
                w = 0
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
        p1500_ranges = map_ranges(a_ijk, mask_1500keV_protons)
        p750_ranges = map_ranges(a_ijk, mask_750keV_protons)
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
        res = opt.minimize(to_minimize, a_ijk_guess)
        with open(fname, 'wb') as file:
            pickle.dump(res, file)
    print(res)
    a_ijk_best = res.x
        

plt.figure()
plt.title('uncorrected RvE')
plt_mask = (ranges>0)&(ranges<150)&(counts>0)
plt.hist2d(counts[plt_mask], ranges[plt_mask], 200, norm=matplotlib.colors.LogNorm())

plt.figure()
plt.title('RvE corrected using r-map')
mapped_ranges = map_ranges(a_ijk_best, ranges==ranges)
plt_mask = (mapped_ranges>0)&(mapped_ranges<150)&(counts>0)
plt.hist2d(counts[plt_mask], mapped_ranges[plt_mask], 200, norm=matplotlib.colors.LogNorm())

plt.figure()
range_hist_bins = np.linspace(30, 70, 80)
plt.hist(ranges[mask_1500keV_protons], bins=range_hist_bins, alpha=0.6, label='uncorrected range')
plt.hist(mapped_ranges[mask_1500keV_protons], bins=range_hist_bins, alpha=0.6, label='corrected range')
plt.legend()

plt.figure()
wsquared = 16
r_obs = np.linspace(0, 50, 10)#radius at which charge was observed
plt.title('r map for track with %f mm width'%wsquared**0.5)
for t in [0, 0.02, 0.05, 0.1]:
    plt.plot(r_obs, map_r(a_ijk_best, r_obs, t, wsquared) - r_obs, label='%f s'%t)
plt.xlabel('position charge was observed')
plt.ylabel('r_dep - r_obs')
plt.legend()

plt.show(block=False)

print('standard deviation of 1500 keV proton ranges: %f mm'%np.std(mapped_ranges[mask_1500keV_protons]))