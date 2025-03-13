import os

import numpy as np
from tqdm import tqdm
import matplotlib.pylab as plt
import matplotlib.colors
import scipy.optimize as opt

from track_fitting.field_distortion import extract_track_axis_info
from track_fitting import build_sim

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

selected_event_mask = (ranges > 31) & (ranges < 65) & (counts > 1.64e5) & (counts < 2.15e5)
true_range = 46.7

selected_ranges = ranges[selected_event_mask]
selected_times_into_decay_window = times_since_start_of_window[selected_event_mask]
selected_angles = angles[selected_event_mask]
selected_track_widths = [] #contains square of variance form pca
for i in range(len(ranges)):
    if selected_event_mask[i]:
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

plt.show(block=False)

'''
plt_mask = (counts>0)&(ranges<150)
plt.figure()
plt.hist2d(counts[plt_mask], ranges[plt_mask], 200, norm=matplotlib.colors.LogNorm())'''
