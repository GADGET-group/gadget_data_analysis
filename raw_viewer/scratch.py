import os

import numpy as np
import matplotlib.pylab as plt
from tqdm import tqdm

from track_fitting import build_sim

run_number = 124
processed_directory = '/egr/research-tpc/shared/Run_Data/run_0124_raw_viewer/run_0124smart_20_length_thresh'
h5file = build_sim.get_rawh5_object('e21072', run_number)
zscale = h5file.zscale

#load histogram arrays
counts = np.load(os.path.join(processed_directory, 'counts.npy'))
dxys = np.load(os.path.join(processed_directory, 'dxy.npy'))
dts = np.load(os.path.join(processed_directory, 'dt.npy'))
max_veto_counts = np.load(os.path.join(processed_directory, 'veto.npy'))
timestamps = np.load(os.path.join(processed_directory, 'timestamps.npy'))

dzs = dts*zscale
ranges = np.sqrt(dzs**2 + dxys**2)
angles = np.arctan2(dzs, dxys)

#1500 keV protons
if False:
    selected_event_mask = (ranges > 35) & (ranges < 65) & (counts > 1.64e5) & (counts < 2.15e5)
    range_histogram_bins = np.linspace(35,65,10)
#4.4 MeV alpha from cathode
if False:
    selected_event_mask = (ranges > 25) & (ranges < 50) & (counts > 4.5e5) & (counts < 5.8e5)
    range_histogram_bins = np.linspace(25,45,10)
#4.4 MeV alpha NOT from cathode
if True:
    selected_event_mask = (ranges > 30) & (ranges < 55) & (counts > 5.9e5) & (counts < 7e5)
    range_histogram_bins = np.linspace(30,55,10)

first_event, last_event = h5file.get_event_num_bounds()
selected_events =  np.nonzero(selected_event_mask)[0] + first_event

ranges = ranges[selected_event_mask]

track_centers, principle_axes, variances_along_axes = [], [], []
widths, widths_xy, widths_z = [], [], []
thetas, phis = [],[]

time_since_last_event = timestamps - np.roll(timestamps, 1)
time_since_last_event[0] = .15 #we don't actuallly know what this is for the first event, so just putting a typical value for start of window
start_of_current_winow = 0

times_since_start_of_window = []
for t, dt in zip(timestamps, time_since_last_event):
    if dt > 0.1:
        start_of_current_winow = t
    times_since_start_of_window.append(t - start_of_current_winow)
times_since_start_of_window = np.array(times_since_start_of_window)

for evt in tqdm(selected_events):
    center, uu,dd,vv = h5file.get_track_axis(evt, threshold=20, return_all_svd_results=True)
    xs, ys, zs, es = h5file.get_xyze(evt, threshold=20, include_veto_pads=False)

    track_centers.append(center)
    principle_axes.append(vv)
    variances_along_axes.append(dd**2/(len(xs)-1))
    dir = vv[0]
    thetas.append(np.arctan2(np.sqrt(dir[0]**2 + dir[1]**2), np.abs(dir[2])))
    phis.append(np.arctan2(np.abs(dir[1]), np.abs(dir[0])))

    
    dir = dir/np.linalg.norm(dir)
    width, width_xy, width_z = 0,0,0 #width will be the standard deviation of the distance of charge from principle axis)
    for x,y,z,e in zip(xs, ys, zs, es):
        distance_from_axis = np.linalg.norm(np.cross(np.array([x,y,z]) - center, dir))
        center_xy = np.array([center[0], center[1], 1])
        distance_xy = np.linalg.norm(np.cross(np.array([x,y,0]) - center, dir))
        width += distance_from_axis**2*e
        width_xy += distance_xy**2*3
        
        distance_z = z - (center + (np.linalg.norm(np.array([x,y,z]) - center)*dir))[2]
        width_z += distance_z**2*e
    widths.append(np.sqrt(width/np.sum(es)))
    widths_xy.append(np.sqrt(width_xy/np.sum(es)))
    widths_z.append(np.sqrt(width_z/np.sum(es)))

widths = np.array(widths)
widths_xy = np.array(widths_xy)
widths_z = np.array(widths_z)
thetas = np.array(thetas)
phis = np.array(phis)
track_centers = np.array(track_centers)
principle_axes = np.array(principle_axes)
variances_along_axes = np.array(variances_along_axes)

#make histogram of track center distance from beam axis vs range
theta_min = np.radians(70)
theta_max = np.radians(90)
theta_mask = (thetas>=theta_min) & (thetas <= theta_max)
plt.figure()
rs = np.sqrt(track_centers[:,0]**2 + track_centers[:, 1]**2)
plt.scatter(rs[theta_mask], ranges[theta_mask], marker='.')
plt.xlabel('track centroid distance from beam axis (mm)')
plt.ylabel('track length (mm)')

#make histogram of range vs width
plt.figure()
plt.scatter(widths[theta_mask], ranges[theta_mask], marker='.')
plt.xlabel('track width (mm)')
plt.ylabel('track length (mm)')

plt.figure()
plt.scatter(widths_xy[theta_mask], ranges[theta_mask], marker='.')
plt.xlabel('track width xy (mm)')
plt.ylabel('track length (mm)')

plt.figure()
plt.scatter(widths_z[theta_mask], ranges[theta_mask], marker='.')
plt.xlabel('track width z (mm)')
plt.ylabel('track length (mm)')


plt.figure()
tsfew = times_since_start_of_window[selected_event_mask]
plt.scatter(np.sqrt(variances_along_axes[:,1][theta_mask]), ranges[theta_mask], c=tsfew[theta_mask], marker='.')
plt.colorbar()
plt.xlabel('track z width from pca')
plt.ylabel('track range (mm)')

plt.figure()
tslice = 0.025
for t in np.arange(tslice, 0.101, tslice):     
    plt.hist(ranges[theta_mask & (tsfew<t)&(tsfew > t-tslice)], label='%d - %d ms'%(int((t-tslice)*1000), int(t*1000)), bins=range_histogram_bins, alpha=(0.5+0.5*(0.1-t)/0.1))
plt.legend()

plt.show(block=False)