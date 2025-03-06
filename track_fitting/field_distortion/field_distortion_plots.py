import os
import pickle

import numpy as np
import cupy as cp
import matplotlib.pylab as plt
import matplotlib.colors
from tqdm import tqdm

from track_fitting import build_sim

run_number = 124
if run_number == 124:
    processed_directory = '/egr/research-tpc/shared/Run_Data/run_0124_raw_viewer/run_0124smart'
elif run_number == 212:
    processed_directory = '/egr/research-tpc/shared/Run_Data/run_0212_raw_viewer/run_0212smart'

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

std_plots = False

def get_track_info():
    '''
    Get information about track direction, width, and charge per pad, which isn't normally stored when processing runs.
    Only redoes processing if a pickled version of this information isn't available.
    '''
    package_directory = os.path.dirname(os.path.abspath(__file__))
    fname = os.path.join(package_directory, 'run%d.pkl'%run_number)
    if os.path.exists(fname):
        return pickle.load(fname)
    else:
        first_event, last_event = h5file.get_event_num_bounds()
        track_centers, uus, vvs, dds, pad_charges = [],[],[],[],[]
        for evt in tqdm(range(first_event, last_event + 1)):
            center, uu,dd,vv = h5file.get_track_axis(evt, return_all_svd_results=True)
            track_centers.append(center)
            uus.append(uus)
            dds.append(dds)
            pad_counts = np.zeros(1024)
            for pad, trace in zip(*h5file.get_pad_traces(evt)):
                pad_counts[pad] = np.sum(trace)
            pad_charges.append(pad_counts)
        track_centers, uus, vvs, dds, pad_charges = np.array(track_centers), np.array(uus), np.array(vvs), np.array(dds), np.array(pad_charges)
        to_return={'track_center':track_centers, 'uu': uus, 'vv':vv, 'dd':dds, 'pad_charge': pad_charges}
        pickle.dump(to_return, fname)
        return to_return

if run_number == 124:
    #1500 keV protons
    if True:
        selected_event_mask = (ranges > 31) & (ranges < 65) & (counts > 1.64e5) & (counts < 2.15e5)
        range_histogram_bins = np.linspace(31,70,25)
    #4.4 MeV alpha from cathode
    if False:
        selected_event_mask = (ranges > 25) & (ranges < 50) & (counts > 4.5e5) & (counts < 5.8e5)
        range_histogram_bins = np.linspace(25,45,10)
    #4.4 MeV alpha NOT from cathode
    if False:
        selected_event_mask = (ranges > 30) & (ranges < 55) & (counts > 5.9e5) & (counts < 7e5)
        range_histogram_bins = np.linspace(30,55,10)
elif run_number == 212:
    #1500 keV protons
    if True:
        selected_event_mask = (ranges > 31) & (ranges < 70) & (counts > 3e5) & (counts < 3.6e5)
        range_histogram_bins = np.linspace(31,70,25)


first_event, last_event = h5file.get_event_num_bounds()
selected_events =  np.nonzero(selected_event_mask)[0] + first_event


all_ranges = np.array(ranges, copy=True)
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

track_info_dict = get_track_info()

for evt in tqdm(selected_events):
    center, uu,dd,vv = track_info_dict['center'][evt], track_info_dict['uu'][evt], track_info_dict['dd'][evt], track_info_dict['vv'][evt], 
    xs, ys, zs, es = h5file.get_xyze(evt, threshold=20, include_veto_pads=False)

    track_centers.append(center)
    principle_axes.append(vv)
    variances_along_axes.append(dd**2/(len(xs)-1))
    dir = vv[0]
    thetas.append(np.arctan2(np.sqrt(dir[0]**2 + dir[1]**2), np.abs(dir[2])))
    phis.append(np.arctan2(np.abs(dir[1]), np.abs(dir[0])))

    
    dir = dir/np.linalg.norm(dir)
    width, width_xy, width_z = 0,0,0 #width will be the standard deviation of the distance of charge from principle axis)
    if std_plots:
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

if std_plots:
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
plt.hist2d(rs[theta_mask], ranges[theta_mask], bins=20)
plt.xlabel('track centroid distance from beam axis (mm)')
plt.ylabel('track length (mm)')

if std_plots:
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

plt.figure()
plt.hist2d(tsfew[tsfew>0], counts[selected_event_mask][tsfew>0], bins=30)
plt.xlabel('time since first event in decay window (s)')
plt.ylabel('adc counts')
plt.colorbar()

plt.figure()
plt.title('events >70 ms after first event in decay window')
hist_mask = (times_since_start_of_window>0.07) & (all_ranges > 0) & (all_ranges<150) 
plt.hist2d(counts[hist_mask], all_ranges[hist_mask], bins=200, norm=matplotlib.colors.LogNorm())
plt.ylabel('range (mm)')
plt.xlabel('adc counts')
plt.colorbar()

plt.figure()
plt.title('all selected events (no theta filter)')
plt.hist2d(tsfew[tsfew>0], np.sqrt(variances_along_axes[:,1][tsfew>0]), bins=100)
plt.colorbar()
plt.ylabel('track z width from pca')
plt.xlabel('time since first event in decay window (s)')

plt.figure()
#selected events centroid vs time
rs = np.sqrt(track_centers[:,0]**2 + track_centers[:, 1]**2)
plt.hist2d(rs[tsfew>0], tsfew[tsfew>0], bins=20)
plt.colorbar()
plt.xlabel('track centroid distance from beam axis (mm)')
plt.ylabel('time since first event in decay window (s)')

plt.show(block=False)