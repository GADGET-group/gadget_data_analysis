import os
import pickle

import numpy as np

import matplotlib.pylab as plt
import matplotlib.colors
from tqdm import tqdm

from track_fitting import build_sim
from track_fitting.field_distortion import extract_track_axis_info

experiment, run = 'e21072', 212


track_info_dict = extract_track_axis_info.get_track_info(experiment, run)
endpoints = np.array(track_info_dict['endpoints'])

processed_directory = '/egr/research-tpc/shared/Run_Data/run_%04d_raw_viewer/run_%04dsmart'%(run, run)

#load histogram arrays
counts = np.load(os.path.join(processed_directory, 'counts.npy'))
dxys = np.load(os.path.join(processed_directory, 'dxy.npy'))
dts = np.load(os.path.join(processed_directory, 'dt.npy'))
max_veto_counts = np.load(os.path.join(processed_directory, 'veto.npy'))
timestamps = np.load(os.path.join(processed_directory, 'timestamps.npy'))

h5file = build_sim.get_rawh5_object(experiment, run)
dzs = dts*h5file.zscale
MeV = build_sim.get_integrated_charge_energy_offset(experiment, run) + counts/build_sim.get_adc_counts_per_MeV(experiment, run)
ranges = np.linalg.norm(endpoints[:,0] - endpoints[:,1], axis=1)

if run == 124:
    veto_mask = max_veto_counts<300
elif run == 212:
    veto_mask = max_veto_counts<150

#use track angle from pca rather than that exported by raw event viewer
angles = []
track_vects = []
for axes in track_info_dict['principle_axes']:
    dir = axes[0]
    angles.append(np.arctan2(np.sqrt(dir[0]**2 + dir[1]**2), np.abs(dir[2])))
angles = np.array(angles)
track_vects = np.array(track_vects)

rve_mask = veto_mask & (ranges<100) & (ranges>0)
plt.figure()
plt.hist2d(MeV[rve_mask], ranges[rve_mask], norm=matplotlib.colors.LogNorm(), bins=100)

plt.figure()
widths_above_thresh = np.array(track_info_dict['width_above_threshold'])
ranges = ranges - widths_above_thresh
rve_mask = veto_mask & (ranges<100) & (ranges>0)
plt.hist2d(MeV[rve_mask], ranges[rve_mask], norm=matplotlib.colors.LogNorm(), bins=100)
plt.show()

