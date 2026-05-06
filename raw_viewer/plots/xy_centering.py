import pickle
import os

import numpy as np
import matplotlib.pylab as plt

from raw_viewer import process_runs

experiment = 'e23035'
run_range = [73,73]#(148,150)#(101, 143)

exclude_runs = []# [1,9, 73, 113,210,216, 225, 226, 227, 228, 229] #210 needs to be transfered by Tyler
runs = []
for run in range(run_range[0], run_range[1]+1):
    if run not in exclude_runs and os.path.exists(process_runs.get_h5_path(experiment, run)):
        runs.append(run)
runs = [250, 253]

#runs=[148,149,150]
print(runs)
#for runs in runs:
#gain_match_path = '/egr/research-tpc/shared/e23035_prep/vault/gm.pkl'
gain_match_path = '/egr/research-tpc/adamsa52/gadget_analysis/fft6_res3.pkl'
with open(gain_match_path, 'rb') as f:
    gain_match_result = pickle.load(f)
pad_gains = np.ones(1024)*np.average(gain_match_result.pad_gains)

veto_max = process_runs.get_max_veto_counts(experiment, runs)
lengths =process_runs.get_lengths(experiment, runs) 
energy = process_runs.get_gm_ic(experiment, runs, pad_gains)
angles = process_runs.get_angle(experiment, runs)
xy_len = np.abs(np.sin(angles)*lengths)


veto_mask = (veto_max<350)&(energy>7)&(energy<8.4)&(lengths>93)&(np.degrees(angles)<20)#&(xy_len < 20)#&(energy<1.5)&(lengths>5)&(lengths<30)#
veto_mask = (veto_max<200) &(xy_len < 10)

centers = process_runs.get_quantity('track_center', experiment, runs)[veto_mask]
endpoints = process_runs.get_quantity('endpoints', experiment, runs)
xs, ys = centers[:,0], centers[:,1]
good_centroid = np.isfinite(xs) & np.isfinite(ys)
plt.figure()
plt.hist2d(xs[good_centroid], ys[good_centroid], 31)
plt.colorbar()

plt.figure()
plt.scatter(xs[good_centroid], ys[good_centroid], marker='.')
plt.xlim(-50, 50)
plt.ylim(-50, 50)
plt.title(runs)

plt.figure()
print(np.shape(endpoints))
xs, ys = endpoints[:,0,0], endpoints[:,0,1]
use_other_endpoints = endpoints[:,0,2]<endpoints[:,1,2]
xs[use_other_endpoints] = endpoints[:,1,0][use_other_endpoints]
ys[use_other_endpoints] = endpoints[:,1,1][use_other_endpoints]
plt.scatter(xs[veto_mask], ys[veto_mask], marker='.')

plt.figure()
plt.hist(np.sqrt(xs[veto_mask]**2+ ys[veto_mask]**2), bins=30)

plt.show(block=False)

evt_runs, evt_nums = process_runs.get_run_and_event_numbers(experiment, runs)