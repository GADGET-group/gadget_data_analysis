import pickle
import os

import numpy as np
import matplotlib.pylab as plt

from raw_viewer import process_runs

experiment = 'e25058'
run_range = (50,50)#(101, 143)

exclude_runs = [1,9, 73, 113,210,216, 225, 226, 227, 228, 229] #210 needs to be transfered by Tyler
runs = []
for run in range(run_range[0], run_range[1]+1):
    if run not in exclude_runs and os.path.exists(process_runs.get_h5_path(experiment, run)):
        runs.append(run)

#for runs in runs:
gain_match_path = '/egr/research-tpc/shared/e23035_prep/vault/gm.pkl'
with open(gain_match_path, 'rb') as f:
    gain_match_result = pickle.load(f)
pad_gains = gain_match_result.x

veto_max = process_runs.get_max_veto_counts(experiment, runs)
lengths =process_runs.get_lengths(experiment, runs) 
energy = process_runs.get_gm_ic(experiment, runs, pad_gains)
angles = process_runs.get_angle(experiment, runs)
xy_len = np.abs(np.sin(angles)*lengths)


veto_mask = (veto_max<500)&(xy_len < 10)#&(energy<1.5)&(lengths>5)&(lengths<30)#

centers = process_runs.get_quantity('track_center', experiment, runs)[veto_mask]
xs, ys = centers[:,0], centers[:,1]
good_centroid = np.isfinite(xs) & np.isfinite(ys)
plt.figure()
plt.hist2d(xs[good_centroid], ys[good_centroid], 11)
plt.colorbar()

plt.figure()
plt.scatter(xs[good_centroid], ys[good_centroid], marker='.')
plt.xlim(-50, 50)
plt.ylim(-50, 50)
plt.title(runs)


plt.show(block=False)