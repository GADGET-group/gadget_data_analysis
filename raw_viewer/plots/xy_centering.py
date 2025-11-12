import numpy as np
import matplotlib.pylab as plt

from raw_viewer import process_runs

experiment = 'e21072'
runs = (99, 100, 101,)
pad_gains = np.ones(1024)

veto_max = process_runs.get_max_veto_counts(experiment, runs)
lengths =process_runs.get_lengths(experiment, runs) 
energy = process_runs.get_gm_ic(experiment, runs, pad_gains)

veto_mask = (lengths>10)&(lengths<20)&(veto_max<450)

centers = process_runs.get_quantity('track_center', experiment, runs)[veto_mask]
xs, ys = centers[:,0], centers[:,1]
good_centroid = np.isfinite(xs) & np.isfinite(ys)
plt.figure()
plt.hist2d(xs[good_centroid], ys[good_centroid], 100)
plt.colorbar()

plt.figure()
plt.scatter(xs[good_centroid], ys[good_centroid], marker='.')
plt.show()