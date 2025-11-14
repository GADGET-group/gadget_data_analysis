import pickle

import numpy as np
import matplotlib.pylab as plt

from raw_viewer import process_runs

experiment = 'e23035'
for runs in [(136, 137, 138, 139)]:
    gain_match_path = '/egr/research-tpc/shared/e23035_prep/vault/gm.pkl'
    with open(gain_match_path, 'rb') as f:
        gain_match_result = pickle.load(f)
    pad_gains = gain_match_result.x

    veto_max = process_runs.get_max_veto_counts(experiment, runs)
    lengths =process_runs.get_lengths(experiment, runs) 
    energy = process_runs.get_gm_ic(experiment, runs, pad_gains)

    veto_mask = (lengths>1)&(lengths<25)&(veto_max<450)&(energy<1.5)

    centers = process_runs.get_quantity('track_center', experiment, runs)[veto_mask]
    xs, ys = centers[:,0], centers[:,1]
    good_centroid = np.isfinite(xs) & np.isfinite(ys)
    plt.figure()
    plt.hist2d(xs[good_centroid], ys[good_centroid], 100)
    plt.colorbar()

    plt.figure()
    plt.scatter(xs[good_centroid], ys[good_centroid], marker='.')
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    plt.title(runs)


plt.show()