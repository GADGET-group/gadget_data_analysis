import pickle

import numpy as np
import matplotlib.pylab as plt

from raw_viewer import process_runs

experiment = 'e23035'
bins = np.linspace(0, 6, 25)

for runs in [(106,),(122,),(125,)]:
    gain_match_path = '/egr/research-tpc/shared/e23035_prep/vault/gm.pkl'
    with open(gain_match_path, 'rb') as f:
        gain_match_result = pickle.load(f)
    pad_gains = gain_match_result.x

    veto_max = process_runs.get_max_veto_counts(experiment, runs)
    lengths =process_runs.get_lengths(experiment, runs) 
    energy = process_runs.get_gm_ic(experiment, runs, pad_gains)

    veto_mask = (lengths>1)&(lengths<30)&(veto_max<450)&(energy<1.5)
    alpha_mask = veto_mask&(energy>1.39)&(energy<4.2)&(lengths<37.5)
    proton_mask = (~alpha_mask)&(energy<2.76)

    charge_widths = process_runs.get_quantity('charge_width', experiment,runs)

    plt.figure()
    plt.hist(charge_widths[proton_mask], bins)
    plt.title(str(runs)+':protons')

    plt.figure()
    plt.hist(charge_widths[alpha_mask], bins)
    plt.title(str(runs)+'alphas')

plt.show(block=False)