import pickle

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.colors

from raw_viewer import process_runs

experiment = 'e23035_prep_vault'
runs = (33,49)
veto_thresh = 350
rve_bins = (300, 300)

#load pad gain match
gain_match_path = '/egr/research-tpc/shared/e23035_prep/vault/gm.pkl'
with open(gain_match_path, 'rb') as f:
    gain_match_result = pickle.load(f)
pad_gains = gain_match_result.x

lengths = process_runs.get_lengths(experiment, runs)
cpp = process_runs.get_quantity('pad_charge', experiment, runs)
#veto_counts = process_runs.get_veto_counts(exp, runs)
veto_max = process_runs.get_max_veto_counts(experiment, runs)
charge_widths = process_runs.get_quantity('charge_width', experiment,runs)
energy = process_runs.get_gm_ic(experiment, runs, pad_gains)

veto_mask = (veto_max < veto_thresh)

fig = plt.figure()
plt_mask = veto_mask&(lengths<400)&(lengths>1)
plt.title('runs: '+str(runs))
plt.hist2d(energy[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
plt.xlabel('energy (MeV)')

