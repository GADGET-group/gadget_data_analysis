import pickle

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.colors
import ROOT

from raw_viewer import process_runs
from  raw_viewer import raw_h5_file

experiment = 'e23035'
runs =  [101,102, 106, 109, 110, 111,112, 114, 117, 119, 120, 122, 123, 124, 125, 126, 127, 128, 130, 131, 132, 133, 134, 135, 136, 137]

veto_thresh = 300
rve_bins = (300, 300)

#load pad gain match
gain_match_path = '/egr/research-tpc/shared/e23035_prep/vault/gm.pkl'
with open(gain_match_path, 'rb') as f:
    gain_match_result = pickle.load(f)
pad_gains = gain_match_result.x
for pad in raw_h5_file.VETO_PADS:
    pad_gains[pad] = 0

lengths = process_runs.get_lengths(experiment, runs)
cpp = process_runs.get_quantity('pad_charge', experiment, runs)
#veto_counts = process_runs.get_veto_counts(exp, runs)
veto_max = process_runs.get_max_veto_counts(experiment, runs)
charge_widths = process_runs.get_quantity('charge_width', experiment,runs)
energy = process_runs.get_gm_ic(experiment, runs, pad_gains)

veto_mask = (veto_max < veto_thresh)

alpha_mask = veto_mask&(energy>1.39)&(energy<4.2)&(lengths<37.5)
proton_mask = (~alpha_mask)&(energy<2.76)

fig, ax = plt.subplots()
plt_mask = veto_mask&(lengths<400)&(lengths>1)
ax.set_title('runs: '+str(runs))
ax.hist2d(energy[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
plt.xlabel('energy (MeV)')
print(str(runs) + "has " + str(len(proton_mask[proton_mask])) + " protons")

# self.poly_selector = matplotlib.widgets.PolygonSelector(ax,self.set_cut_polygon)
# self.poly_selector = matplotlib.widgets.PolygonSelector(ax,self.set_cut_polygon)
# if len(self.rve_cut_verticies) > 0:
#     self.poly_selector.verts = self.rve_cut_verticies
# fig.show()

plt.show(block=False)