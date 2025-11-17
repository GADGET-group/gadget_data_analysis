import os
import pickle
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.colors
import ROOT

from raw_viewer import process_runs
from  raw_viewer import raw_h5_file


experiment = 'e23035'
run = 167
runs=[run]


veto_thresh = 400
rve_bins = (143, 143)

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



fig, ax = plt.subplots()
plt_mask = veto_mask&(lengths<400)&(lengths>1)
ax.set_title('runs: '+str(runs))
ax.hist2d(energy[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
plt.xlabel('energy (MeV)')


# self.poly_selector = matplotlib.widgets.PolygonSelector(ax,self.set_cut_polygon)
# self.poly_selector = matplotlib.widgets.PolygonSelector(ax,self.set_cut_polygon)
# if len(self.rve_cut_verticies) > 0:
#     self.poly_selector.verts = self.rve_cut_verticies
# fig.show()
m = (108.4-32.2)/(3.628-2.25)
alpha_mask = veto_mask&(lengths<(energy*m+32.2 - m*2.25))

m = (159.2-26.2)/(2.81-0.619)
proton_mask = veto_mask&(~alpha_mask)&(lengths<(energy*m+26.6 - m*0.619))&(energy>0.95)

plt.figure()
plt.hist(energy[proton_mask], 200)
plt.title('proton energy spectrum, runs: '+str(runs))
plt.xlabel('energy (MeV)')

plt.figure()
plt.title('protons selected in RVE, runs: '+str(runs))
plt.hist2d(energy[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
plt.scatter(energy[proton_mask], lengths[proton_mask], marker='.', alpha=0.5, color='red')
print(str(runs) + "has " + str(len(proton_mask[proton_mask])) + " protons")

plt.figure()
plt.title('protons')
tsbo=process_runs.get_time_since_beam_off(experiment, runs)
tve_bins = (50, 50)
plt.hist2d(energy[proton_mask&(tsbo>0)], tsbo[proton_mask&(tsbo>0)], bins=tve_bins, norm=matplotlib.colors.LogNorm())

num_protons = len(proton_mask[proton_mask])
print("Total number of protons in run "+str(run)+": ", num_protons)

ts = process_runs.get_quantity('timestamps', experiment, runs)
print('run duration (s): ', (ts[-1]-ts[0]))

print('protons per second = ', num_protons/(ts[-1]-ts[0]))

plt.show(block=False)
