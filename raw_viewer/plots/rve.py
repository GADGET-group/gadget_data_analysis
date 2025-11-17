import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.colors
import ROOT
import numpy as np

from raw_viewer import process_runs
from  raw_viewer import raw_h5_file

if False: #background runs before experiment
    experiment = 'e23035_prep_vault'
    run_range = (68, 73)
else: #during experiment
    experiment = 'e23035'
    run_range = (0,1000)#(0,1000)#(101, 143)

exclude_runs = [1,9, 19, 73, 113,210,216, 225, 226, 227, 228, 229] #210 needs to be transfered by Tyler
runs = []
for run in range(run_range[0], run_range[1]+1):
    if run not in exclude_runs and os.path.exists(process_runs.get_h5_path(experiment, run)):
        runs.append(run)


veto_thresh = 1000
rve_bins = (1000, 1000)
phist_bins = 3000
alphahist_bins = 500

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



plt.figure()
plt_mask = veto_mask&(lengths<400)&(lengths>1)
plt.title('runs: '+str(runs))
plt.hist2d(energy[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
plt.colorbar()
plt.xlabel('energy (MeV)')


# self.poly_selector = matplotlib.widgets.PolygonSelector(ax,self.set_cut_polygon)
# self.poly_selector = matplotlib.widgets.PolygonSelector(ax,self.set_cut_polygon)
# if len(self.rve_cut_verticies) > 0:
#     self.poly_selector.verts = self.rve_cut_verticies
# fig.show()
m = (108.4-32.2)/(3.628-2.25)
alpha_mask = veto_mask&(lengths<(energy*m+32.2 - m*2.25))

m = (159.2-26.2)/(2.81-0.619)
proton_mask = veto_mask&(~alpha_mask)&(lengths<(energy*m+26.6 - m*0.619))

plt.figure()
plt.hist(energy[proton_mask], phist_bins)
plt.title('proton energy spectrum, runs: '+str(runs))
plt.xlabel('energy (MeV)')

plt.figure()
plt.title('protons selected in RVE, runs: '+str(runs))
plt.hist2d(energy[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
plt.scatter(energy[proton_mask], lengths[proton_mask], marker='.', alpha=0.5, color='red')
plt.colorbar()
print(str(runs) + "has " + str(len(proton_mask[proton_mask])) + " protons")


plt.figure()
plt.hist(energy[alpha_mask], alphahist_bins)
plt.title('alpha energy spectrum, runs: '+str(runs))
plt.xlabel('energy (MeV)')

plt.figure()
plt.title('alphas selected in RVE, runs: '+str(runs))
plt.hist2d(energy[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
plt.scatter(energy[alpha_mask], lengths[alpha_mask], marker='.', alpha=0.5, color='red')
plt.colorbar()
print(str(runs) + "has " + str(len(alpha_mask[alpha_mask])) + " alphas")


#TODO: correct for times runs were not instantly started again after previous run ended
run_t_offset = [0]
run_ts = []
for run in runs:
    run_ts.append(process_runs.get_quantity('timestamps', experiment, [run]))
for i in range(1, len(runs)):
    if run_ts[i][0] <= run_ts[i-1][-1]:
        run_t_offset.append(run_t_offset[-1] + run_ts[i-1][-1])
    else:
        run_t_offset.append(run_t_offset[-1])
for i in range(len(runs)):
    run_ts[i] = run_ts[i] + run_t_offset[i]
run_ts = np.concatenate(run_ts)

plt.figure()
plt.title('alphas')
tve_bins = (50, 50)
plt.hist2d(energy[alpha_mask], run_ts[alpha_mask]/3600, bins=tve_bins, norm=matplotlib.colors.LogNorm())
plt.xlabel('energy (MeV)')
plt.ylabel('time since start of experiment (hours)')
plt.colorbar()

if False:
    plt.figure()
    plt.title('protons')
    tsbo=process_runs.get_time_since_beam_off(experiment, runs)
    tve_bins = (50, 50)
    plt.hist2d(energy[proton_mask&(tsbo>0)], tsbo[proton_mask&(tsbo>0)], bins=tve_bins, norm=matplotlib.colors.LogNorm())

    plt.figure()
    plt.title('alphas')
    tve_bins = (50, 50)
    plt.hist2d(energy[alpha_mask&(tsbo>0)], tsbo[alpha_mask&(tsbo>0)], bins=tve_bins, norm=matplotlib.colors.LogNorm())

plt.show(block=False)
