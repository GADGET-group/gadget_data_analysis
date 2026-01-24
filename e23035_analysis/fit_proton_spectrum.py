import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.colors
import ROOT
import numpy as np
from tqdm import tqdm
from scipy import optimize

from raw_viewer import process_runs
from  raw_viewer import raw_h5_file
from raw_viewer import ddas_interface
from e23035_analysis import e23035_runs

experiment = 'e23035'
run_range = e23035_runs.run_df['GET'][(e23035_runs.run_df['Run Type']=='59Zn') & (e23035_runs.run_df['Field Cage Functional?'] == 'yes')]

get_runs = []
for run in run_range:
    if not np.isnan(run):
        # if run not in exclude_runs and os.path.exists(process_runs.get_h5_path(experiment, run)):
        get_runs.append(run)

load_ddas = False

rve_bins = (300, 300)
phist_bins = np.linspace(0, 4, 1001)
alphahist_bins = 100

lengths = process_runs.get_lengths(experiment, get_runs)
cpp = process_runs.get_quantity('pad_charge', experiment, get_runs)
#veto_counts = process_runs.get_veto_counts(exp, runs)
veto_max = process_runs.get_max_veto_counts(experiment, get_runs)
charge_widths = process_runs.get_quantity('charge_width', experiment,get_runs)
#energy = process_runs.get_gm_ic(experiment, get_runs, pad_gains)
energy = e23035_runs.get_energy_MeV(get_runs)
angles = process_runs.get_angle(experiment, get_runs)

#veto_mask = (veto_max < veto_thresh)#&(angles>np.radians(5))#process_runs.get_outer_ring_counts(experiment, runs)<113#
veto_mask = e23035_runs.get_veto_mask(get_runs)


plt.figure()
plt_mask = veto_mask&(lengths>1)&(lengths<400)
plt.title('runs: '+str(get_runs))
plt.hist2d(energy[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
plt.colorbar()
plt.xlabel('energy (MeV)')
plt.ylabel('range (mm)')

proton_mask = e23035_runs.get_proton_mask(get_runs)

plt.figure()
plt.hist(energy[proton_mask], phist_bins)
plt.title('proton energy spectrum, runs: '+str(get_runs))
plt.xlabel('energy (MeV)')
plt.ylabel('counts/keV')
plt.yscale('log')

plt.figure()
plt.title('protons selected in RVE, runs: '+str(get_runs))
plt.hist2d(energy[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
plt.scatter(energy[proton_mask], lengths[proton_mask], marker='.', alpha=0.5, color='red')
plt.colorbar()
print(str(get_runs) + "has " + str(len(proton_mask[proton_mask])) + " protons")

plt.show(block=False)

# hist = ROOT.TH1D('Ep', 'Ep', 4000, 0, 4)
# hist.Fill(energy[proton_mask])
# hist.Draw()

np.save('to_fit.npy', energy[proton_mask])

evt_runs, evt_nums = process_runs.get_run_and_event_numbers(experiment, get_runs)