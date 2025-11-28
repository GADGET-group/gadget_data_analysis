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
    experiment = 'e25058'
    run_range = (121,121)#(0,1000)#(101, 143)

if experiment == 'e23035':
    exclude_runs = [1,9, 19, 73, 113,210,216, 225, 226, 227, 228, 229, #210 needs to be transfered by Tyler
                    289,290, 291, 292, 293, 294, 295, 296, 297, 298]#41 deg angle runs
elif experiment == 'e25058':
    exclude_runs = []
runs = []
for run in range(run_range[0], run_range[1]+1):
    if run not in exclude_runs and os.path.exists(process_runs.get_h5_path(experiment, run)):
        runs.append(run)


veto_thresh = np.inf#250
rve_bins = (300, 300)
phist_bins = np.linspace(0, 4, 4000)
alphahist_bins = 500

#load pad gain match
gain_match_path = '/egr/research-tpc/shared/e23035_prep/vault/gm.pkl'
with open(gain_match_path, 'rb') as f:
    gain_match_result = pickle.load(f)
pad_gains = gain_match_result.x
#pad_gains = np.ones(1024)*np.mean(gain_match_result.x)

lengths = process_runs.get_lengths(experiment, runs)
cpp = process_runs.get_quantity('pad_charge', experiment, runs)
#veto_counts = process_runs.get_veto_counts(exp, runs)
veto_max = process_runs.get_max_veto_counts(experiment, runs)
charge_widths = process_runs.get_quantity('charge_width', experiment,runs)
energy = process_runs.get_gm_ic(experiment, runs, pad_gains)

veto_mask = (veto_max < veto_thresh)



plt.figure()
plt_mask = veto_mask# &(lengths>1)&(lengths<400)
plt.title('runs: '+str(runs))
plt.hist2d(energy[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
plt.colorbar()
plt.xlabel('energy (MeV)')
plt.ylabel('range (mm)')


# self.poly_selector = matplotlib.widgets.PolygonSelector(ax,self.set_cut_polygon)
# self.poly_selector = matplotlib.widgets.PolygonSelector(ax,self.set_cut_polygon)
# if len(self.rve_cut_verticies) > 0:
#     self.poly_selector.verts = self.rve_cut_verticies
# fig.show()
m = (108.4-32.2)/(3.628-2.25)
m2 = (23.3-35.5)/(1-2.4)
alpha_mask = veto_mask&((lengths<(energy*m+32.2 - m*2.25))|((lengths<(energy*m2+35.5 - m2*2.4))&(energy<2.4)))

m = (159.2-26.2)/(2.81-0.619)
proton_mask = veto_mask&(~alpha_mask)&(lengths<(energy*m+26.6 - m*0.619))
#proton_mask = veto_mask&(~alpha_mask)&(lengths<(energy*m+26.6 - m*0.619))&(energy>0.95)&(energy<2.2)&(energy>1.5)&(lengths>55)
palpha_cut = veto_mask&(energy>1.6)&(energy<1.8)&(lengths>27.5)&(lengths<40)
print(np.where(palpha_cut))

plt.figure()
plt.hist(energy[proton_mask], phist_bins)
plt.title('proton energy spectrum, runs: '+str(runs))
plt.xlabel('energy (MeV)')
plt.ylabel('counts/keV')

plt.figure()
plt.title('protons selected in RVE, runs: '+str(runs))
plt.hist2d(energy[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
plt.scatter(energy[proton_mask], lengths[proton_mask], marker='.', alpha=0.5, color='red')
plt.colorbar()
print(str(runs) + "has " + str(len(proton_mask[proton_mask])) + " protons")

timestamps = process_runs.get_quantity('timestamps', experiment, runs)
time_since_last_event = timestamps - np.roll(timestamps, 1)
time_since_last_event[0] = .15 #we don't actuallly know what this is for the first event, so just putting a typical value for start of window
start_of_current_winow = 0
times_since_start_of_window = []
for t, dt in zip(timestamps, time_since_last_event):
    if dt > 0.1 or dt < 0:
        start_of_current_winow = t
    times_since_start_of_window.append(t - start_of_current_winow)
times_since_start_of_window = np.array(times_since_start_of_window)

for t_thresh in []:#[0.02, 0.05, 0.1, 0.15]:
    plt.figure()
    plt_mask = plt_mask&(times_since_start_of_window > t_thresh)
    plt.title('time since start of window > %f ms'%(t_thresh*1000))
    plt.hist2d(energy[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
    plt.colorbar()

if True:
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

if False:
    plt.figure()
    plt.title('alphas')
    tve_bins = (100, 15)
    plt.hist2d(energy[alpha_mask], run_ts[alpha_mask]/3600, bins=tve_bins, norm=matplotlib.colors.LogNorm())
    plt.xlabel('energy (MeV)')
    plt.ylabel('time since start of experiment (hours)')
    plt.colorbar()

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

# hist = ROOT.TH1D('Ep', 'Ep', 4000, 0, 4)
# hist.Fill(energy[proton_mask])
# hist.Draw()

import numpy as np
np.save('to_fit.npy', energy[proton_mask])

plt.figure()
plt.hist(energy[veto_mask], 2000)
plt.show(block=False)
