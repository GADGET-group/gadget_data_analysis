import os
import pickle

import matplotlib.pyplot as plt
import ROOT
import numpy as np

from raw_viewer import process_runs
from  raw_viewer import raw_h5_file


if False: #background runs before experiment
    experiment = 'e23035_prep_vault'
    run_range = (68, 73)
else: #during experiment
    experiment = 'e23035'
    run_range = (140, 279)

if False:
    exclude_runs = [1,9, 19, 73, 113,210,216, 225, 226, 227, 228, 229, #210 needs to be transfered by Tyler
                    289,290, 291, 292, 293, 294, 295, 296, 297, 298]#41 deg angle runs
else:
    exclude_runs = [122, 210]
runs=[]
for run in range(run_range[0], run_range[1]+1):
    if run not in exclude_runs and os.path.exists(process_runs.get_h5_path(experiment, run)):
        runs.append(run)
print(runs)
proton_counts, durations = [], []

veto_thresh = 500
#load pad gain match
gain_match_path = '/egr/research-tpc/shared/e23035_prep/vault/gm.pkl'
with open(gain_match_path, 'rb') as f:
    gain_match_result = pickle.load(f)
pad_gains = gain_match_result.x
print(runs)
total_counts = []
for run in runs:
    lengths = process_runs.get_lengths(experiment, [run])
    #veto_counts = process_runs.get_veto_counts(exp, runs)
    veto_max = process_runs.get_max_veto_counts(experiment, [run])
    energy = process_runs.get_gm_ic(experiment, [run], pad_gains)

    veto_mask = (veto_max < veto_thresh)



    m = (108.4-32.2)/(3.628-2.25)
    alpha_mask = veto_mask&(lengths<(energy*m+32.2 - m*2.25))

    m = (159.2-26.2)/(2.81-0.619)
    #proton_mask = veto_mask&(~alpha_mask)&(lengths<(energy*m+26.6 - m*0.619))&(energy>0.95)&(energy<2.2)&(energy>1.5)
    proton_mask = veto_mask&(~alpha_mask)&(lengths<(energy*m+26.6 - m*0.619))&(energy>0.95)&(energy<2.2)&(energy>1.5)&(lengths>55)
    #proton_mask = veto_mask&(energy>0.67)&(energy<0.84)&(lengths>15)&(lengths<32)
    num_protons = len(proton_mask[proton_mask])
    #print(str(run) + " has " + str(num_protons) + " protons")

    ts = process_runs.get_quantity('timestamps', experiment, [run])
    #print('run duration (s): ', (ts[-1]-ts[0]))

    #print('protons per second = ', num_protons/(ts[-1]-ts[0]))
    proton_counts.append(num_protons)
    total_counts.append(len(energy))
    durations.append(ts[-1]-ts[0])
    print(run, num_protons)

proton_counts = np.array(proton_counts)
total_counts = np.array(total_counts)
durations = np.array(durations)
rates = proton_counts/durations
plt.figure()
plt.title('proton rates')
plt.scatter(runs, rates, c=durations)
plt.xlabel('run number')
plt.ylabel('proton rate (pps)')
plt.colorbar(label='run duration (s)')

plt.figure()
plt.title('trigger rates')
plt.scatter(runs, total_counts/durations, c=durations)
plt.xlabel('run number')
plt.ylabel('trigger rate (cps)')
plt.colorbar(label='run duration (s)')

plt.show(block=False)