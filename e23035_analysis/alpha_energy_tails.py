import os

import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np

from raw_viewer import process_runs
from raw_viewer import get_tsbo
from e23035_analysis import e23035_runs

experiment = 'e23035'
tpc_config = 'smart1_veto2_rpr_fzp.csv'
num_workers = 8
energy_range = (4.2, 9.0) #MeV
max_figure_size = (9, 7.5) #inches; the X server over ssh tends to crash on windows much bigger than this

#load alpha spectra for each of the following run types: background, 60Ga, 59Zn
run_df = e23035_runs.run_df
#PID runs are TPC background: the beam doesn't make it to the TPC when the silicon detectors are inserted
run_type_selections = {'background': run_df['Run Type'].isin(['Background', 'background']) | run_df['Run Type'].str.contains('PID', na=False),
                       '60Ga': (run_df['Run Type'] == '60Ga') & (run_df['final beam settings?'] == 'yes'),
                       '59Zn': (run_df['Run Type'] == '59Zn') & (run_df['Field Cage Functional?'] == 'yes')}
run_type_colors = {'background': 'k', '60Ga': 'tab:blue', '59Zn': 'tab:orange'}

get_runs_by_type = {}
for run_type, selection in run_type_selections.items():
    get_runs = {int(run) for run in run_df['GET'][selection] if np.isfinite(run)}
    get_runs_by_type[run_type] = sorted(run for run in get_runs if os.path.exists(process_runs.get_h5_path(experiment, run)))
    print(run_type, 'GET runs:', get_runs_by_type[run_type])

#process any runs without a ROOT file one at a time, since the parallel processing workers re-run this script when they start
process_runs.ensure_processed(experiment, np.concatenate(list(get_runs_by_type.values())), config_filename=tpc_config, num_workers=1)

alphas = {} #run type -> energy (MeV), time since beam off (s, NaN without a matched DDAS event), GET run, and GET event number of each alpha
for run_type, get_runs in get_runs_by_type.items():
    energy = e23035_runs.get_energy_MeV(get_runs, num_workers=num_workers, tpc_ini_filename=tpc_config)
    alpha_mask = e23035_runs.get_alpha_mask(get_runs, energy=energy, tpc_ini_filename=tpc_config)
    tsbo = get_tsbo.get_time_since_beam_off(experiment, get_runs, tpc_ini_filename=tpc_config, num_workers=num_workers)
    run_numbers, event_numbers = process_runs.get_run_and_event_numbers(experiment, get_runs, config_filename=tpc_config)
    alphas[run_type] = {'energy': energy[alpha_mask], 'tsbo': tsbo[alpha_mask], 'run': run_numbers[alpha_mask].astype(int), 'event': event_numbers[alpha_mask].astype(int)}
    print(run_type, 'has', np.sum(alpha_mask), 'alphas')


#make a 2D histogram where each y bin contains a run, and the x axis is energy.
#the energy range should cover 4.2 MeV - 9 MeV. Color code run numbers by run type.
all_runs = sorted(run for get_runs in get_runs_by_type.values() for run in get_runs)
run_type_of_run = {run: run_type for run_type, get_runs in get_runs_by_type.items() for run in get_runs}
all_energies = np.concatenate([alphas[run_type]['energy'] for run_type in alphas])
all_rows = np.searchsorted(all_runs, np.concatenate([alphas[run_type]['run'] for run_type in alphas]))

fig, ax = plt.subplots(figsize=max_figure_size)
hist = ax.hist2d(all_energies, all_rows, bins=(np.arange(energy_range[0], energy_range[1] + 1e-6, 0.05), np.arange(len(all_runs) + 1) - 0.5),
                 norm=matplotlib.colors.LogNorm())
fig.colorbar(hist[3], ax=ax, pad=0.08, label='alphas / 50 keV')
#run labels alternate between the left and right axes so they stay readable in a window of max_figure_size
rows = np.arange(len(all_runs))
for axis, rows_on_side in ((ax.yaxis, rows[::2]), (ax.secondary_yaxis('right').yaxis, rows[1::2])):
    axis.set_ticks(rows_on_side, labels=[all_runs[i] for i in rows_on_side], fontsize=6.5)
    for label, i in zip(axis.get_ticklabels(), rows_on_side):
        label.set_color(run_type_colors[run_type_of_run[all_runs[i]]])
for run_type, color in run_type_colors.items():
    ax.plot([], [], 's', color=color, label=run_type)
ax.legend(loc='lower right', bbox_to_anchor=(1, 1), ncol=3, frameon=False)
ax.set_xlabel('energy (MeV)')
ax.set_ylabel('GET run')
ax.set_title('alpha energy by run', loc='left')
fig.tight_layout()

#Show the alpha energy spectrum for background, 60Ga, and 59Zn over the previously mentioned energy range
spectrum_bins = np.arange(energy_range[0], energy_range[1] + 1e-6, 0.02)
fig, (ax_counts, ax_normalized) = plt.subplots(2, 1, sharex=True, figsize=max_figure_size)
for run_type, alpha_data in alphas.items():
    energies = alpha_data['energy'][(alpha_data['energy'] >= energy_range[0]) & (alpha_data['energy'] < energy_range[1])]
    if len(energies) == 0:
        continue
    color = run_type_colors[run_type]
    ax_counts.hist(energies, spectrum_bins, histtype='step', color=color, label=f'{run_type} ({len(energies)} alphas)')
    ax_normalized.hist(energies, spectrum_bins, histtype='step', color=color, weights=np.full(len(energies), 1/len(energies)), label=run_type)
ax_counts.set_ylabel('alphas / 20 keV')
ax_normalized.set_ylabel('fraction of alphas / 20 keV')
ax_normalized.set_xlabel('energy (MeV)')
for ax in (ax_counts, ax_normalized):
    ax.set_yscale('log')
    ax.legend()
ax_counts.set_title('alpha energy spectrum by run type')
fig.tight_layout()

#Make a 2D histogram of alpha energy vs time since beam off for the 60Ga runs
ga_alphas = alphas['60Ga']
ga_tsbo = ga_alphas['tsbo']
print(f'{np.sum(np.isnan(ga_tsbo))} of {len(ga_tsbo)} 60Ga alphas have no matched DDAS event')
plot_mask = np.isfinite(ga_tsbo) & (ga_alphas['energy'] >= energy_range[0]) & (ga_alphas['energy'] < energy_range[1])

plt.figure()
plt.hist2d(ga_alphas['energy'][plot_mask], ga_tsbo[plot_mask]*1e3, bins=(np.arange(energy_range[0], energy_range[1] + 1e-6, 0.1), np.arange(0, 100.1, 2.5)))
plt.colorbar(label='alphas')
plt.xlabel('energy (MeV)')
plt.ylabel('time since beam off (ms)')
plt.title('60Ga run alphas')
plt.tight_layout()

plt.show(block=False)
