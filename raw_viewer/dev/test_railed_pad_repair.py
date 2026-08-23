import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

from raw_viewer import process_runs
from e23035_analysis import e23035_runs

# Configurations
repaired_config = 'smart2_rpr.csv'
no_repair_config = 'smart2_nrpr.csv'

# Experiment and run definitions (same as rve.py for e23035_prep_vault)
experiment = 'e23035'
if experiment == 'e23035_prep_vault':
    run_range = [16]#49, 16, 20, 35, 61, 62, 63
else:
    run_range = e23035_runs.run_df['GET'][(e23035_runs.run_df['Run Type']=='60Ga')  & (e23035_runs.run_df['final beam settings?'] == 'yes')] 
exclude_runs = []

GPUs_to_use = [0, 2, 3]
max_workers = len(GPUs_to_use * 3)

# Filter runs to only those that exist
get_runs = []
for run in run_range:
    if not np.isnan(run):
        if run not in exclude_runs and os.path.exists(process_runs.get_h5_path(experiment, run)):
            get_runs.append(run)

get_runs = np.sort(get_runs)
print(f'Experiment: {experiment}')
print(f'Runs to analyze: {get_runs}')

num_workers = min(max_workers, len(get_runs))

# Load gain match calibration
gain_match_path = '/egr/research-tpc/adamsa52/gadget_analysis/raw_viewer/pad_gain_match/gain_match_results/gm_old/fft6_res3.pkl'
with open(gain_match_path, 'rb') as f:
    gain_match_result = pickle.load(f)
pad_gains = gain_match_result.pad_gains


def load_dataset(config_filename):
    print(f'\n--- Loading data for config: {config_filename} ---')
    quantities_to_get = ['charge_width', 'endpoints', 'timestamps']
    if experiment != 'e23035':
        quantities_to_get.append('railed_pads')

    results = process_runs.get_quantity(
        quantities_to_get,
        experiment,
        get_runs,
        show_load_progress=False,
        num_workers=num_workers,
        gpus_to_use=GPUs_to_use,
        config_filename=config_filename
    )

    charge_widths = results[0]
    endpoints = results[1]
    timestamps = results[2]
    pads_railed = results[3] if experiment != 'e23035' else None

    lengths = process_runs.get_lengths(endpoints)
    angles = process_runs.get_angle(endpoints)
    veto_max = process_runs.get_max_veto_counts(
        experiment,
        get_runs,
        num_workers=num_workers,
        config_filename=config_filename
    )

    energy = process_runs.get_gm_ic(
        experiment,
        get_runs,
        pad_gains,
        num_workers=num_workers,
        config_filename=config_filename
    )

    veto_thresh = 200
    veto_mask = (veto_max < veto_thresh)
    plt_mask = veto_mask & (lengths > 1) & (lengths < 250) & (energy < 10)

    print(f'Total events: {len(energy)}, Events passing plot mask: {np.sum(plt_mask)}')
    if pads_railed is not None:
        num_railed = np.array([len(pr) for pr in pads_railed])
        print(f'Events with >=1 railed pad: {np.sum(num_railed > 0)} ({np.sum(num_railed[plt_mask] > 0)} in mask)')
        has_railed = num_railed > 0
    else:
        has_railed = np.zeros(len(energy), dtype=bool)

    return {
        'energy': energy,
        'lengths': lengths,
        'veto_mask': veto_mask,
        'plt_mask': plt_mask,
        'has_railed': has_railed,
        'pads_railed': pads_railed,
        'endpoints': endpoints,
        'timestamps': timestamps
    }


# Load datasets for both repaired and non-repaired configurations
data_rep = load_dataset(repaired_config)
data_norep = load_dataset(no_repair_config)

# ==============================================================================
# 2D Histogram: Range vs Energy (RvE) Comparison & Difference
# ==============================================================================
print('\nComputing 2D Range vs Energy histograms...')
n_ebins = 600
n_rbins = 600
energy_bins = np.linspace(0, 10, n_ebins + 1)
range_bins = np.linspace(0, 250, n_rbins + 1)

# Extract masked data
e_rep = data_rep['energy'][data_rep['plt_mask']]
r_rep = data_rep['lengths'][data_rep['plt_mask']]

e_norep = data_norep['energy'][data_norep['plt_mask']]
r_norep = data_norep['lengths'][data_norep['plt_mask']]

H_rep, xedges, yedges = np.histogram2d(e_rep, r_rep, bins=[energy_bins, range_bins])
H_norep, _, _ = np.histogram2d(e_norep, r_norep, bins=[energy_bins, range_bins])
H_diff = H_rep - H_norep

# Figure 1: 3-panel RvE Comparison (Repaired, No Repair, Difference)
fig_2d, axes_2d = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)

# 1. Repaired RvE
im0 = axes_2d[0].pcolormesh(
    xedges, yedges, np.ma.masked_equal(H_rep.T, 0),
    norm=mcolors.LogNorm(vmin=1, vmax=max(1.1, H_rep.max(), H_norep.max())),
    cmap='viridis'
)
fig_2d.colorbar(im0, ax=axes_2d[0], label='Counts')
axes_2d[0].set_title(f'With Repair ({repaired_config})')
axes_2d[0].set_xlabel('Energy (MeV)')
axes_2d[0].set_ylabel('Range (mm)')

# 2. No Repair RvE
im1 = axes_2d[1].pcolormesh(
    xedges, yedges, np.ma.masked_equal(H_norep.T, 0),
    norm=mcolors.LogNorm(vmin=1, vmax=max(1.1, H_rep.max(), H_norep.max())),
    cmap='viridis'
)
fig_2d.colorbar(im1, ax=axes_2d[1], label='Counts')
axes_2d[1].set_title(f'Without Repair ({no_repair_config})')
axes_2d[1].set_xlabel('Energy (MeV)')

# 3. Difference 2D: RvE(repaired) - RvE(no repair)
max_diff = np.max(np.abs(H_diff))
if max_diff > 0:
    diff_norm = mcolors.SymLogNorm(linthresh=1, linscale=1, vmin=-max_diff, vmax=max_diff, base=10)
else:
    diff_norm = mcolors.Normalize(vmin=-1, vmax=1)

im2 = axes_2d[2].pcolormesh(
    xedges, yedges, H_diff.T,
    norm=diff_norm,
    cmap='coolwarm'
)
fig_2d.colorbar(im2, ax=axes_2d[2], label=r'$\Delta$ Counts (Repaired - No Repair)')
axes_2d[2].set_title('Difference: Repaired - No Repair')
axes_2d[2].set_xlabel('Energy (MeV)')

fig_2d.suptitle(f'{experiment} (Runs: {list(get_runs)}) - 2D Range vs Energy Comparison', fontsize=14)
fig_2d.tight_layout()


# ==============================================================================
# 2D Histogram: Range vs Energy (RvE) for events WITH RAILED PADS
# ==============================================================================
print('\nComputing 2D Range vs Energy histograms for railed pad events...')
railed_mask_rep = data_rep['plt_mask'] & data_rep['has_railed']
railed_mask_norep = data_norep['plt_mask'] & data_norep['has_railed']

e_railed_rep = data_rep['energy'][railed_mask_rep]
r_railed_rep = data_rep['lengths'][railed_mask_rep]

e_railed_norep = data_norep['energy'][railed_mask_norep]
r_railed_norep = data_norep['lengths'][railed_mask_norep]

H_railed_rep, _, _ = np.histogram2d(e_railed_rep, r_railed_rep, bins=[energy_bins, range_bins])
H_railed_norep, _, _ = np.histogram2d(e_railed_norep, r_railed_norep, bins=[energy_bins, range_bins])
H_railed_diff = H_railed_rep - H_railed_norep

# Figure: 3-panel RvE Comparison for Railed Pads Only
fig_railed, axes_railed = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)

# 1. Repaired RvE (railed only)
im0_railed = axes_railed[0].pcolormesh(
    xedges, yedges, np.ma.masked_equal(H_railed_rep.T, 0),
    norm=mcolors.LogNorm(vmin=1, vmax=max(1.1, H_railed_rep.max(), H_railed_norep.max())),
    cmap='viridis'
)
fig_railed.colorbar(im0_railed, ax=axes_railed[0], label='Counts')
axes_railed[0].set_title(f'With Repair ({repaired_config})')
axes_railed[0].set_xlabel('Energy (MeV)')
axes_railed[0].set_ylabel('Range (mm)')

# 2. No Repair RvE (railed only)
im1_railed = axes_railed[1].pcolormesh(
    xedges, yedges, np.ma.masked_equal(H_railed_norep.T, 0),
    norm=mcolors.LogNorm(vmin=1, vmax=max(1.1, H_railed_rep.max(), H_railed_norep.max())),
    cmap='viridis'
)
fig_railed.colorbar(im1_railed, ax=axes_railed[1], label='Counts')
axes_railed[1].set_title(f'Without Repair ({no_repair_config})')
axes_railed[1].set_xlabel('Energy (MeV)')

# 3. Difference 2D (railed only)
max_railed_diff = np.max(np.abs(H_railed_diff))
if max_railed_diff > 0:
    diff_railed_norm = mcolors.SymLogNorm(linthresh=1, linscale=1, vmin=-max_railed_diff, vmax=max_railed_diff, base=10)
else:
    diff_railed_norm = mcolors.Normalize(vmin=-1, vmax=1)

im2_railed = axes_railed[2].pcolormesh(
    xedges, yedges, H_railed_diff.T,
    norm=diff_railed_norm,
    cmap='coolwarm'
)
fig_railed.colorbar(im2_railed, ax=axes_railed[2], label=r'$\Delta$ Counts (Repaired - No Repair)')
axes_railed[2].set_title('Difference: Repaired - No Repair')
axes_railed[2].set_xlabel('Energy (MeV)')

fig_railed.suptitle(f'{experiment} (Runs: {list(get_runs)}) - 2D RvE (Railed Pads ONLY)', fontsize=14)
fig_railed.tight_layout()


# ==============================================================================
# 1D Energy Histogram Comparison & Difference
# ==============================================================================
print('\nComputing 1D histograms...')

# 1D Energy Histograms
e_1d_bins = np.linspace(0, 10, 1001)
e_bin_centers = 0.5 * (e_1d_bins[:-1] + e_1d_bins[1:])
counts_rep_e, _ = np.histogram(e_rep, bins=e_1d_bins)
counts_norep_e, _ = np.histogram(e_norep, bins=e_1d_bins)
counts_diff_e = counts_rep_e - counts_norep_e

fig_e, (ax_e_top, ax_e_bot) = plt.subplots(2, 1, figsize=(7, 5), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

ax_e_top.step(e_bin_centers, counts_rep_e, where='mid', label=f'Repaired ({repaired_config})', color='tab:blue', lw=1.5)
ax_e_top.step(e_bin_centers, counts_norep_e, where='mid', label=f'No Repair ({no_repair_config})', color='tab:orange', lw=1.5, ls='--')
ax_e_top.set_ylabel('Counts')
ax_e_top.set_title(f'1D Energy Spectrum Comparison ({experiment}, Runs: {list(get_runs)})')
ax_e_top.legend()
ax_e_top.grid(True, alpha=0.3)

ax_e_bot.step(e_bin_centers, counts_diff_e, where='mid', color='tab:red', lw=1.5)
ax_e_bot.axhline(0, color='black', linestyle=':', lw=1)
ax_e_bot.set_xlabel('Energy (MeV)')
ax_e_bot.set_ylabel(r'$\Delta$ Counts\n(Rep - NoRep)')
ax_e_bot.grid(True, alpha=0.3)
fig_e.tight_layout()
# 1D Energy Histograms for RAILED PADS ONLY
counts_railed_rep_e, _ = np.histogram(e_railed_rep, bins=e_1d_bins)
counts_railed_norep_e, _ = np.histogram(e_railed_norep, bins=e_1d_bins)
counts_railed_diff_e = counts_railed_rep_e - counts_railed_norep_e

fig_e_railed, (ax_e_railed_top, ax_e_railed_bot) = plt.subplots(2, 1, figsize=(7, 5), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

ax_e_railed_top.step(e_bin_centers, counts_railed_rep_e, where='mid', label=f'Repaired ({repaired_config})', color='tab:blue', lw=1.5)
ax_e_railed_top.step(e_bin_centers, counts_railed_norep_e, where='mid', label=f'No Repair ({no_repair_config})', color='tab:orange', lw=1.5, ls='--')
ax_e_railed_top.set_ylabel('Counts')
ax_e_railed_top.set_title(f'1D Energy Spectrum (Railed Pads ONLY)\n({experiment}, Runs: {list(get_runs)})')
ax_e_railed_top.legend()
ax_e_railed_top.grid(True, alpha=0.3)

ax_e_railed_bot.step(e_bin_centers, counts_railed_diff_e, where='mid', color='tab:red', lw=1.5)
ax_e_railed_bot.axhline(0, color='black', linestyle=':', lw=1)
ax_e_railed_bot.set_xlabel('Energy (MeV)')
ax_e_railed_bot.set_ylabel(r'$\Delta$ Counts\n(Rep - NoRep)')
ax_e_railed_bot.grid(True, alpha=0.3)
fig_e_railed.tight_layout()

if experiment == 'e23035':
    print('\nComputing proton and alpha spectra for e23035...')
    
    alpha_mask_rep = e23035_runs.get_alpha_mask(get_runs, lengths=data_rep['lengths'], energy=data_rep['energy'], veto_mask=data_rep['veto_mask'], tpc_ini_filename=repaired_config)
    proton_mask_rep = e23035_runs.get_proton_mask(get_runs, lengths=data_rep['lengths'], energy=data_rep['energy'], veto_mask=data_rep['veto_mask'], tpc_ini_filename=repaired_config)

    alpha_mask_norep = e23035_runs.get_alpha_mask(get_runs, lengths=data_norep['lengths'], energy=data_norep['energy'], veto_mask=data_norep['veto_mask'], tpc_ini_filename=no_repair_config)
    proton_mask_norep = e23035_runs.get_proton_mask(get_runs, lengths=data_norep['lengths'], energy=data_norep['energy'], veto_mask=data_norep['veto_mask'], tpc_ini_filename=no_repair_config)

    e_alpha_rep = data_rep['energy'][alpha_mask_rep]
    e_alpha_norep = data_norep['energy'][alpha_mask_norep]
    
    e_proton_rep = data_rep['energy'][proton_mask_rep]
    e_proton_norep = data_norep['energy'][proton_mask_norep]

    phist_bins = np.linspace(0, 4, 1000)
    ahist_bins = np.linspace(0, 10, 1000)
    
    # Protons
    fig_p, ax_p = plt.subplots(figsize=(7, 4))
    ax_p.hist(e_proton_rep, bins=phist_bins, histtype='step', label=f'Repaired ({repaired_config})', color='tab:blue', lw=1.5)
    ax_p.hist(e_proton_norep, bins=phist_bins, histtype='step', label=f'No Repair ({no_repair_config})', color='tab:orange', lw=1.5, ls='--')
    ax_p.set_title(f'Proton Energy Spectrum ({experiment})')
    ax_p.set_xlabel('Energy (MeV)')
    ax_p.set_ylabel('Counts')
    ax_p.legend()
    fig_p.tight_layout()

    # Alphas
    fig_a, ax_a = plt.subplots(figsize=(7, 4))
    ax_a.hist(e_alpha_rep, bins=ahist_bins, histtype='step', label=f'Repaired ({repaired_config})', color='tab:blue', lw=1.5)
    ax_a.hist(e_alpha_norep, bins=ahist_bins, histtype='step', label=f'No Repair ({no_repair_config})', color='tab:orange', lw=1.5, ls='--')
    ax_a.set_title(f'Alpha Energy Spectrum ({experiment})')
    ax_a.set_xlabel('Energy (MeV)')
    ax_a.set_ylabel('Counts')
    ax_a.legend()
    fig_a.tight_layout()

print('Plotting complete. Displaying figures...')
plt.show(block=False)
