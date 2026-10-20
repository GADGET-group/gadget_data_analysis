import os

import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import scipy.stats

from raw_viewer import process_runs
from raw_viewer import get_tsbo
from e23035_analysis import e23035_runs

experiment = 'e23035'
tpc_config = 'smart1_veto2_rpr_fzp.csv'
num_workers = 8
energy_range = (2.5, 9.0)#(4.2, 9.0) #MeV
max_figure_size = (9, 7.5) #inches; the X server over ssh tends to crash on windows much bigger than this

#load alpha spectra for each of the following run types: background, 60Ga, 59Zn
run_df = e23035_runs.run_df
#PID runs are TPC background: the beam doesn't make it to the TPC when the silicon detectors are inserted
run_type_selections = {'background': run_df['Run Type'].isin(['Background', 'background']) | run_df['Run Type'].str.contains('PID', na=False),
                       '60Ga': (run_df['Run Type'] == '60Ga') & (run_df['final beam settings?'] == 'yes') & run_df['GET'].between(263, 279),
                       '59Zn': (run_df['Run Type'] == '59Zn') & (run_df['Field Cage Functional?'] == 'yes')}
run_type_colors = {'background': 'k', '60Ga': 'tab:blue', '59Zn': 'tab:orange'}

get_runs_by_type = {}
for run_type, selection in run_type_selections.items():
    get_runs = {int(run) for run in run_df['GET'][selection] if np.isfinite(run)}
    get_runs_by_type[run_type] = sorted(run for run in get_runs if os.path.exists(process_runs.get_h5_path(experiment, run)))
    print(run_type, 'GET runs:', get_runs_by_type[run_type])

#process any runs without a ROOT file one at a time, since the parallel processing workers re-run this script when they start
process_runs.ensure_processed(experiment, np.concatenate(list(get_runs_by_type.values())), config_filename=tpc_config, num_workers=1)

alphas, protons = {}, {} #run type -> energy (MeV), time since beam off (s, NaN without a matched DDAS event), GET run, and GET event number of each alpha / proton
alphas_no_veto = {} #the same with the veto condition (veto pads < 200 ADC and track start > 5 mm into the window) not applied: range vs energy band only
for run_type, get_runs in get_runs_by_type.items():
    energy = e23035_runs.get_energy_MeV(get_runs, num_workers=num_workers, tpc_ini_filename=tpc_config)
    alpha_mask = e23035_runs.get_alpha_mask(get_runs, energy=energy, tpc_ini_filename=tpc_config)
    alpha_mask_no_veto = e23035_runs.get_alpha_mask(get_runs, energy=energy, veto_mask=np.ones(len(energy), dtype=bool), tpc_ini_filename=tpc_config)
    proton_mask = e23035_runs.get_proton_mask(get_runs, energy=energy, tpc_ini_filename=tpc_config)
    tsbo = get_tsbo.get_time_since_beam_off(experiment, get_runs, tpc_ini_filename=tpc_config, num_workers=num_workers)
    run_numbers, event_numbers = process_runs.get_run_and_event_numbers(experiment, get_runs, config_filename=tpc_config)
    for particles, mask in ((alphas, alpha_mask), (alphas_no_veto, alpha_mask_no_veto), (protons, proton_mask)):
        particles[run_type] = {'energy': energy[mask], 'tsbo': tsbo[mask], 'run': run_numbers[mask].astype(int), 'event': event_numbers[mask].astype(int)}
    print(run_type, 'has', np.sum(alpha_mask), 'alphas (', np.sum(alpha_mask_no_veto), 'without the veto condition) and', np.sum(proton_mask), 'protons')


#make a 2D histogram where each y bin contains a run, and the x axis is energy.
#the energy range should cover 4.2 MeV - 9 MeV. Color code run numbers by run type.
all_runs = sorted(run for get_runs in get_runs_by_type.values() for run in get_runs)
run_type_of_run = {run: run_type for run_type, get_runs in get_runs_by_type.items() for run in get_runs}
all_energies = np.concatenate([alphas[run_type]['energy'] for run_type in alphas])
all_rows = np.searchsorted(all_runs, np.concatenate([alphas[run_type]['run'] for run_type in alphas]))

def plot_alphas_by_run(weights, colorbar_label, title):
    fig, ax = plt.subplots(figsize=max_figure_size)
    hist = ax.hist2d(all_energies, all_rows, bins=(np.arange(energy_range[0], energy_range[1] + 1e-6, 0.05), np.arange(len(all_runs) + 1) - 0.5),
                     weights=weights, norm=matplotlib.colors.LogNorm())
    fig.colorbar(hist[3], ax=ax, pad=0.08, label=colorbar_label)
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
    ax.set_title(title, loc='left')
    fig.tight_layout()
    return fig, ax

plot_alphas_by_run(None, 'alphas / 50 keV', 'alpha energy by run')

#The same histogram with each run's counts divided by the length of the run, so that runs of different length can be
#compared. The length is the time from the first to the last GET event of the run (GET timestamps, in seconds), the
#only clock in the GET data; it leaves out any idle time before the first and after the last trigger.
all_timestamps = process_runs.get_quantity('timestamps', experiment, all_runs, config_filename=tpc_config)
all_run_numbers, _ = process_runs.get_run_and_event_numbers(experiment, all_runs, config_filename=tpc_config)
run_length_hours = {}
for run in all_runs:
    timestamps = all_timestamps[all_run_numbers == run]
    run_length_hours[run] = (np.max(timestamps) - np.min(timestamps))/3600 if len(timestamps) > 1 else np.nan
    print(f'GET run {run} ({run_type_of_run[run]}): {60*run_length_hours[run]:.1f} min from first to last event')
all_run_length_hours = np.array([run_length_hours[run] for run in np.concatenate([alphas[run_type]['run'] for run_type in alphas])])
plot_alphas_by_run(1/all_run_length_hours, 'alphas / 50 keV / hour', 'alpha rate by run (counts / run length)')

#Show the alpha energy spectrum for background, 60Ga, and 59Zn over the previously mentioned energy range,
#once with a log y axis and once with a linear one
spectrum_bins = np.arange(energy_range[0], energy_range[1] + 1e-6, 0.02)
def plot_spectrum_by_run_type(yscale, extra_spectra={}, extra_colors={}, split_runs={}):
    '''extra_spectra: label -> alpha energies (MeV) of spectra to draw in addition to the run types
       split_runs: run type -> (GET runs, color) to draw separately from the rest of that run type'''
    fig, (ax_counts, ax_normalized) = plt.subplots(2, 1, sharex=True, figsize=max_figure_size)
    spectra, colors = {}, dict(run_type_colors)
    for run_type, alpha_data in alphas.items():
        if run_type in split_runs:
            runs_out, color_out = split_runs[run_type]
            is_out = np.isin(alpha_data['run'], runs_out)
            label_out = f'{run_type} run{"s" if len(runs_out) > 1 else ""} {", ".join(str(run) for run in runs_out)}'
            spectra[label_out], colors[label_out] = alpha_data['energy'][is_out], color_out
            spectra[f'{run_type} (other runs)'], colors[f'{run_type} (other runs)'] = alpha_data['energy'][~is_out], run_type_colors[run_type]
        else:
            spectra[run_type] = alpha_data['energy']
    spectra |= extra_spectra
    colors |= extra_colors
    for label, all_label_energies in spectra.items():
        energies = all_label_energies[(all_label_energies >= energy_range[0]) & (all_label_energies < energy_range[1])]
        if len(energies) == 0:
            continue
        color = colors[label]
        ax_counts.hist(energies, spectrum_bins, histtype='step', color=color, label=f'{label} ({len(energies)} alphas)')
        ax_normalized.hist(energies, spectrum_bins, histtype='step', color=color, weights=np.full(len(energies), 1/len(energies)), label=label)
    ax_counts.set_ylabel('alphas / 20 keV')
    ax_normalized.set_ylabel('fraction of alphas / 20 keV')
    ax_normalized.set_xlabel('energy (MeV)')
    for ax in (ax_counts, ax_normalized):
        ax.set_yscale(yscale)
        ax.legend()
    ax_counts.set_title(f'alpha energy spectrum by run type ({yscale} y axis)')
    fig.tight_layout()
    return fig, (ax_counts, ax_normalized)

plot_spectrum_by_run_type('log')
plot_spectrum_by_run_type('linear')

#The linear spectrum again with the 220Rn source run that the pad gain match was made from: e23035_prep_vault run 49
#(raw_viewer/pad_gain_match/hybrid_regain/README.md), processed with the same config and gains. e23035_runs assumes
#experiment e23035 everywhere, so the alpha selection of e23035_runs.get_alpha_mask is repeated here for the other experiment.
gain_match_experiment, gain_match_runs = 'e23035_prep_vault', [49]
def get_alpha_energies(experiment, runs):
    process_runs.ensure_processed(experiment, runs, config_filename=tpc_config, num_workers=1)
    energy = process_runs.get_gm_ic(experiment, runs, e23035_runs.get_pad_gains(tpc_config), num_workers=num_workers, config_filename=tpc_config)
    endpoints = np.array(process_runs.get_quantity('endpoints', experiment, runs, config_filename=tpc_config))
    lengths = process_runs.get_lengths(endpoints)
    veto_thresholds = np.full(process_runs.raw_h5_file.NUM_PADS, np.inf)
    for pad in process_runs.raw_h5_file.VETO_PADS:
        veto_thresholds[pad] = 200
    veto_mask = process_runs.get_veto_mask(experiment, runs, veto_thresholds, config_filename=tpc_config) & (np.min(endpoints[:, :, 2], axis=1) > 5)
    lower_band, upper_band = e23035_runs.get_alpha_mask_min_max_range(runs, energy)
    alpha_mask = veto_mask & (lengths > lower_band) & (lengths < upper_band)
    print(f'{experiment} runs {runs}: {np.sum(alpha_mask)} alphas in {len(energy)} events')
    return energy[alpha_mask]
gain_match_label = f'gain-match source ({gain_match_experiment} run {", ".join(str(run) for run in gain_match_runs)})'
gain_match_spectrum = {gain_match_label: get_alpha_energies(gain_match_experiment, gain_match_runs)}
plot_spectrum_by_run_type('linear', gain_match_spectrum, {gain_match_label: 'tab:green'})

#The same again with background run 73 drawn separately from the other background runs
plot_spectrum_by_run_type('linear', gain_match_spectrum, {gain_match_label: 'tab:green'}, split_runs={'background': ([73], 'tab:red')})

#Make a 2D histogram of alpha energy vs time since beam off for the 60Ga runs, with the production alpha selection in
#the top subplot and the selection without the veto condition in the bottom one
print(f"{np.sum(np.isnan(alphas['60Ga']['tsbo']))} of {len(alphas['60Ga']['tsbo'])} 60Ga alphas have no matched DDAS event")

fig, axes = plt.subplots(2, 1, figsize=max_figure_size)
for ax, (ga_alphas, label) in zip(axes, ((alphas['60Ga'], '60Ga run alphas'), (alphas_no_veto['60Ga'], '60Ga run alphas, no veto condition'))):
    ga_tsbo = ga_alphas['tsbo']
    plot_mask = np.isfinite(ga_tsbo) & (ga_alphas['energy'] >= energy_range[0]) & (ga_alphas['energy'] < energy_range[1])
    hist = ax.hist2d(ga_alphas['energy'][plot_mask], ga_tsbo[plot_mask]*1e3, bins=(np.arange(energy_range[0], energy_range[1] + 1e-6, 0.1), np.arange(0, 100.1, 2.5)))
    fig.colorbar(hist[3], ax=ax, label='alphas')
    ax.set_xlabel('energy (MeV)')
    ax.set_ylabel('time since beam off (ms)')
    ax.set_title(f'{label}, {energy_range[0]}-{energy_range[1]} MeV', loc='left')
fig.tight_layout()

#Energy spectra of the particles of a run type in three equal time slices of the beam-off part of the cycle (from the
#first to the last particle inside the decay window), overlaid. The slices have the same duration, so the raw counts
#compare rates directly.
#The decay windows, (start, stop) in s after beam off: GET triggers are released 10.1 ms after beam off and the first
#ones are a surge of events after the gating grid switches, over (excess below 5 % of the rate of all events) by 11 ms
#in the 60Ga runs and by 18 ms in the 59Zn runs. The selected protons do not show it, the 59Zn run alphas do.
decay_windows = {'60Ga': (0.012, 0.1), '59Zn': (0.020, 0.5)}
num_slices = 3
def get_time_slice_edges(particle_data, slice_energy_range, decay_window):
    '''mask of the particles inside the decay window and slice_energy_range, and the edges (s) of the time slices that cover them'''
    energy, tsbo = particle_data['energy'], particle_data['tsbo']
    in_slice_range = (np.isfinite(tsbo) & (tsbo >= decay_window[0]) & (tsbo < decay_window[1])
                      & (energy >= slice_energy_range[0]) & (energy < slice_energy_range[1]))
    return in_slice_range, np.linspace(np.min(tsbo[in_slice_range]), np.max(tsbo[in_slice_range]), num_slices + 1)

def plot_time_slices(particle_data, particle_label, slice_energy_range, bin_width, decay_window, particle_name='alphas'):
    '''particle_data: an entry of alphas or protons; decay_window: (start, stop) of the beam-off time used (s); particle_name for the axis label'''
    energy, tsbo = particle_data['energy'], particle_data['tsbo']
    in_slice_range, slice_edges = get_time_slice_edges(particle_data, slice_energy_range, decay_window)
    plt.figure(figsize=max_figure_size)
    for slice_start, slice_end in zip(slice_edges[:-1], slice_edges[1:]):
        in_slice = in_slice_range & (tsbo >= slice_start) & (tsbo <= slice_end)
        plt.hist(energy[in_slice], np.arange(slice_energy_range[0], slice_energy_range[1] + 1e-6, bin_width), histtype='step',
                 label=f'{slice_start*1e3:.0f}-{slice_end*1e3:.0f} ms after beam off ({np.sum(in_slice)} {particle_name})')
    plt.xlabel('energy (MeV)')
    plt.ylabel(f'{particle_name} / {bin_width*1e3:.0f} keV')
    plt.title(f'{particle_label} in {num_slices} equal slices of time since beam off')
    plt.legend()
    plt.tight_layout()

plot_time_slices(alphas['60Ga'], '60Ga run alphas', (2.5, 9.0), 0.05, decay_windows['60Ga'])
plot_time_slices(alphas_no_veto['60Ga'], '60Ga run alphas, no veto condition', (2.5, 9.0), 0.05, decay_windows['60Ga'])

#The proton plots use only the standard 60Ga runs (GET 263-279, the 0.5 us gate delay / 512 bucket configuration)
standard_runs = range(263, 280)
in_standard_runs = np.isin(protons['60Ga']['run'], list(standard_runs))
standard_protons = {key: values[in_standard_runs] for key, values in protons['60Ga'].items()}
print(f'standard runs {standard_runs.start}-{standard_runs.stop - 1} have {np.sum(in_standard_runs)} protons')
plot_time_slices(standard_protons, 'standard 60Ga run protons', (0.3, 4.0), 0.01, decay_windows['60Ga'], 'protons')
plot_time_slices(protons['59Zn'], '59Zn run protons', (0.3, 4.0), 0.01, decay_windows['59Zn'], 'protons')

#How likely is it that the time slices were all drawn from the same spectrum? Only the shapes are compared (the slices
#hold different numbers of particles because of the decays). Each slice is tested against the other slices combined
#with the Pearson chi2 of their counts in the plot binning, and all slices together with the chi2 of the slices x bins
#table; bins empty in every slice are left out. Most bins hold only a few counts, so the p-values come from shuffling
#the slice labels of the particles; the p-value of the chi2 distribution is printed as well, for discrepancies too
#large for the permutations to reach. The unbinned two-sample Kolmogorov-Smirnov test of each slice against the rest is
#printed alongside: it is the more sensitive to an overall shift such as a change of energy scale, the chi2 to local
#changes such as a changed peak ratio. The per-slice tests share data, so the all-slice test is the one overall answer.
num_permutations = 10000
def test_time_slices(particle_data, particle_label, slice_energy_range, bin_width, decay_window, particle_name='alphas'):
    '''the arguments are those of plot_time_slices, which shows the slices tested'''
    in_slice_range, slice_edges = get_time_slice_edges(particle_data, slice_energy_range, decay_window)
    energy, tsbo = particle_data['energy'][in_slice_range], particle_data['tsbo'][in_slice_range]
    slice_index = np.clip(np.searchsorted(slice_edges, tsbo, side='right') - 1, 0, num_slices - 1)
    bin_edges = np.arange(slice_energy_range[0], slice_energy_range[1] + 1e-6, bin_width)
    num_bins = len(bin_edges) - 1
    bin_index = np.clip(np.searchsorted(bin_edges, energy, side='right') - 1, 0, num_bins - 1)
    def get_counts(slice_index):
        return np.bincount(slice_index*num_bins + bin_index, minlength=num_slices*num_bins).reshape(num_slices, num_bins)
    counts = get_counts(slice_index)
    occupied = np.sum(counts, axis=0) > 0
    num_occupied = np.sum(occupied)
    slice_fractions = np.sum(counts, axis=1)/np.sum(counts)
    def get_chi2(counts):
        '''chi2 of each slice against the rest, and of all slices together'''
        expected = slice_fractions[:, np.newaxis]*np.sum(counts, axis=0)
        terms = (counts - expected)**2/expected
        return np.sum(terms/(1 - slice_fractions[:, np.newaxis]), axis=1), np.sum(terms)
    chi2_slices, chi2_all = get_chi2(counts[:, occupied])
    #the bin and slice totals are the same in every permutation, so the occupied bins are too
    rng = np.random.default_rng(0)
    permuted_chi2_slices, permuted_chi2_all = np.empty((num_permutations, num_slices)), np.empty(num_permutations)
    for i in range(num_permutations):
        permuted_chi2_slices[i], permuted_chi2_all[i] = get_chi2(get_counts(rng.permutation(slice_index))[:, occupied])
    def format_p(num_as_large):
        return f'= {(num_as_large + 1)/(num_permutations + 1):.3g}' if num_as_large > 0 else f'< {1/(num_permutations + 1):.1g}'
    print(f'{particle_label}, {slice_energy_range[0]}-{slice_energy_range[1]} MeV in {bin_width*1e3:.0f} keV bins ({num_occupied} occupied): '
          f'is each of the {num_slices} time slices drawn from the same spectrum as the others?')
    for i in range(num_slices):
        ks = scipy.stats.ks_2samp(energy[slice_index == i], energy[slice_index != i])
        print(f'  {slice_edges[i]*1e3:.0f}-{slice_edges[i + 1]*1e3:.0f} ms ({np.sum(slice_index == i)} {particle_name}) against the rest: '
              f'chi2 {chi2_slices[i]:.1f} / {num_occupied - 1} dof, p {format_p(np.sum(permuted_chi2_slices[:, i] >= chi2_slices[i]))} '
              f'({num_permutations} permutations), {scipy.stats.chi2.sf(chi2_slices[i], num_occupied - 1):.3g} (chi2 distribution); '
              f'KS D = {ks.statistic:.4f}, p = {ks.pvalue:.3g}')
    print(f'  all slices: chi2 {chi2_all:.1f} / {(num_slices - 1)*(num_occupied - 1)} dof, p {format_p(np.sum(permuted_chi2_all >= chi2_all))} '
          f'({num_permutations} permutations), {scipy.stats.chi2.sf(chi2_all, (num_slices - 1)*(num_occupied - 1)):.3g} (chi2 distribution)')

test_time_slices(standard_protons, 'standard 60Ga run protons', (0.3, 4.0), 0.01, decay_windows['60Ga'], 'protons')
test_time_slices(standard_protons, 'standard 60Ga run protons', (1, 2), 0.01, decay_windows['60Ga'], 'protons')
test_time_slices(protons['59Zn'], '59Zn run protons', (0.3, 4.0), 0.01, decay_windows['59Zn'], 'protons')

#The same for the 60Ga beta-delayed alpha triplet, 3.5-4.2 MeV
plot_time_slices(alphas['60Ga'], '60Ga run alphas', (3.5, 4.2), 0.02, decay_windows['60Ga'])
test_time_slices(alphas['60Ga'], '60Ga run alphas', (3.5, 4.2), 0.02, decay_windows['60Ga'])

plt.show(block=False)
