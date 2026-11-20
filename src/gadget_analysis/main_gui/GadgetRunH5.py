'''
Run data for the analysis GUI (python -m gadget_analysis.main_gui): one GET run of an experiment, read through the
library's processing pipeline with a TPC processing config (a preset in tpc_data_processing/tpc_processing_configs/
or the experiment's own copy, see process_runs.get_config_path). Nothing in this module is experiment specific:

- the raw h5 file is process_runs.get_h5_file(experiment, run, config), so zscale, the channel map, the baseline
  subtraction and the thresholds are the config's
- the per-event quantities come from the processed ROOT file of the run and config (process_runs.ensure_processed)
- energies come from the experiment's runs module: get_energy_MeV (which applies the pad gain map named by the
  config's pad_gain_file row) and get_veto_mask; "adc counts" is the integrated charge on the non-veto pads
- event images are the datafused pad plane + summed trace images of the CNN training sets
'''
import io
import os
import math
import importlib.metadata
import importlib.util

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path
from matplotlib.figure import Figure
from scipy.signal import savgol_filter
from tqdm import tqdm

from gadget_analysis.tpc_data_processing import process_runs, raw_h5_file

########################################################
# experiments, configs and processing
########################################################
def list_experiments():
    '''Experiments installed in this env: distributions <exp>_analysis whose package holds an <exp>_runs module.'''
    found = set()
    for dist in importlib.metadata.distributions():
        name = (dist.metadata['Name'] or '').replace('-', '_')
        if not name.endswith('_analysis') or name == 'gadget_analysis':
            continue
        experiment = name[:-len('_analysis')]
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.submodule_search_locations:
            continue
        if any(os.path.isfile(os.path.join(d, f'{experiment}_runs.py')) for d in spec.submodule_search_locations):
            found.add(experiment)
    return sorted(found)

def list_configs(experiment=None):
    '''Processing config names: the library presets plus the experiment's own tpc_processing_configs/ (which take precedence, process_runs.get_config_path).'''
    names = {f for f in os.listdir(process_runs.CONFIG_DIR) if f.endswith('.csv')}
    if experiment:
        spec = importlib.util.find_spec(f'{experiment}_analysis')
        for d in (spec.submodule_search_locations or []) if spec else []:
            local = os.path.join(d, 'tpc_processing_configs')
            if os.path.isdir(local):
                names |= {f for f in os.listdir(local) if f.endswith('.csv')}
    return sorted(names)

def get_default_config(experiment):
    '''The experiment's DEFAULT_TPC_CONFIG, or None.'''
    return process_runs.get_default_config_filename(experiment)

def is_processed(experiment, run_num, config_filename=None):
    '''True when the run has a processed ROOT file for the config (DEFAULT_TPC_CONFIG when None).'''
    config_filename = process_runs.resolve_config_filename(experiment, config_filename)
    return os.path.exists(process_runs.get_processed_file_path(experiment, run_num, config_filename))

def process_run(experiment, run_num, config_filename=None):
    '''
    Process a GET run with the config (DEFAULT_TPC_CONFIG when None) into its processed ROOT file, unless that file
    exists: process_runs.ensure_processed, which splits the run into blocks of events on parallel workers.
    '''
    config_filename = process_runs.resolve_config_filename(experiment, config_filename)
    process_runs.ensure_processed(experiment, [int(run_num)], config_filename=config_filename)

########################################################
# traces
########################################################
TRACE_HALF_WINDOW = 75   # time bins kept on each side of the summed trace's maximum

def smooth_trace(trace, window_length=15, polyorder=3):
    '''Savitzky-Golay smoothing; traces shorter than the window are returned unchanged.'''
    if len(trace) < window_length:
        return np.array(trace, dtype=float)
    return savgol_filter(trace, window_length, polyorder)

def remove_noise(trace, threshold_ratio=0.1):
    '''
    Copy of trace with samples below threshold_ratio x its maximum and negative samples set to zero, and everything
    beyond the first zero on either side of the maximum set to zero.
    '''
    trace = np.array(trace, dtype=float)
    threshold = threshold_ratio * np.max(np.abs(trace)) if len(trace) else 0
    trace[np.abs(trace) < threshold] = 0
    trace[trace < 0] = 0
    if len(trace) == 0:
        return trace
    max_idx = int(np.argmax(trace))
    zeros_left = np.nonzero(trace[:max_idx] == 0)[0]
    if len(zeros_left):
        trace[:zeros_left[-1]] = 0
    zeros_right = np.nonzero(trace[max_idx + 1:] == 0)[0]
    if len(zeros_right):
        trace[max_idx + 1 + zeros_right[0]:] = 0
    return trace

def window_around_peak(trace, half_width=TRACE_HALF_WINDOW):
    '''The part of trace within half_width time bins of its maximum.'''
    peak = int(np.argmax(trace)) if len(trace) else 0
    return trace[max(peak - half_width, 0):peak + half_width]

########################################################
# event images
########################################################
# Format of the CNN training images: a 151 x 224 pixel pad plane (36 x 36 pads as 4 x 4 pixel cells, the micromegas
# outline in black, an energy bar on the left) with the summed trace (224 x 73 pixels) appended below. Layout of the
# outline: https://docs.google.com/spreadsheets/d/1_bbg6svfEph_g_Z002rmzTLu8yjQzuj_p50wqs7mMrI/edit?usp=sharing
PAD_PLANE_IMAGE_SHAPE = (151, 224)   # rows, columns
TRACE_IMAGE_SIZE = (224, 73)         # width, height in pixels
IMAGE_DPI = 96
ENERGY_BAR_FULL_SCALE_MEV = 3        # energy that fills the energy bar
_outline_image = None

def _get_outline_image():
    '''The empty pad plane image: white, micromegas outline and energy bar box in black (int RGB, 0-255).'''
    global _outline_image
    if _outline_image is None:
        row = np.array([63, 47, 39, 31, 27, 23, 19, 15, 15, 11, 11, 7, 7, 7, 7, 3, 3, 3, 3, 3, 3, 3, 7, 7,
                        7, 7, 11, 11, 15, 15, 19, 23, 27, 31, 39, 47, 63])
        to_row = np.array([87, 103, 111, 119, 123, 127, 131, 135, 135, 139, 139, 143, 143, 143, 143, 147,
                           147, 147, 147, 147, 147, 148, 143, 143, 143, 144, 139, 140, 135, 136, 132, 128,
                           124, 120, 112, 104, 88])
        col = np.array([100, 84, 76, 68, 64, 60, 56, 52, 52, 48, 48, 44, 44, 44, 44, 40, 40, 40, 40, 40,
                        40, 40, 44, 44, 44, 44, 48, 48, 52, 52, 56, 60, 64, 68, 76, 84, 100])
        to_col = np.array([124, 140, 148, 156, 160, 164, 168, 172, 172, 176, 176, 180, 180, 180, 180, 184,
                           184, 184, 184, 184, 184, 184, 180, 180, 180, 180, 176, 176, 172, 172, 168, 164,
                           160, 156, 148, 140, 124])
        all_row = np.arange(3, 148, 4)
        all_col = np.arange(40, 185, 4)
        image = np.full(PAD_PLANE_IMAGE_SHAPE + (3,), 255, dtype=int)
        for i in range(len(row)):   #grid columns
            image[row[i]:to_row[i], all_col[i], :] = 0
        for i in range(len(col)):   #grid rows
            image[all_row[i], col[i]:to_col[i], :] = 0
        #energy bar box
        image[4:145, 7, :] = 0
        image[4:146, 17, :] = 0
        image[4, 7:17, :] = 0
        image[145, 7:17, :] = 0
        _outline_image = image
    return _outline_image.copy()

def _fill_energy_bar(image, energy_mev):
    '''Fill the energy bar of a pad plane image: 28 rows (7 each of blue, yellow, orange, red) for ENERGY_BAR_FULL_SCALE_MEV.'''
    if not np.isfinite(energy_mev):
        return image
    total_rows = math.floor(energy_mev / ENERGY_BAR_FULL_SCALE_MEV * 28)
    start_row, low_color, high_color = 140, 0, 35   #blue
    for i in range(max(min(total_rows, 7), 0)):
        image[start_row:start_row+5, 8:17, 0] = low_color
        image[start_row:start_row+5, 8:17, 1] = high_color
        start_row, low_color, high_color = start_row - 5, low_color + 35, high_color + 35
    start_row, color = 105, 220   #yellow
    for i in range(max(min(total_rows - 7, 7), 0)):
        image[start_row:start_row+5, 8:17, 2] = color
        start_row, color = start_row - 5, color - 15
    start_row, color = 70, 210   #orange
    for i in range(max(min(total_rows - 14, 7), 0)):
        image[start_row:start_row+5, 8:17, 1] = color - 15
        image[start_row:start_row+5, 8:17, 2] = color
        start_row, color = start_row - 5, color - 15
    start_row, color = 35, 250   #red
    for i in range(max(min(total_rows - 21, 7), 0)):
        image[start_row:start_row+5, 8:17, 0] = color
        image[start_row:start_row+5, 8:17, 1] = 50
        image[start_row:start_row+5, 8:17, 2] = 50
        start_row, color = start_row - 5, color - 15
    return image

def pad_plane_image(xs, ys, es, energy_mev, padxy):
    '''
    Pad plane image of one event: each pad that has points is a 3 x 3 pixel block colored by its summed charge
    relative to the brightest pad, plus the energy bar for energy_mev. xs, ys: pad centres (mm) as given by
    raw_h5_file.get_xyze; padxy: the pad centre table (raw_h5_file.padxy), whose 36 distinct coordinates are the
    pad columns and rows of the image.
    '''
    image = _get_outline_image()
    centers = np.unique(np.round(padxy, 3))
    if len(xs):
        ix = np.searchsorted(centers, np.round(xs, 3))
        iy = np.searchsorted(centers, np.round(ys, 3))
        charge = np.zeros((len(centers), len(centers)))
        hits = np.zeros((len(centers), len(centers)), dtype=int)
        np.add.at(charge, (ix, iy), es)
        np.add.at(hits, (ix, iy), 1)
        fired_x, fired_y = np.nonzero(hits)
        peak = charge[fired_x, fired_y].max()
        norm = charge / peak if peak > 0 else np.zeros_like(charge)
        for i, j in zip(fired_x, fired_y):
            r, c = 141 - 4*j, 42 + 4*i   #cell centre of pad column i, row j
            image[max(r-1, 0):r+2, c-1:c+2, 0] = norm[i, j] * 205
            image[max(r-1, 0):r+2, c-1:c+2, 1] = norm[i, j] * 240
    return _fill_energy_bar(image, energy_mev)

def trace_image(trace):
    '''The trace drawn as a filled blue curve on a TRACE_IMAGE_SIZE canvas, as uint8 RGB (rows, columns, 3).'''
    from PIL import Image
    width, height = TRACE_IMAGE_SIZE
    fig = Figure(figsize=(width/IMAGE_DPI, height/IMAGE_DPI))
    ax = fig.subplots()
    ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    for side in ('top', 'right', 'bottom', 'left'):
        ax.spines[side].set_visible(False)
    ax.fill_between(np.arange(len(trace)), trace, color='b', alpha=1)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=IMAGE_DPI)
    buf.seek(0)
    with Image.open(buf) as im:
        return np.array(im.convert('RGB'))

def event_image(pad_plane, trace):
    '''Pad plane image with the trace image appended below, as uint8 RGB.'''
    return np.append(pad_plane, trace_image(trace), axis=0).clip(0, 255).astype(np.uint8)

########################################################
# one run
########################################################
class GadgetRunH5:
    '''
    One GET run of an experiment, processed with a TPC config. Per selected event (arrays indexed alike, in event
    order): good_events (event numbers), total_energy (integrated charge on the non-veto pads, adc counts),
    total_energy_MeV (runs module get_energy_MeV), len_list (track length, mm), angle_list (track angle from the
    drift axis, degrees). Selected: events with a track (two or more points above the config's
    length_counts_threshold) and, with apply_veto, kept by the runs module's get_veto_mask. The single-event methods
    (get_hit_lists, get_trace, make_image, get_event_energy_MeV, get_event_length, get_event_status) take the event
    number of any event of the run, selected or not.
    '''
    def __init__(self, experiment, run_num, config_filename=None, apply_veto=True):
        from gadget_analysis.tpc_and_decay_station import ddas_interface   #imports ROOT, so only when a run is loaded
        self.experiment = experiment
        self.run_num = int(run_num)
        self.config_filename = process_runs.resolve_config_filename(experiment, config_filename)
        self.config_path = process_runs.get_config_path(self.config_filename, experiment)
        self.apply_veto = apply_veto
        self.processed_file_path = process_runs.get_processed_file_path(experiment, self.run_num, self.config_filename)
        if not os.path.exists(self.processed_file_path):
            raise FileNotFoundError(f'GET run {self.run_num} of {experiment} has not been processed with {self.config_filename} '
                                    f'({self.processed_file_path} does not exist): process it first')
        self.h5_file = process_runs.get_h5_file(experiment, self.run_num, self.config_filename)
        self.h5_file_path = self.h5_file.file_path
        self._folder_path = None

        runs_module = ddas_interface.get_runs_module(experiment)
        run, cfg = [self.run_num], self.config_filename
        endpoints = process_runs.get_quantity('endpoints', experiment, run, config_filename=cfg, num_workers=1)
        #the processed file has one entry per event from the first event of the run on
        events = self.h5_file.get_event_num_bounds()[0] + np.arange(len(endpoints))
        lengths = process_runs.get_lengths(endpoints)
        angles = np.degrees(process_runs.get_angle(endpoints))
        non_veto_pads = np.ones(raw_h5_file.NUM_PADS)
        non_veto_pads[list(raw_h5_file.VETO_PADS)] = 0
        counts = process_runs.get_gm_ic(experiment, run, non_veto_pads, config_filename=cfg)
        energy_MeV = np.asarray(runs_module.get_energy_MeV(run, tpc_ini_filename=cfg), dtype=float)
        has_track = lengths > 0
        veto_mask = np.asarray(runs_module.get_veto_mask(run, tpc_ini_filename=cfg), dtype=bool)   #True = keep
        keep = has_track & veto_mask if apply_veto else has_track.copy()
        sizes = {len(a) for a in (events, counts, energy_MeV, veto_mask)}
        if len(sizes) != 1:
            raise ValueError(f'run {self.run_num}: per-event arrays of different lengths (endpoints {len(events)}, '
                             f'charge {len(counts)}, MeV {len(energy_MeV)}, veto mask {len(veto_mask)}); a cache in '
                             f'{process_runs.get_save_path(experiment)} may be stale')

        #every event of the run, for looking at single events whether selected or not
        self._first_event = int(events[0]) if len(events) else 0
        self._all = {'energy_MeV': energy_MeV, 'counts': counts, 'length': lengths, 'has_track': has_track,
                     'passes_veto': veto_mask}
        self.num_events = len(events)
        self.num_without_track = int(np.sum(~has_track))
        self.num_vetoed = int(np.sum(has_track & ~keep))
        self.good_events = events[keep]
        self.total_energy = counts[keep]
        self.total_energy_MeV = energy_MeV[keep]
        self.len_list = lengths[keep]
        self.angle_list = angles[keep]
        self._index_of_event = {int(e): i for i, e in enumerate(self.good_events)}
        print(f'{experiment} GET run {self.run_num} ({cfg}): {len(self.good_events)} of {self.num_events} events selected '
              f'({self.num_without_track} without a track, {self.num_vetoed} vetoed)')

    @property
    def folder_path(self):
        '''
        Folder for this run's saved RvE cuts (cut images and event lists): main_gui_cuts/<exp>_run<N>_<config>/ next to
        the processed-data folder, created on first use, writable by everyone in the group.
        '''
        if self._folder_path is None:
            from gadget_analysis.tpc_and_decay_station import cache_maintenance
            save_path = process_runs.get_save_path(self.experiment)
            if save_path is None:
                raise RuntimeError(f'process_runs.get_save_path has no data folder for {self.experiment} on this host')
            config_name = os.path.splitext(self.config_filename)[0]
            path = os.path.join(os.path.dirname(os.path.normpath(save_path)), 'main_gui_cuts',
                                f'{self.experiment}_run{self.run_num}_{config_name}')
            self._folder_path = cache_maintenance.make_shared_dir(path)
        return self._folder_path

    def get_index(self, event_num):
        '''Index of a selected event in the per-event arrays.'''
        try:
            return self._index_of_event[int(event_num)]
        except KeyError:
            raise ValueError(f'event {event_num} is not among the {len(self.good_events)} selected events of run '
                             f'{self.run_num} (no track, vetoed, or outside the run)') from None

    def _run_position(self, event_num):
        '''Position of any event of the run in the whole-run arrays; ValueError outside the run.'''
        i = int(event_num) - self._first_event
        if not 0 <= i < self.num_events:
            raise ValueError(f'event {event_num} is outside run {self.run_num} (events {self._first_event} to '
                             f'{self._first_event + self.num_events - 1})')
        return i

    def get_event_energy_MeV(self, event_num):
        '''Energy (MeV, runs module get_energy_MeV) of any event of the run, selected or not.'''
        return float(self._all['energy_MeV'][self._run_position(event_num)])

    def get_event_length(self, event_num):
        '''Track length (mm) of any event of the run, selected or not.'''
        return float(self._all['length'][self._run_position(event_num)])

    def get_event_status(self, event_num):
        '''"selected", "vetoed" (fails the runs module's get_veto_mask) or "no track"; "vetoed" also when the veto is not applied.'''
        i = self._run_position(event_num)
        if not self._all['has_track'][i]:
            return 'no track'
        return 'selected' if self._all['passes_veto'][i] else 'vetoed'

    def get_hit_lists(self, event_num):
        '''
        x, y, z (mm) and charge of the samples above the config's length_counts_threshold, veto pads excluded: the
        point cloud the processing measured the track on. Any event of the run, selected or not.
        '''
        self._run_position(event_num)
        return self.h5_file.get_xyze(int(event_num), threshold=self.h5_file.length_counts_threshold, include_veto_pads=False)

    def get_trace(self, event_num, smoothen=False, noise_threshold_ratio=0.1):
        '''
        Sum of the baseline-subtracted traces of the non-veto pads (baseline subtraction as the config sets it), cut to
        TRACE_HALF_WINDOW bins around its maximum, optionally smoothed, then remove_noise(noise_threshold_ratio).
        '''
        pads, traces = self.h5_file.get_pad_trace_array(int(event_num), include_veto_pads=False)
        trace = traces.sum(axis=0) if len(pads) else np.zeros(self.h5_file.num_time_bins)
        trace = window_around_peak(np.asarray(trace, dtype=float))
        if smoothen:
            trace = smooth_trace(trace)
        return remove_noise(trace, threshold_ratio=noise_threshold_ratio)

    def _image_parts(self, event_num, use_raw_data, smoothen):
        '''Pad plane image and trace of any event of the run. Point cloud mode: smoothed trace with a 1 % noise cut; raw mode: 10 % noise cut, smoothed only when asked.'''
        xs, ys, zs, es = self.get_hit_lists(event_num)
        if use_raw_data:
            trace = self.get_trace(event_num, smoothen=smoothen, noise_threshold_ratio=0.1)
        else:
            trace = self.get_trace(event_num, smoothen=True, noise_threshold_ratio=0.01)
        return pad_plane_image(xs, ys, es, self.get_event_energy_MeV(event_num), self.h5_file.padxy), trace

    def make_image(self, event_num, use_raw_data=False, save_path=None, show=False, smoothen=False):
        '''
        Datafused image (pad plane and summed trace) of any event of the run (selected, vetoed or without a track),
        drawn in the current pyplot figure; saved to save_path if given.
        '''
        image = event_image(*self._image_parts(event_num, use_raw_data, smoothen))
        source = 'Raw Data' if use_raw_data else 'Point Cloud'
        status = self.get_event_status(event_num)
        note = '' if status == 'selected' else f', {status}'
        plt.title(f'Image {event_num} of Particle Track Event (Using {source}{note})', fontdict={'fontsize': 14})
        plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
        plt.imshow(image)
        if save_path is not None:
            plt.savefig(save_path)
        if show:
            plt.show(block=False)
        else:
            plt.close()

    def get_cut_images(self, cut_indices, use_raw_data=False):
        '''[(image, title, file name), ...] for the selected events at cut_indices; raw mode uses smoothed traces.'''
        to_return = []
        for index in tqdm(cut_indices):
            event_num = self.good_events[index]
            image = event_image(*self._image_parts(event_num, use_raw_data, smoothen=True))
            to_return.append((image, f'Image {event_num} of Particle Track Event', f'run{self.run_num}_image_{event_num}.png'))
        return to_return

    def get_RvE_cut_indexes(self, points):
        '''
        points: list of (energy, range) tuples defining a cut in RvE, energy in MeV, range in mm
        Returns the indices of the selected events inside the polygon.
        '''
        path = matplotlib.path.Path(points)
        inside = path.contains_points(np.column_stack((self.total_energy_MeV, self.len_list)))
        return np.nonzero(inside)[0]
