import datetime
import os
import csv
import json

import tkinter.filedialog
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.widgets import PolygonSelector
from GadgetRunH5 import GadgetRunH5
import numpy as np
from tqdm import tqdm
import pickle

import gadget_widgets
from prev_cut_select_window import PrevCutSelectWindow
from rve_cuts import load_cut, save_cut, validate_cut, polygon_mask
from event_image_io import TraceImageRenderer

class RvE_Frame(ttk.Frame):
    def __init__(self, parent, run_data:GadgetRunH5):
        super().__init__(parent)
        self.run_data = run_data

        # show background image
        self.background_image = gadget_widgets.get_background_image()
        self.background = ttk.Label(self, image=self.background_image)
        self.background.place(relx=0.5, rely=0.5, anchor='center')

        # --- Plot settings ---
        self.plot_settings_frame = ttk.LabelFrame(self, text='plot settings')
        self.plot_settings_frame.grid(row=0)
        self.range_bins_label = ttk.Label(self.plot_settings_frame, text='# range bins:')
        self.range_bins_label.grid(row=0, column=0)
        self.range_bins_entry = gadget_widgets.GEntry(self.plot_settings_frame)
        self.range_bins_entry.grid(row=0, column=1)
        self.range_bins_entry.insert(0,'300')
        self.energy_bins_label = ttk.Label(self.plot_settings_frame, text='# energy bins:')
        self.energy_bins_label.grid(row=0, column=2)
        self.energy_bins_entry = gadget_widgets.GEntry(self.plot_settings_frame)
        self.energy_bins_entry.grid(row=0, column=3)
        self.energy_bins_entry.insert(0,'300')

        self.log_scale_var = tk.BooleanVar(value=True)
        self.scale_label = ttk.Label(self.plot_settings_frame, text='energy scale:')
        self.scale_label.grid(row=1, column=1, sticky=tk.E)
        self.log_scale_radio = ttk.Radiobutton(self.plot_settings_frame,
                                               text='log', variable=self.log_scale_var,
                                               value=True)
        self.lin_scale_radio = ttk.Radiobutton(self.plot_settings_frame,
                                               text='linear', variable=self.log_scale_var,
                                               value=False)
        self.lin_scale_radio.grid(row=1,column=2)
        self.log_scale_radio.grid(row=1,column=3)

        # --- Viewing tools ---
        self.view_frame = ttk.LabelFrame(self, text='viewing tools')
        self.view_frame.grid(row=1)
        self.show_rve_plot_button = ttk.Button(self.view_frame, text='Plot Range vs Energy',
                                               command=self.plot_spectrum)
        self.show_rve_plot_button.grid(row=0, column=0, columnspan=2)
        self.event_num_entry = gadget_widgets.GEntry(self.view_frame, default_text='Event #')
        self.event_num_entry.grid(row=1, column=0)
        self.show_event_button = ttk.Button(self.view_frame, text='show event on RvE plot',
                                            command=self.show_event)
        self.show_event_button.grid(row=1, column=1)
        ttk.Label(self.view_frame, text='Run # for event:').grid(row=2, column=0)
        self.event_run_combo = ttk.Combobox(self.view_frame, state='readonly',
                                           values=self.run_data.run_nums, width=12)
        self.event_run_combo.set(str(self.run_data.run_nums[0]))
        self.event_run_combo.grid(row=2, column=1)
        self.event_status = ttk.Label(self.view_frame, text='', wraplength=650)
        self.event_status.grid(row=3, column=0, columnspan=2)

        # --- Cut tools ---
        self.cut_tools_frame = ttk.LabelFrame(self, text='cut tools')
        self.cut_tools_frame.grid(row=2)

        # Button to open interactive polygon selector
        self.select_polygon_button = ttk.Button(self.cut_tools_frame, text='Select Polygon Region',
                                                command=self.plot_spectrum_polygon)
        self.select_polygon_button.grid(row=0, column=0)

        # Button to save images from last selected polygon
        self.save_polygon_images_button = ttk.Button(self.cut_tools_frame, text='Save Polygon Images',
                                                     command=self.save_selected_polygon_images)
        self.save_polygon_images_button.grid(row=0, column=1)

        # Optional: previous cuts
        self.prev_cut_button = ttk.Button(self.cut_tools_frame, 
                                          text='Previous Cuts',
                                          command=self.prev_cut)
        self.prev_cut_button.grid(row=2, column=0, columnspan=2)
        ttk.Button(self.cut_tools_frame, text='Save Cut', command=self.save_cut_definition).grid(row=1, column=0)
        ttk.Button(self.cut_tools_frame, text='Load Cut', command=self.load_cut_definition).grid(row=1, column=1)

        # Optional projection buttons
        self.project_cut_x_ax_button = ttk.Button(self.cut_tools_frame, text='Project Cut to X-axis',
                                                 command=lambda: self.project_cut('energy'))
        self.project_cut_x_ax_button.grid(row=3, column=0)
        self.project_cut_y_ax_button = ttk.Button(self.cut_tools_frame, text='Project Cut to Y-axis',
                                                 command=lambda: self.project_cut('range'))
        self.project_cut_y_ax_button.grid(row=3, column=1)
        self.cut_status = ttk.Label(self.cut_tools_frame, text='No polygon selected.', wraplength=520)
        self.cut_status.grid(row=4, column=0, columnspan=2)

        # --- Filter / Selection settings ---
        self.filter_frame = ttk.LabelFrame(self, text='filter settings')
        self.filter_frame.grid(row=3, pady=10)

        # Veto threshold
        ttk.Label(self.filter_frame, text="Veto max < ").grid(row=0, column=0)
        self.veto_threshold_entry = gadget_widgets.GEntry(self.filter_frame)
        self.veto_threshold_entry.grid(row=0, column=1)
        self.veto_threshold_entry.insert(0, "150")  # default, as in raw_viewer/plots/rve.py

        # Range min / max
        ttk.Label(self.filter_frame, text="Range min:").grid(row=1, column=0)
        self.range_min_entry = gadget_widgets.GEntry(self.filter_frame)
        self.range_min_entry.grid(row=1, column=1)
        self.range_min_entry.insert(0, "0")

        ttk.Label(self.filter_frame, text="Range max:").grid(row=1, column=2)
        self.range_max_entry = gadget_widgets.GEntry(self.filter_frame)
        self.range_max_entry.grid(row=1, column=3)
        self.range_max_entry.insert(0, "400")

        # Angle min / max
        ttk.Label(self.filter_frame, text="Angle min:").grid(row=2, column=0)
        self.angle_min_entry = gadget_widgets.GEntry(self.filter_frame)
        self.angle_min_entry.grid(row=2, column=1)
        self.angle_min_entry.insert(0, "-inf")

        ttk.Label(self.filter_frame, text="Angle max:").grid(row=2, column=2)
        self.angle_max_entry = gadget_widgets.GEntry(self.filter_frame)
        self.angle_max_entry.grid(row=2, column=3)
        self.angle_max_entry.insert(0, "inf")

        # IC min / max
        ttk.Label(self.filter_frame, text=f"Energy min ({self.run_data.energy_units}):").grid(row=3, column=0)
        self.ic_min_entry = gadget_widgets.GEntry(self.filter_frame)
        self.ic_min_entry.grid(row=3, column=1)
        self.ic_min_entry.insert(0, "-inf")

        ttk.Label(self.filter_frame, text=f"Energy max ({self.run_data.energy_units}):").grid(row=3, column=2)
        self.ic_max_entry = gadget_widgets.GEntry(self.filter_frame)
        self.ic_max_entry.grid(row=3, column=3)
        self.ic_max_entry.insert(0, "inf")

        # railed pads / time since beam off (defaults as in raw_viewer/plots/rve.py)
        ttk.Label(self.filter_frame, text="Railed pads max:").grid(row=4, column=0)
        self.railed_pads_max_entry = gadget_widgets.GEntry(self.filter_frame)
        self.railed_pads_max_entry.grid(row=4, column=1)
        self.railed_pads_max_entry.insert(0, "0")

        ttk.Label(self.filter_frame, text="t since beam off > (s):").grid(row=4, column=2)
        self.beam_off_min_entry = gadget_widgets.GEntry(self.filter_frame)
        self.beam_off_min_entry.grid(row=4, column=3)
        self.beam_off_min_entry.insert(0, "0.1")

        # Storage for selected vertices and mask
        self.rve_cut_verticies = []
        self.rve_cut_select_mask = None
        self.selected_rve_path = None
        self.cut_source_runs = list(self.run_data.run_nums)
        self._plot_figures = []
        self.poly_selector = None

    def close_plots(self):
        if self.poly_selector is not None:
            self.poly_selector.disconnect_events()
        for fig in self._plot_figures:
            plt.close(fig)
        self._plot_figures.clear()

    def get_cut_definition(self):
        return validate_cut({
            'experiment': self.run_data.experiment,
            'source_runs': self.cut_source_runs,
            'vertices': self.rve_cut_verticies,
        }, self.run_data.experiment)

    def apply_cut_definition(self, definition):
        definition = validate_cut(definition, self.run_data.experiment)
        self.rve_cut_verticies = definition['vertices']
        self.cut_source_runs = definition['source_runs']
        self.selected_rve_path = Path(self.rve_cut_verticies)
        self.refresh_selection()
        if self.poly_selector is not None:
            self.poly_selector.verts = self.rve_cut_verticies

    def refresh_selection(self, base_mask=None):
        inside = polygon_mask(self.run_data.counts, self.run_data.ranges, self.rve_cut_verticies)
        self.rve_cut_select_mask = inside & (self.get_processed_event_mask() if base_mask is None else base_mask)
        count = int(self.rve_cut_select_mask.sum())
        source = ', '.join(str(r) for r in self.cut_source_runs)
        self.cut_status.configure(text=f'Cut source: {self.run_data.experiment}, runs {source}. '
                                       f'{count:,} selected with current filters.')
        print(int(inside.sum()), 'events inside polygon before filters')
        print(count, 'events inside polygon after filters')
        return self.rve_cut_select_mask

    def set_cut_polygon(self, verticies):
        try:
            definition = validate_cut({'experiment': self.run_data.experiment,
                                       'source_runs': self.run_data.run_nums,
                                       'vertices': verticies})
        except ValueError as exc:
            print('Incomplete polygon:', exc)
            return
        # An edit creates a new cut on the currently loaded runs.
        self.rve_cut_verticies = definition['vertices']
        self.cut_source_runs = definition['source_runs']
        self.selected_rve_path = Path(self.rve_cut_verticies)
        self.refresh_selection()

    def get_filter_values(self):
        values = {name: float(getattr(self, name + '_entry').get()) for name in (
            'veto_threshold', 'range_min', 'range_max', 'angle_min', 'angle_max',
            'ic_min', 'ic_max', 'railed_pads_max', 'beam_off_min')}
        if any(np.isnan(value) for value in values.values()):
            raise ValueError('Filter values cannot be NaN.')
        for lo, hi in (('range_min', 'range_max'), ('angle_min', 'angle_max'), ('ic_min', 'ic_max')):
            if values[lo] >= values[hi]:
                raise ValueError(f'{lo} must be smaller than {hi}.')
        return values

    def get_filter_checks(self, values=None, index=None):
        """Use the same comparisons for plotting, exporting and event inspection."""
        values = self.get_filter_values() if values is None else values
        data = self.run_data
        checks = []
        for field, setting, label, compare, symbol in (
                ('max_veto_counts', 'veto_threshold', 'Max veto', np.less, '<'),
                ('ranges', 'range_min', 'Range (mm)', np.greater, '>'),
                ('ranges', 'range_max', 'Range (mm)', np.less, '<'),
                ('angles', 'angle_min', 'Angle (degrees)', np.greater, '>'),
                ('angles', 'angle_max', 'Angle (degrees)', np.less, '<'),
                ('counts', 'ic_min', f'Energy ({data.energy_units})', np.greater, '>'),
                ('counts', 'ic_max', f'Energy ({data.energy_units})', np.less, '<'),
                ('num_railed_pads', 'railed_pads_max', 'Railed pads', np.less_equal, '<='),
                ('time_since_beam_off', 'beam_off_min', 'Time since beam off (s)', np.greater, '>')):
            actual = getattr(data, field)
            if index is not None:
                actual = actual[index]
            rule = f'{label} must be {symbol} {values[setting]:.9g}'
            if index is not None:
                rule += f'; event value = {actual:.9g}'
            checks.append((rule, compare(actual, values[setting])))
        return checks

    def get_processed_event_mask(self, values=None):
        return np.logical_and.reduce([passed for _, passed in self.get_filter_checks(values)])

    def get_event_selection_details(self, index):
        failures = [rule for rule, passed in self.get_filter_checks(index=index) if not passed]
        inside = None
        if self.rve_cut_verticies:
            inside = bool(polygon_mask(self.run_data.counts[index:index + 1],
                                       self.run_data.ranges[index:index + 1], self.rve_cut_verticies)[0])
        return {'passes_filters': not failures, 'inside_polygon': inside,
                'selected': inside and not failures, 'failures': failures}

    def get_bins(self):
        bins = (int(self.energy_bins_entry.get()), int(self.range_bins_entry.get()))
        if min(bins) <= 0:
            raise ValueError('Histogram bin counts must be positive integers.')
        return bins

    def _draw_histogram(self, ax, mask, vertices=None):
        if np.any(mask):
            norm = colors.LogNorm() if self.log_scale_var.get() else None
            ax.hist2d(self.run_data.counts[mask], self.run_data.ranges[mask],
                      bins=self.get_bins(), norm=norm)
        else:
            ax.text(0.5, 0.5, 'No events pass the current filters.', transform=ax.transAxes,
                    ha='center', va='center')
        ax.set_xlabel(f'Energy ({self.run_data.energy_units})')
        ax.set_ylabel('Range (mm)')
        if vertices is not None:
            ax.add_patch(patches.Polygon(vertices, closed=True, fill=False, color='red'))

    def plot_spectrum(self, fig_name='RvE', clear=True, show=True):
        mask = self.get_processed_event_mask()
        if self.rve_cut_verticies:
            self.refresh_selection(base_mask=mask)
        print(int(mask.sum()), 'events after mask applied', len(mask), 'total events')
        fig, ax = plt.subplots(num=fig_name, clear=clear)
        self._draw_histogram(ax, mask)
        ax.set_title(f'{self.run_data.experiment}: runs {self.run_data.runs_label}')
        if self.rve_cut_verticies:
            ax.add_patch(patches.Polygon(self.rve_cut_verticies, closed=True,
                                         fill=False, color='red'))
        if fig not in self._plot_figures:
            self._plot_figures.append(fig)
        if show:
            plt.show(block=False)
        return fig, ax

    def plot_spectrum_polygon(self):
        if self.poly_selector is not None:
            self.poly_selector.disconnect_events()
        fig, ax = self.plot_spectrum(fig_name='RvE polygon', show=False)
        # PolygonSelector draws the outline itself; remove the static preview.
        for patch in list(ax.patches):
            patch.remove()
        self.poly_selector = PolygonSelector(ax, self.set_cut_polygon)
        if self.rve_cut_verticies:
            self.poly_selector.verts = self.rve_cut_verticies
            self.refresh_selection()
        ax.set_title(f'Runs {self.run_data.runs_label}: draw or edit the polygon')
        plt.show(block=False)

    def save_cut_definition(self):
        try:
            definition = self.get_cut_definition()
            directory = os.path.join(self.run_data.folder_path, 'cut_definitions')
            os.makedirs(directory, exist_ok=True)
            filename = tkinter.filedialog.asksaveasfilename(
                parent=self, title='Save reusable RvE cut', initialdir=directory,
                initialfile=f"{definition['experiment']}_run{definition['source_runs'][0]}_cut.json",
                defaultextension='.json', filetypes=[('RvE cut definition', '*.json')])
            if not filename:
                return
            save_cut(filename, definition)
            print('Reusable polygon saved:', filename)
            self.cut_status.configure(text=f'Saved cut: {os.path.basename(filename)}')
        except (OSError, ValueError) as exc:
            messagebox.showerror('Cannot save cut', str(exc), parent=self)

    def load_cut_definition(self):
        filename = tkinter.filedialog.askopenfilename(
            parent=self, title='Load reusable RvE cut',
            initialdir=os.path.join(self.run_data.folder_path, 'cut_definitions'),
            filetypes=[('RvE cut definition', '*.json')])
        if not filename:
            return
        try:
            # Validate completely before changing any current cut state.
            definition = load_cut(filename, self.run_data.experiment)
            self.apply_cut_definition(definition)
            self.plot_spectrum_polygon()
            print('Loaded polygon; current GUI filters retained:', filename)
        except (OSError, ValueError) as exc:
            messagebox.showerror('Cannot load cut', str(exc), parent=self)

    def project_cut(self, axis):
        if not self.rve_cut_verticies:
            messagebox.showinfo('No polygon', 'Draw or load a polygon first.', parent=self)
            return
        indices = np.flatnonzero(self.refresh_selection())
        values = self.run_data.counts[indices] if axis == 'energy' else self.run_data.ranges[indices]
        bins = self.get_bins()[0 if axis == 'energy' else 1]
        fig, ax = plt.subplots()
        ax.hist(values, bins=bins, histtype='step')
        ax.set_xlabel(f'Energy ({self.run_data.energy_units})' if axis == 'energy' else 'Range (mm)')
        ax.set_ylabel('Events')
        ax.set_title(f'Runs {self.run_data.runs_label}: {len(indices):,} selected events')
        self._plot_figures.append(fig)
        plt.show(block=False)

    def show_event(self):
        try:
            run_num = int(self.event_run_combo.get())
            event_num = int(self.event_num_entry.get())
            index = self.run_data.get_index(event_num, run_num=run_num)
            fig, ax = self.plot_spectrum()
            energy, track_range = self.run_data.counts[index], self.run_data.ranges[index]
            details = self.get_event_selection_details(index)
            if not details['passes_filters']:
                state = 'Excluded by filters'
            elif details['inside_polygon'] is False:
                state = 'Outside polygon'
            elif details['inside_polygon'] is None:
                state = 'Passes filters; no polygon selected'
            else:
                state = 'Selected by polygon and filters'
            marker = 'rx' if not details['passes_filters'] else 'go'
            ax.plot(energy, track_range, marker, picker=5)
            ax.annotate(f'Run {run_num}, event {event_num}\n{state}', (energy, track_range),
                        textcoords='offset points', xytext=(-15, 7), ha='center', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', edgecolor='black'))
            explanation = f'Run {run_num}, event {event_num}: {state}.'
            if details['inside_polygon'] is not None:
                explanation += ' Inside polygon.' if details['inside_polygon'] else ' Outside polygon.'
            if details['failures']:
                explanation += '\n' + '\n'.join(details['failures'])
            self.event_status.configure(text=explanation)
            print(explanation)
            fig.canvas.draw_idle()
        except (ValueError, IndexError) as exc:
            messagebox.showerror('Cannot locate event', str(exc), parent=self)

    def prev_cut(self):
        PrevCutSelectWindow(self, self.run_data)

    def save_selected_polygon_images(self):
        if not self.rve_cut_verticies:
            messagebox.showinfo('No polygon', 'Draw or load a polygon first.', parent=self)
            return
        base = tkinter.filedialog.askdirectory(parent=self, title='Choose export parent folder',
                                               initialdir=self.run_data.folder_path)
        if not base:
            return
        try:
            self.configure(cursor='watch')
            self.update_idletasks()
            result = self.save_cut_files(self.rve_cut_verticies, output_base=base)
            if result:
                lines, incomplete = self.export_summary(result)
                if incomplete:
                    messagebox.showwarning('Export needs attention', '\n'.join(lines), parent=self)
                else:
                    messagebox.showinfo('Images saved', '\n'.join(lines), parent=self)
        except Exception as exc:
            messagebox.showerror('Image export failed', str(exc), parent=self)
        finally:
            self.configure(cursor='')

    @staticmethod
    def export_summary(folders):
        lines, incomplete = [], False
        for folder in folders:
            with open(os.path.join(folder, 'export_status.json'), encoding='utf-8') as handle:
                status = json.load(handle)
            incomplete |= status['status'] != 'complete'
            lines.append(f"Run {status['run']}: {status['selected_count']} selected, "
                         f"{status['exported_count']} saved, "
                         f"{len(status['skipped_empty_events'])} empty, "
                         f"{len(status.get('failed_events', []))} failed.\n{folder}")
        return lines, incomplete

    def save_cut_files(self, points, output_base=None, date=None):
        """Export each selected event and record its outcome incrementally."""
        from PIL import Image
        definition = validate_cut({'experiment': self.run_data.experiment,
                                   'source_runs': self.cut_source_runs, 'vertices': points})
        # Snapshot filters once: selection, manifests and metadata must agree.
        filter_values = self.get_filter_values()
        base_mask = self.get_processed_event_mask(filter_values)
        selected_mask = polygon_mask(self.run_data.counts, self.run_data.ranges, points) & base_mask
        self.rve_cut_select_mask = selected_mask
        indices = np.flatnonzero(selected_mask)
        print(len(indices), 'filtered events selected for image export')
        if not len(indices):
            print('No images were written: the cut is empty after filtering.')
            return []
        output_base = self.run_data.folder_path if output_base is None else os.fspath(output_base)
        os.makedirs(output_base, exist_ok=True)
        date = datetime.datetime.now().strftime('%Y_%m_%d') if date is None else date
        targets = []
        for run_num in self.run_data.run_nums:
            run_indices = indices[self.run_data.event_runs[indices] == run_num]
            if not len(run_indices):
                print(f'Run {run_num}: no selected events')
                continue
            name = f'run{run_num}_{date}'
            folder = os.path.join(output_base, name)
            preview = os.path.join(output_base, name + '.jpg')
            if os.path.exists(folder) or os.path.exists(preview):
                raise FileExistsError(f'An export already exists: {folder}. '
                                      'Choose a different export parent folder to keep both cuts.')
            targets.append((run_num, run_indices, folder, preview))
        renderer = TraceImageRenderer()
        outputs = []
        for run_num, run_indices, folder, preview in targets:
            os.mkdir(folder)
            print('NEW DIRECTORY', folder)
            selected_events = np.asarray(self.run_data.get_event_num(run_indices), dtype=np.int64)
            run = self.run_data.get_run(run_num)
            # Infinity is represented as text so this remains standard JSON.
            settings = {name: value if np.isfinite(value) else str(value)
                        for name, value in filter_values.items()}
            status = {'format_version': 2, 'experiment': self.run_data.experiment, 'run': run_num,
                      'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      'loaded_runs': list(self.run_data.run_nums),
                      'filter_settings': settings, 'energy_units': self.run_data.energy_units,
                      'filter_comparisons': 'veto < max; range/angle/energy strictly between bounds; '
                                            'railed pads <= max; time since beam off > min',
                      'beam_off_method': 'framework timestamp gaps > 0.1 s, separately for each run',
                      'image_hit_threshold_adc': float(run.image_hit_threshold),
                      'image_include_veto_pads': False,
                      'raw_h5': os.fspath(run.file_path),
                      'selected_count': len(run_indices), 'exported_count': 0,
                      'skipped_empty_events': [], 'failed_events': [],
                      'unprocessed_events': [int(evt) for evt in selected_events],
                      'status': 'in_progress'}
            status_path = os.path.join(folder, 'export_status.json')
            exported_events = []
            expected = {f'run{run_num}_event{int(evt)}.png': int(evt) for evt in selected_events}

            def write_status():
                status['exported_count'] = len(exported_events)
                status['unprocessed_events'] = list(expected.values())
                temporary = status_path + '.tmp'
                with open(temporary, 'w', encoding='utf-8') as handle:
                    json.dump(status, handle, indent=2, allow_nan=False)
                    handle.write('\n')
                os.replace(temporary, status_path)
                # The H5-copy tool must only receive events with saved PNGs,
                # including when an export is interrupted by a later failure.
                pickle_path = os.path.join(folder, 'cut_indices_H5list.pkl')
                with open(pickle_path + '.tmp', 'wb') as handle:
                    pickle.dump(np.asarray(exported_events, dtype=np.int64), handle)
                os.replace(pickle_path + '.tmp', pickle_path)

            write_status()
            fig = None
            try:
                save_cut(os.path.join(folder, 'cut_definition.json'), definition)
                with open(os.path.join(folder, 'selected_events.csv'), 'w', newline='', encoding='utf-8') as handle:
                    writer = csv.writer(handle)
                    writer.writerow(('run', 'event', 'filename'))
                    writer.writerows((run_num, int(evt), f'run{run_num}_event{int(evt)}.png')
                                     for evt in selected_events)
                fig, ax = plt.subplots()
                self._draw_histogram(ax, base_mask & (self.run_data.event_runs == run_num), points)
                ax.set_title(f'{self.run_data.experiment}: run {run_num}')
                fig.savefig(preview)
                plt.close(fig)
                fig = None
                with open(os.path.join(folder, 'export_results.csv'), 'w', newline='', encoding='utf-8') as journal:
                    writer = csv.writer(journal)
                    writer.writerow(('run', 'event', 'filename', 'status', 'error'))
                    journal.flush()

                    def record(filename, outcome, error=''):
                        event = expected.pop(filename)
                        writer.writerow((run_num, event, filename, outcome, error))
                        journal.flush()
                        if outcome == 'exported':
                            exported_events.append(event)
                        elif outcome == 'empty':
                            status['skipped_empty_events'].append(event)
                        else:
                            status['failed_events'].append({'event': event, 'filename': filename,
                                                            'stage': outcome, 'error': error})
                        # Keep progress recoverable even if a later event fails.
                        write_status()

                    def build_error(source_run, event, exc):
                        if source_run != run_num:
                            raise ValueError(f'Image error routed to wrong run: {source_run}')
                        record(f'run{run_num}_event{event}.png', 'build_failed',
                               f'{type(exc).__name__}: {exc}')

                    for pad_plane, trace, title, filename in self.run_data.iter_cutImages(
                            run_indices, on_error=build_error):
                        if filename not in expected:
                            raise ValueError(f'Unexpected or duplicate event filename: {filename}')
                        path = os.path.join(folder, filename)
                        created = False
                        try:
                            image = renderer.compose(pad_plane, trace)
                            with open(path, 'xb') as handle:
                                created = True
                                Image.fromarray(image).save(handle, format='PNG')
                        except Exception as exc:
                            if created:
                                os.unlink(path)  # Remove our incomplete image only.
                            record(filename, 'image_failed', f'{type(exc).__name__}: {exc}')
                            continue
                        record(filename, 'exported')
                    # A successful traversal omitted only events with zero image hits.
                    for filename in list(expected):
                        record(filename, 'empty')
                if status['failed_events']:
                    status['status'] = 'complete_with_errors'
                elif status['skipped_empty_events']:
                    status['status'] = 'complete_with_skips'
                else:
                    status['status'] = 'complete'
                write_status()
            except Exception as exc:
                status['status'] = 'failed'
                status['error'] = f'{type(exc).__name__}: {exc}'
                write_status()
                raise RuntimeError(f'Export stopped for run {run_num}; progress is saved in '
                                   f'{folder}. {exc}') from exc
            finally:
                if fig is not None:
                    plt.close(fig)
            print(f"Run {run_num}: {len(selected_events)} selected, {len(exported_events)} PNGs saved, "
                  f"{len(status['skipped_empty_events'])} empty, {len(status['failed_events'])} failed")
            outputs.append(folder)
        return outputs

    def cut_from_file(self):
        self.load_cut_definition()
