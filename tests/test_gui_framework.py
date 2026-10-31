"""Synthetic-data integration checks; no detector files or desktop are required."""
import csv
import importlib
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import types
import unittest
from unittest import mock

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    import tqdm
except ImportError:
    # Progress-bar presentation is optional in this headless test environment.
    progress_module = types.ModuleType('tqdm')
    class Progress:
        def __init__(self, iterable=(), **kwargs): self.iterable = iterable
        def __iter__(self): return iter(self.iterable)
        def update(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
    progress_module.tqdm = Progress
    sys.modules['tqdm'] = progress_module

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Only the external detector backend is replaced. Selection, collection routing,
# the original image builder, PNG rendering and file writes run as shipped.
backend = types.ModuleType('raw_viewer.process_runs')
raw_module = types.ModuleType('raw_viewer.raw_h5_file')
raw_module.VETO_PADS = (253, 254, 508, 509, 763, 764, 1018, 1019)
raw_package = types.ModuleType('raw_viewer')
raw_package.process_runs = backend
raw_package.raw_h5_file = raw_module
sys.modules['raw_viewer'] = raw_package
sys.modules['raw_viewer.process_runs'] = backend
sys.modules['raw_viewer.raw_h5_file'] = raw_module

from GadgetRunCollection import GadgetRunCollection
from GadgetRunH5 import GadgetRunH5
from RvE_Frame import RvE_Frame
from EnergySpectrumFrame import EnergySpectrumFrame
from rve_cuts import parse_run_numbers, validate_cut, load_cut, save_cut


class Entry:
    def __init__(self, value): self.value = str(value)
    def get(self): return self.value
    def configure(self, **kwargs): self.__dict__.update(kwargs)


class RawFile:
    def __init__(self, run, calls, empty):
        self.run = run
        self.calls = calls
        self.empty = empty
        self.file_path = f'/synthetic/run_{run:04d}.h5'
        self.closed = False
        self.h5_file = types.SimpleNamespace(close=self.close)
    def close(self): self.closed = True
    def get_event_num_bounds(self): return 100, 102
    def get_xyze(self, event, threshold=20, include_veto_pads=False):
        self.calls.append((self.run, int(event)))
        if (self.run, int(event)) in self.empty:
            return tuple(np.array([]) for _ in range(4))
        return (np.array([-3., 1.]), np.array([1., 3.]), np.array([10., 11.]), np.array([150., 200.]))
    def get_pad_traces(self, event):
        trace = np.zeros((2, 512))
        trace[:, 245:255] = [100.] * 10
        return np.array([0, 1]), trace


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.calls = []
        self.timing_calls = []
        self.empty = set()
        self.raws = []
        def raw_factory(exp, run):
            raw = RawFile(run, self.calls, self.empty)
            self.raws.append(raw)
            return raw
        backend.get_save_path = lambda exp: self.temp.name
        backend.get_h5_file = raw_factory
        backend.get_gm_ic = lambda exp, runs, gains: np.array([.8, 1.9, 2.1]) if np.median(gains) < .01 else np.array([100., 200., 300.])
        backend.get_lengths = lambda exp, runs: np.array([25., 30., 25.])
        backend.get_angle = lambda exp, runs: np.radians(np.array([0., 90., 45.]))
        backend.get_max_veto_counts = lambda exp, runs: np.array([0., 200., 0.]) if runs == [145] else np.zeros(3)
        def timing(exp, runs):
            self.timing_calls.append(list(runs))
            return np.array([.15, .15, .05]) if runs == [145] else np.array([.15, .15, .15])
        backend.get_time_since_beam_off = timing
        def quantity(name, exp, runs):
            if name == 'timestamps': return np.array([1., 1.01, 1.02])
            if name == 'railed_pads': return [[], [], []]
            raise AssertionError(name)
        backend.get_quantity = quantity
        patcher = mock.patch('GadgetRunH5.get_default_pad_gains', return_value=np.full(1024, 5e-6))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(plt.close, 'all')

    def data(self, runs=(145, 147)):
        data = GadgetRunCollection(runs, folder_path=os.path.join(self.temp.name, 'exports'),
                                   require_processed=False)
        self.addCleanup(data.close)
        return data

    def frame(self, data):
        frame = RvE_Frame.__new__(RvE_Frame)
        frame.run_data = data
        defaults = dict(veto_threshold=150, range_min=1, range_max=400,
                        angle_min='-inf', angle_max='inf', ic_min='-inf', ic_max='inf',
                        railed_pads_max=0, beam_off_min=.1, energy_bins=300, range_bins=200)
        for name, value in defaults.items(): setattr(frame, name + '_entry', Entry(value))
        frame.log_scale_var = types.SimpleNamespace(get=lambda: True)
        frame.cut_status = Entry('')
        frame.event_status = Entry('')
        frame.poly_selector = None
        frame.rve_cut_verticies = []
        frame.rve_cut_select_mask = None
        frame.cut_source_runs = list(data.run_nums)
        frame._plot_figures = []
        return frame

    def cut(self, sources=(145,)):
        return {'experiment': 'e25058', 'source_runs': list(sources),
                'vertices': [[.7, 20.], [2.2, 20.], [2.2, 35.], [.7, 35.]]}

    def test_parser_spaces_duplicates_and_bad_inputs(self):
        self.assertEqual(parse_run_numbers('145, 147,145,149'), [145, 147, 149])
        for value in ('', '145,', ',145', '145,,147', '-1', '145-147', 'abc', '1.2'):
            with self.subTest(value=value), self.assertRaises(ValueError): parse_run_numbers(value)

    def test_timing_and_duplicate_event_identity(self):
        data = self.data()
        self.assertEqual(self.timing_calls, [[145], [147]])
        self.assertEqual(data.get_index(100, run_num=145), 0)
        self.assertEqual(data.get_index(100, run_num=147), 3)
        self.assertEqual(data.get_event_identity(3), (147, 100))
        with self.assertRaises(ValueError): data.get_index(100)
        with self.assertRaises(IndexError): data.get_index(99, run_num=147)
        with self.assertRaises(ValueError): data.get_index(100, run_num=999)
        np.testing.assert_array_equal(data.time_since_beam_off, [.15, .15, .05, .15, .15, .15])
        self.assertTrue(np.shares_memory(data.counts, data.get_run(147).counts))

    def test_single_run_compatibility(self):
        data = self.data((145,))
        self.assertEqual(data.run_num, 145)
        self.assertEqual(data.get_index(101), 1)
        self.assertEqual(data.get_event_num(1), 101)
        self.assertEqual(data.file_path, '/synthetic/run_0145.h5')
        self.assertEqual(data.get_event_num_bounds(), (100, 102))

    def test_plot_mask_and_polygon_exclude_rejected_events(self):
        frame = self.frame(self.data())
        frame.apply_cut_definition(self.cut())
        np.testing.assert_array_equal(frame.rve_cut_select_mask, [True, False, False, True, True, True])
        frame.angle_min_entry.value = '0'
        frame.angle_max_entry.value = '90'
        np.testing.assert_array_equal(frame.refresh_selection(), [False, False, False, False, False, True])
        fig, ax = frame.plot_spectrum(show=False)
        self.assertEqual(frame.get_bins(), (300, 200))
        self.assertIn('145, 147', ax.get_title())

    def test_cut_json_has_only_geometry_and_origin(self):
        frame = self.frame(self.data())
        frame.apply_cut_definition(self.cut())
        filename = os.path.join(self.temp.name, 'cut.json')
        save_cut(filename, frame.get_cut_definition())
        result = load_cut(filename, 'e25058')
        self.assertEqual(set(result), {'experiment', 'source_runs', 'vertices'})
        self.assertEqual(result, self.cut())
        with self.assertRaises(ValueError): load_cut(filename, 'e23035')
        with self.assertRaises(ValueError): validate_cut(dict(result, vertices=[[0, 0], [1, 1], [2, 2]]))
        with self.assertRaises(ValueError): validate_cut(dict(result, vertices=[[0, 0], [1, 1], [float('nan'), 2]]))

    def test_apply_cut_to_new_run_preserves_current_filters_and_origin(self):
        frame = self.frame(self.data((147,)))
        frame.beam_off_min_entry.value = '0.2'
        frame.veto_threshold_entry.value = '99'
        before = (frame.beam_off_min_entry.get(), frame.veto_threshold_entry.get())
        frame.apply_cut_definition(self.cut((145,)))
        self.assertEqual(frame.get_cut_definition()['source_runs'], [145])
        self.assertEqual((frame.beam_off_min_entry.get(), frame.veto_threshold_entry.get()), before)
        self.assertEqual(frame.rve_cut_select_mask.sum(), 0)
        frame.beam_off_min_entry.value = '0.1'
        self.assertEqual(frame.refresh_selection().sum(), 3)

    def test_actual_image_builder_routes_same_event_in_two_runs(self):
        data = self.data()
        images = data.save_cutImages(np.array([0, 3]))
        self.assertEqual([item[3] for item in images], ['run145_event100.png', 'run147_event100.png'])
        self.assertEqual(self.calls, [(145, 100), (147, 100)])
        self.assertEqual(images[0][0].shape, (151, 224, 3))
        with self.assertRaises(IndexError): data.save_cutImages(np.array([-1]))

    def test_multi_run_export_filters_pngs_manifests_and_no_overwrite(self):
        self.empty.add((147, 102))
        data = self.data()
        frame = self.frame(data)
        frame.apply_cut_definition(self.cut())
        outputs = frame.save_cut_files(frame.rve_cut_verticies, date='2026_10_18')
        self.assertEqual([Path(folder).name for folder in outputs], ['run145_2026_10_18', 'run147_2026_10_18'])
        self.assertEqual(sorted(Path(outputs[0]).glob('*.png'))[0].name, 'run145_event100.png')
        self.assertEqual(sorted(p.name for p in Path(outputs[1]).glob('*.png')),
                         ['run147_event100.png', 'run147_event101.png'])
        for directory in outputs:
            for png in Path(directory).glob('*.png'):
                with Image.open(png) as image:
                    self.assertEqual(image.size, (224, 224))
                    self.assertEqual(image.mode, 'RGBA')
            with open(Path(directory) / 'cut_indices_H5list.pkl', 'rb') as handle:
                events = pickle.load(handle)
            self.assertEqual(len(events), len(list(Path(directory).glob('*.png'))))
            self.assertEqual(load_cut(Path(directory) / 'cut_definition.json'), self.cut())
        status = json.loads((Path(outputs[1]) / 'export_status.json').read_text())
        self.assertEqual(status['selected_count'], 3)
        self.assertEqual(status['exported_count'], 2)
        self.assertEqual(status['skipped_empty_events'], [102])
        self.assertEqual(status['status'], 'complete_with_skips')
        self.assertEqual(status['unprocessed_events'], [])
        self.assertEqual(status['filter_settings']['beam_off_min'], .1)
        self.assertEqual(status['filter_settings']['angle_min'], '-inf')
        self.assertEqual(status['image_hit_threshold_adc'], 20)
        self.assertTrue(frame.export_summary(outputs)[1])
        before = {p: p.read_bytes() for folder in outputs for p in Path(folder).iterdir() if p.is_file()}
        with self.assertRaises(FileExistsError):
            frame.save_cut_files(frame.rve_cut_verticies, date='2026_10_18')
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.assertNotIn((145, 101), self.calls)
        self.assertNotIn((145, 102), self.calls)

    def test_preflight_checks_all_runs_before_creating_first_folder(self):
        frame = self.frame(self.data())
        frame.apply_cut_definition(self.cut())
        base = Path(self.temp.name) / 'preflight'
        (base / 'run147_2026_10_18').mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            frame.save_cut_files(frame.rve_cut_verticies, output_base=base, date='2026_10_18')
        self.assertFalse((base / 'run145_2026_10_18').exists())

    def test_empty_selection_creates_no_export(self):
        data = self.data()
        frame = self.frame(data)
        frame.beam_off_min_entry.value = '0.5'
        frame.apply_cut_definition(self.cut())
        self.assertEqual(frame.save_cut_files(frame.rve_cut_verticies), [])
        self.assertFalse(list(Path(data.folder_path).glob('run*')))

    def test_spectrum_uses_all_runs_and_correct_unit_dataset(self):
        frame = EnergySpectrumFrame.__new__(EnergySpectrumFrame)
        frame.run_data = self.data()
        frame.units_combobox = Entry('MeV')
        np.testing.assert_array_equal(frame.get_dataset(), [.8, 1.9, 2.1, .8, 1.9, 2.1])
        frame.units_combobox.value = 'adc counts'
        np.testing.assert_array_equal(frame.get_dataset(), [100, 200, 300, 100, 200, 300])

    def test_load_and_save_dialogs_do_not_replace_filter_entries(self):
        frame = self.frame(self.data((147,)))
        frame.apply_cut_definition(self.cut())
        filename = os.path.join(self.temp.name, 'dialog_cut.json')
        with mock.patch('RvE_Frame.tkinter.filedialog.asksaveasfilename', return_value=filename):
            frame.save_cut_definition()
        frame.beam_off_min_entry.value = '0.2'
        with mock.patch('RvE_Frame.tkinter.filedialog.askopenfilename', return_value=filename), mock.patch.object(frame, 'plot_spectrum_polygon'):
            frame.load_cut_definition()
        self.assertEqual(frame.beam_off_min_entry.get(), '0.2')
        self.assertEqual(frame.cut_source_runs, [145])
        self.assertFalse(frame.rve_cut_select_mask.any())

    def test_invalid_load_does_not_replace_existing_polygon(self):
        frame = self.frame(self.data())
        frame.apply_cut_definition(self.cut())
        filename = os.path.join(self.temp.name, 'wrong_exp.json')
        save_cut(filename, dict(self.cut(), experiment='e23035'))
        old = frame.get_cut_definition()
        with mock.patch('RvE_Frame.tkinter.filedialog.askopenfilename', return_value=filename), mock.patch('RvE_Frame.messagebox.showerror') as error:
            frame.load_cut_definition()
        error.assert_called_once()
        self.assertEqual(frame.get_cut_definition(), old)

    def test_missing_processed_run_fails_before_loading_any_h5(self):
        with self.assertRaises(FileNotFoundError): GadgetRunCollection([145, 147])
        self.assertEqual(self.raws, [])

    def individual_module(self):
        # These unchanged scientific viewers are supplied by the full repository.
        fit = types.ModuleType('fit_gui.FitFrame')
        fit.FitFrame = mock.Mock()
        spatial = types.ModuleType('skspatial.objects')
        spatial.Line = mock.Mock()
        viewer = types.ModuleType('raw_viewer.RawEventViewerFrame')
        viewer.RawEventViewerFrame = mock.Mock()
        with mock.patch.dict(sys.modules, {'fit_gui': types.ModuleType('fit_gui'),
                    'fit_gui.FitFrame': fit, 'skspatial': types.ModuleType('skspatial'),
                    'skspatial.objects': spatial, 'raw_viewer.RawEventViewerFrame': viewer}):
            module = importlib.import_module('IndividualEventFrame')
        sys.modules['IndividualEventFrame'] = module
        return module

    def test_individual_event_and_raw_viewer_use_run_dropdown(self):
        module = self.individual_module()
        frame = module.IndividualEventFrame.__new__(module.IndividualEventFrame)
        frame.run_data = self.data()
        frame.run_num_combo = Entry(147)
        frame.event_num_entry = Entry(100)
        frame.show_plot = mock.Mock()
        frame.show_point_cloud()
        self.assertEqual(self.calls, [(147, 100)])
        frame.show_plot.assert_called_once()
        with mock.patch.object(frame.selected_run(), 'make_image') as make_image:
            frame.track_w_trace_raw()
            make_image.assert_called_once_with(100, show=True)
        with mock.patch.object(module.tk, 'Toplevel'), mock.patch.object(module, 'RawEventViewerFrame') as viewer:
            frame.open_raw_viewer()
            self.assertEqual(viewer.call_args.kwargs['file_path'], '/synthetic/run_0147.h5')

    def test_rve_event_annotation_identifies_run(self):
        frame = self.frame(self.data())
        frame.apply_cut_definition(self.cut())
        frame.event_run_combo = Entry(147)
        frame.event_num_entry = Entry(100)
        with mock.patch('RvE_Frame.plt.show'):
            frame.show_event()
        self.assertTrue(any('Run 147, event 100' in text.get_text() for text in plt.gca().texts))
        self.assertIn('Selected by polygon and filters', frame.event_status.text)

    def test_inspection_explains_event_inside_polygon_failing_time(self):
        frame = self.frame(self.data())
        frame.apply_cut_definition(self.cut())
        frame.event_run_combo = Entry(145)
        frame.event_num_entry = Entry(102)
        details = frame.get_event_selection_details(2)
        self.assertTrue(details['inside_polygon'])
        self.assertFalse(details['selected'])
        self.assertEqual(len(details['failures']), 1)
        self.assertIn('Time since beam off (s) must be > 0.1; event value = 0.05', details['failures'][0])
        with mock.patch('RvE_Frame.plt.show'):
            frame.show_event()
        self.assertIn('Excluded by filters', frame.event_status.text)
        self.assertIn('Inside polygon', frame.event_status.text)
        self.assertEqual(plt.gca().lines[-1].get_marker(), 'x')

    def test_build_error_keeps_good_events_before_and_after_bad_event(self):
        data = self.data((147,))
        frame = self.frame(data)
        frame.apply_cut_definition(self.cut())
        raw = data.h5_file
        original = raw.get_xyze
        def fail_middle(event, **kwargs):
            if event == 101:
                raise ValueError('synthetic corrupt event')
            return original(event, **kwargs)
        with mock.patch.object(raw, 'get_xyze', side_effect=fail_middle):
            folder, = frame.save_cut_files(frame.rve_cut_verticies, date='test')
        folder = Path(folder)
        self.assertEqual(sorted(p.name for p in folder.glob('*.png')),
                         ['run147_event100.png', 'run147_event102.png'])
        status = json.loads((folder / 'export_status.json').read_text())
        self.assertEqual(status['status'], 'complete_with_errors')
        self.assertEqual(status['selected_count'], 3)
        self.assertEqual(status['exported_count'], 2)
        self.assertEqual(status['skipped_empty_events'], [])
        self.assertEqual(status['unprocessed_events'], [])
        self.assertEqual(status['failed_events'][0]['event'], 101)
        self.assertEqual(status['failed_events'][0]['stage'], 'build_failed')
        with (folder / 'export_results.csv').open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([(row['event'], row['status']) for row in rows],
                         [('100', 'exported'), ('101', 'build_failed'), ('102', 'exported')])

    def test_render_error_is_recorded_without_losing_neighbor_images(self):
        from event_image_io import TraceImageRenderer
        frame = self.frame(self.data((147,)))
        frame.apply_cut_definition(self.cut())
        original = TraceImageRenderer.compose
        calls = []
        def fail_middle(renderer, pad, trace):
            calls.append(1)
            if len(calls) == 2:
                raise ValueError('synthetic invalid trace')
            return original(renderer, pad, trace)
        with mock.patch.object(TraceImageRenderer, 'compose', fail_middle):
            folder, = frame.save_cut_files(frame.rve_cut_verticies, date='test')
        status = json.loads((Path(folder) / 'export_status.json').read_text())
        self.assertEqual(status['exported_count'], 2)
        self.assertEqual(status['failed_events'][0]['event'], 101)
        self.assertEqual(status['failed_events'][0]['stage'], 'image_failed')
        self.assertEqual(len(list(Path(folder).glob('*.png'))), 2)

    def test_fatal_stream_error_preserves_progress_and_marks_unprocessed(self):
        data = self.data((147,))
        frame = self.frame(data)
        frame.apply_cut_definition(self.cut())
        original = data.iter_cutImages
        def interrupted(indices, **kwargs):
            stream = original(indices, **kwargs)
            try:
                yield next(stream)
                raise RuntimeError('synthetic stream interruption')
            finally:
                stream.close()
        with mock.patch.object(data, 'iter_cutImages', interrupted):
            with self.assertRaisesRegex(RuntimeError, 'stream interruption'):
                frame.save_cut_files(frame.rve_cut_verticies, date='test')
        folder = Path(data.folder_path) / 'run147_test'
        status = json.loads((folder / 'export_status.json').read_text())
        self.assertEqual(status['status'], 'failed')
        self.assertEqual(status['exported_count'], 1)
        self.assertEqual(status['unprocessed_events'], [101, 102])
        self.assertEqual(status['skipped_empty_events'], [])
        self.assertTrue((folder / 'run147_event100.png').exists())
        with (folder / 'cut_indices_H5list.pkl').open('rb') as handle:
            np.testing.assert_array_equal(pickle.load(handle), [100])

    def test_filter_edits_update_plot_selection_count_and_export_snapshot(self):
        frame = self.frame(self.data((147,)))
        frame.apply_cut_definition(self.cut())
        frame.beam_off_min_entry.value = '.2'
        frame.plot_spectrum(show=False)
        self.assertIn('0 selected with current filters', frame.cut_status.text)
        frame.beam_off_min_entry.value = '.075'
        folder, = frame.save_cut_files(frame.rve_cut_verticies, date='test')
        frame.beam_off_min_entry.value = '.9'
        status = json.loads((Path(folder) / 'export_status.json').read_text())
        self.assertEqual(status['filter_settings']['beam_off_min'], .075)
        self.assertEqual(status['status'], 'complete')
        self.assertFalse(frame.export_summary([folder])[1])
        self.assertEqual(set(load_cut(Path(folder) / 'cut_definition.json')),
                         {'experiment', 'source_runs', 'vertices'})

    def test_legacy_cut_loader_alias_does_not_raise_after_loading(self):
        frame = self.frame(self.data())
        with mock.patch.object(frame, 'load_cut_definition') as load:
            frame.cut_from_file()
        load.assert_called_once()

    def test_run_load_success_closes_previous_dataset(self):
        from RunSelectFrame import RunSelectFrame
        frame = RunSelectFrame.__new__(RunSelectFrame)
        previous = self.data((145,))
        frame.run_data = previous
        frame.experiment_entry = Entry('e25058')
        frame.load_run_number_entry = Entry('145,147,145')
        frame.load_status = Entry('')
        frame.load_button = types.SimpleNamespace(state=mock.Mock())
        frame.update_idletasks = mock.Mock()
        frame.main_gui = types.SimpleNamespace(new_run_loaded=mock.Mock())
        with mock.patch('RunSelectFrame.GadgetRunCollection', side_effect=lambda runs, **kw: self.data(runs)):
            frame.load_button_clicked()
        self.assertEqual(frame.run_numbers, [145, 147])
        frame.main_gui.new_run_loaded.assert_called_once()
        self.assertTrue(previous.get_run(145).h5_file.closed)
        self.assertEqual(frame.run_data.run_nums, [145, 147])

    def test_failed_run_load_keeps_previous_dataset_open(self):
        from RunSelectFrame import RunSelectFrame
        frame = RunSelectFrame.__new__(RunSelectFrame)
        previous = self.data((145,))
        frame.run_data = previous
        frame.experiment_entry = Entry('e25058')
        frame.load_run_number_entry = Entry('145,999')
        frame.load_status = Entry('')
        frame.load_button = types.SimpleNamespace(state=mock.Mock())
        frame.update_idletasks = mock.Mock()
        frame.main_gui = types.SimpleNamespace(new_run_loaded=mock.Mock())
        with mock.patch('RunSelectFrame.GadgetRunCollection', side_effect=FileNotFoundError('missing run 999')), mock.patch('RunSelectFrame.messagebox.showerror'):
            frame.load_button_clicked()
        self.assertIs(frame.run_data, previous)
        self.assertFalse(previous.get_run(145).h5_file.closed)
        frame.main_gui.new_run_loaded.assert_not_called()

    def test_run_reload_keeps_polygon_without_accumulating_tabs(self):
        self.individual_module()
        import main_gui
        old_frame = self.frame(self.data((145,)))
        old_frame.apply_cut_definition(self.cut())
        new_data = self.data((147,))
        gui = main_gui.GadgetAnalysisGUI.__new__(main_gui.GadgetAnalysisGUI)
        gui.run_specific_tabs = [old_frame]
        gui.run_select_frame = types.SimpleNamespace(run_data=new_data)
        gui.notebook = types.SimpleNamespace(forget=mock.Mock(), add=mock.Mock())
        gui.title = mock.Mock()
        class OtherTab:
            def __init__(self, parent, data): self.run_data = data
            def destroy(self): pass
        def rve_init(frame, parent, data):
            frame.__dict__.update(self.frame(data).__dict__)
        with mock.patch.object(main_gui, 'EnergySpectrumFrame', OtherTab), mock.patch.object(main_gui, 'IndividualEventFrame', OtherTab), mock.patch.object(RvE_Frame, '__init__', rve_init), mock.patch.object(RvE_Frame, 'destroy'):
            gui.new_run_loaded()
            self.assertEqual(len(gui.run_specific_tabs), 3)
            new_rve = gui.run_specific_tabs[1]
            self.assertEqual(new_rve.get_cut_definition(), self.cut())
            self.assertEqual(int(new_rve.rve_cut_select_mask.sum()), 3)
            gui.new_run_loaded()
            self.assertEqual(len(gui.run_specific_tabs), 3)
        gui.title.assert_called_with('GADGET II Analysis: Runs 147')


if __name__ == '__main__':
    unittest.main(verbosity=2)
