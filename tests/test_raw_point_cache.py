"""Exercise the shipped raw-reader methods without optional H5/GPU packages.

Only dataset access is synthetic. AST extraction avoids importing unrelated
plotting/ML dependencies; all cache, trace processing and point code is real.
"""
import ast
import copy
from pathlib import Path
import unittest

import numpy as np


source = Path(__file__).resolve().parents[1] / 'raw_viewer' / 'raw_h5_file.py'
tree = ast.parse(source.read_text())
raw_class = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == 'raw_h5_file')
methods = {'invalidate_cache', '_data_cache_key', 'get_data',
           'calculate_background', 'get_xyte', 'get_xyze'}
raw_class.body = [node for node in raw_class.body
                  if isinstance(node, ast.FunctionDef) and node.name in methods]
namespace = {'np': np, 'FIRST_DATA_BIN': 6, 'VETO_PADS': (253,)}
exec(compile(ast.fix_missing_locations(ast.Module(body=[raw_class], type_ignores=[])),
             str(source), 'exec'), namespace)
Raw = namespace['raw_h5_file']


class PointCacheTests(unittest.TestCase):
    def raw(self):
        raw = Raw()
        settings = dict(background_subtract_mode='none', background_convolution_kernel=None,
                        remove_outliers=False, num_background_bins=(0, 1),
                        ic_counts_threshold=0, data_select_mode='all data',
                        near_peak_window_width=1, num_smart_background_ave_bins=2,
                        smart_bins_away_to_check=1, smart2_min_bins_in_peak=2,
                        smart2_min_sigma=2, smart2_ransac_percentile=100,
                        require_peak_within=(-np.inf, np.inf), apply_gain_match=False,
                        asads='all', cobos='all', pads='all', cache_enable=True,
                        num_time_bins=3, zscale=1.088)
        raw.__dict__.update(settings)
        raw.pad_gains = np.ones(1024)
        raw.chnls_to_pad = {(0, 0, 0, 0): 0, (0, 0, 0, 1): 253}
        raw.chnls_to_xy_coord = {(0, 0, 0, 0): (1., 2.), (0, 0, 0, 1): (3., 4.)}
        raw.dataset = np.array([[0, 0, 0, 0, 0, 0, 10, 50, 150],
                                [0, 0, 0, 1, 0, 0, 20, 70, 170]], dtype=float)
        raw.read_count = 0
        def get_dataset(event, kind='data'):
            raw.read_count += 1
            return raw.dataset
        raw._get_event_dataset = get_dataset
        raw.invalidate_cache()
        return raw

    def assert_fresh(self, raw, method='get_xyze', **kwargs):
        actual = getattr(raw, method)(649226, **kwargs)
        reference = copy.copy(raw)
        reference.cache_enable = False
        expected = getattr(reference, method)(649226, **kwargs)
        for got, want in zip(actual, expected):
            np.testing.assert_array_equal(got, want)
        return actual

    def test_threshold_change_recovers_hits_from_an_empty_cached_selection(self):
        raw = self.raw()
        self.assertEqual(len(raw.get_xyze(649226, threshold=200)[3]), 0)
        self.assertEqual(self.assert_fresh(raw, threshold=20)[3].tolist(), [50, 150])
        self.assertEqual(self.assert_fresh(raw, threshold=100)[3].tolist(), [150])

    def test_veto_flag_is_part_of_both_point_cache_keys(self):
        for method in ('get_xyte', 'get_xyze'):
            raw = self.raw()
            getattr(raw, method)(649226, threshold=100, include_veto_pads=False)
            self.assertEqual(self.assert_fresh(raw, method, threshold=100,
                                              include_veto_pads=True)[3].tolist(), [150, 170])
            self.assertEqual(self.assert_fresh(raw, method, threshold=100,
                                              include_veto_pads=False)[3].tolist(), [150])

    def test_zscale_change_updates_z_without_reloading_traces(self):
        raw = self.raw()
        raw.get_xyze(649226, threshold=100)
        reads = raw.read_count
        raw.zscale = 2.
        np.testing.assert_array_equal(raw.get_xyze(649226, threshold=100)[2], [4.])
        self.assertEqual(raw.read_count, reads)

    def test_baseline_window_changes_invalidate_traces_and_points(self):
        raw = self.raw()
        raw.get_xyze(649226, threshold=20)
        raw.background_subtract_mode = 'fixed window'
        self.assertEqual(self.assert_fresh(raw, threshold=20)[3].tolist(), [40, 140])
        raw.num_background_bins = [0, 1]
        raw.get_xyze(649226, threshold=20)
        raw.num_background_bins[1] = 2
        self.assertEqual(self.assert_fresh(raw, threshold=20)[3].tolist(), [120])

    def test_in_place_gain_and_pad_selection_changes_invalidate_cache(self):
        raw = self.raw()
        raw.apply_gain_match = True
        raw.get_xyze(649226, threshold=100)
        raw.pad_gains[0] = 2
        self.assertEqual(self.assert_fresh(raw, threshold=100)[3].tolist(), [300])
        raw.pads = [253]
        self.assertEqual(len(self.assert_fresh(raw, threshold=100)[3]), 0)

    def test_public_and_legacy_invalidation_recompute_derived_points(self):
        for invalidate in (lambda raw: raw.invalidate_cache(),
                           lambda raw: setattr(raw, 'cached_event', np.inf)):
            raw = self.raw()
            raw.get_xyze(649226, threshold=100)
            raw.dataset[0, -1] = 250
            invalidate(raw)
            self.assertEqual(self.assert_fresh(raw, threshold=100)[3].tolist(), [250])

    def test_caller_mutation_cannot_corrupt_trace_or_point_cache(self):
        raw = self.raw()
        raw.get_data(649226)[:, 6:] = 0
        self.assertEqual(raw.get_xyze(649226, threshold=100)[3].tolist(), [150])
        for _ in range(2):
            raw.get_xyze(649226, threshold=100)[3][:] = 0
            self.assertEqual(raw.get_xyze(649226, threshold=100)[3].tolist(), [150])

    def test_unchanged_settings_reuse_processed_traces(self):
        raw = self.raw()
        raw.get_xyze(649226, threshold=100)
        raw.get_xyze(649226, threshold=100)
        raw.get_xyte(649226, threshold=20, include_veto_pads=True)
        self.assertEqual(raw.read_count, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
