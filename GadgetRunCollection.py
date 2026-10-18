"""A GUI data set with unambiguous (run, original H5 event) identities."""
import os
from numbers import Integral

import numpy as np

from GadgetRunH5 import GadgetRunH5
from raw_viewer import process_runs
from rve_cuts import polygon_mask


class GadgetRunCollection:
    # Timing is computed by each single-run loader, before concatenation.
    ARRAY_NAMES = ('counts', 'ranges', 'angles', 'max_veto_counts', 'timestamps',
                   'time_since_beam_off', 'num_railed_pads', 'event_nums', 'event_runs')

    def __init__(self, run_nums, folder_path=None, experiment='e25058', gains=None,
                 require_processed=True, run_factory=None):
        run_factory = GadgetRunH5 if run_factory is None else run_factory
        requested = list(run_nums)
        if (not requested or any(isinstance(r, bool) or not isinstance(r, Integral) or r < 0
                                 for r in requested)):
            raise ValueError('Supply at least one nonnegative integer run number.')
        self.run_nums = list(dict.fromkeys(int(r) for r in requested))
        self.experiment = experiment
        self._runs = {}
        self._offsets = {}
        self._raw_total_counts = None
        if require_processed:
            save_path = process_runs.get_save_path(experiment)
            missing = [r for r in self.run_nums if not os.path.isfile(
                os.path.join(save_path, f'{experiment}_run{r}.root'))]
            if missing:
                raise FileNotFoundError(f'Process these runs before loading them: {missing}')
        try:
            offset = 0
            for run in self.run_nums:
                data = run_factory(run, folder_path=folder_path, experiment=experiment,
                                   gains=gains, require_processed=require_processed)
                self._runs[run] = data
                self._offsets[run] = offset
                offset += len(data.counts)
                print(f'Loaded run {run}: {len(data.counts)} events')
            first = self._runs[self.run_nums[0]]
            if any(data.energy_units != first.energy_units for data in self._runs.values()):
                raise ValueError('All loaded runs must have matching energy units.')
            self.energy_units = first.energy_units
            self.gains = first.gains
            self.folder_path = first.folder_path
            for name in self.ARRAY_NAMES:
                arrays = [getattr(self._runs[r], name) for r in self.run_nums]
                # Avoid doubling array memory for the common single-run case.
                setattr(self, name, arrays[0] if len(arrays) == 1 else np.concatenate(arrays))
            self.total_energy = self.counts
            self.total_energy_MeV = self.counts
            self.len_list = self.ranges
            if len(self.run_nums) > 1:
                # Children retain views for image/event lookup, not duplicate arrays.
                for run_num, data in self._runs.items():
                    start = self._offsets[run_num]
                    stop = start + len(data.counts)
                    for name in self.ARRAY_NAMES:
                        setattr(data, name, getattr(self, name)[start:stop])
                    data.total_energy = data.counts
                    data.total_energy_MeV = data.counts
                    data.len_list = data.ranges
        except Exception:
            self.close()
            raise

    @property
    def runs_label(self):
        return ', '.join(str(r) for r in self.run_nums)

    @property
    def run_num(self):
        if len(self.run_nums) != 1:
            raise ValueError('Several runs are loaded; select a run explicitly.')
        return self.run_nums[0]

    @property
    def file_path(self):
        return self.get_run(self.run_num).file_path

    @property
    def h5_file(self):
        return self.get_run(self.run_num).h5_file

    def get_run(self, run_num):
        try:
            return self._runs[int(run_num)]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'Run {run_num} is not loaded; loaded runs: {self.runs_label}.') from exc

    def get_index(self, event_num, run_num=None):
        if run_num is None:
            run_num = self.run_num
        run = self.get_run(run_num)
        return run.get_index(event_num) + self._offsets[int(run_num)]

    def get_event_num(self, index):
        result = self.event_nums[index]
        return int(result) if np.ndim(result) == 0 else result

    def get_event_identity(self, index):
        return int(self.event_runs[index]), int(self.event_nums[index])

    def get_event_num_bounds(self, run_num=None):
        return self.get_run(self.run_num if run_num is None else run_num).get_event_num_bounds()

    def get_hit_lists(self, event_num, run_num=None):
        return self.get_run(self.run_num if run_num is None else run_num).get_hit_lists(event_num)

    def make_image(self, event_num, run_num=None, **kwargs):
        return self.get_run(self.run_num if run_num is None else run_num).make_image(event_num, **kwargs)

    def get_raw_total_counts(self):
        if self._raw_total_counts is None:
            arrays = [self._runs[r].get_raw_total_counts() for r in self.run_nums]
            self._raw_total_counts = arrays[0] if len(arrays) == 1 else np.concatenate(arrays)
        return self._raw_total_counts

    def to_MeV(self, counts):
        return self._runs[self.run_nums[0]].to_MeV(counts)

    def to_counts(self, energy):
        return self._runs[self.run_nums[0]].to_counts(energy)

    def save_cutImages(self, cut_indices):
        """Route global event-array indices to their original H5 files."""
        indices = np.asarray(cut_indices)
        if indices.ndim != 1 or (indices.size and not np.issubdtype(indices.dtype, np.integer)):
            raise ValueError('Image selection must be a one-dimensional integer index array.')
        indices = indices.astype(np.int64)
        if np.any(indices < 0) or np.any(indices >= len(self.counts)):
            raise IndexError('Image selection contains an out-of-bounds event index.')
        result = []
        for run_num in self.run_nums:
            local = indices[self.event_runs[indices] == run_num] - self._offsets[run_num]
            if len(local):
                result.extend(self._runs[run_num].save_cutImages(local))
        return result

    def get_RvE_cut_indexes(self, vertices):
        self.rve_cut_select_mask = polygon_mask(self.counts, self.ranges, vertices)
        return np.flatnonzero(self.rve_cut_select_mask)

    def close(self):
        for data in self._runs.values():
            data.close()
