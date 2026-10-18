"""Reusable RvE geometry. Filter settings deliberately do not belong to a cut."""
import json
import os
import tempfile
from numbers import Integral

import numpy as np
from matplotlib.path import Path


def parse_run_numbers(text):
    """Parse comma-separated runs, retaining order and removing duplicates."""
    parts = str(text).split(',')
    if not parts or any(not part.strip().isdigit() for part in parts):
        raise ValueError('Enter run numbers separated by commas, for example 145,147,149.')
    return list(dict.fromkeys(int(part.strip()) for part in parts))


def validate_cut(definition, experiment=None):
    if not isinstance(definition, dict):
        raise ValueError('The cut file must contain a JSON object.')
    exp = definition.get('experiment')
    if not isinstance(exp, str) or not exp.strip():
        raise ValueError('The cut is missing its experiment identifier.')
    if experiment is not None and exp != experiment:
        raise ValueError(f'This cut belongs to {exp}, but the loaded experiment is {experiment}.')
    runs = definition.get('source_runs')
    if (not isinstance(runs, (list, tuple)) or not runs or
            any(isinstance(run, bool) or not isinstance(run, Integral) or run < 0
                for run in runs)):
        raise ValueError('The cut must identify the source run number(s).')
    try:
        vertices = np.asarray(definition.get('vertices'), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('Polygon vertices must be numeric energy/range pairs.') from exc
    if (vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3 or
            not np.all(np.isfinite(vertices)) or len(np.unique(vertices, axis=0)) < 3):
        raise ValueError('A polygon needs at least three distinct, finite energy/range pairs.')
    centered = vertices - vertices[0]
    if np.linalg.matrix_rank(centered) < 2:
        raise ValueError('Polygon vertices must enclose a region, not a straight line.')
    # Construct a fresh dictionary: no saved GUI filters are imported or written.
    return {'experiment': exp, 'source_runs': list(dict.fromkeys(int(r) for r in runs)),
            'vertices': vertices.tolist()}


def load_cut(path, experiment=None):
    with open(path, encoding='utf-8') as handle:
        return validate_cut(json.load(handle), experiment)


def save_cut(path, definition):
    """Atomically replace a cut definition after the save dialog has confirmed it."""
    definition = validate_cut(definition)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=directory,
                                         suffix='.tmp', delete=False) as handle:
            temporary = handle.name
            json.dump(definition, handle, indent=2, allow_nan=False)
            handle.write('\n')
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def polygon_mask(energy, ranges, vertices):
    """Return a mask in full event-array order, never histogram-bin order."""
    energy = np.asarray(energy)
    ranges = np.asarray(ranges)
    if energy.shape != ranges.shape or energy.ndim != 1:
        raise ValueError('Energy and range must be matching one-dimensional arrays.')
    finite = np.isfinite(energy) & np.isfinite(ranges)
    result = np.zeros(len(energy), dtype=bool)
    result[finite] = Path(vertices).contains_points(np.column_stack((energy[finite], ranges[finite])))
    return result
