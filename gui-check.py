#!/usr/bin/env python3
"""Read-only GADGET GUI export audit for one or more (run, event) pairs.

Save this file as gui-check.py next to main_gui.py. Run in e21072_root from the
framework directory. Works before or after the GUI export bug-fix patch.

ONE EVENT (existing command remains valid):
 python gui-check.py --folder /path/to/Nandhalalan/run147_2026_10_22 \
   --run 147 --event 649226 --inspect-data

AUTOMATIC RUN-FOLDER LOOKUP:
 python gui-check.py --folder /path/to/Nandhalalan \
   --run 147 --event 649226 --inspect-data

SEVERAL EVENTS FROM ONE RUN:
 python gui-check.py --folder /path/to/Nandhalalan \
   --run 147 --events 649226 649227 --inspect-data

EVENTS FROM DIFFERENT RUNS (replace the second example pair with your event):
 python gui-check.py --folder /path/to/Nandhalalan \
   --targets 147:649226 149:123456 --inspect-data

If several exports exist for a run, all matching exports are checked; use
--date 2026_10_22 to restrict the export date. Each event is identified by BOTH
run and original H5 event number, never by its position in the cut.

Without --inspect-data only the Python standard library is needed. With it,
use the normal framework dependencies. No PNGs, ROOT/H5 files, NPY caches or
export files are written. Image generation is tested in memory only.

Old exports did not save GUI filters. Optional --beam-off-min 0.1,
--veto-threshold 150 and the other filter flags compare user-supplied settings;
they do NOT recover historical GUI settings. New exports use filter_settings
from export_status.json automatically. CLI filter flags override those values
for a clearly labelled hypothetical comparison.

Missing images are audit findings, not script errors. Exit 0 means all requested
audits ran; exit 2 means at least one target could not be audited. Other targets
continue even when one run, event or export folder causes an error.
"""
import argparse
import csv
import hashlib
import json
import math
import operator
from pathlib import Path
import re
import sys

# Importing framework modules must not create bytecode files during this audit.
sys.dont_write_bytecode = True
FILTERS = (
    ('veto_threshold', 'veto', 'Max veto', '<', operator.lt),
    ('range_min', 'range', 'Range (mm)', '>', operator.gt),
    ('range_max', 'range', 'Range (mm)', '<', operator.lt),
    ('angle_min', 'angle', 'Angle (degrees)', '>', operator.gt),
    ('angle_max', 'angle', 'Angle (degrees)', '<', operator.lt),
    ('ic_min', 'energy', 'Energy (MeV)', '>', operator.gt),
    ('ic_max', 'energy', 'Energy (MeV)', '<', operator.lt),
    ('railed_pads_max', 'railed', 'Railed pads', '<=', operator.le),
    ('beam_off_min', 'beam_time', 'Time since beam off (s)', '>', operator.gt),
)


def load_json(path):
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError(f'Cannot read {path}: {exc}') from exc
    if not isinstance(data, dict):
        raise ValueError(f'Expected a JSON object in {path}')
    return data


def folder_run(folder):
    path = folder / 'export_status.json'
    if path.is_file():
        status = load_json(path)
        if status.get('run') is not None:
            return int(status['run'])
    match = re.fullmatch(r'run(\d+)_.*', folder.name)
    return int(match.group(1)) if match else None


def find_exports(base, run, date=None):
    base = Path(base).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(f'Folder not found: {base}')
    is_export = any((base / name).is_file() for name in
                    ('export_status.json', 'selected_events.csv', 'cut_definition.json'))
    is_export |= re.fullmatch(r'run\d+_\d{4}_\d{2}_\d{2}', base.name) is not None
    if is_export:
        actual = folder_run(base)
        if actual is not None and actual != run:
            raise ValueError(f'Target is run {run}, but this export belongs to run {actual}: {base}. '
                             f'Use --folder "{base.parent}" for automatic run-folder lookup.')
        folders = [base]
    else:
        folders = sorted(p for p in base.glob(f'run{run}_*') if p.is_dir())
    if date:
        folders = [p for p in folders if p.name.endswith('_' + date)]
    if not folders:
        suffix = f' on {date}' if date else ''
        raise FileNotFoundError(f'No export folder for run {run}{suffix} under {base}. '
                                'Choose the directory containing the run###_YYYY_MM_DD folders.')
    return folders


def csv_rows_for_event(path, run, event):
    if not path.is_file():
        return None
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not {'run', 'event'}.issubset(reader.fieldnames or []):
            raise ValueError(f'{path} needs run and event columns.')
        return [row for row in reader if int(row['run']) == run and int(row['event']) == event]


def event_matches(items, event):
    return any(int(item['event'] if isinstance(item, dict) else item) == event for item in items)


def audit_metadata(folder, run, event, experiment=None):
    folder = Path(folder)
    actual = folder_run(folder)
    if actual is not None and actual != run:
        raise ValueError(f'Folder metadata says run {actual}, but requested run is {run}: {folder}')
    print(f'Target: run {run}, event {event}')
    print(f'Export folder: {folder}')
    filename = f'run{run}_event{event}.png'
    present = (folder / filename).is_file()
    print(f'PNG exists: {present}')
    status_path = folder / 'export_status.json'
    status = load_json(status_path) if status_path.is_file() else {}
    definition_path = folder / 'cut_definition.json'
    definition = load_json(definition_path) if definition_path.is_file() else None
    experiments = {str(value) for value in (experiment, status.get('experiment'),
                   definition.get('experiment') if definition else None) if value is not None}
    if len(experiments) > 1:
        raise ValueError(f'Experiment mismatch between --experiment, cut and status: {sorted(experiments)}')
    experiment = next(iter(experiments), None)
    for key in ('experiment', 'run', 'status', 'selected_count', 'exported_count', 'error'):
        if key in status:
            print(f'{key}: {status[key]}')
    if not status:
        print('No export_status.json found; export completion is unknown.')
    skipped = status.get('skipped_empty_events', [])
    failed = status.get('failed_events', [])
    unprocessed = status.get('unprocessed_events', [])
    is_empty = event_matches(skipped, event)
    is_failed = event_matches(failed, event)
    is_unprocessed = event_matches(unprocessed, event)
    print(f'Skipped-empty count: {len(skipped)}; failed count: {len(failed)}')
    print(f'Target listed as empty: {is_empty}; failed: {is_failed}; unprocessed: {is_unprocessed}')
    rows = csv_rows_for_event(folder / 'selected_events.csv', run, event)
    selected = bool(rows) if rows is not None else None
    print(f'Target selected at export time (CSV): {selected if selected is not None else "unknown (CSV absent)"}')
    outcomes = csv_rows_for_event(folder / 'export_results.csv', run, event)
    if outcomes:
        for row in outcomes:
            print(f'Event journal: {row.get("status", "unknown")}; error: {row.get("error", "")}')
    if present:
        print('RESULT: The event PNG is present in this export folder.')
        if selected is False or is_empty or is_failed:
            print('METADATA CONFLICT: The PNG exists but its selection/outcome records disagree.')
    elif selected is False:
        print('RESULT: Not selected at export time. Check the polygon AND the GUI filters.')
    elif is_failed:
        print('RESULT: Selected, but building or saving the image failed.')
        for failure in failed:
            if isinstance(failure, dict) and int(failure['event']) == event:
                print(f'Failure details: {failure}')
    elif is_empty:
        print('RESULT: Selected, but the image builder found no hits above the image threshold.')
    elif is_unprocessed:
        print('RESULT: Selected, but not processed at the last saved export checkpoint.')
    elif outcomes and any(row.get('status') == 'exported' for row in outcomes):
        print('RESULT: Recorded as exported, but the PNG is now missing from this folder.')
    elif status.get('status') in ('failed', 'in_progress'):
        print('RESULT: This export was not marked complete; this event has no saved PNG.')
    elif selected:
        print('RESULT: Selected, but missing. Inspect the event data and export records.')
    else:
        print('RESULT: PNG missing; export-time selection is unknown.')
    return {'status': status, 'definition': definition, 'experiment': experiment}


def compare_filters(quantities, status, overrides):
    saved = status.get('filter_settings') or {}
    if not isinstance(saved, dict):
        raise ValueError('filter_settings must be a JSON object.')
    values = dict(saved)
    values.update(overrides)
    for lo, hi in (('range_min', 'range_max'), ('angle_min', 'angle_max'), ('ic_min', 'ic_max')):
        if lo in values and hi in values and float(values[lo]) >= float(values[hi]):
            raise ValueError(f'{lo} must be smaller than {hi}.')
    if not saved:
        print('Historical GUI filters: NOT RECORDED in this old export.')
    else:
        print('Filter comparison uses export_status.json filter_settings.')
    if overrides:
        print('USER-SUPPLIED COMPARISON (not proof of historical settings): ' +
              ', '.join(f'{key}={value}' for key, value in overrides.items()))
    failures, known = [], 0
    for setting, field, label, symbol, compare in FILTERS:
        if setting not in values:
            continue
        boundary = float(values[setting])
        if math.isnan(boundary):
            raise ValueError(f'Invalid NaN filter setting: {setting}')
        actual = quantities[field]
        passed = bool(compare(actual, boundary))
        known += 1
        print(f'  {"PASS" if passed else "FAIL"}: {label}: {actual:.9g} {symbol} {boundary:.9g}')
        if not passed:
            failures.append(setting)
    if failures:
        print('FILTER RESULT: Rejected under the compared settings: ' + ', '.join(failures))
    elif known == len(FILTERS):
        print('FILTER RESULT: Passes all compared GUI filters.')
    elif known:
        print(f'FILTER RESULT: Passes the {known} specified rules; {len(FILTERS)-known} rules are unknown.')
    else:
        print('Filter result is unknown. Supply the settings you used, e.g. --beam-off-min 0.1 --veto-threshold 150.')
    return failures


def inspect_event(folder, run, event, metadata, overrides):
    # These imports are deliberately optional for metadata-only audits.
    import numpy as np
    import uproot
    from matplotlib.path import Path as PolygonPath
    from raw_viewer import process_runs
    from GadgetRunH5 import GadgetRunH5, get_default_pad_gains

    experiment = metadata['experiment']
    if not experiment:
        raise ValueError('No experiment in this export; supply --experiment e25058.')
    status, definition = metadata['status'], metadata['definition']
    units = status.get('energy_units', 'MeV')
    if units != 'MeV':
        raise ValueError(f'This export uses {units!r}. Data inspection needs its actual energy gains; '
                         'the built-in inspector uses the framework default MeV calibration. '
                         'Metadata checks above are still valid.')
    gains = np.asarray(get_default_pad_gains(experiment), dtype=np.float64)
    print('Energy calibration: current framework default gains; custom historical gains were not recorded.')
    raw = process_runs.get_h5_file(experiment, run)
    try:
        raw.cache_enable = False
        first, last = raw.get_event_num_bounds()
        if not first <= event <= last:
            raise IndexError(f'Event {event} is outside run {run} raw H5 bounds: {first}..{last}')
        index = event - first
        root_path = Path(process_runs.get_save_path(experiment)) / f'{experiment}_run{run}.root'
        print(f'Raw H5: {raw.file_path}')
        print(f'Processed ROOT: {root_path}')
        with uproot.open(root_path) as root:
            tree = root['events']
            if index >= tree.num_entries:
                raise IndexError(f'ROOT has {tree.num_entries} entries; event maps to index {index}.')
            def read(name, library='np'):
                return tree[name].array(entry_start=index, entry_stop=index+1, library=library)[0]
            pad_charge = np.asarray(read('pad_charge'))
            pad_max = np.asarray(read('pad_max'))
            endpoints = np.asarray(read('endpoints'))
            rail_count = len(read('railed_pads', 'ak'))
            timestamp = float(read('timestamps'))
            prefix = tree['timestamps'].array(entry_stop=index+1, library='np')
        gaps = np.flatnonzero(np.diff(prefix) > 0.1) + 1
        window_start = prefix[gaps[-1]] if len(gaps) else prefix[0]
        beam_time = timestamp - float(window_start)
        energy = float(np.dot(pad_charge, gains))
        dr = endpoints[0] - endpoints[1]
        track_range = float(np.linalg.norm(dr))
        angle = float(np.degrees(np.arctan2(np.hypot(dr[0], dr[1]), abs(dr[2]))))
        veto = float(np.max(pad_max[list(process_runs.raw_h5_file.VETO_PADS)]))
        inside = None
        if definition is not None:
            vertices = np.asarray(definition['vertices'], dtype=float)
            if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3 or not np.isfinite(vertices).all():
                raise ValueError('Saved polygon vertices must be finite (energy, range) pairs.')
            inside = bool(PolygonPath(vertices).contains_points([[energy, track_range]])[0])
        print(f'Energy from ROOT pad_charge: {energy:.9g} MeV')
        print(f'Range: {track_range:.9g} mm; inside saved polygon: {inside if inside is not None else "unknown (cut absent)"}')
        print(f'Max veto: {veto:.9g}; angle: {angle:.9g} degrees; railed pads: {rail_count}')
        print(f'Time since beam off (framework gap method): {beam_time:.9g} s ({beam_time*1000:.6g} ms)')
        quantities = dict(energy=energy, range=track_range, angle=angle,
                          veto=veto, railed=rail_count, beam_time=beam_time)
        failures = compare_filters(quantities, status, overrides)
        if inside is False:
            print('POLYGON RESULT: Outside the saved cut.')
        elif inside is True and not failures:
            print('POLYGON RESULT: Inside; export also requires every GUI filter to pass.')
        gain_hash = hashlib.sha256(gains.tobytes()).hexdigest()[:16]
        cache = root_path.parent / f'gm_ic_{experiment}_run{run}_{gain_hash}.npy'
        if cache.is_file():
            try:
                cache_array = np.load(cache, mmap_mode='r')
                try:
                    cached_energy = float(cache_array[index])
                    print(f'Existing GUI energy cache value: {cached_energy:.9g} MeV')
                    if not np.isclose(cached_energy, energy, rtol=1e-9, atol=1e-12):
                        print('MISMATCH: Existing cached energy differs from current ROOT pad_charge and gains.')
                finally:
                    del cache_array
            except Exception as exc:
                print(f'ENERGY CACHE WARNING: {type(exc).__name__}: {exc}; continuing with fresh raw data.')
        image_threshold = float(status.get('image_hit_threshold_adc', 20))
        if not math.isfinite(image_threshold):
            raise ValueError('Image threshold must be finite.')
        print(f'Fresh image dry-run threshold: {image_threshold:g} ADC '
              f'({"saved setting" if "image_hit_threshold_adc" in status else "framework default; not historically recorded"})')
        for threshold in dict.fromkeys((image_threshold, 100.)):
            hits = raw.get_xyze(event, threshold=threshold, include_veto_pads=False)
            print(f'Fresh non-veto hit count above {threshold:g} ADC: {len(hits[3])}')
        data = GadgetRunH5.__new__(GadgetRunH5)
        data.h5_file = raw
        data.run_num = run
        data.image_hit_threshold = image_threshold
        data.get_event_num = lambda unused: event
        try:
            images = data.save_cutImages(np.array([0], dtype=np.int64))
            print(f'Fresh single-event builder produced: {[item[3] for item in images]}')
            if images:
                from event_image_io import TraceImageRenderer
                image = TraceImageRenderer().compose(images[0][0], images[0][1])
                print(f'PNG pixel construction succeeded: shape={image.shape}; no file was written.')
            else:
                print('IMAGE RESULT: Fresh builder returned no image (empty image hit set).')
        except Exception as exc:
            raise RuntimeError(f'SINGLE-EVENT IMAGE FAILURE: {type(exc).__name__}: {exc}') from exc
    finally:
        raw.h5_file.close()


def positive_or_zero(text):
    number = int(text)
    if number < 0:
        raise argparse.ArgumentTypeError('Run and event numbers must be nonnegative.')
    return number


def target_pair(text):
    try:
        run, event = text.split(':')
        return positive_or_zero(run), positive_or_zero(event)
    except (ValueError, argparse.ArgumentTypeError) as exc:
        raise argparse.ArgumentTypeError(f'Use RUN:EVENT, for example 147:649226; got {text!r}') from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--folder', required=True, help='One run export folder OR its Nandhalalan parent.')
    parser.add_argument('--run', type=positive_or_zero)
    parser.add_argument('--event', type=positive_or_zero, help='One original H5 event number.')
    parser.add_argument('--events', type=positive_or_zero, nargs='+', help='Several event numbers for --run.')
    parser.add_argument('--targets', type=target_pair, nargs='+', help='Pairs such as 147:649226 149:123456.')
    parser.add_argument('--date', help='Optional export date YYYY_MM_DD (or YYYY-MM-DD).')
    parser.add_argument('--experiment', help='Experiment override for exports lacking this metadata.')
    parser.add_argument('--inspect-data', action='store_true', help='Read ROOT/H5 and test image construction in memory.')
    for name, _, _, _, _ in FILTERS:
        flag = {'ic_min': 'energy-min', 'ic_max': 'energy-max'}.get(name, name.replace('_', '-'))
        parser.add_argument('--' + flag, dest=name, type=float, help='User-supplied filter comparison override.')
    args = parser.parse_args(argv)
    if args.targets:
        if args.run is not None or args.event is not None or args.events is not None:
            parser.error('Use --targets OR --run with --event/--events, not both.')
        targets = args.targets
    else:
        if args.run is None or (args.event is None and args.events is None):
            parser.error('Supply --targets RUN:EVENT ... OR --run RUN with --event EVENT / --events EVENT ...')
        if args.event is not None and args.events is not None:
            parser.error('Use --event or --events, not both.')
        targets = [(args.run, evt) for evt in (args.events if args.events is not None else [args.event])]
    targets = list(dict.fromkeys(targets))
    date = args.date.replace('-', '_') if args.date else None
    if date and not re.fullmatch(r'\d{4}_\d{2}_\d{2}', date):
        parser.error('--date must be YYYY_MM_DD or YYYY-MM-DD.')
    overrides = {name: getattr(args, name) for name, _, _, _, _ in FILTERS if getattr(args, name) is not None}
    if any(math.isnan(value) for value in overrides.values()):
        parser.error('Filter overrides cannot be NaN.')
    if overrides and not args.inspect_data:
        parser.error('Filter comparison flags require --inspect-data.')
    errors, completed = 0, 0
    for run, event in targets:
        try:
            folders = find_exports(args.folder, run, date)
        except Exception as exc:
            print(f'ERROR for run {run}, event {event}: {type(exc).__name__}: {exc}')
            errors += 1
            continue
        if len(folders) > 1:
            print(f'Run {run}: found {len(folders)} exports; checking each. Use --date to restrict.')
        for folder in folders:
            print('\n' + '=' * 72)
            try:
                metadata = audit_metadata(folder, run, event, args.experiment)
                if args.inspect_data:
                    inspect_event(folder, run, event, metadata, overrides)
                else:
                    print('Add --inspect-data for polygon, filter values and a fresh image dry-run.')
                completed += 1
            except Exception as exc:
                print(f'AUDIT ERROR for run {run}, event {event} in {folder}: {type(exc).__name__}: {exc}')
                print('Any metadata findings printed above still apply; continuing with other targets.')
                errors += 1
    print(f'\nAudit finished: {completed} completed event/export checks; {errors} audit errors.')
    return 2 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
