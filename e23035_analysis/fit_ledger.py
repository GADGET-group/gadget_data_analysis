"""
One row per saved fit: what it is, where it came from, and how well it fit.

try_fit already records everything needed -- the configuration in fit_<hash>_info.json and,
for fits made by add_peak_to_fit / remove_peak_from_fit / recenter_peak_bounds /
change_knots_from_fit, a `provenance` block naming the parent hash and the operation. Nothing
read that back until now, so the relationship between the fits in a folder had to be
reconstructed from timestamps. This module turns a folder into a table and a tree.

Reading is direct ROOT access to the saved TFitResult and TF1 rather than
load_spectrum_fitter_from_file, which is ~15x faster (0.3 s per fit, so a whole folder in
seconds) and enough for everything reported here.

    python -m e23035_analysis.fit_ledger protons_le_10keV_bins
    python -m e23035_analysis.fit_ledger protons_le_10keV_bins --tree --csv ledger.csv
"""
import os
import json
import glob
import datetime

import numpy as np
import pandas as pd
import ROOT

from e23035_analysis import fit_diagnostics
from e23035_analysis.fit_tpc_spectrum_simultaneous import fit_path

# Provenance keys that name what an operation did, in the order they are worth printing.
DETAIL_KEYS = ('peaks_added', 'new_peak_locs', 'peaks_removed', 'peaks_recentered', 'wiggle',
               'bg_knots', 'knots_added', 'knots_removed')


def _summarize_root(root_path, window_idx=0):
    '''chi2/ndf/p/pinned straight from the saved fit objects. {} if the file has no result.'''
    f = ROOT.TFile.Open(root_path)
    if not f or f.IsZombie():
        return {}
    try:
        fit_res = f.Get(f'peak_{window_idx}_fit_res')
        f_to_fit = f.Get(f'peak_{window_idx}_f_to_fit_2d') or f.Get(f'peak_{window_idx}_f_to_fit')
        if not fit_res:
            return {}
        chi2 = 2.0 * fit_res.MinFcnValue()
        root_chi2 = fit_res.Chi2()
        out = {'chi2': chi2, 'ndf': fit_res.Ndf(), 'npar': fit_res.NPar(), 'p': fit_res.Prob(),
               'chi2_ndf': chi2 / fit_res.Ndf() if fit_res.Ndf() else np.nan,
               'chi2_trustworthy': abs(root_chi2 - chi2) <= 1e-6 * max(abs(chi2), 1.0),
               'valid': bool(fit_res.IsValid()), 'status': fit_res.Status()}
        if f_to_fit:
            pinned = fit_diagnostics.pinned_from_function(f_to_fit)
            out['n_pinned'] = len(pinned)
            out['pinned_mu'] = ';'.join(p['name'] for p in pinned if p['name'].startswith('mu'))
        spec = f.Get('spectrum_0')
        if spec:
            out['bin_width'] = spec.GetBinWidth(1)
        return out
    finally:
        f.Close()


def _detail_string(provenance):
    parts = []
    for key in DETAIL_KEYS:
        if key in provenance:
            value = provenance[key]
            if key == 'bg_knots' and isinstance(value, list):
                parts.append(f'{len(value)} knots')
            else:
                parts.append(f'{key}={value}')
    return ', '.join(parts)


def scan_folder(folder_name, window_idx=0):
    '''
    Every fit in one folder as a DataFrame, newest last.

    Fits made by hand (no provenance) get operation 'manual' and parent '-'. A fit whose
    _info.json exists but whose .root does not (a run that died, or is still going) is kept,
    with empty fit statistics, so the gaps are visible rather than silently dropped.
    '''
    folder = os.path.join(fit_path, folder_name)
    rows = []
    for info_path in sorted(glob.glob(os.path.join(folder, 'fit_*_info.json'))):
        hash_str = os.path.basename(info_path)[len('fit_'):-len('_info.json')]
        with open(info_path) as fh:
            info = json.load(fh)
        provenance = info.get('provenance') or {}
        args = info.get('args', {})
        peaks = info.get('peaks') or []
        knots = args.get('bg_knots')
        root_path = os.path.join(folder, f'fit_{hash_str}.root')

        # A screening fit froze most of the model, so its chi2 is a lower bound on what the
        # same change is worth and must never be compared with a full fit's. try_fit stores
        # the predicate by name, which is where that shows up.
        fix_params = provenance.get('fix_params')
        screen = isinstance(fix_params, str) and fix_params.startswith('local_unfreeze')

        row = {
            'hash': hash_str,
            'parent': provenance.get('parent_hash', '-'),
            'operation': ('screen' if screen else provenance.get('operation', 'manual')),
            'details': _detail_string(provenance),
            'screen': screen,
            'n_peaks': sum(len(w[0]) for w in peaks),
            'n_knots': len(knots) if isinstance(knots, list) else (knots if knots else 0),
            'bg': f"{args.get('bg_model')}{args.get('bg_order')}",
            'bin_width': info.get('bin_width'),
            'mtime': datetime.datetime.fromtimestamp(os.path.getmtime(info_path)),
            'has_root': os.path.exists(root_path),
        }
        if row['has_root']:
            row.update(_summarize_root(root_path, window_idx))
            # A fit's wall time is the gap between its config being written and its result
            # being saved. Close enough to spot a fit that took much longer than its siblings.
            row['minutes'] = (os.path.getmtime(root_path) - os.path.getmtime(info_path)) / 60.0
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if 'bin_width' in df:
        df['bin_width'] = df['bin_width'].fillna(df.get('bin_width'))
    return df.sort_values('mtime').reset_index(drop=True)


def print_table(df, sort_by='chi2'):
    if df.empty:
        print('no fits found')
        return
    cols = ['hash', 'parent', 'operation', 'n_peaks', 'n_knots', 'bin_width', 'chi2', 'ndf',
            'chi2_ndf', 'p', 'n_pinned', 'minutes', 'details']
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()
    if sort_by in out.columns:
        out = out.sort_values(sort_by)
    with pd.option_context('display.width', 250, 'display.max_colwidth', 40):
        print(out.to_string(index=False, float_format=lambda v: f'{v:.4g}'))


def print_tree(df):
    '''
    The parent -> child structure, so a chain of modified fits reads top to bottom.

    Roots are fits whose parent is not in this folder (hand-made ones, and any fit whose
    parent was deleted). Children are ordered by when they were made.
    '''
    if df.empty:
        print('no fits found')
        return
    by_parent = {}
    known = set(df['hash'])
    for _, r in df.iterrows():
        parent = r['parent'] if r['parent'] in known and r['parent'] != r['hash'] else None
        by_parent.setdefault(parent, []).append(r)

    def show(row, depth):
        chi2 = f"{row['chi2']:.1f}/{int(row['ndf'])}" if row.get('has_root') and not pd.isna(row.get('chi2')) else 'no result'
        pinned = '' if pd.isna(row.get('n_pinned')) else f" pinned {int(row['n_pinned'])}"
        detail = f"  {row['details']}" if row['details'] else ''
        print(f"{'    ' * depth}{row['hash']}  {row['operation']:<10} {chi2:>12}"
              f"  peaks {row['n_peaks']:>3} knots {row['n_knots']:>2}{pinned}{detail}")
        for child in by_parent.get(row['hash'], []):
            show(child, depth + 1)

    for root in by_parent.get(None, []):
        show(root, 0)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('folder', help='folder under tpc_spectrum_fitting')
    parser.add_argument('--tree', action='store_true', help='print the parent/child tree instead of the table')
    parser.add_argument('--sort', default='chi2', help='table column to sort by (default chi2)')
    parser.add_argument('--csv', help='also write the table here (relative to the fit folder)')
    parser.add_argument('--no-screens', action='store_true',
                        help='leave out screening fits, whose chi2 is not comparable')
    parser.add_argument('--window', type=int, default=0)
    args = parser.parse_args()

    ROOT.gROOT.SetBatch(True)
    df = scan_folder(args.folder, window_idx=args.window)
    if args.no_screens and 'screen' in df.columns:
        df = df[~df['screen']].reset_index(drop=True)
    if args.tree:
        print_tree(df)
    else:
        print_table(df, sort_by=args.sort)
    if args.csv:
        out = os.path.join(fit_path, args.folder, args.csv)
        df.to_csv(out, index=False)
        print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
