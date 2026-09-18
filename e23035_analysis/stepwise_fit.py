"""
Choose the peak list and the background knots by rule instead of by hand, and leave a record
of every decision.

The loop, from a starting fit:

    1. diagnose            residual scan, pinned parameters
    2. recenter            any peak sitting at a mu bound is re-centred on where it ended up
    3. add peaks           screen the top candidates cheaply, then refit the best properly
    4. prune peaks         periodically, try dropping each non-core peak
    5. knots               once peaks are stable, try removing each knot and adding one where
                           the residuals run broadly the same sign
    6. stop                when a full sweep changes nothing, or the fit budget runs out

Every fit goes through try_fit, so it is hashed, cached and carries a provenance block naming
its parent and the operation -- fit_ledger turns a folder back into the tree. On top of that
this module appends one record per candidate *considered*, accepted or not, to decisions.jsonl.
Rejections are what make the run reproducible and resumable: without them a rerun would
re-try, at six minutes a fit, everything it had already ruled out.

Decisions use delta-chi2 = chi2(parent) - chi2(child) on 2*MinFcnValue(), never fit_res.Chi2(),
and never across bin widths. The thresholds sit above the nominal Wilks values because peak
locations are picked by scanning residuals (a look-elsewhere penalty) and because an amplitude
bounded at zero is a boundary case; they are config, not constants.

Screening freezes everything a new peak cannot interact with (fit_diagnostics.local_unfreeze),
which is ~200x faster than a full refit. Freezing can only make a fit look worse, so a screen
ranks candidates and nothing else: only a full refit accepts, rejects, or rules a region out.

    python -m e23035_analysis.stepwise_fit --folder protons_le_10keV_bins --start d8f7935d --dry-run
    python -m e23035_analysis.stepwise_fit --folder protons_le_10keV_bins --start d8f7935d --max-fits 40
"""
import os
import json
import time
import datetime
import argparse

import numpy as np
import ROOT

from e23035_analysis import fit_tpc_spectrum_simultaneous as fts
from e23035_analysis import fit_diagnostics as fd
from e23035_analysis import fitting_tools
from e23035_analysis.fit_tpc_spectrum_simultaneous import fit_path

DECISIONS_FILE = 'decisions.jsonl'

DEFAULT_CONFIG = {
    # Accept an added peak only for this much delta-chi2 (2 free parameters: mu, total_amp).
    'add_peak_dchi2': 16.0,
    # Keep a removed peak out unless putting it back is worth this much.
    'remove_peak_dchi2': 9.0,
    # Adding a knot costs one coefficient per spectrum.
    'add_knot_dchi2': 12.0,
    'remove_knot_dchi2': 9.0,
    # A new peak must also be this significant, and off its bound, to be kept.
    'min_amp_significance': 3.0,
    # Residual scan: only candidates this significant are worth a screen at all.
    'candidate_sig_min': fd.PEAK_SIG_MIN,
    # Screen candidates whose screened dchi2 reaches this fraction of the accept bar; below
    # it a candidate is deferred rather than rejected, since a screen underestimates.
    'screen_queue_fraction': 0.5,
    'screen_top_k': 3,
    'screen_n_sigmas': 3.0,
    # Skip screening and refit the top candidates directly. Slower, always correct.
    'screen_disabled': False,
    # Sweeps with nothing added before the knot pass may run. While peaks are still missing,
    # their counts have to go somewhere, and a background with another knot is happy to take
    # them: giving the background more freedom exactly when the peak search has stalled is how
    # the first rebuild run buried its own candidates.
    'knot_pass_after_stalls': 2,
    'prune_every': 4,
    # Only bother trying to remove a peak whose amplitude is this uncertain. Removing a 20-sigma
    # peak costs hundreds of chi2 and the answer is never in doubt, so spending a full refit to
    # confirm it is waste; this reads the significance off the current fit for free.
    'prune_significance_max': 5.0,
    'max_recenter_rounds': 3,
    # Whether a core peak may be re-centred. Off: a core peak sits at a known energy, so when
    # one ends up against its bound the cause is a missing neighbour absorbing its counts, not
    # a bad guess -- re-centring would walk it away, and two of them are the calibration
    # anchors. They come back on their own as the peak list fills in.
    'recenter_core': False,
    # Peaks that are never pruned: the ones whose position and counts were stable across all
    # 25 better hand fits, which includes both calibration anchors (59Zn 904 and 1778). As
    # [energy, isotope] so a peak of the other isotope nearby cannot be protected by mistake:
    # 60Ga 1795 sits 17 keV from the 59Zn 1778 anchor and is one of the least stable peaks
    # there is. The tolerance covers the drift between a guess and its fitted value.
    'core_peaks': [[716, '60Ga'], [1110, '60Ga'], [1400, '60Ga'],
                   [820, '59Zn'], [904, '59Zn'], [940, '59Zn'],
                   [1063, '59Zn'], [1376, '59Zn'], [1778, '59Zn']],
    'core_tolerance': 15.0,
    'max_fits': 40,
}


def load_config(path=None):
    config = dict(DEFAULT_CONFIG)
    if path:
        with open(path) as fh:
            config.update(json.load(fh))
    return config


class Decisions:
    '''
    The append-only record of every candidate considered, and the memory that makes a run
    resumable: regions a full refit has rejected are not offered again.
    '''

    def __init__(self, folder_name):
        self.path = os.path.join(fit_path, folder_name, DECISIONS_FILE)
        self.records = []
        if os.path.exists(self.path):
            with open(self.path) as fh:
                self.records = [json.loads(line) for line in fh if line.strip()]

    def append(self, **record):
        record['time'] = datetime.datetime.now().isoformat(timespec='seconds')
        self.records.append(record)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'a') as fh:
            fh.write(json.dumps(record) + '\n')
        return record

    def rejected_regions(self, sigma_pad=30.0):
        '''(low, high) around every peak a full refit rejected, to keep off them on a rerun.'''
        out = []
        for r in self.records:
            if r.get('operation') == 'add' and r.get('verdict') == 'reject' and r.get('kind') == 'full':
                energy = r.get('energy')
                if energy is not None:
                    out.append((energy - sigma_pad, energy + sigma_pad))
        return out

    def last_accepted_child(self):
        for r in reversed(self.records):
            if r.get('verdict') == 'accept' and r.get('child'):
                return r['child']
        return None

    def n_fits(self):
        return sum(1 for r in self.records if r.get('child'))


def chi2_of(fitter):
    chi2, ndf, trustworthy = fd.fit_stat(fitter)
    if not trustworthy:
        print('  WARNING: fit_res.Chi2() disagrees with 2*MinFcnValue(); using MinFcnValue')
    return chi2, ndf


def is_core(energy, isotope, config):
    '''Whether a fitted peak is one of the protected ones. Isotope must match too.'''
    return any(abs(energy - c_energy) <= config['core_tolerance'] and isotope == c_iso
               for c_energy, c_iso in config['core_peaks'])


def peak_amp_significance(fitter, peak_idx, window_idx=0):
    '''total_amp / its error for one peak, the "is this peak real" number.'''
    _, f_to_fit = fd._result(fitter, window_idx)
    name = f'total_amp_{peak_idx}' if f_to_fit.GetParNumber(f'total_amp_{peak_idx}') >= 0 else 'total_amp'
    j = f_to_fit.GetParNumber(name)
    if j < 0:
        return np.nan
    val, err = f_to_fit.GetParameter(j), f_to_fit.GetParError(j)
    return val / err if err > 0 else np.inf


def try_recenter(fitter, hash_str, folder, decisions, config, budget):
    '''
    Re-centre pinned peaks until none are pinned or the round limit is reached.

    A pinned mu is not a fitted value -- the fit ran out of room and stopped at the edge -- so
    this runs before any candidate is judged, and its fits are not a matter of accept/reject:
    the peak list does not change, so the result is simply adopted.
    '''
    for round_idx in range(config['max_recenter_rounds']):
        pinned = [p for p in fd.pinned_params(fitter) if p['name'].startswith('mu')]
        if not config['recenter_core']:
            positions = {idx: (energy, iso) for idx, energy, iso in fd.peak_positions(fitter)}
            kept = []
            for p in pinned:
                idx = 0 if p['name'] == 'mu' else int(p['name'].split('_')[1])
                energy, iso = positions.get(idx, (p['value'], 'unknown'))
                if is_core(energy, iso, config):
                    print(f"  leaving core peak {p['name']}={p['value']:.0f} ({iso}) at its bound")
                else:
                    kept.append(p)
            pinned = kept
        if not pinned:
            return fitter, hash_str
        if budget.exhausted():
            return fitter, hash_str
        print(f'  recentering {len(pinned)} pinned peaks (round {round_idx + 1}): '
              + ', '.join(f"{p['name']}={p['value']:.0f}" for p in pinned))
        chi2_before, _ = chi2_of(fitter)
        budget.spend()
        # Pass the peaks explicitly: left to itself recenter_peak_bounds would take every
        # pinned mu, including the core ones just filtered out.
        to_recenter = [(0, 0 if p['name'] == 'mu' else int(p['name'].split('_')[1])) for p in pinned]
        new_hash, new_fitter = fts.recenter_peak_bounds(fitter, peaks_to_recenter=to_recenter)
        chi2_after, _ = chi2_of(new_fitter)
        still = [p['name'] for p in fd.pinned_params(new_fitter) if p['name'].startswith('mu')]
        decisions.append(operation='recenter', kind='full', parent=hash_str, child=new_hash,
                         details={'pinned': [p['name'] for p in pinned], 'still_pinned': still},
                         dchi2=chi2_before - chi2_after, verdict='accept',
                         reason='recentering does not change the peak list')
        fitter, hash_str = new_fitter, new_hash
    print(f'  still pinned after {config["max_recenter_rounds"]} rounds; leaving them')
    return fitter, hash_str


class Budget:
    '''
    Counts full refits against the budget, screens separately.

    A full refit of this model takes ~6 minutes and a screen ~2 seconds, so one limit for both
    would let a few sweeps of screening consume a run's whole allowance. Screens have their own,
    much larger cap purely as a runaway guard.
    '''

    def __init__(self, max_fits, max_screens=None, already_spent=0):
        self.max_fits = max_fits
        self.max_screens = max_screens if max_screens is not None else 20 * max_fits
        self.spent = already_spent
        self.screens = 0

    def spend(self, n=1):
        self.spent += n

    def spend_screen(self, n=1):
        self.screens += n

    def exhausted(self):
        return self.spent >= self.max_fits or self.screens >= self.max_screens

    def __str__(self):
        return f'{self.spent}/{self.max_fits} fits, {self.screens} screens'


def add_peak_pass(fitter, hash_str, folder, decisions, config, budget):
    '''
    One attempt at adding a peak. Returns (fitter, hash, changed).

    Candidates come from the residual scan, are screened (cheap, frozen neighbourhood) to rank
    them, and the best is refit properly. Only that full refit decides.
    '''
    exclude = decisions.rejected_regions()
    candidates = fd.peak_candidates(fitter, sig_min=config['candidate_sig_min'], exclude=exclude)
    if not candidates:
        print('  no peak candidates above '
              f'{config["candidate_sig_min"]} sigma (excluding {len(exclude)} rejected regions)')
        return fitter, hash_str, False

    print(f'  {len(candidates)} candidates; top: '
          + ', '.join(f"{c['energy']:.0f} ({c['significance']:.1f} sigma, {c['isotope']})"
                      for c in candidates[:config['screen_top_k']]))

    queue = []
    if config['screen_disabled']:
        queue = [(c, None) for c in candidates[:config['screen_top_k']]]
    else:
        chi2_parent, _ = chi2_of(fitter)
        for candidate in candidates[:config['screen_top_k']]:
            if budget.exhausted():
                break
            budget.spend_screen()
            predicate = fd.local_unfreeze(fitter, candidate['energy'], n_sigmas=config['screen_n_sigmas'])
            screen_hash, screen_fitter = fts.add_peak_to_fit(
                fitter, new_peak_loc=candidate['energy'], new_peak_iso=candidate['isotope'],
                fix_params=predicate)
            chi2_screen, _ = chi2_of(screen_fitter)
            dchi2 = chi2_parent - chi2_screen
            threshold = config['screen_queue_fraction'] * config['add_peak_dchi2']
            verdict = 'queue' if dchi2 >= threshold else 'defer'
            print(f"    screen {candidate['energy']:.0f}: dchi2 {dchi2:+.1f} -> {verdict}")
            decisions.append(operation='add', kind='screen', parent=hash_str, child=screen_hash,
                             energy=candidate['energy'], isotope=candidate['isotope'],
                             dchi2=dchi2, verdict=verdict,
                             reason=f'screened dchi2 {dchi2:.1f} vs queue threshold {threshold:.1f}',
                             details={'free_params': sorted(predicate.free_params)})
            if verdict == 'queue':
                queue.append((candidate, dchi2))
        queue.sort(key=lambda item: -(item[1] or 0))

    if not queue:
        print('  nothing worth a full refit')
        return fitter, hash_str, False

    candidate, screened = queue[0]
    if budget.exhausted():
        return fitter, hash_str, False
    print(f"  full refit adding {candidate['energy']:.0f} ({candidate['isotope']})")
    chi2_parent, _ = chi2_of(fitter)
    budget.spend()
    new_hash, new_fitter = fts.add_peak_to_fit(
        fitter, new_peak_loc=candidate['energy'], new_peak_iso=candidate['isotope'], fix_params=False)
    chi2_child, _ = chi2_of(new_fitter)
    dchi2 = chi2_parent - chi2_child

    # Where did the new peak land in the new numbering, and is it a real peak?
    # The new peak can have moved up to loc_wiggle from the guess it was given.
    new_idx = current_index_of(new_fitter, candidate['energy'],
                               tolerance=getattr(new_fitter, 'location_wiggle', 15.0))
    significance = peak_amp_significance(new_fitter, new_idx) if new_idx is not None else np.nan
    pinned_names = [p['name'] for p in fd.pinned_params(new_fitter)]
    # The two bounds mean opposite things. An amplitude driven to its floor is a peak with no
    # counts -- reject it. A mu at its bound only means the peak wants to sit further from the
    # guess than loc_wiggle allows: the residual scan puts a candidate at the centre of an
    # excess, which is biased when a neighbour is still missing. That is a mis-centred window,
    # not a spurious peak, and the recenter pass at the top of the next sweep fixes it. The
    # first rebuild run rejected a 20-sigma peak worth dchi2 450 this way.
    mu_pinned = new_idx is not None and f'mu_{new_idx}' in pinned_names
    amp_pinned = new_idx is not None and f'total_amp_{new_idx}' in pinned_names

    reasons = []
    if dchi2 < config['add_peak_dchi2']:
        reasons.append(f'dchi2 {dchi2:.1f} < {config["add_peak_dchi2"]}')
    if not np.isnan(significance) and significance < config['min_amp_significance']:
        reasons.append(f'amplitude {significance:.1f} sigma < {config["min_amp_significance"]}')
    if amp_pinned:
        reasons.append('amplitude driven to its floor: the peak has no counts')
    verdict = 'accept' if not reasons else 'reject'
    note = ' [at its mu bound; the next recenter pass will move it]' if mu_pinned else ''

    print(f"  -> {verdict}: dchi2 {dchi2:+.1f}, amplitude {significance:.1f} sigma"
          + (f" ({'; '.join(reasons)})" if reasons else '') + note)
    decisions.append(operation='add', kind='full', parent=hash_str, child=new_hash,
                     energy=candidate['energy'], isotope=candidate['isotope'],
                     dchi2=dchi2, screened_dchi2=screened, amp_significance=float(significance),
                     mu_at_bound=bool(mu_pinned),
                     verdict=verdict, reason='; '.join(reasons) or 'passes every rule')
    if verdict == 'accept':
        return new_fitter, new_hash, True
    return fitter, hash_str, False


def current_index_of(fitter, energy, tolerance=5.0):
    '''
    Where a peak of this energy sits in the current numbering, or None.

    Peak parameters are indexed by position in the sorted list, so removing one renumbers
    everything above it, and each refit moves the survivors a little. Identifying peaks by
    energy and re-resolving the index before every operation is what keeps a prune pass from
    removing the wrong peak after its first success.
    '''
    peaks = fd.peak_positions(fitter)
    if not peaks:
        return None
    idx, mu, _ = min(peaks, key=lambda p: abs(p[1] - energy))
    return idx if abs(mu - energy) <= tolerance else None


def prune_pass(fitter, hash_str, folder, decisions, config, budget):
    '''Try dropping each non-core peak; keep it dropped when it costs little.'''
    changed = False
    to_try = []
    for idx, energy, iso in fd.peak_positions(fitter):
        if is_core(energy, iso, config):
            continue
        significance = peak_amp_significance(fitter, idx)
        if np.isfinite(significance) and significance > config['prune_significance_max']:
            continue
        to_try.append((energy, iso, significance))
    if not to_try:
        print(f'  no peak below {config["prune_significance_max"]} sigma to try removing')
        return fitter, hash_str, False

    for energy, iso, significance in to_try:
        if budget.exhausted():
            break
        idx = current_index_of(fitter, energy)
        if idx is None:
            continue    # it was removed, or moved too far to identify
        chi2_parent, _ = chi2_of(fitter)
        budget.spend()
        new_hash, new_fitter = fts.remove_peak_from_fit(fitter, (0, idx))
        chi2_child, _ = chi2_of(new_fitter)
        cost = chi2_child - chi2_parent   # removing can only make chi2 worse or equal
        verdict = 'accept' if cost < config['remove_peak_dchi2'] else 'reject'
        print(f'  remove {energy:.0f} ({iso}, {significance:.1f} sigma): costs {cost:+.1f} -> {verdict}')
        decisions.append(operation='remove', kind='full', parent=hash_str, child=new_hash,
                         energy=energy, isotope=iso, amp_significance=float(significance),
                         dchi2=-cost, verdict=verdict,
                         reason=f'removal costs {cost:.1f} vs {config["remove_peak_dchi2"]}')
        if verdict == 'accept':
            fitter, hash_str, changed = new_fitter, new_hash, True
    return fitter, hash_str, changed


def knot_pass(fitter, hash_str, folder, decisions, config, budget):
    '''
    Try removing each interior knot, then adding one wherever the residuals run broadly the
    same sign. Removal first: a background that is too free is the failure mode seen in the
    hand fits, where spare knots let the background grow peak-shaped bumps.
    '''
    changed = False
    knots = list(getattr(fitter, 'fit_multi_peaks_kwargs', {}).get('bg_knots') or [])
    if not knots:
        print('  no interior knots to work with')
        return fitter, hash_str, False

    for knot in list(knots):
        if budget.exhausted():
            break
        trial = [k for k in knots if k != knot]
        chi2_parent, _ = chi2_of(fitter)
        budget.spend()
        new_hash, new_fitter = fts.change_knots_from_fit(fitter, trial)
        chi2_child, _ = chi2_of(new_fitter)
        cost = chi2_child - chi2_parent
        verdict = 'accept' if cost < config['remove_knot_dchi2'] else 'reject'
        print(f'  remove knot {knot:.0f}: costs {cost:+.1f} -> {verdict}')
        decisions.append(operation='knots', kind='full', parent=hash_str, child=new_hash,
                         energy=knot, dchi2=-cost, verdict=verdict,
                         details={'removed_knot': knot, 'bg_knots': trial},
                         reason=f'removal costs {cost:.1f} vs {config["remove_knot_dchi2"]}')
        if verdict == 'accept':
            fitter, hash_str, knots, changed = new_fitter, new_hash, trial, True

    for run in fd.broad_runs(fitter):
        if budget.exhausted():
            break
        centre = round(run['center'] / 10.0) * 10.0
        if any(abs(centre - k) < 50 for k in knots):
            continue
        trial = sorted(knots + [centre])
        chi2_parent, _ = chi2_of(fitter)
        budget.spend()
        new_hash, new_fitter = fts.change_knots_from_fit(fitter, trial)
        chi2_child, _ = chi2_of(new_fitter)
        dchi2 = chi2_parent - chi2_child
        verdict = 'accept' if dchi2 >= config['add_knot_dchi2'] else 'reject'
        print(f'  add knot {centre:.0f} (run z {run["z"]:+.1f}): dchi2 {dchi2:+.1f} -> {verdict}')
        decisions.append(operation='knots', kind='full', parent=hash_str, child=new_hash,
                         energy=centre, dchi2=dchi2, verdict=verdict,
                         details={'added_knot': centre, 'bg_knots': trial, 'run_z': run['z']},
                         reason=f'dchi2 {dchi2:.1f} vs {config["add_knot_dchi2"]}')
        if verdict == 'accept':
            fitter, hash_str, knots, changed = new_fitter, new_hash, trial, True

    return fitter, hash_str, changed


def dry_run(fitter, hash_str, decisions, config):
    '''Everything the loop would try next, without fitting anything.'''
    print(f'=== dry run from {hash_str}')
    fd.report(fitter)
    exclude = decisions.rejected_regions()
    print(f'\nwould try, in order:')
    pinned = [p for p in fd.pinned_params(fitter) if p['name'].startswith('mu')]
    if pinned:
        print(f'  1. recenter {len(pinned)} pinned peaks: '
              + ', '.join(f"{p['name']}={p['value']:.0f}" for p in pinned))
    candidates = fd.peak_candidates(fitter, sig_min=config['candidate_sig_min'], exclude=exclude)
    if candidates:
        for c in candidates[:config['screen_top_k']]:
            print(f"  2. screen a {c['isotope']} peak at {c['energy']:.0f} ({c['significance']:.1f} sigma)")
    else:
        print(f'  2. no peak candidates (excluding {len(exclude)} previously rejected regions)')
    knots = list(getattr(fitter, 'fit_multi_peaks_kwargs', {}).get('bg_knots') or [])
    print(f'  3. try removing each of {len(knots)} knots: {knots}')
    runs = fd.broad_runs(fitter)
    if runs:
        for r in runs:
            print(f"  4. try adding a knot at {r['center']:.0f} (run z {r['z']:+.1f})")
    else:
        print('  4. no broad residual runs, so no knot to add')
    non_core = [e for _, e, iso in fd.peak_positions(fitter) if not is_core(e, iso, config)]
    print(f'  5. prune pass over {len(non_core)} non-core peaks '
          f'({len(fd.peak_positions(fitter)) - len(non_core)} protected)')


def run(folder, start_hash, config, resume=False):
    decisions = Decisions(folder)
    if resume:
        last = decisions.last_accepted_child()
        if last:
            print(f'resuming from {last} ({len(decisions.records)} decisions on record)')
            start_hash = last
    budget = Budget(config['max_fits'])

    fitter = fts.load_fit(start_hash, folder_name=folder)
    hash_str = start_hash
    chi2, ndf = chi2_of(fitter)
    print(f'start {hash_str}: chi2 {chi2:.1f} / {ndf} = {chi2 / ndf:.3f}, '
          f'{len(fd.peak_positions(fitter))} peaks')

    sweep = 0
    since_prune = 0
    stalls = 0
    while not budget.exhausted():
        sweep += 1
        print(f'\n=== sweep {sweep} ({budget})')
        changed = False

        fitter, hash_str = try_recenter(fitter, hash_str, folder, decisions, config, budget)

        print(' peaks:')
        fitter, hash_str, added = add_peak_pass(fitter, hash_str, folder, decisions, config, budget)
        changed |= added
        since_prune += 1 if added else 0

        if since_prune >= config['prune_every'] or not added:
            print(' pruning:')
            fitter, hash_str, pruned = prune_pass(fitter, hash_str, folder, decisions, config, budget)
            changed |= pruned
            since_prune = 0

        stalls = 0 if added else stalls + 1
        if not added and stalls >= config['knot_pass_after_stalls']:
            print(' knots:')
            fitter, hash_str, knotted = knot_pass(fitter, hash_str, folder, decisions, config, budget)
            changed |= knotted
        elif not added:
            print(f' knots: deferred ({stalls} of {config["knot_pass_after_stalls"]} stalled sweeps)')

        # Only a sweep that got as far as the knot pass can end the run: a stalled sweep whose
        # knot pass is still deferred has not tried everything yet.
        if not changed and stalls >= config['knot_pass_after_stalls']:
            print(f'\nno change in a full sweep; stopping at {hash_str} ({budget})')
            decisions.append(operation='stop', kind='terminal', parent=hash_str, child=None,
                             verdict='stop', reason='a full sweep changed nothing')
            break
    else:
        print(f'\nfit budget exhausted ({budget}); stopping at {hash_str}')
        decisions.append(operation='stop', kind='terminal', parent=hash_str, child=None,
                         verdict='stop', reason=f'fit budget exhausted ({budget})')

    chi2, ndf = chi2_of(fitter)
    print(f'final {hash_str}: chi2 {chi2:.1f} / {ndf} = {chi2 / ndf:.3f}, '
          f'{len(fd.peak_positions(fitter))} peaks, '
          f'knots {getattr(fitter, "fit_multi_peaks_kwargs", {}).get("bg_knots")}')
    return hash_str, fitter


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--folder', required=True, help='folder under tpc_spectrum_fitting')
    parser.add_argument('--start', help='hash of the fit to start from')
    parser.add_argument('--config', help='json file overriding the defaults')
    parser.add_argument('--max-fits', type=int, help='fit budget for this run')
    parser.add_argument('--dry-run', action='store_true', help='report what would be tried, fit nothing')
    parser.add_argument('--resume', action='store_true', help='continue from the last accepted fit')
    args = parser.parse_args()

    ROOT.gROOT.SetBatch(True)
    # The fits being continued were made with no background floor (non-negative spline
    # coefficients do that job) and the exact-Hessian strategy; both are compiled into the
    # model or the minimiser, so a child must use them too.
    fitting_tools.BG_FLOOR_SCALE = 0
    ROOT.Math.MinimizerOptions.SetDefaultStrategy(2)

    config = load_config(args.config)
    if args.max_fits is not None:
        config['max_fits'] = args.max_fits

    decisions = Decisions(args.folder)
    start = args.start or decisions.last_accepted_child()
    if not start:
        parser.error('no --start given and no accepted fit on record to resume from')

    if args.dry_run:
        fitter = fts.load_fit(start, folder_name=args.folder)
        dry_run(fitter, start, decisions, config)
        return

    started = time.time()
    run(args.folder, start, config, resume=args.resume)
    print(f'{(time.time() - started) / 60:.1f} minutes total')
    print(f'decisions: {os.path.join(fit_path, args.folder, DECISIONS_FILE)}')
    print(f'ledger:    python -m e23035_analysis.fit_ledger {args.folder} --tree')


if __name__ == '__main__':
    main()
