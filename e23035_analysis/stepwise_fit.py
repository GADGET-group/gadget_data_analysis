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
Rejections are what make the run resumable: a rejected add or removal is not offered again
while the peaks and knots around it are unchanged (it was a test against that local model, and
lapses when the model there changes).

Decisions use delta-chi2 = chi2(parent) - chi2(child) on the Baker-Cousins statistic (2*MinFcnValue()
on a MINUIT fit, recomputed from the saved histograms on a CMA-ES-only one), never fit_res.Chi2(),
and never across bin widths. The gates are set for an inclusive list -- a peak that plausibly
exists goes in and the wider intervals where peaks overlap are the price -- so they sit near the
nominal Wilks values rather than above them; they are config, not constants.

Trial fits use warm, seeded CMA-ES alone (fitting_tools.CMAES_DEFAULTS): ~11 s, reproducible,
within 0.2 chi2 of the MIGRAD minimum. An accepted trial is polished by MIGRAD before the
error-based gates read it and before it becomes the next parent.

Screening freezes everything a new peak cannot interact with (fit_diagnostics.local_unfreeze),
which is ~200x faster than a full refit. Freezing can only make a fit look worse, so a screen
ranks candidates and nothing else: only a full refit accepts, rejects, or rules a region out.

    python -m e23035_analysis.stepwise_fit --folder protons_le_10keV_bins --start d8f7935d --dry-run
    python -m e23035_analysis.stepwise_fit --folder protons_le_10keV_bins --start d8f7935d --max-fits 150
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
    # The peak list is meant to be inclusive. A peak that might be there goes in, and the
    # price is wider mu and amplitude intervals where peaks overlap -- which the MCMC error
    # propagation reports honestly. A peak wrongly left out biases its neighbours, and nothing
    # downstream can see that. So these gates mean "plausibly there", not "required by the
    # data"; the table of claimed lines is a separate, stricter cut on the result.
    # Accept an added peak for this much delta-chi2 (2 free parameters: mu, total_amp).
    'add_peak_dchi2': 9.0,
    # Drop a peak only when removing it costs less than this. Well under the add gate, so a
    # peak near the threshold cannot flip in and out between sweeps.
    'remove_peak_dchi2': 4.0,
    # Adding a knot costs one coefficient per spectrum.
    'add_knot_dchi2': 12.0,
    'remove_knot_dchi2': 9.0,
    # A new peak must also be this significant, and off its amplitude floor, to be kept.
    'min_amp_significance': 2.0,
    # Residual scan: only candidates this significant are offered. A proposal filter, not an
    # acceptance rule: run 4 stopped at 23 peaks with a 2.9 sigma excess still there, and two
    # of its four sub-3-sigma candidates were ENSDF lines.
    'candidate_sig_min': 2.0,
    # When the residual scan finds nothing, ask the likelihood directly: screen a peak at
    # every point on this grid (both isotopes) and rank by the screened delta-chi2. A weak
    # peak between two fitted ones leaves almost no residual -- its counts have already gone
    # into the neighbours and the spline -- so the residual scan goes blind long before the
    # list is complete (run 5 stalled at 27 peaks with the hand fit at 43). None disables it.
    'grid_scan_keV': 25.0,
    # Grid points closer than this to an existing peak of the same isotope are not offered.
    'grid_min_sep_sigmas': 0.5,
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
    # Trial fits (adds, removals, knot changes) use warm, seeded CMA-ES alone: ~11 s against
    # 60+ for MIGRAD on this model, reproducibly within 0.2 chi2 of the MIGRAD minimum, and in
    # the perturbed-start tests it reached every minimum MIGRAD reached. What it lacks is a
    # covariance, so an accepted trial is polished by MIGRAD from its own solution before the
    # error-based gates read it and before it becomes the next parent. 'minuit' restores the
    # old behaviour.
    'trial_optimiser': 'cmaes',
    # popsize is what the optimiser and robustness tests ran with (there it came from the
    # worker count); pinned here so it cannot drift with the machine. A 17-member default
    # population in 96 dimensions missed a dchi2-15 peak its own screen had found.
    'cmaes_opts': {'seed': 1, 'popsize': 200},
    # A rejected add or removal is remembered only while the peaks within this distance and
    # the knots are unchanged: it was a test of one peak against one local model.
    'reject_memory_keV': 100.0,
    # After the search converges, refit once with sigma(E) free as a Bernstein of this order
    # (None: skip). The search holds or constrains the curve so the candidate scan stays
    # sharp; the reported fit must not, or every error is understated. 3 or 4 is enough.
    'final_sigma_order': 3,
    # With a knot-free (Bernstein polynomial) background there is nothing to place, so the
    # knot pass instead tries one degree up (accept for add_knot_dchi2, one coefficient per
    # spectrum) and one degree down (accept when it costs less than remove_knot_dchi2).
    'degree_pass': True,
    'max_fits': 150,
}


def load_config(path=None):
    config = dict(DEFAULT_CONFIG)
    if path:
        with open(path) as fh:
            config.update(json.load(fh))
    return config


def local_context(fitter, energy, radius=100.0):
    '''
    What a decision at this energy depended on: the peaks within `radius` (to the nearest
    5 keV, so ordinary refit jitter does not count as a change) and the knot list. A
    rejection recorded with this context is honoured only while the context still matches.
    '''
    peaks = sorted((int(5 * round(mu / 5.0)), iso)
                   for _, mu, iso in fd.peak_positions(fitter) if abs(mu - energy) <= radius)
    knots = getattr(fitter, 'fit_multi_peaks_kwargs', {}).get('bg_knots') or []
    return {'peaks': [list(p) for p in peaks], 'knots': [float(k) for k in knots]}


class Decisions:
    '''
    The append-only record of every candidate considered, and the memory that makes a run
    resumable: an add or removal a full refit rejected is not offered again while the
    neighbourhood it was judged in is unchanged.
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

    def rejected_regions(self, fitter, radius=100.0, pad_sigmas=1.0, min_pad=15.0, gate=None):
        '''
        (low, high, isotope) around every add a full refit rejected whose neighbourhood is
        unchanged since. A rejection was a test of one peak against one local model -- the
        peaks and knots around it, and through them sigma(E) and the fraction curves -- so it
        is worth remembering only while that model stands; once a neighbour is added or
        removed or a knot moves, the same energy is a new question. Keyed by isotope: a 60Ga
        rejection says nothing about a 59Zn peak at the same energy. Records without a
        context (older logs) are honoured as they were, permanently.
        '''
        out = []
        for r in self.records:
            if r.get('operation') != 'add' or r.get('verdict') != 'reject' or r.get('kind') != 'full':
                continue
            energy = r.get('energy')
            if energy is None:
                continue
            if r.get('context') is not None and r['context'] != local_context(fitter, energy, radius):
                continue
            # A rejection only vetoes while it would still be a rejection: a lower gate on a
            # resumed run must be able to revisit a +7 that was turned away at 9.
            if gate is not None and r.get('dchi2') is not None and r['dchi2'] >= gate:
                continue
            pad = max(pad_sigmas * float(r.get('sigma') or 0.0), min_pad)
            out.append((energy - pad, energy + pad, r.get('isotope')))
        return out

    def removal_vetoed(self, fitter, energy, isotope, radius=100.0, tolerance=5.0, gate=None):
        '''Whether removing this peak was already rejected under the current neighbourhood
        (and would still be rejected at the current gate; dchi2 is minus the removal cost).'''
        context = local_context(fitter, energy, radius)
        return any(r.get('operation') == 'remove' and r.get('verdict') == 'reject'
                   and r.get('kind') == 'full' and r.get('isotope') == isotope
                   and abs((r.get('energy') if r.get('energy') is not None else -1e9) - energy) <= tolerance
                   and r.get('context') == context
                   and (gate is None or r.get('dchi2') is None or -r['dchi2'] >= gate)
                   for r in self.records)

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


def trial_kwargs(config):
    '''Optimiser settings for a fit whose only job is a delta-chi2.'''
    if config.get('trial_optimiser', 'minuit') == 'cmaes':
        return {'use_cmaes': True, 'cmaes_only': True, 'cmaes_opts': dict(config.get('cmaes_opts') or {})}
    return {'use_cmaes': False}


def needs_polish(fitter):
    '''Whether the fit has no minimiser result (a CMA-ES-only trial), so no parameter errors.'''
    fit_res = fitter.fit_results[0].get('fit_res')
    return fit_res is None or (hasattr(fit_res, 'Get') and not fit_res.Get())


def polish(fitter, hash_str, decisions, config, budget, why):
    '''
    MIGRAD from a CMA-ES-only trial's solution. Gives the errors the significance and
    pinned-parameter gates read, and the fit the next step builds on. A fit that already has
    a minimiser result is returned as it is.
    '''
    if not needs_polish(fitter):
        return fitter, hash_str
    chi2_trial, _ = chi2_of(fitter)
    budget.spend()
    new_hash, new_fitter = fts.refit_from_fit(fitter, kwargs_override={'use_cmaes': False},
                                              operation='polish', details={'why': why})
    chi2_polished, _ = chi2_of(new_fitter)
    print(f'  polish {hash_str} -> {new_hash}: chi2 {chi2_trial:.1f} -> {chi2_polished:.1f}')
    decisions.append(operation='polish', kind='full', parent=hash_str, child=new_hash,
                     dchi2=chi2_trial - chi2_polished, verdict='accept', reason=why)
    return new_fitter, new_hash


def pinned_amp_energies(fitter):
    '''Energies of the peaks whose total_amp sits at its floor.'''
    positions = {idx: mu for idx, mu, _ in fd.peak_positions(fitter)}
    out = []
    for p in fd.pinned_params(fitter):
        if p['name'].startswith('total_amp') and p['side'] == 'low':
            idx = 0 if p['name'] == 'total_amp' else int(p['name'].split('_')[2])
            if idx in positions:
                out.append(positions[idx])
    return out


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
        # A linked component's offset at the edge of its box is reported, never re-centred:
        # the box is the literature's uncertainty, and a fit pushing against it is a finding
        # (a hidden neighbour, or a literature energy that is off).
        for p in fd.pinned_params(fitter):
            if p['name'].startswith('dmu'):
                print(f"  note: {p['name']}={p['value']:.1f} sits at its {p['side']} bound "
                      f"[{p['low']:.1f}, {p['high']:.1f}] (the literature spacing box)")
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

    A full refit (a CMA-ES trial or a MIGRAD polish) takes ~11-60 s and a screen ~2 s, so one
    limit for both would let a few sweeps of screening consume a run's whole allowance. Screens
    have their own, much larger cap purely as a runaway guard.
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


def grid_candidates(fitter, config, exclude=()):
    '''
    Every place a peak could go, whether or not the residuals show one: a grid across the
    fit window, both isotopes, skipping points within grid_min_sep_sigmas of an existing peak
    of the same isotope and the vetoed regions. Ordered by local residual excess so the
    likeliest are screened first, but all of them are screened; the screen is the detector.
    Same dict shape as fd.peak_candidates.
    '''
    step = float(config['grid_scan_keV'])
    x, S = fd.local_excess(fitter)
    sigma_at = fd.sigma_function(fitter)
    positions = fd.peak_positions(fitter)
    out = []
    for E in np.arange(x[0] + step, x[-1] - step / 2.0, step):
        sig = sigma_at(E)
        i = int(np.argmin(np.abs(x - E)))
        for spec in range(S.shape[0]):
            iso = fd.isotope_of_spectrum(fitter, spec)
            if any(abs(E - mu) < config['grid_min_sep_sigmas'] * sig
                   for _, mu, p_iso in positions if p_iso == iso):
                continue
            if any(r[0] <= E <= r[1] and (len(r) < 3 or r[2] in (None, iso)) for r in exclude):
                continue
            out.append({'energy': float(E), 'significance': float(S[spec, i]), 'isotope': iso,
                        'spectrum': spec, 'sigma': float(sig),
                        'per_spectrum': [float(v) for v in S[:, i]]})
    out.sort(key=lambda c: -c['significance'])
    return out


def add_peak_pass(fitter, hash_str, folder, decisions, config, budget):
    '''
    One attempt at adding a peak. Returns (fitter, hash, changed).

    Candidates come from the residual scan, are screened (cheap, frozen neighbourhood) to rank
    them, and the best is refit properly. Only that full refit decides.
    '''
    exclude = decisions.rejected_regions(fitter, config['reject_memory_keV'], gate=config['add_peak_dchi2'])
    chi2_parent, _ = chi2_of(fitter)

    def screen(cands, source):
        '''Screen candidates in order; return [(candidate, screened dchi2)] worth a full refit.'''
        queue = []
        if config['screen_disabled']:
            return [(c, None) for c in cands]
        for candidate in cands:
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
                             dchi2=dchi2, verdict=verdict, source=source,
                             reason=f'screened dchi2 {dchi2:.1f} vs queue threshold {threshold:.1f}',
                             details={'free_params': sorted(predicate.free_params)})
            if verdict == 'queue':
                queue.append((candidate, dchi2))
        queue.sort(key=lambda item: -(item[1] or 0))
        return queue

    # The residual scan first: cheap, and right when the excess is visible. When it offers
    # nothing worth a full refit -- nothing at all, or only candidates that screen below the
    # queue bar -- fall back to the grid, which asks the likelihood at every point. Run 6
    # spent three sweeps re-screening one 2-sigma residual candidate at +4.1 while the grid
    # scan never ran, because the fallback only fired on an empty candidate list.
    source = 'residual scan'
    candidates = fd.peak_candidates(fitter, sig_min=config['candidate_sig_min'], exclude=exclude)
    if candidates:
        print(f'  {len(candidates)} candidates ({source}); top: '
              + ', '.join(f"{c['energy']:.0f} ({c['significance']:.1f} sigma, {c['isotope']})"
                          for c in candidates[:config['screen_top_k']]))
        queue = screen(candidates[:config['screen_top_k']], source)
    else:
        print(f'  no peak candidates above {config["candidate_sig_min"]} sigma '
              f'(excluding {len(exclude)} rejected regions)')
        queue = []
    if not queue and config.get('grid_scan_keV') and not budget.exhausted():
        source = 'grid scan'
        candidates = grid_candidates(fitter, config, exclude)
        print(f'  nothing worth a full refit from the residual scan; {len(candidates)} grid candidates'
              + (', top: ' + ', '.join(f"{c['energy']:.0f} ({c['significance']:.1f} sigma, {c['isotope']})"
                                       for c in candidates[:config['screen_top_k']]) if candidates else ''))
        queue = screen(candidates, source)
    if not queue:
        print('  nothing worth a full refit')
        return fitter, hash_str, False

    candidate, screened = queue[0]
    if budget.exhausted():
        return fitter, hash_str, False
    print(f"  full refit adding {candidate['energy']:.0f} ({candidate['isotope']})")
    chi2_parent, _ = chi2_of(fitter)
    context = local_context(fitter, candidate['energy'], config['reject_memory_keV'])
    parent_floor = pinned_amp_energies(fitter)
    budget.spend()
    trial_hash, trial_fitter = fts.add_peak_to_fit(
        fitter, new_peak_loc=candidate['energy'], new_peak_iso=candidate['isotope'], fix_params=False,
        kwargs_override=trial_kwargs(config))
    chi2_trial, _ = chi2_of(trial_fitter)
    dchi2 = chi2_parent - chi2_trial
    if screened is not None and dchi2 < screened - 1.0:
        # The screen is this same fit with most parameters frozen, so it cannot do better than
        # the full fit: a trial below its own screen means the optimiser missed the minimum.
        # Redo it with MIGRAD from the parent, the way the screen was done.
        print(f'  trial dchi2 {dchi2:+.1f} is below its own screen ({screened:+.1f}): '
              f'the optimiser missed the minimum; retrying with MIGRAD')
        budget.spend()
        retry_hash, retry_fitter = fts.add_peak_to_fit(
            fitter, new_peak_loc=candidate['energy'], new_peak_iso=candidate['isotope'], fix_params=False,
            kwargs_override={'use_cmaes': False})
        chi2_retry, _ = chi2_of(retry_fitter)
        decisions.append(operation='add', kind='retry', parent=hash_str, child=retry_hash,
                         energy=candidate['energy'], isotope=candidate['isotope'],
                         dchi2=chi2_parent - chi2_retry, verdict='replace',
                         reason=f'CMA-ES trial {trial_hash} gave {dchi2:.1f}, below its screen {screened:.1f}')
        trial_hash, trial_fitter, chi2_trial = retry_hash, retry_fitter, chi2_retry
        dchi2 = chi2_parent - chi2_trial
        print(f'  MIGRAD trial: dchi2 {dchi2:+.1f}')
    record = dict(operation='add', kind='full', parent=hash_str, trial=trial_hash,
                  energy=candidate['energy'], isotope=candidate['isotope'], sigma=candidate['sigma'],
                  context=context, dchi2=dchi2, screened_dchi2=screened, source=source)

    if dchi2 < config['add_peak_dchi2']:
        reason = f'dchi2 {dchi2:.1f} < {config["add_peak_dchi2"]}'
        print(f'  -> reject: dchi2 {dchi2:+.1f} ({reason})')
        decisions.append(child=trial_hash, verdict='reject', reason=reason, **record)
        return fitter, hash_str, False

    # Worth keeping on chi2 alone. The remaining gates read parameter errors, so polish first.
    new_fitter, new_hash = polish(trial_fitter, trial_hash, decisions, config, budget,
                                  why=f"adding {candidate['energy']:.0f} passed the dchi2 gate")
    chi2_child, _ = chi2_of(new_fitter)

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
    # A neighbour whose amplitude the new peak drove to the floor is the degenerate case an
    # inclusive gate has to guard against: two peaks where there was one. A linked multiplet
    # component is exempt: it sits where the literature puts it, so an amplitude at zero is
    # an upper limit on that line, not a peak that lost its counts to the newcomer.
    new_mu = dict((i, mu) for i, mu, _ in fd.peak_positions(new_fitter)).get(new_idx)
    linked_mu = [mu for i, mu, _ in fd.peak_positions(new_fitter) if i in fd.linked_indices(new_fitter)]
    collapsed = [e for e in pinned_amp_energies(new_fitter)
                 if not any(abs(e - p) <= 5.0 for p in parent_floor)
                 and (new_mu is None or abs(e - new_mu) > 5.0)
                 and not any(abs(e - m) <= 5.0 for m in linked_mu)]

    reasons = []
    if not np.isnan(significance) and significance < config['min_amp_significance']:
        reasons.append(f'amplitude {significance:.1f} sigma < {config["min_amp_significance"]}')
    if amp_pinned:
        reasons.append('amplitude driven to its floor: the peak has no counts')
    if collapsed:
        reasons.append('neighbour amplitude driven to its floor at '
                       + ', '.join(f'{e:.0f}' for e in collapsed))
    verdict = 'accept' if not reasons else 'reject'
    note = ' [at its mu bound; the next recenter pass will move it]' if mu_pinned else ''

    print(f"  -> {verdict}: dchi2 {dchi2:+.1f} (polished {chi2_parent - chi2_child:+.1f}), "
          f"amplitude {significance:.1f} sigma"
          + (f" ({'; '.join(reasons)})" if reasons else '') + note)
    decisions.append(child=new_hash, verdict=verdict, reason='; '.join(reasons) or 'passes every rule',
                     dchi2_polished=chi2_parent - chi2_child, amp_significance=float(significance),
                     mu_at_bound=bool(mu_pinned), **record)
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
    '''Try dropping each non-core peak; keep it dropped when it costs little. Multiplet
    members (linked components and their references) are never offered: the multiplet is one
    object, and a component with no counts is an upper limit, not a peak to remove.'''
    changed = False
    to_try = []
    members = fd.multiplet_indices(fitter)
    for idx, energy, iso in fd.peak_positions(fitter):
        if is_core(energy, iso, config) or idx in members:
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
        if idx is None or idx in fd.multiplet_indices(fitter):
            continue    # it was removed, or moved too far to identify
        if decisions.removal_vetoed(fitter, energy, iso, config['reject_memory_keV'], gate=config['remove_peak_dchi2']):
            print(f'  remove {energy:.0f} ({iso}): already rejected in this neighbourhood; skipping')
            continue
        chi2_parent, _ = chi2_of(fitter)
        context = local_context(fitter, energy, config['reject_memory_keV'])
        budget.spend()
        trial_hash, trial_fitter = fts.remove_peak_from_fit(fitter, (0, idx),
                                                            kwargs_override=trial_kwargs(config))
        chi2_trial, _ = chi2_of(trial_fitter)
        cost = chi2_trial - chi2_parent   # removing can only make chi2 worse or equal
        verdict = 'accept' if cost < config['remove_peak_dchi2'] else 'reject'
        print(f'  remove {energy:.0f} ({iso}, {significance:.1f} sigma): costs {cost:+.1f} -> {verdict}')
        new_hash, new_fitter = trial_hash, trial_fitter
        if verdict == 'accept':
            new_fitter, new_hash = polish(trial_fitter, trial_hash, decisions, config, budget,
                                          why=f'removing {energy:.0f} accepted')
        decisions.append(operation='remove', kind='full', parent=hash_str, child=new_hash,
                         trial=trial_hash, energy=energy, isotope=iso, context=context,
                         amp_significance=float(significance), dchi2=-cost, verdict=verdict,
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
    if not knots and config.get('degree_pass'):
        return degree_pass(fitter, hash_str, folder, decisions, config, budget)
    if not knots:
        print('  no interior knots to remove')   # a knot-free start still gets the add loop below

    for knot in list(knots):
        if budget.exhausted():
            break
        trial = [k for k in knots if k != knot]
        chi2_parent, _ = chi2_of(fitter)
        budget.spend()
        new_hash, new_fitter = fts.change_knots_from_fit(fitter, trial, kwargs_override=trial_kwargs(config))
        chi2_child, _ = chi2_of(new_fitter)
        cost = chi2_child - chi2_parent
        verdict = 'accept' if cost < config['remove_knot_dchi2'] else 'reject'
        print(f'  remove knot {knot:.0f}: costs {cost:+.1f} -> {verdict}')
        if verdict == 'accept':
            new_fitter, new_hash = polish(new_fitter, new_hash, decisions, config, budget,
                                          why=f'removing knot {knot:.0f} accepted')
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
        new_hash, new_fitter = fts.change_knots_from_fit(fitter, trial, kwargs_override=trial_kwargs(config))
        chi2_child, _ = chi2_of(new_fitter)
        dchi2 = chi2_parent - chi2_child
        verdict = 'accept' if dchi2 >= config['add_knot_dchi2'] else 'reject'
        print(f'  add knot {centre:.0f} (run z {run["z"]:+.1f}): dchi2 {dchi2:+.1f} -> {verdict}')
        if verdict == 'accept':
            new_fitter, new_hash = polish(new_fitter, new_hash, decisions, config, budget,
                                          why=f'adding knot {centre:.0f} accepted')
        decisions.append(operation='knots', kind='full', parent=hash_str, child=new_hash,
                         energy=centre, dchi2=dchi2, verdict=verdict,
                         details={'added_knot': centre, 'bg_knots': trial, 'run_z': run['z']},
                         reason=f'dchi2 {dchi2:.1f} vs {config["add_knot_dchi2"]}')
        if verdict == 'accept':
            fitter, hash_str, knots, changed = new_fitter, new_hash, trial, True

    return fitter, hash_str, changed


def degree_pass(fitter, hash_str, folder, decisions, config, budget):
    '''
    The knot pass for a Bernstein-polynomial (knot-free) background: try the degree one
    lower (drop it when that costs less than remove_knot_dchi2) and one higher (take it when
    it is worth add_knot_dchi2). Each step is one coefficient per spectrum, like a knot.
    '''
    changed = False
    degree = int(getattr(fitter, 'fit_multi_peaks_kwargs', {}).get('bg_order') or 3)
    for new_degree, gate, direction in ((degree - 1, config['remove_knot_dchi2'], 'down'),
                                        (degree + 1, config['add_knot_dchi2'], 'up')):
        if new_degree < 1 or budget.exhausted():
            continue
        chi2_parent, _ = chi2_of(fitter)
        budget.spend()
        new_hash, new_fitter = fts.change_knots_from_fit(
            fitter, [], kwargs_override={**trial_kwargs(config), 'bg_order': new_degree})
        chi2_child, _ = chi2_of(new_fitter)
        dchi2 = chi2_parent - chi2_child
        verdict = ('accept' if -dchi2 < gate else 'reject') if direction == 'down' else ('accept' if dchi2 >= gate else 'reject')
        print(f'  background degree {degree} -> {new_degree}: dchi2 {dchi2:+.1f} -> {verdict}')
        if verdict == 'accept':
            new_fitter, new_hash = polish(new_fitter, new_hash, decisions, config, budget,
                                          why=f'background degree {new_degree} accepted')
        decisions.append(operation='degree', kind='full', parent=hash_str, child=new_hash,
                         dchi2=dchi2, verdict=verdict, details={'degree_before': degree, 'degree_after': new_degree},
                         reason=f'degree {direction}: dchi2 {dchi2:.1f} vs {gate}')
        if verdict == 'accept':
            fitter, hash_str, degree, changed = new_fitter, new_hash, new_degree, True
    return fitter, hash_str, changed


def dry_run(fitter, hash_str, decisions, config):
    '''Everything the loop would try next, without fitting anything.'''
    print(f'=== dry run from {hash_str}')
    fd.report(fitter)
    exclude = decisions.rejected_regions(fitter, config['reject_memory_keV'], gate=config['add_peak_dchi2'])
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
    fitter, hash_str = polish(fitter, hash_str, decisions, config, budget,
                              why='the start fit has no minimiser result')
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
    if config.get('final_sigma_order'):
        order = int(config['final_sigma_order'])
        free_hash, free_fitter = fts.free_sigma_refit(fitter, order=order)
        chi2_f, ndf_f = chi2_of(free_fitter)
        sig = fd.sigma_function(free_fitter)
        # Recorded as 'final', not 'accept': a --resume must continue from the search state
        # (sigma constrained), never from the free-sigma polish, whose sigma runs away wherever
        # peaks are still missing.
        decisions.append(operation='free_sigma', kind='full', parent=hash_str, child=free_hash,
                         dchi2=chi2 - chi2_f, verdict='final',
                         reason=f'final polish with sigma(E) free (Bernstein order {order})',
                         details={'sigma_order': order})
        print(f'free-sigma polish {free_hash}: chi2 {chi2_f:.1f} / {ndf_f} = {chi2_f / ndf_f:.3f}; '
              'sigma(E): ' + ', '.join(f'{E}: {sig(E):.1f}' for E in (720, 1000, 1500, 2000, 2500, 2800)))
        return free_hash, free_fitter
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
