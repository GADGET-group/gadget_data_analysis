"""
Read-only diagnostics on a saved simultaneous fit: where it fits badly, what is pinned, and
how two fits compare.

Everything here works from a fitter returned by fit_tpc_spectrum_simultaneous.load_fit(), so it
never touches the raw data and never fits anything. The stepwise driver uses these functions to
choose what to try next; they are also useful on their own for judging a hand-made fit.

Two kinds of candidate come out of the residuals, and they are deliberately kept apart:
  * peak candidates -- a narrow excess, about as wide as the fitted resolution, which a peak
    would explain;
  * broad runs -- a long same-sign stretch of pulls, which is a background-shape problem and
    should be answered with a knot, not a peak.

Goodness of fit is 2*fit_res.MinFcnValue(), the Baker-Cousins likelihood-ratio chi2 for the 'L'
fits used here. fit_res.Chi2() should equal it and does on the 2D fits, but it went stale on the
1D path when Minos failed, so fit_stat() reports both and flags any disagreement rather than
trusting either blindly.
"""
import os
import csv
import ctypes
import uuid

import numpy as np
import ROOT

from e23035_analysis import fit_tpc_spectrum_simultaneous as fts
from e23035_analysis import fitting_tools

# A parameter is "pinned" when the bound sits within this fraction of its own uncertainty:
# the minimiser wanted to go past it and stopped. Measuring against the error rather than
# against the bound range (which the fit's own warning loop does) is what makes this usable
# here, where amplitudes are bounded at [1e-3, 1e6] -- a tenth of a percent of that range is
# 1000 counts, so every weak peak would look pinned.
PIN_ERR_FRACTION = 0.1

# Significance a broad run of same-sign pulls must reach to count as a background problem.
# The mean of n pulls has standard deviation 1/sqrt(n), so the run's z is mean * sqrt(n).
BROAD_Z_MIN = 3.0

# Defaults for the residual scan. All in units of the local fitted sigma, or keV where noted.
EXCESS_HALF_WIDTH_SIGMAS = 1.5
PEAK_SIG_MIN = 3.0
# How close to an existing peak a candidate may sit. This spectrum is dense: in the 43-peak
# hand fit, 23 of the 42 gaps between neighbouring peaks are under 2 sigma and the tightest is
# 0.18 sigma, so a 2-sigma exclusion would block more than half the real peaks from ever being
# proposed (and blankets 70% of the window). A duplicate peak on top of an existing one is
# caught by the accept rules instead -- it gains no chi2 and its amplitude goes to the floor.
MIN_SEP_SIGMAS = 0.5
BROAD_SMOOTH_KEV = 200.0
BROAD_MIN_RUN_KEV = 150.0


def _result(fitter, window_idx=0):
    res = fitter.fit_results[window_idx]
    f_to_fit = res.get('f_to_fit_2d') or res.get('f_to_fit')
    if f_to_fit is None:
        raise ValueError(f'fit result {window_idx} has no fitted function')
    return res, f_to_fit


def isotope_of_spectrum(fitter, spec_idx):
    '''Isotope label a peak in spectrum spec_idx belongs to, from the histogram name.'''
    name = fitter.spectra[spec_idx].GetName()
    for iso in ('59Zn', '60Ga', '61Ge'):
        if iso in name:
            return iso
    return 'unknown'


def fit_stat(fitter, window_idx=0):
    '''
    (chi2, ndf, agrees) for one fit window.

    chi2 is 2*MinFcnValue(), the quantity the minimiser actually converged on. `agrees` is
    whether fit_res.Chi2() matches it to 1e-6 relative; when it does not, Chi2() is stale and
    must not be used for a nested comparison.
    '''
    res, f_to_fit = _result(fitter, window_idx)
    fit_res = res.get('fit_res')
    if fit_res is None or (hasattr(fit_res, 'Get') and not fit_res.Get()):
        # A CMA-ES-only fit has no minimiser result. It minimised the same Baker-Cousins
        # statistic, recomputed here from the saved fit histogram (matches 2*MinFcnValue() on
        # a MINUIT fit to ~1e-6).
        chi2, n_bins = chi2_baker_cousins(fitter, window_idx)
        return chi2, n_bins - n_free_params(f_to_fit), True
    chi2 = 2.0 * fit_res.MinFcnValue()
    root_chi2 = fit_res.Chi2()
    agrees = abs(root_chi2 - chi2) <= 1e-6 * max(abs(chi2), 1.0)
    return chi2, fit_res.Ndf(), agrees


def chi2_baker_cousins(fitter, window_idx=0):
    '''(chi2, n_bins): the Poisson likelihood-ratio chi2 of the saved fit against the data.'''
    _, data, fit, _ = residual_arrays(fitter, window_idx)
    d, f = data.ravel(), np.maximum(fit.ravel(), 1e-300)
    with np.errstate(divide='ignore', invalid='ignore'):
        term = np.where(d > 0, 2.0 * (f - d + d * np.log(np.where(d > 0, d, 1.0) / f)), 2.0 * f)
    return float(term.sum()), int(d.size)


def n_free_params(f_to_fit):
    '''Parameters the fit could move. TF1::FixParameter stores a fixed value v as limits (v, v),
    or (1, 1) when v is 0; limits (0, 0) mean unbounded.'''
    low, high = ctypes.c_double(0), ctypes.c_double(0)
    n = 0
    for j in range(f_to_fit.GetNpar()):
        f_to_fit.GetParLimits(j, low, high)
        fixed = low.value == high.value and low.value != 0.0
        n += 0 if fixed else 1
    return n


def pinned_from_function(f_to_fit, err_fraction=PIN_ERR_FRACTION):
    '''
    pinned_params for a TF1 read straight out of a fit file, without building a fitter.
    The ledger uses this; see pinned_params for what counts as pinned.
    '''
    out = []
    low, high = ctypes.c_double(0), ctypes.c_double(0)
    for j in range(f_to_fit.GetNpar()):
        f_to_fit.GetParLimits(j, low, high)
        lo, hi = low.value, high.value
        if not (lo < hi) or not np.isfinite(hi - lo):
            continue
        val = f_to_fit.GetParameter(j)
        err = f_to_fit.GetParError(j)
        tol = max(err_fraction * abs(err), 1e-9 * max(abs(val), 1.0))
        at_low = (val - lo) <= tol
        at_high = (hi - val) <= tol
        if at_low or at_high:
            out.append({'name': f_to_fit.GetParName(j), 'value': val, 'err': err,
                        'low': lo, 'high': hi, 'side': 'low' if at_low else 'high'})
    return out


def pinned_params(fitter, window_idx=0, err_fraction=PIN_ERR_FRACTION):
    '''
    Parameters stopped at a finite bound, as a list of
    {name, value, err, low, high, side}.

    A bound counts as reached when it lies within err_fraction of the parameter's own
    uncertainty: the fit could not move away from it. A parameter whose error is zero (fixed,
    or a failed error estimate) is reported only if it sits on the bound exactly. Fixed
    parameters (low == high) and infinite sides are never reported.
    '''
    _, f_to_fit = _result(fitter, window_idx)
    return pinned_from_function(f_to_fit, err_fraction)


def peak_positions(fitter, window_idx=0):
    '''[(peak_idx, mu, isotope)] for the peaks of one window, in peak-index order.'''
    _, f_to_fit = _result(fitter, window_idx)
    isotopes = (getattr(fitter, 'peak_isotopes', None)
                or getattr(fitter, 'fit_multi_peaks_kwargs', {}).get('peak_isotopes') or [])
    window_isos = isotopes[window_idx] if len(isotopes) > window_idx else []
    out = []
    for j in range(f_to_fit.GetNpar()):
        name = f_to_fit.GetParName(j)
        if not name.startswith('mu'):
            continue
        idx = 0 if name == 'mu' else int(name.split('_')[1])
        iso = window_isos[idx] if len(window_isos) > idx else 'unknown'
        out.append((idx, f_to_fit.GetParameter(j), iso))
    return sorted(out)


def sigma_function(fitter, window_idx=0):
    '''
    sigma(E) as a python callable, from the fit's sigma parameterization.

    The parameterization holds a formula string in terms of {mu} and [param] names (Bernstein,
    Chebyshev or the monotonic form); substituting the fitted values gives a TFormula that
    evaluates it exactly as the fit did, whatever the form. A python-callable parameterization
    is called directly, and a fit with a plain free sigma gives a constant.
    '''
    _, f_to_fit = _result(fitter, window_idx)
    values = {f_to_fit.GetParName(j): f_to_fit.GetParameter(j) for j in range(f_to_fit.GetNpar())}
    spec = getattr(fitter, 'parameterizations', {}).get('sigma')

    if spec is None:
        sigma = values.get('sigma')
        if sigma is None:
            raise ValueError('fit has neither a sigma parameterization nor a sigma parameter')
        return lambda E: sigma

    params = list(spec['params'])
    formula = spec['formula']
    if callable(formula):
        vals = [values[name] for name in params]
        pass_mu = spec.get('pass_mu', False)
        return lambda E: formula(*(([E] if pass_mu else []) + vals))

    expr = formula.replace('{mu}', '(x)')
    for name in params:
        expr = expr.replace(f'[{name}]', f'({values[name]!r})')
    window = fitter.peaks_to_fit[window_idx]
    tf = ROOT.TF1(f'sigma_{uuid.uuid4().hex[:8]}', expr, float(window[1]), float(window[2]))
    if not tf.IsValid():
        raise ValueError(f'could not build sigma formula: {expr[:200]}')

    def sigma_at(E):
        return tf.Eval(E)
    sigma_at._tf = tf  # keep the TF1 alive as long as the callable is
    return sigma_at


def residual_arrays(fitter, window_idx=0):
    '''
    Per-bin data, fit and residual for each spectrum.

    Returns (x, data, fit, resid) where x is (nbins,) and the rest are (nspectra, nbins).
    The residual histogram the fit saved is data - fit, recomputed here from the two so the
    arrays are consistent even for a fit saved before h_resid_2d existed.
    '''
    res, _ = _result(fitter, window_idx)
    h_data, h_fit = res['sub_hist_2d'], res['h_fit_2d']
    nx, ny = h_data.GetNbinsX(), h_data.GetNbinsY()
    x = np.array([h_data.GetXaxis().GetBinCenter(i) for i in range(1, nx + 1)])
    data = np.array([[h_data.GetBinContent(i, j) for i in range(1, nx + 1)] for j in range(1, ny + 1)])
    fit = np.array([[h_fit.GetBinContent(i, j) for i in range(1, nx + 1)] for j in range(1, ny + 1)])
    return x, data, fit, data - fit


def local_excess(fitter, window_idx=0, half_width_sigmas=EXCESS_HALF_WIDTH_SIGMAS):
    '''
    Significance of a local excess at every bin, per spectrum.

    S_j(E) = sum(data - fit) / sqrt(sum(fit)) over |x - E| < half_width_sigmas * sigma(E): the
    counts a peak at E would have to account for, in units of their own uncertainty. Summing
    the fit (not the data) in the denominator keeps the statistic from being inflated by the
    very excess it is measuring.

    Returns (x, S) with S of shape (nspectra, nbins).
    '''
    x, _, fit, resid = residual_arrays(fitter, window_idx)
    sigma_at = sigma_function(fitter, window_idx)
    widths = np.array([half_width_sigmas * sigma_at(E) for E in x])
    S = np.zeros_like(fit)
    for i, (E, w) in enumerate(zip(x, widths)):
        m = np.abs(x - E) < w
        expected = fit[:, m].sum(axis=1)
        S[:, i] = np.where(expected > 0, resid[:, m].sum(axis=1) / np.sqrt(np.maximum(expected, 1e-9)), 0.0)
    return x, S


def peak_candidates(fitter, window_idx=0, sig_min=PEAK_SIG_MIN, min_sep_sigmas=MIN_SEP_SIGMAS,
                    exclude=()):
    '''
    Places a new peak might belong, most significant first.

    A candidate is a local maximum of the excess significance that clears sig_min, sits at
    least min_sep_sigmas * sigma away from every fitted peak, and is outside every (low, high)
    range in `exclude` (the regions a full refit has already rejected). The isotope is the
    spectrum carrying the larger excess, which is what add_peak_to_fit wants.

    Returns [{energy, significance, isotope, spectrum, sigma, per_spectrum}].
    '''
    x, S = local_excess(fitter, window_idx)
    sigma_at = sigma_function(fitter, window_idx)
    best_spec = np.argmax(S, axis=0)
    best = S[best_spec, np.arange(S.shape[1])]
    mus = [mu for _, mu, _ in peak_positions(fitter, window_idx)]

    candidates = []
    for i in range(1, len(x) - 1):
        if best[i] < sig_min or best[i] < best[i - 1] or best[i] < best[i + 1]:
            continue
        E, sig = x[i], sigma_at(x[i])
        if any(abs(E - mu) < min_sep_sigmas * sig for mu in mus):
            continue
        spec = int(best_spec[i])
        iso = isotope_of_spectrum(fitter, spec)
        # An exclude entry is (low, high) or (low, high, isotope); without an isotope it
        # applies to both.
        if any(r[0] <= E <= r[1] and (len(r) < 3 or r[2] in (None, iso)) for r in exclude):
            continue
        candidates.append({'energy': float(E), 'significance': float(best[i]),
                           'isotope': iso, 'spectrum': spec,
                           'sigma': float(sig),
                           'per_spectrum': [float(v) for v in S[:, i]]})

    # Collapse candidates closer together than one sigma: they are one feature, and the fit
    # would only be offered the same peak twice.
    candidates.sort(key=lambda c: -c['significance'])
    kept = []
    for c in candidates:
        if all(abs(c['energy'] - k['energy']) > max(c['sigma'], k['sigma']) for k in kept):
            kept.append(c)
    return kept


def broad_runs(fitter, window_idx=0, smooth_keV=BROAD_SMOOTH_KEV, min_run_keV=BROAD_MIN_RUN_KEV,
               z_min=BROAD_Z_MIN):
    '''
    Long same-sign stretches of pulls: background shape, not peaks.

    Pulls are smoothed over smooth_keV so a peak-width wiggle cannot start a run, then runs of
    one sign longer than min_run_keV are scored by the mean of their *unsmoothed* pulls:
    z = mean * sqrt(n), since the mean of n unit-variance pulls has standard deviation
    1/sqrt(n). A run clearing z_min is somewhere the background cannot follow the data over a
    stretch far wider than a peak, i.e. a place to try a knot.

    Returns [{center, low, high, width, mean_pull, z, spectrum, isotope}], most significant first.
    '''
    x, _, fit, resid = residual_arrays(fitter, window_idx)
    pull = np.where(fit > 0, resid / np.sqrt(np.maximum(fit, 1e-9)), 0.0)
    bin_width = x[1] - x[0] if len(x) > 1 else 1.0
    k = max(1, int(round(smooth_keV / bin_width)))
    kernel = np.ones(k) / k

    runs = []
    for j in range(pull.shape[0]):
        smooth = np.convolve(pull[j], kernel, mode='same')
        sign = np.sign(smooth)
        start = 0
        for i in range(1, len(sign) + 1):
            if i < len(sign) and sign[i] == sign[start]:
                continue
            width = (i - start) * bin_width
            seg = pull[j][start:i]
            z = seg.mean() * np.sqrt(len(seg)) if len(seg) else 0.0
            if sign[start] != 0 and width >= min_run_keV and abs(z) >= z_min:
                runs.append({'center': float(x[start:i].mean()), 'low': float(x[start]),
                             'high': float(x[i - 1]), 'width': float(width),
                             'mean_pull': float(seg.mean()), 'z': float(z), 'spectrum': j,
                             'isotope': isotope_of_spectrum(fitter, j)})
            start = i
    return sorted(runs, key=lambda r: -abs(r['z']))


def sigma_bound_overrides(reference_fitter, window_idx=0):
    '''
    additional_param_bounds entries that hold the resolution curve at a reference fit's values.

    Building a peak list stepwise with a free shared sigma(E) is unstable: every peak still
    missing inflates sigma so the fitted peaks can cover for it, each fitted peak becomes a
    blob wider than the real resolution, and the residual scan -- whose windows and exclusion
    radius are measured in sigma -- goes blind to the next missing peak. Rebuild runs saw
    sigma(2000) reach 53 against 27 in the converged fit.

    The resolution is a property of the detector, not of how complete the peak list is, so
    during a rebuild it is taken from a converged fit and held. Release it (refit without
    these) once the peak list is settled.

    The bounds are 3-tuples (value, value, value), which ParamManager reads as a fixed
    parameter, and _extract_fitter_bounds carries that forward, so every child of a fit made
    this way inherits the fixed curve.
    '''
    _, f_to_fit = _result(reference_fitter, window_idx)
    spec = getattr(reference_fitter, 'parameterizations', {}).get('sigma')
    if spec is None:
        raise ValueError('reference fit has no sigma parameterization to copy')
    overrides = {}
    for name in spec['params']:
        j = f_to_fit.GetParNumber(name)
        if j < 0:
            raise ValueError(f'reference fit has no parameter {name}')
        value = f_to_fit.GetParameter(j)
        overrides[name] = (lambda v: (lambda E: (v, v, v)))(value)
    return overrides


def local_unfreeze(fitter, new_peak_locs, n_sigmas=3.0, window_idx=0):
    '''
    A fix_params predicate that frees only what a new peak can actually interact with.

    Pass the result as add_peak_to_fit(..., fix_params=local_unfreeze(...)). Free are: the new
    peak's own parameters, those of every peak whose fitted mu lies within n_sigmas of it, and
    the background coefficients whose spline basis functions cover that stretch. Everything
    else -- distant peaks, the sigma and fraction curves -- is held at the parent's values, so
    the screening fit has ~10 free parameters instead of ~130.

    This is for ranking candidates cheaply, never for accepting one: freezing can only make a
    fit worse, so a screened dchi2 is a lower bound on the full one. The caller must refit
    properly (fix_params=False) before believing any improvement.

    Peak parameters are indexed by position in the sorted peak list, which adding a peak
    renumbers; the new numbering is reconstructed here the same way add_peak_to_fit builds it.
    '''
    if not isinstance(new_peak_locs, (list, tuple)):
        new_peak_locs = [new_peak_locs]
    new_peak_locs = [float(loc) for loc in new_peak_locs]

    sigma_at = sigma_function(fitter, window_idx)
    reach = max(n_sigmas * sigma_at(loc) for loc in new_peak_locs)

    # The new peak list, sorted by location, as add_peak_to_fit will build it.
    existing = [mu for _, mu, _ in peak_positions(fitter, window_idx)]
    all_locs = sorted(existing + new_peak_locs)
    n_peaks = len(all_locs)

    free = set()
    for idx, loc in enumerate(all_locs):
        if min(abs(loc - target) for target in new_peak_locs) > reach:
            continue
        suffix = '' if n_peaks == 1 else f'_{idx}'
        free.add(f'mu{suffix}')
        free.add(f'total_amp{suffix}')
        for spec_idx in range(len(fitter.spectra)):
            free.add(f'amplitude_{idx}_{spec_idx}')
        if n_peaks == 1:
            free.add('amplitude')

    # Background coefficients covering the region: basis k is non-zero only on knot spans
    # k .. k+degree+1 of the clamped knot vector, so the rest cannot respond to the new peak.
    kwargs = getattr(fitter, 'fit_multi_peaks_kwargs', {})
    if kwargs.get('bg_model') == 'bspline':
        window = fitter.peaks_to_fit[window_idx]
        degree = kwargs.get('bg_order', 3)
        knot_vector, n_basis = fitting_tools.bspline_knot_vector(
            degree, kwargs.get('bg_knots'), float(window[1]), float(window[2]))
        lo, hi = min(new_peak_locs) - reach, max(new_peak_locs) + reach
        for k in range(n_basis):
            if knot_vector[k] < hi and knot_vector[k + degree + 1] > lo:
                for spec_idx in range(len(fitter.spectra)):
                    free.add(f'bg_p{k}_{spec_idx}')

    def should_fix(name, E=None):
        return name not in free

    # try_fit hashes a callable by its __qualname__, so the freed set has to be part of the
    # name: two screens that free different parameters must not collide in the fit cache.
    should_fix.__qualname__ = 'local_unfreeze[' + ','.join(sorted(free)) + ']'
    should_fix.free_params = free
    return should_fix


def compare_fits(parent, child, folder_name, window_idx=0):
    '''
    Nested comparison of two fits, by hash or as already-loaded fitters.

    dchi2 is parent - child, so a better child is positive. dnpar counts the parameters the
    child added (ndf falls as parameters are added, at fixed binning).
    '''
    def load(f):
        return fts.load_fit(f, folder_name=folder_name) if isinstance(f, str) else f

    fp, fc = load(parent), load(child)
    chi2_p, ndf_p, ok_p = fit_stat(fp, window_idx)
    chi2_c, ndf_c, ok_c = fit_stat(fc, window_idx)
    return {'parent': parent if isinstance(parent, str) else '<fitter>',
            'child': child if isinstance(child, str) else '<fitter>',
            'chi2_parent': chi2_p, 'ndf_parent': ndf_p,
            'chi2_child': chi2_c, 'ndf_child': ndf_c,
            'dchi2': chi2_p - chi2_c, 'dnpar': ndf_p - ndf_c,
            'chi2_trustworthy': ok_p and ok_c,
            'n_pinned_child': len(pinned_params(fc, window_idx))}


def report(fitter, window_idx=0, max_candidates=10):
    '''Print the full diagnostic picture of one fit. Returns the pieces as a dict.'''
    chi2, ndf, agrees = fit_stat(fitter, window_idx)
    pinned = pinned_params(fitter, window_idx)
    peaks = peak_positions(fitter, window_idx)
    candidates = peak_candidates(fitter, window_idx)
    runs = broad_runs(fitter, window_idx)
    # Sub-threshold maxima, so a report shows how close the next candidate came rather than
    # just "none" -- the difference between "nothing there" and "nearly something".
    near = [c for c in peak_candidates(fitter, window_idx, sig_min=0.0) if c['significance'] < PEAK_SIG_MIN]

    print(f'chi2 {chi2:.1f} / ndf {ndf} = {chi2 / ndf:.3f}'
          + ('' if agrees else '   [WARNING: fit_res.Chi2() disagrees, it is stale]'))
    print(f'{len(peaks)} peaks, {len(pinned)} parameters at a bound')
    for p in pinned:
        print(f"    {p['name']:<18} {p['value']:>10.4g} +- {p['err']:<9.3g} at {p['side']} bound "
              f"[{p['low']:.4g}, {p['high']:.4g}]")
    print(f'peak candidates (excess >= {PEAK_SIG_MIN} sigma, >= {MIN_SEP_SIGMAS} sigma from any peak):')
    for c in candidates[:max_candidates]:
        per = ' '.join(f'{isotope_of_spectrum(fitter, j)}:{v:+.1f}' for j, v in enumerate(c['per_spectrum']))
        print(f"    {c['energy']:>7.0f}  {c['significance']:>5.1f} sigma  -> {c['isotope']:<6} ({per})")
    if not candidates:
        print('    none; largest sub-threshold excesses:')
        for c in near[:3]:
            print(f"        {c['energy']:>7.0f}  {c['significance']:>5.1f} sigma  ({c['isotope']})")
    print(f'broad residual runs (background shape; knot candidates), |z| >= {BROAD_Z_MIN}:')
    for r in runs:
        print(f"    {r['low']:>7.0f}-{r['high']:<7.0f} centre {r['center']:>7.0f}  "
              f"z {r['z']:+.1f} (mean pull {r['mean_pull']:+.2f})  in {r['isotope']}")
    if not runs:
        print('    none')
    if not candidates and not runs and chi2 / max(ndf, 1) > 1.05:
        print(f'note: chi2/ndf is {chi2 / ndf:.2f} but the misfit is not localized -- no narrow excess and\n'
              f'      no broad run. Adding peaks or knots here is not indicated by the residuals.')
    return {'chi2': chi2, 'ndf': ndf, 'chi2_trustworthy': agrees, 'pinned': pinned,
            'peaks': peaks, 'peak_candidates': candidates, 'near_candidates': near,
            'broad_runs': runs}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('folder', help='folder under tpc_spectrum_fitting, e.g. protons_le_10keV_bins')
    parser.add_argument('hash', help='fit hash')
    parser.add_argument('--window', type=int, default=0)
    parser.add_argument('--candidates', type=int, default=10)
    args = parser.parse_args()

    ROOT.gROOT.SetBatch(True)
    fitter = fts.load_fit(args.hash, folder_name=args.folder)
    report(fitter, window_idx=args.window, max_candidates=args.candidates)


if __name__ == '__main__':
    main()
