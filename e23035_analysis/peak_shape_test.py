"""
Peak-shape test on the clean 59Zn proton peaks (904 and 1063, raw units).

The simultaneous 60Ga/59Zn fits plateau at p ~ 1e-3 with a chi2/ndf excess spread evenly over
the window, which points at the line shape rather than a missing peak. Protons that deposit
incomplete energy in the gas put a low-energy tail on every peak; the full fits use a pure
Gaussian. This script asks whether a tailed shape is required, on the two peaks that do not
overlap other zinc lines.

Each window is fit as a multiplet (the target plus every neighbour whose flank reaches into
the window: [820, 904, 940] and [904, 940, 1063, 1110]) on the 59Zn spectrum stored in the
existing fit files. One width parameter per window, scaled by (mu_i/mu_target)^0.8 to follow
the energy dependence seen in the full fits. Four
nested models come from one compiled function (below) by fixing parameters:

    gaus       tau fixed at 0 (exact Gaussian branch), bg_shift fixed at 0
    gaus+step  tau fixed at 0, bg_shift free (low-side step under each peak)
    emg        tau free (exponential low-side tail), bg_shift fixed at 0
    emg+step   both free

The peak shapes follow fitting_tools.fit_emg_w_bg_shift (amp = counts, tail on the low-energy
side, same step) but the background is a non-negative linear Bernstein instead of a Chebyshev
under a sqrt smoothing: with the smoothing, the 'step' fits found a spurious minimum where the
background went negative and reflected back up. The function is C++ because ROOT 6.36 no longer
takes a Python callable in the TF1 constructor. Fits run through fitting_tools.fit_hist with the
'L' (Poisson likelihood) and 'I' (bin integral) options.

Goodness of fit is 2*MinFcnValue(), which equals the Baker-Cousins likelihood-ratio chi2
recomputed by hand from the fitted bin contents. fit_res.Chi2() is NOT used: on these fits it
came back stale (equal to the Gaussian's value after the EMG had clearly moved). Every model is
fit from several starting points and the best minimum is kept, because single starts were seen
to land in local minima (a step-free fit missing the step solution its neighbour found).

Decision rule
-------------
A tail model is adopted only if, in BOTH windows and at BOTH binnings:
  * delta-chi2 versus gaus >= 9 per extra free parameter (1 for gaus+step or emg, 2 for emg+step);
  * the tail parameter (tau, bg_shift) is not sitting at a bound;
  * the tail is physically ordered: its value at 1063 >= its value at 904, within errors.
delta-mu (tail model minus gaus) is reported for both peaks regardless: a Gaussian fitted to a
tailed peak pulls mu low, so it is a calibration systematic whatever the decision.

Outputs go to tpc_spectrum_fitting/peak_shape_test/: peak_shape_test.csv (one row per binning,
window, model) and one PNG per window and binning with the four models overlaid on the data and
a pull panel per model.
"""
import os
import csv
import ctypes
import itertools

import numpy as np
from scipy.special import erfc
import ROOT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from e23035_analysis import fitting_tools

HERE = os.path.dirname(os.path.abspath(__file__))
FIT_DIR = os.path.join(HERE, 'tpc_spectrum_fitting')
OUT_DIR = os.path.join(FIT_DIR, 'peak_shape_test')

# The 59Zn spectrum is saved as spectrum_1 in every fit file; these two carry the same data at
# the two bin widths the full fits were run with.
SPECTRUM_FILES = {
    5: os.path.join(FIT_DIR, 'protons_le_5keV_bins', 'fit_d8f7935d.root'),
    10: os.path.join(FIT_DIR, 'protons_le_10keV_bins', 'fit_d8f7935d.root'),
}

# Each window holds the target and every neighbour whose flank reaches into it, on both sides:
# the low side of the target is where a tail would show, so nothing unmodelled may sit there.
WINDOWS = {
    '904': {'window': (760, 1000), 'guesses': [820, 904, 940], 'target': 1},
    '1063': {'window': (890, 1150), 'guesses': [904, 940, 1063, 1110], 'target': 2},
}
# Width grows with energy in the full fits (17.7 at 820 to 22.6 at 1110, about E^0.8); a single
# shared width across a window would otherwise leave residuals that look like tails.
SIGMA_EXPONENT = 0.8

MODELS = {'gaus': (False, False), 'gaus+step': (False, True), 'emg': (True, False), 'emg+step': (True, True)}
N_EXTRA = {'gaus': 0, 'gaus+step': 1, 'emg': 1, 'emg+step': 2}
TAIL_PARAM = {'gaus+step': 'bg_shift', 'emg': 'tau', 'emg+step': 'tau'}
MU_WIGGLE = 15
SIGMA_START, SIGMA_BOUNDS = 18.0, (10.0, 60.0)
TAU_STARTS, TAU_BOUNDS = (2.0, 8.0, 20.0, 50.0), (0.5, 300.0)
STEP_STARTS, STEP_BOUNDS = (0.01, 0.05, 0.15), (0.0, 1.0)
FIT_OPTIONS = 'LS0QI'
DCHI2_PER_PARAM = 9.0
PIN_TOL = 1e-3

# Categorical slots 1-4 of the validated reference palette, assigned in fixed model order.
COLORS = {'gaus': '#2a78d6', 'gaus+step': '#eb6834', 'emg': '#1baf7a', 'emg+step': '#eda100'}

# N peaks, each a Gaussian (tau <= 0) or a low-side EMG, plus a low-side step per peak and a
# non-negative linear background. exp(-g^2/2 + z^2) erfc(z) is exp(-g^2/2) erfcx(z); the
# asymptotic series stands in for erfcx where exp(z^2) would overflow. Parameter layout:
#   0 bg_lo  1 bg_hi  2 bg_shift  3 sigma (of the target peak)  4 tau
#   5 e_low  6 e_high  7 bin_width  8 n_peaks  9 target index  10 sigma exponent   (5-10 fixed)
#   11+2i amplitude_i  12+2i mu_i
EMG_CPP = r"""
#include <cmath>
double pst_peak_model(double *x, double *p) {
    const double xv = x[0];
    const double e_low = p[5], e_high = p[6], bw = p[7];
    const int n_peaks = (int) std::lround(p[8]);
    const int target = (int) std::lround(p[9]);
    const double sigma_t = p[3], tau = p[4], sig_exp = p[10];
    if (sigma_t <= 0.0) return 1e10;
    const double mu_t = p[12 + 2 * target];
    const double u = (xv - e_low) / (e_high - e_low);
    double total = p[0] * (1.0 - u) + p[1] * u;
    for (int i = 0; i < n_peaks; ++i) {
        const double amp = p[11 + 2 * i], mu = p[12 + 2 * i];
        const double sigma = sigma_t * std::pow(mu / mu_t, sig_exp);
        const double g = (xv - mu) / sigma;
        total += 0.5 * amp * p[2] * std::erfc(g / 1.41421356);
        if (tau <= 0.0) {
            total += amp * bw / (sigma * 2.50662827) * std::exp(-0.5 * g * g);
            continue;
        }
        const double z = (g + sigma / tau) / 1.41421356;
        double val;
        if (z < 5.0) {
            val = std::exp(-0.5 * g * g + z * z) * std::erfc(z);
        } else {
            const double z2 = z * z;
            const double s = 1.0 - 1.0 / (2.0 * z2) + 3.0 / (4.0 * z2 * z2)
                             - 15.0 / (8.0 * z2 * z2 * z2) + 105.0 / (16.0 * z2 * z2 * z2 * z2);
            val = std::exp(-0.5 * g * g) * s / (z * 1.7724538509);
        }
        total += amp * bw / (2.0 * tau) * val;
    }
    return total;
}
"""


def load_spectrum(path):
    f = ROOT.TFile.Open(path)
    h = f.Get('spectrum_1')
    h.SetDirectory(0)
    f.Close()
    return h


def param_table(f_to_fit):
    '''name -> (value, error, pinned). Fixed parameters and infinite sides are never pinned.'''
    out = {}
    lo, hi = ctypes.c_double(0), ctypes.c_double(0)
    for j in range(f_to_fit.GetNpar()):
        f_to_fit.GetParLimits(j, lo, hi)
        val = f_to_fit.GetParameter(j)
        pinned = False
        if lo.value < hi.value and np.isfinite(hi.value - lo.value):
            width = hi.value - lo.value
            pinned = min(val - lo.value, hi.value - val) < PIN_TOL * width
        out[f_to_fit.GetParName(j)] = (val, f_to_fit.GetParError(j), pinned)
    return out


def data_seeds(spectrum, guesses, window):
    '''Background from the window edges, amplitudes from the local maxima, as fitting_tools does.'''
    ax = spectrum.GetXaxis()
    bw = spectrum.GetBinWidth(1)
    bg_lo = spectrum.GetBinContent(ax.FindBin(window[0]))
    bg_hi = spectrum.GetBinContent(ax.FindBin(window[1]))
    amps = []
    for g in guesses:
        b = ax.FindBin(g)
        local_max = max(spectrum.GetBinContent(b + k) for k in range(-3, 4))
        amps.append(max((local_max - 0.5 * (bg_lo + bg_hi)) * SIGMA_START * 2.50662827 / bw, 1.0))
    return bg_lo, bg_hi, amps


def fit_model(spectrum, model, guesses, window, target):
    '''Fit one model from every start in the tau x step grid; return the best valid minimum.'''
    tau_free, step_free = MODELS[model]
    e_low, e_high = window
    bw = spectrum.GetBinWidth(1)
    bg_lo, bg_hi, amps = data_seeds(spectrum, guesses, window)
    best = None
    starts = itertools.product(TAU_STARTS if tau_free else (0.0,), STEP_STARTS if step_free else (0.0,))
    for tau0, step0 in starts:
        pm = fitting_tools.ParamManager()
        pm.add('bg_lo', bg_lo, (0.0, np.inf))
        pm.add('bg_hi', bg_hi, (0.0, np.inf))
        pm.add('bg_shift', step0, STEP_BOUNDS if step_free else (0.0, 0.0))
        pm.add('sigma', SIGMA_START, SIGMA_BOUNDS)
        pm.add('tau', tau0, TAU_BOUNDS if tau_free else (0.0, 0.0))
        for name, v in (('e_low', e_low), ('e_high', e_high), ('bin_width', bw),
                        ('n_peaks', len(guesses)), ('target', target), ('sigma_exp', SIGMA_EXPONENT)):
            pm.add(name, float(v), (v, v))
        for i, g in enumerate(guesses):
            pm.add(f'amplitude_{i}', amps[i], (1.0, np.inf))
            pm.add(f'mu_{i}', float(g), (g - MU_WIGGLE, g + MU_WIGGLE))
        fit_res, _rp, canvas, sub_hist, f_to_fit, h_fit = fitting_tools.fit_hist(
            spectrum, ROOT.pst_peak_model, pm.initial_values, pm.bounds, window,
            names=pm.names, fit_options=FIT_OPTIONS)
        chi2 = 2.0 * fit_res.MinFcnValue()
        if fit_res.IsValid() and (best is None or chi2 < best['chi2']):
            if best is not None:
                best['canvas'].Close()
            best = dict(fit_res=fit_res, sub_hist=sub_hist, f_to_fit=f_to_fit, h_fit=h_fit,
                        canvas=canvas, chi2=chi2, start=(tau0, step0))
        else:
            canvas.Close()
    if best is None:
        raise RuntimeError(f'no valid fit for {model} in window {window}')
    return best


def background_curve(f_to_fit, xs):
    p = [f_to_fit.GetParameter(j) for j in range(f_to_fit.GetNpar())]
    xs = np.asarray(xs, dtype=float)
    u = (xs - p[5]) / (p[6] - p[5])
    t = p[0] * (1.0 - u) + p[1] * u
    n, target = int(round(p[8])), int(round(p[9]))
    for i in range(n):
        sigma = p[3] * (p[12 + 2 * i] / p[12 + 2 * target]) ** p[10]
        t = t + 0.5 * p[11 + 2 * i] * p[2] * erfc((xs - p[12 + 2 * i]) / (1.41421356 * sigma))
    return t


def baker_cousins(sub_hist, h_fit):
    _, d = hist_arrays(sub_hist)
    _, f = hist_arrays(h_fit)
    m = f > 0
    return 2.0 * np.sum((f[m] - d[m]) + np.where(d[m] > 0, d[m] * np.log(np.where(d[m] > 0, d[m], 1.0) / f[m]), 0.0))


def hist_arrays(h):
    n = h.GetNbinsX()
    x = np.array([h.GetXaxis().GetBinCenter(i) for i in range(1, n + 1)])
    y = np.array([h.GetBinContent(i) for i in range(1, n + 1)])
    return x, y


def fit_window(spectrum, bw, name, cfg):
    rows, curves = [], {}
    window, guesses, t = cfg['window'], cfg['guesses'], cfg['target']
    for model in MODELS:
        r = fit_model(spectrum, model, guesses, window, t)
        fit_res, f_to_fit = r['fit_res'], r['f_to_fit']
        params = param_table(f_to_fit)
        get = lambda k: params.get(k, (np.nan, np.nan, False))
        x, data = hist_arrays(r['sub_hist'])
        _, fit = hist_arrays(r['h_fit'])
        xf = np.linspace(window[0], window[1], 600)
        curves[model] = dict(x=x, data=data, fit=fit, xf=xf,
                             total=np.array([f_to_fit.Eval(v) for v in xf]),
                             bg=background_curve(f_to_fit, xf))
        rows.append(dict(
            binning_keV=bw, window=name, model=model, valid=int(fit_res.IsValid()),
            status=fit_res.Status(), start_tau=r['start'][0], start_step=r['start'][1],
            chi2=r['chi2'], chi2_by_hand=baker_cousins(r['sub_hist'], r['h_fit']),
            root_chi2=fit_res.Chi2(), ndf=fit_res.Ndf(),
            chi2_ndf=r['chi2'] / fit_res.Ndf() if fit_res.Ndf() else np.nan,
            n_extra=N_EXTRA[model],
            mu=get(f'mu_{t}')[0], mu_err=get(f'mu_{t}')[1],
            sigma=get('sigma')[0], sigma_err=get('sigma')[1],
            tau=get('tau')[0], tau_err=get('tau')[1],
            bg_shift=get('bg_shift')[0], bg_shift_err=get('bg_shift')[1],
            amp=get(f'amplitude_{t}')[0], amp_err=get(f'amplitude_{t}')[1],
            amps=' '.join(f"{g}:{get(f'amplitude_{i}')[0]:.0f}@{get(f'mu_{i}')[0]:.1f}" for i, g in enumerate(guesses)),
            bg_lo=get('bg_lo')[0], bg_hi=get('bg_hi')[0],
            pinned=';'.join(k for k, v in params.items() if v[2]),
        ))
        r['canvas'].Close()
    base = rows[0]
    for r in rows:
        r['dchi2_vs_gaus'] = base['chi2'] - r['chi2']
        r['dchi2_per_extra'] = r['dchi2_vs_gaus'] / r['n_extra'] if r['n_extra'] else np.nan
        r['dmu_vs_gaus'] = r['mu'] - base['mu']
    return rows, curves


def plot_window(bw, name, rows, curves, path):
    target = rows[0]
    fig, axes = plt.subplots(5, 1, figsize=(8, 10), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1, 1, 1, 1], 'hspace': 0.08})
    ax = axes[0]
    ref = curves['gaus']
    ax.errorbar(ref['x'], ref['data'], yerr=np.sqrt(np.maximum(ref['data'], 1)), fmt='o',
                ms=3, color='#0b0b0b', ecolor='#52514e', elinewidth=0.8, label='data')
    for model in MODELS:
        c = curves[model]
        ax.plot(c['xf'], c['total'], color=COLORS[model], lw=1.6, label=model)
        ax.plot(c['xf'], c['bg'], color=COLORS[model], lw=0.9, ls='--')
    mu, sig = target['mu'], target['sigma']
    ax.axvspan(mu - 6 * sig, mu - 1.5 * sig, color='#f0efec', zorder=0)
    ax.axvline(mu, color='#52514e', lw=0.8, ls=':')
    ax.set_ylabel(f'counts / {bw} keV')
    ax.set_yscale('log')
    ax.set_ylim(bottom=max(0.5, 0.3 * ref['data'][ref['data'] > 0].min()))
    ax.legend(frameon=False, fontsize=9, ncol=3)
    ax.set_title(f'59Zn {name} window, {bw} keV bins  (shaded: low side of target peak)', fontsize=10)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for axr, (model, row) in zip(axes[1:], zip(MODELS, rows)):
        c = curves[model]
        ok = c['fit'] > 0
        pull = np.where(ok, (c['data'] - c['fit']) / np.sqrt(np.where(ok, c['fit'], 1)), 0)
        axr.axhline(0, color='#52514e', lw=0.6)
        for lvl in (-2, 2):
            axr.axhline(lvl, color='#c3c2b7', lw=0.6, ls=':')
        axr.axvspan(mu - 6 * sig, mu - 1.5 * sig, color='#f0efec', zorder=0)
        axr.plot(c['x'], pull, color=COLORS[model], lw=1.2, marker='o', ms=2.5)
        axr.set_ylim(-4, 4)
        axr.set_ylabel('pull', fontsize=8)
        axr.text(0.01, 0.82, f"{model}: chi2/ndf {row['chi2']:.1f}/{row['ndf']}", transform=axr.transAxes,
                 fontsize=8, color='#0b0b0b')
        for spine in ('top', 'right'):
            axr.spines[spine].set_visible(False)
    axes[-1].set_xlabel('raw energy')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def decide(rows):
    '''Apply the docstring's rule to every tail model; returns {model: (adopted, reasons)}.'''
    by = {(r['binning_keV'], r['window'], r['model']): r for r in rows}
    verdicts = {}
    for model in list(MODELS)[1:]:
        reasons = []
        tp = TAIL_PARAM[model]
        for bw in SPECTRUM_FILES:
            vals = {}
            for name in WINDOWS:
                r = by[(bw, name, model)]
                if not r['valid']:
                    reasons.append(f'{bw} keV/{name}: fit not valid')
                if r['dchi2_per_extra'] < DCHI2_PER_PARAM:
                    reasons.append(f"{bw} keV/{name}: dchi2/extra {r['dchi2_per_extra']:.1f} < {DCHI2_PER_PARAM:g}")
                if tp in r['pinned'].split(';'):
                    reasons.append(f'{bw} keV/{name}: {tp} at a bound')
                vals[name] = (r[tp], r[tp + '_err'])
            (v9, e9), (v10, e10) = vals['904'], vals['1063']
            if v10 + e10 < v9 - e9:
                reasons.append(f'{bw} keV: {tp} smaller at 1063 ({v10:.3g}) than at 904 ({v9:.3g}) beyond errors')
        verdicts[model] = (not reasons, reasons)
    return verdicts


def main():
    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kWarning
    if not ROOT.gInterpreter.Declare(EMG_CPP):
        raise RuntimeError('could not compile the peak model')
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for bw, path in SPECTRUM_FILES.items():
        spectrum = load_spectrum(path)
        for name, cfg in WINDOWS.items():
            print(f'--- {bw} keV bins, window {name} {cfg["window"]} peaks {cfg["guesses"]}')
            wrows, curves = fit_window(spectrum, bw, name, cfg)
            rows.extend(wrows)
            plot_window(bw, name, wrows, curves, os.path.join(OUT_DIR, f'peak_shape_{name}_{bw}keV.png'))

    fields = list(rows[0].keys())
    with open(os.path.join(OUT_DIR, 'peak_shape_test.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'bins':>4} {'win':>5} {'model':<10} {'st':>3} {'chi2/ndf':>12} {'byhand':>7} {'dchi2':>7} {'/extra':>7} "
          f"{'mu':>8} {'dmu':>6} {'sigma':>6} {'tau':>7} {'bg_shift':>9} {'amp':>7} {'bg_lo':>6} {'bg_hi':>6}  pinned")
    for r in rows:
        print(f"{r['binning_keV']:>4} {r['window']:>5} {r['model']:<10} {r['status']:>3} "
              f"{r['chi2']:>7.1f}/{r['ndf']:<4} {r['chi2_by_hand']:>7.1f} {r['dchi2_vs_gaus']:>7.1f} {r['dchi2_per_extra']:>7.1f} "
              f"{r['mu']:>8.2f} {r['dmu_vs_gaus']:>6.2f} {r['sigma']:>6.2f} {r['tau']:>7.2f} {r['bg_shift']:>9.4f} "
              f"{r['amp']:>7.0f} {r['bg_lo']:>6.1f} {r['bg_hi']:>6.1f}  {r['pinned']}")

    print('\nDecision (rule in module docstring):')
    for model, (adopted, reasons) in decide(rows).items():
        print(f"  {model:<10} {'ADOPT' if adopted else 'reject'}")
        for why in reasons:
            print(f'      - {why}')
    print(f'\nOutputs in {OUT_DIR}')


if __name__ == '__main__':
    main()
