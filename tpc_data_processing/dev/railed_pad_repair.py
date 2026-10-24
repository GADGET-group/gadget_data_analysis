import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
from raw_viewer import process_runs 

import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
from scipy.special import erf
import math
import numba
# from raw_viewer import process_runs

def skew_normal(x, amplitude, center, sigma, alpha, baseline):
    """
    Skew normal distribution model for asymmetric pulses.
    center: The location parameter (not exactly the peak if alpha != 0)
    sigma: The scale/width parameter
    alpha: The skewness parameter (alpha > 0 means right-skewed, alpha < 0 means left-skewed)
    """
    # Standardize x
    t = (x - center) / sigma
    
    # Gaussian PDF component (unnormalized, peaks at 1.0)
    pdf = np.exp(-0.5 * t**2)
    
    # Skew CDF component
    cdf = 0.5 * (1.0 + erf(alpha * t / np.sqrt(2)))
    
    # The factor of 2 ensures that when alpha=0, the peak height equals 'amplitude'
    return amplitude * pdf * (2.0 * cdf) + baseline

def repair(railed_trace, repair_start_offset, repair_stop_offset, rail_val=4095):
    '''
    railed_trace: array containing trace
    repair_start_offset: int, number of bins before saturation to use for the fit
    repair_stop_offset: int, number of bins after saturation to use for the fit

    Fit a single skew normal distribution to the bins just before and just after 
    the pad rails. Exclude the railed bins from the fit. Fill the railed region 
    with the fitted skew normal curve.
    '''
    # Find indices where the trace is saturated
    railed_indecies = np.where(railed_trace >= rail_val)[0]
    
    # If the trace isn't actually railed, return it as-is
    if len(railed_indecies) == 0:
        return np.copy(railed_trace)
        
    railed_start = railed_indecies[0]
    railed_end = railed_indecies[-1]
    
    # Define indices for left and right regions (excluding the railed bins)
    left_start = max(railed_start - repair_start_offset, 0)
    left_end = railed_start
    
    right_start = railed_end + 1
    right_end = min(railed_end + 1 + repair_stop_offset, len(railed_trace))

    # Extract data for fitting
    x_left = np.arange(left_start, left_end)
    y_left = railed_trace[left_start:left_end]
    
    x_right = np.arange(right_start, right_end)
    y_right = railed_trace[right_start:right_end]
    
    # Combine left and right sides into a single dataset for the fit
    x_fit = np.concatenate((x_left, x_right))
    y_fit = np.concatenate((y_left, y_right))
    
    # Ensure we have enough data points (need at least 5 for a 5-parameter fit)
    if len(x_fit) < 5:
        print("Warning: Not enough data points to perform the Skew Normal fit.")
        return np.copy(railed_trace)

    # --- Initial Guesses ---
    baseline_guess = np.min(y_fit) 
    center_guess = (railed_start + railed_end) / 2.0 
    amplitude_guess = 5000.0 - baseline_guess 
    sigma_guess = len(railed_indecies) + (repair_start_offset + repair_stop_offset) / 4.0
    alpha_guess = 0.0 # Start by assuming no skew (symmetric pulse)
    
    # p0 matches the argument order of skew_normal
    p0 = [amplitude_guess, center_guess, sigma_guess, alpha_guess, baseline_guess]
    
    # --- Bounds ---
    # Alpha bounded roughly between -10 (highly left-skewed) and 10 (highly right-skewed)
    # to prevent mathematical overflow in the error function.
    lower_bounds = [0, left_start, 0.1, -10.0, -np.inf]
    upper_bounds = [np.inf, right_end, np.inf, 10.0, rail_val]
    
    repaired_trace = railed_trace.astype(float).copy()
    
    try:
        # Fit the single Skew Normal to the combined, non-railed data
        popt, _ = curve_fit(skew_normal, x_fit, y_fit, p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=15000)
        
        # Replace ONLY the railed trace bins with the Skew Normal extrapolation
        repaired_trace[railed_indecies] = skew_normal(railed_indecies, *popt)
        
    except RuntimeError:
        print("Skew Normal fit failed to converge. Returning original trace.")

    return repaired_trace


#repair_lm: the same repair with the skew normal fit done by a numba Levenberg-Marquardt, ~60x faster than curve_fit
#(~100 us instead of ~6.5 ms per railed trace). Checked 2026-10-23 on 7106 railed traces of e23035 and e25058 runs
#against repair: never a higher residual sum; in 7093 well determined fits (no parameter on a bound, >= 10 fit points)
#the repaired bins agree to 2e-5 ADC (median) and 1e-3 ADC (99 %), except 19 fits (0.3 %) where curve_fit stopped at
#the symmetric (alpha = 0) saddle point and repair_lm finds a 10-600x lower residual sum.
LM_MAX_ITERATIONS = 500
LM_INITIAL_DAMPING = 10.0         #in units of the scaled normal matrix (unit diagonal): small first steps
LM_EXTRA_ALPHA_STARTS = (2.0, -2.0) #starting from alpha = 0 alone often ends at the alpha = 0 saddle point
_SQRT2 = np.sqrt(2.0)
_SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)

@numba.njit(cache=True)
def _skew_normal_eval(x, p, f, J, with_jac):
    '''f = skew_normal(x, *p), and the Jacobian in J if with_jac'''
    a, c, s, al, b = p[0], p[1], p[2], p[3], p[4]
    for i in range(x.shape[0]):
        t = (x[i] - c) / s
        g = math.exp(-0.5 * t * t)
        e = 1.0 + math.erf(al * t / _SQRT2)
        f[i] = a * g * e + b
        if with_jac:
            k = _SQRT_2_OVER_PI * math.exp(-0.5 * (al * t) * (al * t))
            dfdt = a * g * (al * k - t * e)
            J[i, 0] = g * e
            J[i, 1] = -dfdt / s
            J[i, 2] = -dfdt * t / s
            J[i, 3] = a * g * t * k
            J[i, 4] = 1.0

@numba.njit(cache=True)
def _sum_sq(f, y):
    c = 0.0
    for i in range(y.shape[0]):
        r = f[i] - y[i]
        c += r * r
    return c

@numba.njit(cache=True)
def _normal_equations(J, f, y, A, g):
    '''A = J^T J, g = J^T (f - y)'''
    n, m = J.shape
    for i in range(m):
        gi = 0.0
        for k in range(n):
            gi += J[k, i] * (f[k] - y[k])
        g[i] = gi
        for j in range(i + 1):
            aij = 0.0
            for k in range(n):
                aij += J[k, i] * J[k, j]
            A[i, j] = aij
            A[j, i] = aij

@numba.njit(cache=True)
def _cholesky_solve(M, b, out):
    '''solve M out = b for a small symmetric positive definite M; False if M is not positive definite'''
    m = M.shape[0]
    L = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1):
            s = M[i, j]
            for k in range(j):
                s -= L[i, k] * L[j, k]
            if i == j:
                if not s > 0.0:
                    return False
                L[i, i] = math.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    z = np.empty(m)
    for i in range(m):
        s = b[i]
        for k in range(i):
            s -= L[i, k] * z[k]
        z[i] = s / L[i, i]
    for i in range(m - 1, -1, -1):
        s = z[i]
        for k in range(i + 1, m):
            s -= L[k, i] * out[k]
        out[i] = s / L[i, i]
    return True

@numba.njit(cache=True)
def _fit_skew_normal(x, y, p0, lo, hi, max_iter, mu0):
    '''
    Least squares fit of skew_normal to (x, y) inside the box [lo, hi]: Levenberg-Marquardt with Marquardt scaling,
    steps clipped onto the bounds, parameters on a bound that the step pushes outward held there.
    Returns params, sum of squared residuals, converged.
    '''
    n = x.shape[0]
    m = 5
    p = np.empty(m)
    for k in range(m):
        p[k] = min(max(p0[k], lo[k]), hi[k])
    f = np.empty(n)
    fn = np.empty(n)
    J = np.empty((n, m))
    A = np.empty((m, m))
    g = np.empty(m)
    M = np.empty((m, m))
    rhs = np.empty(m)
    h = np.empty(m)
    pn = np.empty(m)
    scale = np.empty(m)
    free = np.empty(m, np.bool_)
    _skew_normal_eval(x, p, f, J, True)
    cost = _sum_sq(f, y)
    _normal_equations(J, f, y, A, g)
    mu = mu0
    nu = 2.0
    for it in range(max_iter):
        dmax = 0.0
        for k in range(m):
            dmax = max(dmax, A[k, k])
        for k in range(m):
            free[k] = not ((p[k] <= lo[k] and g[k] > 0.0) or (p[k] >= hi[k] and g[k] < 0.0))
            scale[k] = math.sqrt(max(A[k, k], 1e-12 * dmax + 1e-300))
        for i in range(m):
            for j in range(m):
                M[i, j] = A[i, j] / (scale[i] * scale[j]) if (free[i] and free[j]) else 0.0
            if free[i]:
                M[i, i] += mu
                rhs[i] = -g[i] / scale[i]
            else:
                M[i, i] = 1.0
                rhs[i] = 0.0
        if not _cholesky_solve(M, rhs, h):
            mu *= nu
            nu *= 2.0
            continue
        pred = 0.0
        small = True
        for k in range(m):
            pred += h[k] * (mu * h[k] + rhs[k])  #predicted decrease of the sum of squares
            h[k] /= scale[k]
            if (p[k] <= lo[k] and h[k] < 0.0) or (p[k] >= hi[k] and h[k] > 0.0):
                h[k] = 0.0
            pn[k] = min(max(p[k] + h[k], lo[k]), hi[k])
            if abs(pn[k] - p[k]) > 1e-12 * (abs(p[k]) + 1e-12):
                small = False
        if small:
            return p, cost, True
        _skew_normal_eval(x, pn, fn, J, False)
        cost_n = _sum_sq(fn, y)
        if cost_n < cost:
            dec = cost - cost_n
            rho = dec / pred if pred > 0.0 else 0.0
            for k in range(m):
                p[k] = pn[k]
            _skew_normal_eval(x, p, f, J, True)
            cost = cost_n
            _normal_equations(J, f, y, A, g)
            mu *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
            nu = 2.0
            if dec <= 1e-15 * cost or cost == 0.0:
                return p, cost, True
        else:
            mu *= nu
            nu *= 2.0
            if mu > 1e30:
                return p, cost, True
    return p, cost, False

@numba.njit(cache=True)
def _repair_lm(railed_trace, repair_start_offset, repair_stop_offset, rail_val, max_iter, mu0, alpha_starts):
    '''
    Core of repair_lm (float64 trace). Returns the repaired trace and a status: 1 repaired, 0 not railed,
    -1 fewer than 5 fit points, -2 fit did not converge (the last two return the trace unchanged).
    '''
    N = railed_trace.shape[0]
    out = railed_trace.copy()
    nr = 0
    rs = -1
    re = -1
    for i in range(N):
        if railed_trace[i] >= rail_val:
            nr += 1
            if rs < 0:
                rs = i
            re = i
    if nr == 0:
        return out, 0
    ls = max(rs - repair_start_offset, 0)
    re2 = min(re + 1 + repair_stop_offset, N)
    n = (rs - ls) + (re2 - re - 1)
    if n < 5:
        return out, -1
    x = np.empty(n)
    y = np.empty(n)
    k = 0
    for i in range(ls, rs):
        x[k] = i
        y[k] = railed_trace[i]
        k += 1
    for i in range(re + 1, re2):
        x[k] = i
        y[k] = railed_trace[i]
        k += 1
    #initial guesses and bounds as in repair
    b0 = y.min()
    p0 = np.array([5000.0 - b0, (rs + re) / 2.0, nr + (repair_start_offset + repair_stop_offset) / 4.0, 0.0, b0])
    lo = np.array([0.0, float(ls), 0.1, -10.0, -np.inf])
    hi = np.array([np.inf, float(re2), np.inf, 10.0, float(rail_val)])
    p, cost, converged = _fit_skew_normal(x, y, p0, lo, hi, max_iter, mu0)
    for a0 in alpha_starts:
        p0[3] = a0
        p2, cost2, converged2 = _fit_skew_normal(x, y, p0, lo, hi, max_iter, mu0)
        if cost2 < cost:
            p, cost, converged = p2, cost2, converged2
    if not converged:
        return out, -2
    xr = np.empty(nr)
    k = 0
    for i in range(N):
        if railed_trace[i] >= rail_val:
            xr[k] = i
            k += 1
    fr = np.empty(nr)
    _skew_normal_eval(xr, p, fr, np.empty((nr, 5)), False)
    k = 0
    for i in range(N):
        if railed_trace[i] >= rail_val:
            out[i] = fr[k]
            k += 1
    return out, 1

def repair_lm(railed_trace, repair_start_offset, repair_stop_offset, rail_val=4095):
    '''
    Same as repair (same fit points, model, initial guesses and bounds), with the fit done by _fit_skew_normal from
    alpha = 0 and from LM_EXTRA_ALPHA_STARTS, keeping the lowest residual sum. Returns a float copy of the trace with
    the railed bins replaced, or unchanged if there are fewer than 5 fit points or the fit does not converge.
    '''
    repaired_trace, status = _repair_lm(np.asarray(railed_trace, dtype=np.float64), int(repair_start_offset),
                                        int(repair_stop_offset), float(rail_val), LM_MAX_ITERATIONS,
                                        LM_INITIAL_DAMPING, np.array(LM_EXTRA_ALPHA_STARTS))
    if status == -1:
        print("Warning: Not enough data points to perform the Skew Normal fit.")
    elif status == -2:
        print("Skew Normal fit failed to converge. Returning original trace.")
    return repaired_trace

if False:
    # Test Execution
    f = process_runs.get_h5_file('e23035', 131)
    f.background_subtract_mode = 'none'
    pads, traces = f.get_pad_traces(2745)#(2745)
    traces = np.array(traces)
    railed_trace = traces[np.where(traces==4095)[0][0]]
    #repaired_trace = repair(railed_trace, 20, 3)
    plt.plot(railed_trace)
    plt.plot(repair(railed_trace, 20, 20))
    plt.show()

    #try "reparing" a non railed pads
    def g(i):
        plt.plot(traces[i])
        threshold = (np.max(traces[i]) - np.min(traces[i]))*0.75 + np.min(traces[i])
        plt.plot(repair(traces[i], 8, 8, threshold))
        plt.plot((0, len(traces[i])), (threshold, threshold))
        plt.show()

