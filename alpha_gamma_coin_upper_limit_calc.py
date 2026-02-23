import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

# -----------------------------
# INPUT DATA
# -----------------------------
# Energy (keV), observed counts, background counts
data = [
    (550, 756.625, 12.0389),
    (728, 3271.46, 923.06),
    (786, 793.342, 302.39),
    (1080, 239.184, 60.1028),
    (1621, 431.24, 172.269),
    (419, 36.3673, 6.02728e-07),
    (359, 67.3981, 7.85854),
    (223, 5.26975e-08, 8.4782),
    (805, 185.648, 203.661),
    (315, 93.836, 20.9285),
    (157, 165.717, 3.52929e-06),
    (58.1, 6.4716e-10, 1.68539e-07),
]

CL = 0.90
DELTA = chi2.ppf(CL, 1)  # likelihood threshold

def upper_limit_on_off(n_on, n_off, tau=1.0):
    """
    Profile likelihood upper limit for signal
    Poisson ON/OFF measurement.
    """

    def neg_logL(params):
        s, b = params
        if s < 0 or b < 0:
            return np.inf
        mu_on = s + b
        mu_off = tau * b
        return mu_on - n_on*np.log(mu_on) + mu_off - n_off*np.log(mu_off)

    # best fit
    res = minimize(neg_logL, [max(0, n_on - n_off/tau), n_off/tau])
    Lmin = res.fun

    # scan signal to find UL
    s_vals = np.linspace(0, max(10, n_on*2), 2000)
    for s in s_vals:
        def nll_b(b):
            if b < 0:
                return np.inf
            mu_on = s + b
            mu_off = tau * b
            return mu_on - n_on*np.log(mu_on) + mu_off - n_off*np.log(mu_off)

        b_hat = minimize(nll_b, [n_off/tau]).x[0]
        if nll_b(b_hat) - Lmin > DELTA/2:
            return s

    return s_vals[-1]


# -----------------------------
# Compute limits
# -----------------------------

for E, n_on, n_off in data:
    ul = upper_limit_on_off(n_on, n_off, tau=1)
    print(f"{E} keV: N_signal < {ul:.2f} (90% CL)")

