'''
Polynomial bases for the empirical field-distortion map of polynomial_field_dist_corrections.py.

The radial map (observed endpoint radius -> deposited radius) is

    r -> r' = r + delta(r, t, w),    delta = sum_{(i,j,k) in S} a_ijk phi_i(r) psi_j(t) chi_k(w)

with r the endpoint radius about the beam axis (mm), t the time since the start of the decay window (s) and w the
track charge width (mm, a proxy for the drift distance). The z offset uses the same basis functions with its own
coefficients b_ijk. The index set S is either the total-degree set {i + j + k <= N} of the original script or the
tensor set {i <= Nr, j <= Nt, k <= Nw}; with force_0_to_0 the i = 0 terms are dropped, so that delta(0, t, w) = 0.

One-dimensional bases (`basis` can be one name for all three variables or a tuple of three):
    'monomial'        (v/scale)^i with the fixed scales of the original script (20 mm, 0.05 s, 3 mm); reproduces the
                      old map_r exactly, kept for comparison.
    'monomial_mapped' monomials of the variable mapped onto its domain: (r/rmax)^i, x_t^j, x_w^k with x in [-1, 1].
                      Separates the effect of the scaling from the effect of the basis.
    'chebyshev'       T_i(x) of the variable mapped onto [-1, 1]; for r with force_0_to_0 the shifted T_i(x) - T_i(-1),
                      which vanishes at r = 0. Same function space as the monomials, condition number grows linearly
                      instead of exponentially with the degree.
    'bernstein'       B_{i,n}(u) = C(n,i) u^i (1-u)^(n-i) of the variable mapped onto [0, 1], n the degree of that
                      variable; tensor index set only (the Bernstein basis of degree n is not nested in degree n+1).
                      B_{i,n}(0) = 0 for i >= 1, so dropping i = 0 gives force_0_to_0.

Monotone parameterisation (`monotone=True`, needs 'bernstein' in all three variables and the tensor set): the map is
written directly as
    r' = sum_{i,j,k} C_ijk B_{i,Nr}(u_r) B_{j,Nt}(u_t) B_{k,Nw}(u_w),   C_ijk = C_0jk + sum_{m=1..i} d_mjk,
    d_mjk = s * softplus(theta_mjk / s) > 0,   s = rmax / Nr,   C_0jk = 0 with force_0_to_0 (else free).
Because the Bernstein functions in t and w are non-negative and sum to one, C_i(t, w) - C_{i-1}(t, w) =
sum_jk d_ijk B_j B_k >= 0 for every (t, w) inside the domain, and a Bernstein polynomial with non-decreasing
coefficients is non-decreasing: r' is monotone in r at fixed (t, w) for any real theta, so an unconstrained optimiser
can be used. The identity map is C_ijk = rmax i/Nr (sum_i (i/n) B_{i,n}(u) = u), i.e. theta = s log(e - 1); the
softplus is linear for d >> s, so near the identity the parameters are ordinary step sizes in mm with a soft floor at 0.
The sufficient condition is not necessary: monotone maps whose Bernstein coefficients are not ordered are excluded,
which for smooth maps costs nothing at moderate degree (Bernstein coefficients of a monotone polynomial become ordered
after degree elevation).

Interface:
    b = DistortionBasis(N, basis, ...)        N an int (total-degree set) or (Nr, Nt, Nw) (tensor set)
    b.n_params, b.n_z_params                  parameter counts of the r map and of the z offset (same basis functions)
    b.identity_params(), b.identity_z_params()
    b.design(r, t, w)                         (n_events, n_basis) matrix of phi_i psi_j chi_k
    b.map_r(params, r, t, w)                  r'
    b.z_offset(bparams, r, t, w)
    b.delta_coefficients(params)              a_ijk of delta (for the monotone map C_ijk minus the identity)
    b.delta_coefficients_jac(params)          d a_ijk / d params, (n_basis, n_params)
    b.min_slope(params, t, w)                 smallest d r'/d r on a grid: monotonicity check for any basis
    b.tag()                                   short string for file names, e.g. 'cheb_N4', 'bern_r4t2w2_monotone'
'''
import numpy as np
from numpy.polynomial import chebyshev as _cheb
from scipy.special import comb

BASES = ('monomial', 'monomial_mapped', 'chebyshev', 'bernstein')
SHORT = {'monomial': 'mono', 'monomial_mapped': 'monomap', 'chebyshev': 'cheb', 'bernstein': 'bern'}


def softplus(x):
    return np.logaddexp(0.0, x)


def softplus_inv(y):
    return np.log(np.expm1(y))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def bernstein_vander(u, n):
    '''B_{i,n}(u), i = 0..n, shape (len(u), n + 1). A polynomial, so defined (but not non-negative) outside [0, 1].'''
    u = np.asarray(u, dtype=float)[:, None]
    i = np.arange(n + 1)
    return comb(n, i) * u**i * (1 - u)**(n - i)


def total_degree_indices(N, i_min):
    '''(i, j, k) with i + j + k <= N and i >= i_min, in the order of the original script.'''
    ijk = []
    for n in range(N + 1):
        for i in range(i_min, n + 1):
            for j in range(n - i + 1):
                ijk.append((i, j, n - i - j))
    return np.array(ijk, dtype=int).reshape(-1, 3)


def tensor_indices(Nr, Nt, Nw, i_min):
    '''(i, j, k) with i_min <= i <= Nr, j <= Nt, k <= Nw; i is the slowest index (cumulative sums along i reshape).'''
    ijk = [(i, j, k) for i in range(i_min, Nr + 1) for j in range(Nt + 1) for k in range(Nw + 1)]
    return np.array(ijk, dtype=int).reshape(-1, 3)


class DistortionBasis:
    def __init__(self, N, basis='chebyshev', index_set=None, force_0_to_0=True, monotone=False,
                 r_domain=(0., 50.), t_domain=(0., 0.1), w_domain=(1., 5.),
                 monomial_scales=(20., 0.05, 3.), clip_tw=True):
        '''
        N: total degree (int, index_set 'total') or degrees (Nr, Nt, Nw) (index_set 'tensor').
        basis: one of BASES, or a tuple of three (for r, t, w).
        r_domain, t_domain, w_domain: (lo, hi) of each variable, used by the mapped bases; r_domain[0] must be 0 when
            force_0_to_0 (the mapped r bases vanish at the lower edge). t and w are clipped to their domains when
            clip_tw (the Bernstein functions are non-negative only inside), r is never clipped.
        monomial_scales: (rscale, tscale, wscale) of the 'monomial' basis, the original script's values.
        '''
        self.basis = (basis,) * 3 if isinstance(basis, str) else tuple(basis)
        if len(self.basis) != 3 or any(b not in BASES for b in self.basis):
            raise ValueError('basis must be one of %s or a tuple of three' % (BASES,))
        if np.isscalar(N):
            self.degrees = (int(N),) * 3
            index_set = 'total' if index_set is None else index_set
        else:
            self.degrees = tuple(int(n) for n in N)
            index_set = 'tensor' if index_set is None else index_set
            if index_set != 'tensor':
                raise ValueError('per-variable degrees need the tensor index set')
        if index_set not in ('total', 'tensor'):
            raise ValueError("index_set must be 'total' or 'tensor'")
        if 'bernstein' in self.basis and index_set != 'tensor':
            raise ValueError('the Bernstein basis needs the tensor index set (pass N as (Nr, Nt, Nw))')
        if monotone and (self.basis != ('bernstein',) * 3 or index_set != 'tensor'):
            raise ValueError('the monotone parameterisation needs the Bernstein basis in r, t and w')
        if force_0_to_0 and self.basis[0] != 'monomial' and r_domain[0] != 0:
            raise ValueError('force_0_to_0 needs r_domain[0] == 0')
        self.N = max(self.degrees)
        self.index_set = index_set
        self.force_0_to_0 = bool(force_0_to_0)
        self.monotone = bool(monotone)
        self.i_min = 1 if force_0_to_0 else 0
        self.domains = (tuple(map(float, r_domain)), tuple(map(float, t_domain)), tuple(map(float, w_domain)))
        self.monomial_scales = tuple(map(float, monomial_scales))
        self.clip_tw = bool(clip_tw)
        if index_set == 'total':
            self.ijk = total_degree_indices(self.N, self.i_min)
        else:
            self.ijk = tensor_indices(*self.degrees, i_min=self.i_min)
        self.n_basis = len(self.ijk)
        if self.n_basis == 0:
            raise ValueError('empty basis')
        # the monotone map has one theta (or C_0jk) per basis function, the linear maps one coefficient
        self.n_params = self.n_basis
        self.n_z_params = self.n_basis
        # Bernstein step scale of the monotone parameterisation
        self._s = (self.domains[0][1] - self.domains[0][0]) / max(self.degrees[0], 1)

    # ---- one-dimensional bases ---------------------------------------------------------------------------------
    def _mapped(self, var, v):
        lo, hi = self.domains[var]
        v = np.asarray(v, dtype=float)
        if self.clip_tw and var > 0:
            v = np.clip(v, lo, hi)
        return v, lo, hi

    def _vander(self, var, v):
        '''1D basis matrix of variable var (0 r, 1 t, 2 w), shape (len(v), degree + 1).'''
        kind, deg = self.basis[var], self.degrees[var]
        v = np.asarray(v, dtype=float)
        if kind == 'monomial':
            x = v / self.monomial_scales[var]
            return x[:, None] ** np.arange(deg + 1)
        v, lo, hi = self._mapped(var, v)
        if kind == 'monomial_mapped':
            x = (v - lo) / (hi - lo) if var == 0 else (2 * v - lo - hi) / (hi - lo)
            return x[:, None] ** np.arange(deg + 1)
        if kind == 'chebyshev':
            x = (2 * v - lo - hi) / (hi - lo)
            V = _cheb.chebvander(x, deg)
            if var == 0 and self.force_0_to_0:
                V = V - (-1.0) ** np.arange(deg + 1)  # T_i(-1) = (-1)^i: every column vanishes at r = 0
            return V
        if kind == 'bernstein':
            return bernstein_vander((v - lo) / (hi - lo), deg)
        raise ValueError(kind)

    def design(self, r, t, w):
        '''(n_events, n_basis) matrix of the basis functions phi_i(r) psi_j(t) chi_k(w); t and w may be scalars.'''
        r = np.atleast_1d(np.asarray(r, dtype=float))
        t = np.broadcast_to(np.asarray(t, dtype=float), r.shape)
        w = np.broadcast_to(np.asarray(w, dtype=float), r.shape)
        Vr, Vt, Vw = self._vander(0, r), self._vander(1, t), self._vander(2, w)
        return Vr[:, self.ijk[:, 0]] * Vt[:, self.ijk[:, 1]] * Vw[:, self.ijk[:, 2]]

    # ---- parameters -> coefficients of delta ------------------------------------------------------------------
    def _block_shape(self):
        Nr, Nt, Nw = self.degrees
        return (Nr + 1 - self.i_min, Nt + 1, Nw + 1)

    def _identity_C(self):
        Nr = self.degrees[0]
        rmax = self.domains[0][1]
        return rmax * np.arange(self.i_min, Nr + 1) / Nr

    def delta_coefficients(self, params):
        '''a_ijk such that r' = r + design(r, t, w) @ a_ijk.'''
        params = np.asarray(params, dtype=float)
        if params.shape != (self.n_params,):
            raise ValueError('expected %d parameters, got shape %s' % (self.n_params, params.shape))
        if not self.monotone:
            return params
        th = params.reshape(self._block_shape())
        s = self._s
        if self.i_min == 1:
            C = np.cumsum(s * softplus(th / s), axis=0)
        else:
            C = np.concatenate([th[:1], th[:1] + np.cumsum(s * softplus(th[1:] / s), axis=0)], axis=0)
        return (C - self._identity_C()[:, None, None]).ravel()

    def delta_coefficients_jac(self, params):
        '''d a_ijk / d params, shape (n_basis, n_params).'''
        if not self.monotone:
            return np.eye(self.n_params)
        th = np.asarray(params, dtype=float).reshape(self._block_shape())
        ni, nj, nk = th.shape
        s = self._s
        J = np.zeros((ni, nj, nk, ni, nj, nk))
        jk = np.arange(nj)[:, None], np.arange(nk)[None, :]
        if self.i_min == 1:
            slope = sigmoid(th / s)  # d d_m / d theta_m
            for i in range(ni):
                for m in range(i + 1):
                    J[i, jk[0], jk[1], m, jk[0], jk[1]] = slope[m]
        else:
            slope = sigmoid(th[1:] / s)
            for i in range(ni):
                J[i, jk[0], jk[1], 0, jk[0], jk[1]] = 1.0
                for m in range(1, i + 1):
                    J[i, jk[0], jk[1], m, jk[0], jk[1]] = slope[m - 1]
        return J.reshape(self.n_basis, self.n_params)

    def identity_params(self):
        if not self.monotone:
            return np.zeros(self.n_params)
        th = np.full(self._block_shape(), self._s * softplus_inv(1.0))
        if self.i_min == 0:
            th[0] = 0.0
        return th.ravel()

    def identity_z_params(self):
        return np.zeros(self.n_z_params)

    # ---- maps --------------------------------------------------------------------------------------------------
    def map_r(self, params, r, t, w):
        r = np.atleast_1d(np.asarray(r, dtype=float))
        return r + self.design(r, t, w) @ self.delta_coefficients(params)

    def z_offset(self, bparams, r, t, w):
        bparams = np.asarray(bparams, dtype=float)
        if bparams.shape != (self.n_z_params,):
            raise ValueError('expected %d z parameters, got shape %s' % (self.n_z_params, bparams.shape))
        return self.design(r, t, w) @ bparams

    def min_slope(self, params, t_values=None, w_values=None, r_grid=None, h=1e-3):
        '''Smallest d r'/d r (central differences) over a grid of (r, t, w); < 0 means a non-monotone map.'''
        lo_r, hi_r = self.domains[0]
        r_grid = np.linspace(lo_r, hi_r, 201) if r_grid is None else np.asarray(r_grid, dtype=float)
        t_values = np.linspace(*self.domains[1], 11) if t_values is None else np.atleast_1d(t_values)
        w_values = np.linspace(*self.domains[2], 11) if w_values is None else np.atleast_1d(w_values)
        R, T, W = np.meshgrid(r_grid, t_values, w_values, indexing='ij')
        R, T, W = R.ravel(), T.ravel(), W.ravel()
        slope = (self.map_r(params, R + h, T, W) - self.map_r(params, R - h, T, W)) / (2 * h)
        return slope.min()

    # ---- bookkeeping -------------------------------------------------------------------------------------------
    def tag(self):
        names = [SHORT[b] for b in self.basis]
        name = names[0] if len(set(names)) == 1 else '-'.join(names)
        if self.index_set == 'total':
            deg = 'N%d' % self.N
        else:
            deg = 'r%dt%dw%d' % self.degrees
        return name + '_' + deg + ('_monotone' if self.monotone else '')

    def __repr__(self):
        return ('DistortionBasis(%s, index_set=%r, force_0_to_0=%r, monotone=%r, n_basis=%d, domains=%r)'
                % (self.tag(), self.index_set, self.force_0_to_0, self.monotone, self.n_basis, self.domains))
