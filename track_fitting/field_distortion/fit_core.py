'''
The objective of the empirical field-distortion fit, written against a DistortionBasis.

Same model and objective as polynomial_field_dist_corrections.py (map_endpoints / map_ranges / to_minimize):
    both endpoints of every track are moved radially about the beam axis, r -> basis.map_r(a, r, t, w), the z of
    each endpoint by basis.z_offset(b, r, t, w); the corrected ranges of the selected lines enter
        f = sum_lines weight * var(range) + sum_spacings weight * (mean(range_1) - mean(range_2) - (L_1 - L_2))^2
    ('absolute' is a pseudo line with mean range 0 and true range 0, so (w, 'absolute', 'p') pins the mean range of p).
Options kept from the script: z_field_dist (b_ijk), allow_beam_off_axis (free beam_xy), offset_endpoints (endpoints
pulled towards each other by offset_multiplier * w before the map). Parameter vector x = [a, b, beam_xy, offset_mult].

New here: the gradient of f with respect to a and b is analytic (d range / d a_m is a product of the design matrix
and the direction cosines), the two or three extra parameters use central differences; `residual_jacobian` gives
the Jacobian of the least-squares residuals, whose condition number is the conditioning diagnostic of the stability
study (the Gauss-Newton Hessian is 2 J^T J).

    fit = FieldDistortionFit(basis, endpoints, t, w, masks, true_ranges, peak_widths_to_minimize,
                             peak_spacings_to_preserve, z_field_dist=..., allow_beam_off_axis=..., offset_endpoints=...)
    res = fit.minimize()                       scipy BFGS from the identity map, analytic gradient
    fit.map_ranges(res.x, mask)                corrected ranges of any event selection
    fit.unpack(res.x) -> a, b, beam_xy, offset_multiplier
'''
import numpy as np
import scipy.optimize as opt


class FieldDistortionFit:
    def __init__(self, basis, endpoints, t, w, masks, true_ranges, peak_widths_to_minimize, peak_spacings_to_preserve,
                 z_field_dist=False, allow_beam_off_axis=False, offset_endpoints=False):
        '''
        endpoints (n, 2, 3) mm, t (n,) s, w (n,) mm; masks {label: bool array (n,)}; true_ranges {label: mm} with
        'absolute': 0; peak_widths_to_minimize [(weight, label)]; peak_spacings_to_preserve [(weight, label1, label2)].
        '''
        self.basis = basis
        self.endpoints = np.asarray(endpoints, dtype=float)
        self.t = np.asarray(t, dtype=float)
        self.w = np.asarray(w, dtype=float)
        self.true_ranges = dict(true_ranges)
        self.true_ranges.setdefault('absolute', 0.0)
        self.peak_widths_to_minimize = list(peak_widths_to_minimize)
        self.peak_spacings_to_preserve = list(peak_spacings_to_preserve)
        self.z_field_dist = bool(z_field_dist)
        self.allow_beam_off_axis = bool(allow_beam_off_axis)
        self.offset_endpoints = bool(offset_endpoints)
        self.labels = []
        for _, lab in self.peak_widths_to_minimize:
            self.labels.append(lab)
        for _, l1, l2 in self.peak_spacings_to_preserve:
            self.labels += [l1, l2]
        self.labels = [lab for lab in dict.fromkeys(self.labels) if lab != 'absolute']
        self.data = {lab: (self.endpoints[masks[lab]], self.t[masks[lab]], self.w[masks[lab]]) for lab in self.labels}
        self.n_events = {lab: len(self.data[lab][0]) for lab in self.labels}
        self.n_a = basis.n_params
        self.n_b = basis.n_z_params if self.z_field_dist else 0
        self.n_extra = (2 if self.allow_beam_off_axis else 0) + (1 if self.offset_endpoints else 0)
        self.n_params = self.n_a + self.n_b + self.n_extra
        # finite-difference steps of the extra parameters: beam_xy (mm), offset_multiplier (dimensionless)
        self.extra_step = ([0.05, 0.05] if self.allow_beam_off_axis else []) + ([1e-3] if self.offset_endpoints else [])
        self.nfev = 0

    # ---- parameters ----------------------------------------------------------------------------------------------
    def unpack(self, x):
        x = np.asarray(x, dtype=float)
        a = x[:self.n_a]
        b = x[self.n_a:self.n_a + self.n_b] if self.z_field_dist else np.zeros(0)
        rest = x[self.n_a + self.n_b:]
        beam_xy = np.zeros(2)
        offset_multiplier = 0.0
        if self.allow_beam_off_axis:
            beam_xy = rest[:2]
            rest = rest[2:]
        if self.offset_endpoints:
            offset_multiplier = rest[0]
        return a, b, beam_xy, offset_multiplier

    def initial_guess(self):
        x = [self.basis.identity_params()]
        if self.z_field_dist:
            x.append(self.basis.identity_z_params())
        if self.allow_beam_off_axis:
            x.append(np.zeros(2))
        if self.offset_endpoints:
            x.append(np.zeros(1))
        return np.concatenate(x)

    # ---- the map -------------------------------------------------------------------------------------------------
    def _prepare(self, endpoints, w, beam_xy, offset_multiplier):
        '''Endpoint offset of the script, then the two endpoints relative to the beam axis.'''
        P = np.array(endpoints, dtype=float)
        if self.offset_endpoints:
            d = P[:, 0, :] - P[:, 1, :]
            d /= np.linalg.norm(d, axis=1)[:, None]
            shift = (w * offset_multiplier)[:, None] * d
            P[:, 0, :] -= shift
            P[:, 1, :] += shift
        p1 = P[:, 0, :2] - beam_xy
        p2 = P[:, 1, :2] - beam_xy
        r1 = np.linalg.norm(p1, axis=1)
        r2 = np.linalg.norm(p2, axis=1)
        return P, p1, p2, r1, r2

    def _ranges(self, a, b, beam_xy, offset_multiplier, endpoints, t, w, with_jac=False):
        P, p1, p2, r1, r2 = self._prepare(endpoints, w, beam_xy, offset_multiplier)
        D1 = self.basis.design(r1, t, w)
        D2 = self.basis.design(r2, t, w)
        coef = self.basis.delta_coefficients(a)
        r1n = r1 + D1 @ coef
        r2n = r2 + D2 @ coef
        s1 = np.where(r1 > 0, r1n / np.where(r1 > 0, r1, 1), 0.0)
        s2 = np.where(r2 > 0, r2n / np.where(r2 > 0, r2, 1), 0.0)
        new = np.copy(P)
        new[:, 0, :2] = p1 * s1[:, None] + beam_xy
        new[:, 1, :2] = p2 * s2[:, None] + beam_xy
        if self.z_field_dist:
            new[:, 0, 2] += D1 @ b
            new[:, 1, 2] += D2 @ b
        delta = new[:, 0, :] - new[:, 1, :]
        R = np.linalg.norm(delta, axis=1)
        if not with_jac:
            return R, new
        Rsafe = np.where(R > 0, R, 1.0)
        u1 = p1 / np.where(r1 > 0, r1, 1)[:, None]  # radial unit vectors
        u2 = p2 / np.where(r2 > 0, r2, 1)[:, None]
        c1 = np.einsum('ij,ij->i', delta[:, :2], u1) / Rsafe
        c2 = np.einsum('ij,ij->i', delta[:, :2], u2) / Rsafe
        J_coef = c1[:, None] * D1 - c2[:, None] * D2            # d R / d coef
        J_a = J_coef @ self.basis.delta_coefficients_jac(a)      # d R / d a (chain rule for the monotone map)
        parts = [J_a]
        if self.z_field_dist:
            parts.append((delta[:, 2] / Rsafe)[:, None] * (D1 - D2))
        return R, new, np.concatenate(parts, axis=1)

    def map_endpoints(self, x, mask=None):
        a, b, beam_xy, off = self.unpack(x)
        sel = slice(None) if mask is None else mask
        return self._ranges(a, b, beam_xy, off, self.endpoints[sel], self.t[sel], self.w[sel])[1]

    def map_ranges(self, x, mask=None):
        a, b, beam_xy, off = self.unpack(x)
        sel = slice(None) if mask is None else mask
        return self._ranges(a, b, beam_xy, off, self.endpoints[sel], self.t[sel], self.w[sel])[0]

    def _line_ranges(self, x, with_jac=False):
        '''{label: ranges} (and {label: Jacobian (n_events, n_params)}): the extra parameters by central differences.'''
        a, b, beam_xy, off = self.unpack(x)
        ranges, jacs = {}, {}
        for lab in self.labels:
            E, t, w = self.data[lab]
            if with_jac:
                ranges[lab], _, J = self._ranges(a, b, beam_xy, off, E, t, w, with_jac=True)
                jacs[lab] = J
            else:
                ranges[lab] = self._ranges(a, b, beam_xy, off, E, t, w)[0]
        if with_jac and self.n_extra:
            x = np.asarray(x, dtype=float)
            extra = {lab: np.zeros((self.n_events[lab], self.n_extra)) for lab in self.labels}
            for m in range(self.n_extra):
                h = self.extra_step[m]
                e = np.zeros_like(x)
                e[self.n_a + self.n_b + m] = h
                rp = self._line_ranges(x + e)
                rm = self._line_ranges(x - e)
                for lab in self.labels:
                    extra[lab][:, m] = (rp[lab] - rm[lab]) / (2 * h)
            for lab in self.labels:
                jacs[lab] = np.concatenate([jacs[lab], extra[lab]], axis=1)
        return (ranges, jacs) if with_jac else ranges

    # ---- objective -----------------------------------------------------------------------------------------------
    def objective(self, x):
        self.nfev += 1
        R = self._line_ranges(x)
        means = {lab: R[lab].mean() for lab in self.labels}
        means['absolute'] = 0.0
        f = 0.0
        for wt, lab in self.peak_widths_to_minimize:
            f += wt * np.var(R[lab])
        for wt, l1, l2 in self.peak_spacings_to_preserve:
            f += wt * (means[l1] - means[l2] - (self.true_ranges[l1] - self.true_ranges[l2]))**2
        return f

    def objective_and_gradient(self, x):
        self.nfev += 1
        R, J = self._line_ranges(x, with_jac=True)
        means = {lab: R[lab].mean() for lab in self.labels}
        means['absolute'] = 0.0
        g = {lab: np.zeros(self.n_events[lab]) for lab in self.labels}  # d f / d range_i
        f = 0.0
        for wt, lab in self.peak_widths_to_minimize:
            f += wt * np.var(R[lab])
            g[lab] += 2 * wt * (R[lab] - means[lab]) / self.n_events[lab]
        for wt, l1, l2 in self.peak_spacings_to_preserve:
            miss = means[l1] - means[l2] - (self.true_ranges[l1] - self.true_ranges[l2])
            f += wt * miss**2
            if l1 != 'absolute':
                g[l1] += 2 * wt * miss / self.n_events[l1]
            if l2 != 'absolute':
                g[l2] -= 2 * wt * miss / self.n_events[l2]
        grad = np.zeros(self.n_params)
        for lab in self.labels:
            grad += J[lab].T @ g[lab]
        return f, grad

    def residual_jacobian(self, x):
        '''Jacobian of the residual vector whose sum of squares is the objective, shape (n_residuals, n_params).'''
        R, J = self._line_ranges(x, with_jac=True)
        rows = []
        for wt, lab in self.peak_widths_to_minimize:
            rows.append(np.sqrt(wt / self.n_events[lab]) * (J[lab] - J[lab].mean(axis=0)))
        for wt, l1, l2 in self.peak_spacings_to_preserve:
            row = np.zeros(self.n_params)
            if l1 != 'absolute':
                row += J[l1].mean(axis=0)
            if l2 != 'absolute':
                row -= J[l2].mean(axis=0)
            rows.append(np.sqrt(wt) * row[None, :])
        return np.concatenate(rows, axis=0)

    def minimize(self, x0=None, method='BFGS', callback=None, **options):
        '''scipy.optimize.minimize with the analytic gradient; options are passed through (gtol, maxiter, ...).'''
        x0 = self.initial_guess() if x0 is None else np.asarray(x0, dtype=float)
        self.nfev = 0
        res = opt.minimize(self.objective_and_gradient, x0, jac=True, method=method, callback=callback, options=options)
        res.nfev_total = self.nfev
        return res
