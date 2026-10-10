'''
Synthetic monoenergetic tracks with a known smooth field distortion, for testing the field-distortion fit without data.

Physical ("forward") distortion, deposited radius -> observed radius about the beam axis:
    r_obs = r_dep + A(t, w) * r_dep / (1 + (r_dep / R0)^2),   A(t, w) = A0 * exp(-t / tau) * (w / w_ref)^2
i.e. an outward push that peaks at r = R0, decays after the start of the decay window (ion space charge draining)
and grows with the drift distance (w^2 - w0^2 is proportional to the drift length). The map is monotone for A < 1 and
its inverse (what the fit has to find) is not a polynomial. The z coordinate is left alone (z_offset = 0).

generate(lines, n_per_line, ...) returns a dict with
    endpoints_true, endpoints (observed, distorted and smeared), t, w, label (line label per event), masks,
    true_ranges, the distortion parameters, and inverse_map(r_obs, t, w) (numerical inverse, for the map error).
'''
import numpy as np

DEFAULT_DISTORTION = dict(A0=0.25, R0=15.0, tau=0.05, w_ref=3.0)


def forward_map(r_dep, t, w, A0=0.25, R0=15.0, tau=0.05, w_ref=3.0):
    A = A0 * np.exp(-t / tau) * (w / w_ref)**2
    return r_dep + A * r_dep / (1 + (r_dep / R0)**2)


def inverse_map(r_obs, t, w, n_iter=60, **dist):
    '''r_dep such that forward_map(r_dep, t, w) = r_obs, by bisection (the forward map is monotone for A0 < 1).'''
    r_obs = np.asarray(r_obs, dtype=float)
    lo = np.zeros_like(r_obs)
    hi = np.maximum(r_obs, 1.0) * 1.0 + 1e-9
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        f = forward_map(mid, t, w, **dist)
        lo = np.where(f < r_obs, mid, lo)
        hi = np.where(f < r_obs, hi, mid)
    return 0.5 * (lo + hi)


def generate(lines, n_per_line, seed=0, r_source=25.0, r_accept=45.0, z_range=(100., 400.), t_max=0.1,
             w0=1.5, dw2_dz=0.025, w_noise=0.1, endpoint_noise=0.7, distortion=None):
    '''
    lines: {label: true range (mm)}; n_per_line: {label: n} or int. Track start points uniform in a disk of radius
    r_source about the beam axis, isotropic directions, both endpoints inside r_accept; t uniform in [0, t_max];
    w = sqrt(w0^2 + dw2_dz * z) + noise (z of the track centre); observed endpoints = forward map + Gaussian noise.
    '''
    rng = np.random.default_rng(seed)
    dist = dict(DEFAULT_DISTORTION, **(distortion or {}))
    E_true, T, W, labels = [], [], [], []
    for lab, L in lines.items():
        n = n_per_line[lab] if isinstance(n_per_line, dict) else int(n_per_line)
        chunks, got = [], 0
        while got < n:
            m = 2 * (n - got) + 10
            rho = r_source * np.sqrt(rng.uniform(size=m))
            phi = rng.uniform(0, 2 * np.pi, m)
            p0 = np.stack([rho * np.cos(phi), rho * np.sin(phi), rng.uniform(*z_range, m)], axis=1)
            u = rng.normal(size=(m, 3))
            u /= np.linalg.norm(u, axis=1)[:, None]
            p1 = p0 + L * u
            ok = np.linalg.norm(p1[:, :2], axis=1) < r_accept
            keep = np.stack([p0[ok], p1[ok]], axis=1)[:n - got]
            chunks.append(keep)
            got += len(keep)
        E_line = np.concatenate(chunks, axis=0)
        E_true.append(E_line)
        T.append(rng.uniform(0, t_max, n))
        zc = E_line[:, :, 2].mean(axis=1)
        W.append(np.sqrt(w0**2 + dw2_dz * zc) + w_noise * rng.normal(size=n))
        labels += [lab] * n
    E_true = np.concatenate(E_true, axis=0)
    T, W = np.concatenate(T), np.concatenate(W)
    labels = np.array(labels)
    E_obs = np.copy(E_true)
    for k in range(2):
        r = np.linalg.norm(E_true[:, k, :2], axis=1)
        scale = forward_map(r, T, W, **dist) / np.where(r > 0, r, 1)
        E_obs[:, k, :2] = E_true[:, k, :2] * scale[:, None]
    E_obs += endpoint_noise * rng.normal(size=E_obs.shape)
    masks = {lab: labels == lab for lab in lines}
    return dict(endpoints_true=E_true, endpoints=E_obs, t=T, w=W, label=labels, masks=masks,
                true_ranges=dict(lines), distortion=dist,
                inverse_map=lambda r, t, w: inverse_map(r, t, w, **dist),
                forward_map=lambda r, t, w: forward_map(r, t, w, **dist))
