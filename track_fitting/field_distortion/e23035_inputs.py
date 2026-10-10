'''
Inputs of the field-distortion fit for e23035, from the processed files (raw_viewer.process_runs) and the
time-since-beam-off cache (raw_viewer.get_tsbo). Nothing is reprocessed here.

    data = load(get_runs, config_filename='smart1_veto2_rpr_fzp.csv')
        endpoints (n, 2, 3) mm (z = bucket x 1.088), t (n,) s = time since beam off (NaN if no DDAS match),
        w (n,) mm = charge width along the second PCA axis, pca_width (n,) mm, angle (n,) rad from the beam axis,
        energy (n,) MeV (gain-matched integrated charge), ranges (n,) mm, veto / proton / alpha masks (the production
        selections of e23035_analysis.e23035_runs), run (n,) GET run of each event.
    masks = line_masks(data, LINES)   {label: bool mask} of the known lines (energy window inside the species mask,
                                      range within +-range_halfwidth of the SRIM value, finite t)
    true_ranges(get_run)              {label: SRIM range in mm} (track_fitting.srim_interface, P10 at the run's density)

A script that calls load() with num_workers > 1 must guard its top level with `if __name__ == '__main__':`
(process_runs loads with spawned worker processes).
'''
import numpy as np

from raw_viewer import process_runs, get_tsbo
from e23035_analysis import e23035_runs
from track_fitting import srim_interface, build_sim

DEFAULT_CONFIG = 'smart1_veto2_rpr_fzp.csv'
# the standard 60Ga runs (DDAS 241-257): 512 time buckets, 0.5 us gate delay, SCA ~300 keV
STANDARD_60GA_GET_RUNS = list(range(263, 280))

# label: (species, energy window in MeV on the detector's alpha-scale axis, SRIM energy in MeV)
LINES = {
    'p1110': ('proton', (1.06, 1.16), 1.11),   # strong 60Ga proton line
    'p2050': ('proton', (1.98, 2.10), 2.05),   # the 2 MeV 60Ga multiplet (1963-2043 keV lines)
    'a5850': ('alpha', (5.70, 5.98), 5.85),    # 212Bi
    'a6110': ('alpha', (5.98, 6.30), 6.11),    # 220Rn
    'a6540': ('alpha', (6.40, 6.70), 6.54),    # 216Po
    'a8460': ('alpha', (8.20, 8.70), 8.46),    # 212Po
}
range_halfwidth = 20.0  # mm about the SRIM range, to drop outliers (the distortion moves ranges by a few mm)


def true_ranges(get_run, lines=LINES):
    rho = build_sim.get_gas_density('e23035', get_run)
    tables = {'proton': srim_interface.SRIM_Table('track_fitting/stopping_powers/1H_in_P10.txt', rho),
              'alpha': srim_interface.SRIM_Table('track_fitting/stopping_powers/4He_in_P10.txt', rho)}
    out = {lab: float(tables[sp].get_stopping_distance(E)) for lab, (sp, _, E) in lines.items()}
    out['absolute'] = 0.0
    return out


def load(get_runs, config_filename=DEFAULT_CONFIG, num_workers=1):
    get_runs = [int(r) for r in get_runs]
    q = process_runs.get_quantity(['endpoints', 'charge_width', 'variance_along_axes', 'principle_axes'],
                                  'e23035', get_runs, config_filename=config_filename, num_workers=num_workers)
    endpoints, charge_width, variances, axes = q
    t = get_tsbo.get_time_since_beam_off('e23035', get_runs, config_filename, num_workers=max(1, min(num_workers, 8)))
    energy = e23035_runs.get_energy_MeV(get_runs, tpc_ini_filename=config_filename)
    veto = e23035_runs.get_veto_mask(get_runs, endpoints=endpoints, tpc_ini_filename=config_filename)
    ranges = np.linalg.norm(endpoints[:, 0] - endpoints[:, 1], axis=1)
    proton = e23035_runs.get_proton_mask(get_runs, lengths=ranges, energy=energy, veto_mask=veto, tpc_ini_filename=config_filename)
    alpha = e23035_runs.get_alpha_mask(get_runs, lengths=ranges, energy=energy, veto_mask=veto, tpc_ini_filename=config_filename)
    d = axes[:, 0, :]
    angle = np.arctan2(np.sqrt(d[:, 0]**2 + d[:, 1]**2), np.abs(d[:, 2]))
    run = process_runs.get_run_and_event_numbers('e23035', get_runs, config_filename=config_filename)[0].astype(int)
    assert len(run) == len(endpoints)
    return dict(endpoints=endpoints, t=t, w=charge_width, pca_width=np.sqrt(np.maximum(variances[:, 1], 0)),
                angle=angle, energy=energy, ranges=ranges, veto=veto, proton=proton, alpha=alpha, run=run,
                get_runs=get_runs, config_filename=config_filename)


def line_masks(data, lines=LINES, halfwidth=range_halfwidth):
    tr = true_ranges(data['get_runs'][0], lines)
    masks = {}
    for lab, (species, (lo, hi), _) in lines.items():
        masks[lab] = (data[species] & (data['energy'] > lo) & (data['energy'] < hi) & np.isfinite(data['t'])
                      & (np.abs(data['ranges'] - tr[lab]) < halfwidth))
    return masks


def correct_width_for_angle(data, mask, degree=2):
    '''The script's correct_width_for_angle: remove the angle dependence of w fitted on one line (mask).'''
    w, angle = data['w'], data['angle']
    poly = np.polyfit(angle[mask], w[mask], degree)
    return w - np.polyval(poly, angle) + np.mean(w[mask]), poly
