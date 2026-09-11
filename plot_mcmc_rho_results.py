#!/usr/bin/env python3
"""
Plot the 13-parameter proton-alpha MCMC results for:

    e25058
    run 71
    event 4007
    rho_scale = k = 1 fixed

Creates:
    forward_chain.png
    backward_chain.png

    forward_corner_raw.png
    backward_corner_raw.png

    forward_corner_physical.png
    backward_corner_physical.png

It can also open the simulated-versus-real event comparison GUI.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Configuration
# ============================================================

EXPERIMENT = "e25058"
RUN = 71
EVENT = 4007
FIXED_RHO_SCALE = 1.0

BASE_DIR = Path(
    "e25058_mcmc/run71_palpha_mcmc/event4007_rho"
)

PLOT_DIR = BASE_DIR / "plots"

RAW_PARAMETER_LABELS = [
    r"$E$",
    r"$f_{\alpha}$",
    r"$x$",
    r"$y$",
    r"$z$",
    r"$p_x$",
    r"$p_y$",
    r"$p_z$",
    r"$a_x$",
    r"$a_y$",
    r"$a_z$",
    r"$\sigma_{xy}$",
    r"$\sigma_z$",
]

PHYSICAL_PARAMETER_LABELS = [
    r"$E_{\alpha}$",
    r"$E_p$",
    r"$x$",
    r"$y$",
    r"$z$",
    r"$\theta_p$",
    r"$\phi_p$",
    r"$\theta_{\alpha}$",
    r"$\phi_{\alpha}$",
    r"$\sigma_{xy}$",
    r"$\sigma_z$",
]


# ============================================================
# Reading and validating chains
# ============================================================

def load_backend(direction: str) -> emcee.backends.HDFBackend:
    """Open the forward or backward emcee HDF backend."""

    path = BASE_DIR / f"{direction}.h5"

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find MCMC file:\n{path}"
        )

    backend = emcee.backends.HDFBackend(
        str(path),
        read_only=True,
    )

    chain = backend.get_chain()

    if chain.ndim != 3:
        raise ValueError(
            f"{path} has unexpected chain shape {chain.shape}"
        )

    if chain.shape[-1] != 13:
        raise ValueError(
            f"{path} contains {chain.shape[-1]} parameters. "
            "Expected 13 parameters for the fixed-rho run."
        )

    return backend


def print_backend_information(
    direction: str,
    backend: emcee.backends.HDFBackend,
) -> None:
    """Print basic chain information and autocorrelation estimates."""

    chain = backend.get_chain()
    log_prob = backend.get_log_prob()

    print(f"\n{'=' * 60}")
    print(f"{direction.upper()} CHAIN")
    print(f"{'=' * 60}")
    print(f"Iterations:        {backend.iteration}")
    print(f"Chain shape:       {chain.shape}")
    print(f"Log-prob shape:    {log_prob.shape}")
    print(f"Number of walkers: {chain.shape[1]}")
    print(f"Number parameters: {chain.shape[2]}")
    print(f"Fixed rho_scale:   {FIXED_RHO_SCALE}")

    try:
        tau = backend.get_autocorr_time(tol=0)
        print("\nAutocorrelation times:")
        for label, value in zip(RAW_PARAMETER_LABELS, tau):
            print(f"  {label:20s} {value:10.3f}")

        print(f"\nLargest tau: {np.max(tau):.3f}")
        print(
            "Iterations / largest tau: "
            f"{backend.iteration / np.max(tau):.2f}"
        )

    except Exception as error:
        print(
            "\nCould not calculate reliable autocorrelation "
            f"times: {error}"
        )


# ============================================================
# Parameter transformations
# ============================================================

def vector_to_angles(
    vector_x: np.ndarray,
    vector_y: np.ndarray,
    vector_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert Cartesian direction vectors to theta and phi.

    theta: polar angle from +z
    phi: azimuthal angle in the x-y plane
    """

    magnitude = np.sqrt(
        vector_x**2
        + vector_y**2
        + vector_z**2
    )

    if np.any(magnitude <= 0):
        raise ValueError(
            "At least one direction vector has zero magnitude."
        )

    theta = np.arccos(
        np.clip(vector_z / magnitude, -1.0, 1.0)
    )

    phi = np.arctan2(vector_y, vector_x)

    return theta, phi


def transform_to_physical_parameters(
    samples: np.ndarray,
) -> np.ndarray:
    """
    Convert the 13 raw MCMC parameters into more physical parameters.

    Raw:
        E, Ea_frac,
        x, y, z,
        p_x, p_y, p_z,
        a_x, a_y, a_z,
        sigma_xy, sigma_z

    Returned:
        Ea, Ep,
        x, y, z,
        theta_p, phi_p,
        theta_a, phi_a,
        sigma_xy, sigma_z
    """

    if samples.ndim != 2 or samples.shape[1] != 13:
        raise ValueError(
            "Expected flattened samples with shape "
            f"(number_of_samples, 13), got {samples.shape}"
        )

    E = samples[:, 0]
    Ea_frac = samples[:, 1]

    x = samples[:, 2]
    y = samples[:, 3]
    z = samples[:, 4]

    p_x = samples[:, 5]
    p_y = samples[:, 6]
    p_z = samples[:, 7]

    a_x = samples[:, 8]
    a_y = samples[:, 9]
    a_z = samples[:, 10]

    sigma_xy = samples[:, 11]
    sigma_z = samples[:, 12]

    Ea = E * Ea_frac
    Ep = E * (1.0 - Ea_frac)

    theta_p, phi_p = vector_to_angles(
        p_x,
        p_y,
        p_z,
    )

    theta_a, phi_a = vector_to_angles(
        a_x,
        a_y,
        a_z,
    )

    return np.column_stack([
        Ea,
        Ep,
        x,
        y,
        z,
        theta_p,
        phi_p,
        theta_a,
        phi_a,
        sigma_xy,
        sigma_z,
    ])


def limit_corner_samples(
    samples: np.ndarray,
    maximum: int = 100_000,
) -> np.ndarray:
    """Limit the number of points sent to corner for performance."""

    if len(samples) <= maximum:
        return samples

    rng = np.random.default_rng(12345)

    indices = rng.choice(
        len(samples),
        size=maximum,
        replace=False,
    )

    return samples[indices]


# ============================================================
# Plotting
# ============================================================

def plot_chain(
    direction: str,
    backend: emcee.backends.HDFBackend,
) -> None:
    """Plot every walker as a function of MCMC step."""

    chain = backend.get_chain()

    number_of_parameters = chain.shape[2]

    fig, axes = plt.subplots(
        number_of_parameters,
        1,
        figsize=(13, 2.0 * number_of_parameters),
        sharex=True,
    )

    for parameter_index, axis in enumerate(axes):
        axis.plot(
            chain[:, :, parameter_index],
            alpha=0.25,
        )

        median = np.median(
            chain[:, :, parameter_index],
            axis=1,
        )

        lower = np.percentile(
            chain[:, :, parameter_index],
            16,
            axis=1,
        )

        upper = np.percentile(
            chain[:, :, parameter_index],
            84,
            axis=1,
        )

        axis.plot(
            median,
            linewidth=2,
            label="Walker median",
        )

        axis.plot(
            lower,
            linestyle="--",
            linewidth=1,
        )

        axis.plot(
            upper,
            linestyle="--",
            linewidth=1,
        )

        axis.set_ylabel(
            RAW_PARAMETER_LABELS[parameter_index]
        )

        axis.grid(alpha=0.2)

    axes[0].set_title(
        f"{direction.capitalize()} chain — "
        f"run {RUN}, event {EVENT}, k = 1"
    )

    axes[-1].set_xlabel("MCMC step")

    fig.tight_layout()

    output_path = PLOT_DIR / f"{direction}_chain.png"

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


def plot_raw_corner(
    direction: str,
    backend: emcee.backends.HDFBackend,
    discard: int,
    thin: int,
) -> None:
    """Create the corner plot in the original 13 parameters."""

    samples = backend.get_chain(
        discard=discard,
        thin=thin,
        flat=True,
    )

    samples = limit_corner_samples(samples)

    figure = corner.corner(
        samples,
        labels=RAW_PARAMETER_LABELS,
        quantiles=[0.16, 0.50, 0.84],
        show_titles=True,
        title_fmt=".4f",
        title_kwargs={"fontsize": 8},
        label_kwargs={"fontsize": 10},
        plot_datapoints=False,
        fill_contours=True,
    )

    figure.suptitle(
        f"{direction.capitalize()} raw parameters\n"
        f"run {RUN}, event {EVENT}, k = 1",
        fontsize=16,
    )

    output_path = (
        PLOT_DIR / f"{direction}_corner_raw.png"
    )

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved: {output_path}")


def plot_physical_corner(
    direction: str,
    backend: emcee.backends.HDFBackend,
    discard: int,
    thin: int,
) -> None:
    """Create a corner plot using energies and direction angles."""

    raw_samples = backend.get_chain(
        discard=discard,
        thin=thin,
        flat=True,
    )

    physical_samples = transform_to_physical_parameters(
        raw_samples
    )

    physical_samples = limit_corner_samples(
        physical_samples
    )

    figure = corner.corner(
        physical_samples,
        labels=PHYSICAL_PARAMETER_LABELS,
        quantiles=[0.16, 0.50, 0.84],
        show_titles=True,
        title_fmt=".4f",
        title_kwargs={"fontsize": 8},
        label_kwargs={"fontsize": 10},
        plot_datapoints=False,
        fill_contours=True,
    )

    figure.suptitle(
        f"{direction.capitalize()} physical parameters\n"
        f"run {RUN}, event {EVENT}, k = 1",
        fontsize=16,
    )

    output_path = (
        PLOT_DIR / f"{direction}_corner_physical.png"
    )

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved: {output_path}")


def print_energy_summary(
    direction: str,
    backend: emcee.backends.HDFBackend,
    discard: int,
    thin: int,
) -> None:
    """Print posterior percentiles for alpha and proton energy."""

    samples = backend.get_chain(
        discard=discard,
        thin=thin,
        flat=True,
    )

    E = samples[:, 0]
    Ea_frac = samples[:, 1]

    Ea = E * Ea_frac
    Ep = E * (1.0 - Ea_frac)

    percentiles = [2.5, 16, 50, 84, 97.5]

    print(f"\n{direction.capitalize()} energy percentiles")
    print("Percentiles:", percentiles)
    print("Ea [MeV]:", np.percentile(Ea, percentiles))
    print("Ep [MeV]:", np.percentile(Ep, percentiles))


# ============================================================
# Construct the best simulated event
# ============================================================

def get_best_last_step_parameters(
    backend: emcee.backends.HDFBackend,
) -> np.ndarray:
    """
    Select the highest-log-probability walker from the final step.

    This matches the selection method used by Alex's original loader.
    """

    final_samples = backend.get_chain()[-1]
    final_log_prob = backend.get_log_prob()[-1]

    best_index = int(np.nanargmax(final_log_prob))

    best_parameters = np.asarray(
        final_samples[best_index],
        dtype=float,
    )

    if best_parameters.shape != (13,):
        raise ValueError(
            f"Expected 13 parameters, got "
            f"{best_parameters.shape}"
        )

    return best_parameters


def create_best_simulation(
    direction: str,
):
    """
    Build and simulate the best final-step model with rho_scale fixed to 1.
    """

    from track_fitting import build_sim

    backend = load_backend(direction)

    best = get_best_last_step_parameters(backend)

    (
        E,
        Ea_frac,
        x,
        y,
        z,
        p_x,
        p_y,
        p_z,
        a_x,
        a_y,
        a_z,
        sigma_xy,
        sigma_z,
    ) = best

    Ea = E * Ea_frac
    Ep = E * (1.0 - Ea_frac)

    sim = build_sim.create_multi_particle_decay(
        EXPERIMENT,
        RUN,
        EVENT,
        ["1H", "4He"],
        [1.0, 4.0],
        "16O",
        16.0,
    )

    vertex = (x, y, z)

    # Set the parent and all particle starting points.
    sim.initial_point = vertex

    for particle_sim in sim.sims:
        particle_sim.initial_point = vertex

    sim.sims[0].initial_energy = Ep
    sim.sims[1].initial_energy = Ea

    proton_magnitude = np.linalg.norm(
        [p_x, p_y, p_z]
    )

    alpha_magnitude = np.linalg.norm(
        [a_x, a_y, a_z]
    )

    if proton_magnitude <= 0:
        raise ValueError(
            "Best proton direction has zero magnitude."
        )

    if alpha_magnitude <= 0:
        raise ValueError(
            "Best alpha direction has zero magnitude."
        )

    sim.sims[0].theta = np.arccos(
        np.clip(
            p_z / proton_magnitude,
            -1.0,
            1.0,
        )
    )

    sim.sims[0].phi = np.arctan2(
        p_y,
        p_x,
    )

    sim.sims[1].theta = np.arccos(
        np.clip(
            a_z / alpha_magnitude,
            -1.0,
            1.0,
        )
    )

    sim.sims[1].phi = np.arctan2(
        a_y,
        a_x,
    )

    sim.sigma_xy = sigma_xy
    sim.sigma_z = sigma_z

    # Explicitly enforce nominal density: rho_scale = 1.
    nominal_density = build_sim.get_gas_density(
        EXPERIMENT,
        RUN,
    )

    for particle_sim in sim.sims:
        particle_sim.load_srim_table(
            particle_sim.particle,
            particle_sim.material,
            nominal_density * FIXED_RHO_SCALE,
        )

    sim.name = (
        f"{EXPERIMENT} run {RUN} event {EVENT} "
        f"{direction}, rho_scale fixed at 1"
    )

    print(f"\nBest {direction} final-step model")
    print(f"E total  = {E:.6f} MeV")
    print(f"E alpha  = {Ea:.6f} MeV")
    print(f"E proton = {Ep:.6f} MeV")
    print(f"Vertex   = ({x:.4f}, {y:.4f}, {z:.4f})")
    print(f"sigma_xy = {sigma_xy:.4f}")
    print(f"sigma_z  = {sigma_z:.4f}")
    print("rho_scale = 1.0 fixed")

    sim.simulate_event()

    return sim


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--discard",
        type=int,
        default=400,
        help="Number of initial MCMC steps discarded as burn-in.",
    )

    parser.add_argument(
        "--thin",
        type=int,
        default=1,
        help="Keep every Nth post-burn-in sample.",
    )

    parser.add_argument(
        "--sim",
        choices=["none", "forward", "backward"],
        default="none",
        help="Open a simulation comparison for one direction.",
    )

    parser.add_argument(
        "--view",
        choices=["gui", "3d"],
        default="gui",
        help="Comparison display to use with --sim.",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating chain and corner plots.",
    )

    args = parser.parse_args()

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not args.no_plots:
        for direction in ("forward", "backward"):
            backend = load_backend(direction)

            print_backend_information(
                direction,
                backend,
            )

            if args.discard >= backend.iteration:
                raise ValueError(
                    f"discard={args.discard} is not smaller "
                    f"than the {backend.iteration} completed steps."
                )

            plot_chain(
                direction,
                backend,
            )

            plot_raw_corner(
                direction,
                backend,
                args.discard,
                args.thin,
            )

            plot_physical_corner(
                direction,
                backend,
                args.discard,
                args.thin,
            )

            print_energy_summary(
                direction,
                backend,
                args.discard,
                args.thin,
            )

    if args.sim != "none":
        from track_fitting import build_sim

        sim = create_best_simulation(args.sim)

        if args.view == "gui":
            build_sim.open_gui(sim)

        else:
            build_sim.show_3d_plots(
                sim,
                view_thresh=20,
            )

            plt.show()


if __name__ == "__main__":
    main()
