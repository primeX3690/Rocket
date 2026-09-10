"""
analysis/monte_carlo.py

Monte Carlo dispersion analysis for launch-vehicle ascent guidance.

Real launch providers never certify a guidance design on a single
"nominal" simulation — they run hundreds to thousands of Monte Carlo
trials with randomized off-nominal conditions (wind, engine performance
variation, mass/CG uncertainty, sensor noise) and report statistics like
"99.7% of trials achieve orbit insertion within tolerance." This module
brings that certification-style analysis to the AscentGNC stack, wired
to the PEG closed-loop guidance module (guidance/peg.py) so the actual
dispersion-rejection capability of closed-loop guidance can be
quantified, not just asserted.

Dispersed parameters modeled (each independently randomized per trial,
Gaussian, standard aerospace dispersion-analysis practice):
    - Initial velocity error      (stage-1 burnout dispersion)
    - Initial flight-path-angle error (wind/steering dispersion)
    - Initial radius error         (stage-1 altitude dispersion)
    - Propellant mass error        (loading/measurement uncertainty)
    - Specific impulse error       (engine performance variation)

Zero external dependencies beyond NumPy — CPU-only. Designed to run
comfortably on a CPU-only laptop (default 500 trials completes in
well under a minute for this point-mass model).
"""

import numpy as np
from guidance.peg import PEGTarget, simulate_peg_guided_burn


class DispersionModel:
    """
    Gaussian 1-sigma dispersion magnitudes for each randomized parameter.
    Defaults are representative order-of-magnitude values for a small-
    to-medium launch vehicle upper stage (not vehicle-specific — meant
    to be tuned once real vehicle data is available).
    """
    def __init__(self,
                 sigma_v0_m_s=15.0,
                 sigma_gamma0_deg=0.3,
                 sigma_r0_m=1500.0,
                 sigma_mass_frac=0.01,
                 sigma_isp_frac=0.008):
        self.sigma_v0 = sigma_v0_m_s
        self.sigma_gamma0 = np.radians(sigma_gamma0_deg)
        self.sigma_r0 = sigma_r0_m
        self.sigma_mass_frac = sigma_mass_frac
        self.sigma_isp_frac = sigma_isp_frac


def run_monte_carlo(
    nominal_r0_m, nominal_v0_m_s, nominal_gamma0_rad, nominal_m0_kg,
    thrust_n, nominal_isp_s, target: PEGTarget,
    dispersion: DispersionModel = None,
    n_trials: int = 500,
    success_velocity_tol_m_s: float = 30.0,
    success_gamma_tol_deg: float = 1.0,
    seed: int = 123,
):
    """
    Run n_trials Monte Carlo simulations of the PEG-guided burn with
    randomized initial conditions and vehicle parameters, and compute
    mission-success statistics.

    A trial is counted "successful" if BOTH the final velocity error and
    flight-path-angle error at cutoff are within the given tolerances —
    the standard two-sided pass/fail criterion used in real insertion-
    accuracy certification.

    Returns a dict with per-trial results plus aggregate statistics:
    success rate, mean/std/percentiles of insertion errors.
    """
    if dispersion is None:
        dispersion = DispersionModel()

    rng = np.random.default_rng(seed)

    vel_errors = np.zeros(n_trials)
    gamma_errors = np.zeros(n_trials)
    radius_errors = np.zeros(n_trials)
    successes = np.zeros(n_trials, dtype=bool)

    for i in range(n_trials):
        v0 = nominal_v0_m_s + rng.normal(0.0, dispersion.sigma_v0)
        gamma0 = nominal_gamma0_rad + rng.normal(0.0, dispersion.sigma_gamma0)
        r0 = nominal_r0_m + rng.normal(0.0, dispersion.sigma_r0)
        m0 = nominal_m0_kg * (1.0 + rng.normal(0.0, dispersion.sigma_mass_frac))
        isp = nominal_isp_s * (1.0 + rng.normal(0.0, dispersion.sigma_isp_frac))

        result = simulate_peg_guided_burn(
            r0_m=r0, v0_m_s=v0, gamma0_rad=gamma0, m0_kg=m0,
            thrust_n=thrust_n, isp_s=isp, target=target,
        )

        err = result["insertion_error"]
        vel_errors[i] = err["velocity_error_m_s"]
        gamma_errors[i] = err["flight_path_angle_error_deg"]
        radius_errors[i] = err["radius_error_m"]

        successes[i] = (abs(vel_errors[i]) <= success_velocity_tol_m_s and
                         abs(gamma_errors[i]) <= success_gamma_tol_deg)

    success_rate = np.mean(successes)

    return {
        "n_trials": n_trials,
        "success_rate": success_rate,
        "success_count": int(np.sum(successes)),
        "velocity_error_m_s": vel_errors,
        "gamma_error_deg": gamma_errors,
        "radius_error_m": radius_errors,
        "velocity_error_mean": np.mean(vel_errors),
        "velocity_error_std": np.std(vel_errors),
        "velocity_error_p99": np.percentile(np.abs(vel_errors), 99),
        "gamma_error_mean": np.mean(gamma_errors),
        "gamma_error_std": np.std(gamma_errors),
        "gamma_error_p99": np.percentile(np.abs(gamma_errors), 99),
    }


def compare_open_vs_closed_loop_dispersion(
    nominal_r0_m, nominal_v0_m_s, nominal_gamma0_rad, nominal_m0_kg,
    thrust_n, nominal_isp_s, target: PEGTarget,
    fixed_chi_deg: float,
    dispersion: DispersionModel = None,
    n_trials: int = 500,
    seed: int = 123,
):
    """
    The key comparison that justifies building PEG at all: run the SAME
    dispersed trials through (a) closed-loop PEG guidance and (b) a
    naive open-loop guidance that just holds a fixed steering angle
    (representing what you'd get flying a pre-computed profile with no
    re-targeting). Compare final velocity-error distributions to
    quantify PEG's dispersion-rejection advantage numerically.
    """
    if dispersion is None:
        dispersion = DispersionModel()

    rng = np.random.default_rng(seed)
    MU_EARTH = 3.986004418e14

    closed_loop_errs = np.zeros(n_trials)
    open_loop_errs = np.zeros(n_trials)

    ve = nominal_isp_s * 9.80665
    mdot_nominal = thrust_n / ve
    chi_fixed = np.radians(fixed_chi_deg)

    for i in range(n_trials):
        v0 = nominal_v0_m_s + rng.normal(0.0, dispersion.sigma_v0)
        gamma0 = nominal_gamma0_rad + rng.normal(0.0, dispersion.sigma_gamma0)
        r0 = nominal_r0_m + rng.normal(0.0, dispersion.sigma_r0)
        m0 = nominal_m0_kg * (1.0 + rng.normal(0.0, dispersion.sigma_mass_frac))
        isp = nominal_isp_s * (1.0 + rng.normal(0.0, dispersion.sigma_isp_frac))

        # --- closed-loop PEG ---
        cl_result = simulate_peg_guided_burn(
            r0_m=r0, v0_m_s=v0, gamma0_rad=gamma0, m0_kg=m0,
            thrust_n=thrust_n, isp_s=isp, target=target,
        )
        closed_loop_errs[i] = cl_result["insertion_error"]["velocity_error_m_s"]

        # --- naive open-loop: fixed chi, fixed burn duration (from nominal
        #     rocket-equation time-to-go), no re-targeting whatsoever ---
        ve_trial = isp * 9.80665
        mdot = thrust_n / ve_trial
        dv_nominal_plan = target.v_t - nominal_v0_m_s
        t_burn_fixed = (nominal_m0_kg / mdot_nominal) * (1.0 - np.exp(-dv_nominal_plan / (nominal_isp_s * 9.80665)))

        r, v, gamma, m = r0, v0, gamma0, m0
        dt = 0.05
        steps = int(t_burn_fixed / dt)
        for _ in range(steps):
            if m <= 0:
                break
            g = MU_EARTH / r ** 2
            dv_dt = (thrust_n * np.cos(chi_fixed)) / m - g * np.sin(gamma)
            dgamma_dt = (thrust_n * np.sin(chi_fixed)) / (m * v) - (g * np.cos(gamma)) / v + (v * np.cos(gamma)) / r
            dr_dt = v * np.sin(gamma)
            v += dv_dt * dt
            gamma += dgamma_dt * dt
            r += dr_dt * dt
            m -= mdot * dt

        open_loop_errs[i] = v - target.v_t

    return {
        "closed_loop_velocity_error_m_s": closed_loop_errs,
        "open_loop_velocity_error_m_s": open_loop_errs,
        "closed_loop_std": np.std(closed_loop_errs),
        "open_loop_std": np.std(open_loop_errs),
        "closed_loop_p99_abs": np.percentile(np.abs(closed_loop_errs), 99),
        "open_loop_p99_abs": np.percentile(np.abs(open_loop_errs), 99),
        "improvement_factor": np.std(open_loop_errs) / max(np.std(closed_loop_errs), 1e-9),
    }
