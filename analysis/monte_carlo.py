"""
analysis/monte_carlo.py

Monte Carlo dispersion analysis for launch-vehicle ascent/insertion
guidance, wired to the PEG closed-loop guidance module (guidance/peg.py).

Dispersed parameters (each independently randomized per trial, Gaussian):
    - Initial velocity, flight-path-angle, radius error (stage-1 burnout dispersion)
    - Propellant mass error (loading/measurement uncertainty)
    - Specific impulse error (engine performance variation)

Success/comparison metrics use ALL THREE insertion errors (radius,
velocity, flight-path-angle) via a normalized RSS score, not velocity
alone -- a trial that nails velocity but misses radius by 100 km is not
a success just because gamma_dot happened to help.

Zero external dependencies beyond NumPy - CPU-only. Each trial calls the
numerical predictor-corrector PEG solve (guidance/peg.py), roughly
0.1-0.3 s/trial on a Ryzen 3 laptop; budget ~1-2 minutes for 500 trials
(slower than the earlier closed-form 2-constraint solver, but it now
actually converges on radius too). Use a smaller `n_trials` (50-150) for
interactive/test use.
"""

import numpy as np
from guidance.peg import PEGTarget, simulate_peg_guided_burn, rk4_step

MU_EARTH = 3.986004418e14
G0 = 9.80665

# Normalizers turning (radius_m, velocity_m/s, gamma_rad) errors into a
# single comparable RSS score -- 1 km radius ~ 10 m/s velocity ~ 0.3 deg gamma.
_R_NORM_M = 1000.0
_V_NORM_M_S = 10.0
_GAMMA_NORM_RAD = np.radians(0.3)


def _rss_score(radius_err_m, vel_err_m_s, gamma_err_rad):
    return float(np.sqrt((radius_err_m / _R_NORM_M) ** 2 +
                         (vel_err_m_s / _V_NORM_M_S) ** 2 +
                         (gamma_err_rad / _GAMMA_NORM_RAD) ** 2))


class DispersionModel:
    """Gaussian 1-sigma dispersion magnitudes for each randomized parameter."""
    def __init__(self, sigma_v0_m_s=15.0, sigma_gamma0_deg=0.3, sigma_r0_m=1500.0,
                 sigma_mass_frac=0.01, sigma_isp_frac=0.008):
        self.sigma_v0 = sigma_v0_m_s
        self.sigma_gamma0 = np.radians(sigma_gamma0_deg)
        self.sigma_r0 = sigma_r0_m
        self.sigma_mass_frac = sigma_mass_frac
        self.sigma_isp_frac = sigma_isp_frac

    def sample(self, rng, nominal_r0, nominal_v0, nominal_gamma0, nominal_m0, nominal_isp):
        return (nominal_r0 + rng.normal(0.0, self.sigma_r0),
                nominal_v0 + rng.normal(0.0, self.sigma_v0),
                nominal_gamma0 + rng.normal(0.0, self.sigma_gamma0),
                nominal_m0 * (1.0 + rng.normal(0.0, self.sigma_mass_frac)),
                nominal_isp * (1.0 + rng.normal(0.0, self.sigma_isp_frac)))


def run_monte_carlo(
    nominal_r0_m, nominal_v0_m_s, nominal_gamma0_rad, nominal_m0_kg,
    thrust_n, nominal_isp_s, target: PEGTarget,
    dispersion: DispersionModel = None, n_trials: int = 500,
    success_rss_tol: float = 3.0, seed: int = 123,
):
    """
    Run n_trials Monte Carlo closed-loop PEG burns with randomized initial
    conditions/vehicle parameters. A trial succeeds if its combined
    radius/velocity/gamma RSS score (see `_rss_score`) is within
    `success_rss_tol` (roughly: within a few km AND a few tens of m/s AND
    under a degree, simultaneously).
    """
    dispersion = dispersion or DispersionModel()
    rng = np.random.default_rng(seed)

    vel_errors = np.zeros(n_trials)
    gamma_errors = np.zeros(n_trials)
    radius_errors = np.zeros(n_trials)
    rss = np.zeros(n_trials)
    successes = np.zeros(n_trials, dtype=bool)

    for i in range(n_trials):
        r0, v0, gamma0, m0, isp = dispersion.sample(
            rng, nominal_r0_m, nominal_v0_m_s, nominal_gamma0_rad, nominal_m0_kg, nominal_isp_s)
        result = simulate_peg_guided_burn(r0_m=r0, v0_m_s=v0, gamma0_rad=gamma0, m0_kg=m0,
                                          thrust_n=thrust_n, isp_s=isp, target=target)
        err = result["insertion_error"]
        vel_errors[i] = err["velocity_error_m_s"]
        gamma_errors[i] = err["flight_path_angle_error_deg"]
        radius_errors[i] = err["radius_error_m"]
        rss[i] = _rss_score(radius_errors[i], vel_errors[i], np.radians(gamma_errors[i]))
        successes[i] = rss[i] <= success_rss_tol

    return {
        "n_trials": n_trials, "success_rate": float(np.mean(successes)),
        "success_count": int(np.sum(successes)),
        "velocity_error_m_s": vel_errors, "gamma_error_deg": gamma_errors,
        "radius_error_m": radius_errors, "rss_score": rss,
        "velocity_error_mean": float(np.mean(vel_errors)), "velocity_error_std": float(np.std(vel_errors)),
        "velocity_error_p99": float(np.percentile(np.abs(vel_errors), 99)),
        "radius_error_mean": float(np.mean(radius_errors)), "radius_error_std": float(np.std(radius_errors)),
        "radius_error_p99": float(np.percentile(np.abs(radius_errors), 99)),
        "gamma_error_mean": float(np.mean(gamma_errors)), "gamma_error_std": float(np.std(gamma_errors)),
        "gamma_error_p99": float(np.percentile(np.abs(gamma_errors), 99)),
    }


def _fixed_steering_burn(r0, v0, gamma0, m0, isp, thrust_n, chi_fixed, t_burn, mu=MU_EARTH, dt=0.05):
    """Open-loop reference: fixed thrust-to-velocity angle `chi_fixed`, fixed duration `t_burn`."""
    mdot = thrust_n / (isp * G0)
    n = max(1, int(round(t_burn / dt)))
    h = t_burn / n
    r, v, gamma, m = r0, v0, gamma0, m0
    for _ in range(n):
        if m - mdot * h <= 0:
            break
        theta = gamma + chi_fixed
        r, v, gamma, m = rk4_step((r, v, gamma, m), thrust_n, mdot, theta, h, mu)
    return r, v, gamma, m


def compare_open_vs_closed_loop_dispersion(
    nominal_r0_m, nominal_v0_m_s, nominal_gamma0_rad, nominal_m0_kg,
    thrust_n, nominal_isp_s, target: PEGTarget,
    fixed_chi_deg: float = None, dispersion: DispersionModel = None,
    n_trials: int = 500, seed: int = 123,
):
    """
    Apples-to-apples PEG-vs-open-loop dispersion comparison. Both fly the
    SAME dispersed trial and the SAME burn duration (the nominal PEG
    solve's T_go), and are scored on the SAME combined radius/velocity/
    gamma RSS metric -- not velocity alone, which understates open-loop's
    disadvantage because PEG deliberately trades duration to hit velocity.

    `fixed_chi_deg`: open-loop's fixed thrust-to-velocity angle. Defaults
    to the nominal PEG solve's own FIRST commanded angle, i.e. "the best
    open-loop profile you could have pre-computed from the nominal plan".
    """
    dispersion = dispersion or DispersionModel()
    rng = np.random.default_rng(seed)

    nominal = simulate_peg_guided_burn(nominal_r0_m, nominal_v0_m_s, nominal_gamma0_rad,
                                       nominal_m0_kg, thrust_n, nominal_isp_s, target)
    t_burn_fixed = nominal["burn_time_s"]
    if fixed_chi_deg is None:
        chi_fixed = np.radians(nominal["chi_deg"][0]) if len(nominal["chi_deg"]) else 0.0
    else:
        chi_fixed = np.radians(fixed_chi_deg)

    closed_rss = np.zeros(n_trials); open_rss = np.zeros(n_trials)
    closed_v = np.zeros(n_trials); open_v = np.zeros(n_trials)
    closed_r = np.zeros(n_trials); open_r = np.zeros(n_trials)
    closed_g = np.zeros(n_trials); open_g = np.zeros(n_trials)

    for i in range(n_trials):
        r0, v0, gamma0, m0, isp = dispersion.sample(
            rng, nominal_r0_m, nominal_v0_m_s, nominal_gamma0_rad, nominal_m0_kg, nominal_isp_s)

        cl = simulate_peg_guided_burn(r0, v0, gamma0, m0, thrust_n, isp, target)
        ce = cl["insertion_error"]
        closed_v[i], closed_r[i], closed_g[i] = ce["velocity_error_m_s"], ce["radius_error_m"], ce["flight_path_angle_error_deg"]
        closed_rss[i] = _rss_score(closed_r[i], closed_v[i], np.radians(closed_g[i]))

        r, v, gamma, m = _fixed_steering_burn(r0, v0, gamma0, m0, isp, thrust_n, chi_fixed, t_burn_fixed)
        open_v[i], open_r[i], open_g[i] = v - target.v_t, r - target.r_t, float(np.degrees(gamma - target.gamma_t))
        open_rss[i] = _rss_score(open_r[i], open_v[i], np.radians(open_g[i]))

    return {
        "closed_loop_velocity_error_m_s": closed_v, "open_loop_velocity_error_m_s": open_v,
        "closed_loop_radius_error_m": closed_r, "open_loop_radius_error_m": open_r,
        "closed_loop_gamma_error_deg": closed_g, "open_loop_gamma_error_deg": open_g,
        "closed_loop_rss": closed_rss, "open_loop_rss": open_rss,
        "closed_loop_std": float(np.std(closed_v)), "open_loop_std": float(np.std(open_v)),
        "closed_loop_p99_abs": float(np.percentile(np.abs(closed_v), 99)),
        "open_loop_p99_abs": float(np.percentile(np.abs(open_v), 99)),
        "closed_loop_rss_mean": float(np.mean(closed_rss)), "open_loop_rss_mean": float(np.mean(open_rss)),
        "closed_loop_rss_p99": float(np.percentile(closed_rss, 99)), "open_loop_rss_p99": float(np.percentile(open_rss, 99)),
        "improvement_factor": float(np.std(open_v) / max(np.std(closed_v), 1e-9)),
        "rss_improvement_factor": float(np.mean(open_rss) / max(np.mean(closed_rss), 1e-9)),
    }
