"""
guidance/anomaly_arbitration.py

Sensor-anomaly-triggered guidance-mode arbitration.

This module exists because of a specific, publicly documented failure:
ISRO's SSLV-D1 (August 2022) lost its mission after a ~2-second anomaly
in ONE of several onboard accelerometers during second-stage separation.
Per ISRO Chairman S. Somanath's own account (The Hindu, 11 Aug 2022) and
the public post-flight review, the flight software responded by dropping
ENTIRELY from closed-loop guidance to open-loop guidance (accelerometer
data fully isolated, a pre-computed path flown blind) and STAYED in that
mode for the rest of the burn, producing a velocity shortfall that an
uncorrected "salvage mode" could not fix.

The structural question this module tests: should one transient,
detected sensor fault force a PERMANENT, full guidance-law fallback, or
can the bad channel be isolated for just its faulty samples while
closed-loop guidance keeps running? This is simulated end-to-end, not
asserted:

  1. A single accelerometer channel is given real, injected fault
     samples (a bias glitch of `fault_bias_m_s2` for `fault_duration_s`).
  2. `AccelerometerAnomalyDetector` runs on the (noisy, faulted) samples
     against the guidance model's own predicted accel and decides,
     online, whether each sample is anomalous, using the same
     consecutive-sample debounce for both architectures.
  3. The two architectures differ ONLY in what they do once a fault is
     flagged:
       - `simulate_sslv_style_full_open_loop_fallback`: freezes the
         steering command forever and stops using any further sensor
         update -- guidance never re-solves again.
       - `simulate_graceful_degradation_guidance`: for flagged samples
         only, the navigation estimate dead-reckons through the glitch
         using the guidance model's OWN predicted accel instead of the
         bad measurement; once the fault clears, real measurements
         resume and PEG keeps closing the loop every cycle for the rest
         of the burn.

Both burns fly the same true dynamics, see the same fault, and run to
the same propellant-exhaustion stopping condition, so the only thing
that differs is guidance architecture's response to the fault.

Simplification: only the along-track (tangential) accelerometer channel
is modeled as faulted; radius/mass are assumed known (matches the public
SSLV-D1 narrative of a single accelerometer channel issue, not a full
IMU failure). Gyro/attitude and multi-sensor voting are out of scope --
see navigation/state_estimation.py for the attitude/position EKF.

Zero external dependencies beyond NumPy - CPU-only.
"""

import numpy as np
from guidance.peg import (
    PEGTarget, solve_peg_predictor_corrector, rk4_step, point_mass_derivs, G0,
)
from guidance.orbit_utils import orbit_from_state

MU_EARTH = 3.986004418e14


class AccelerometerAnomalyDetector:
    """
    Residual-based fault detector: compares a measured accelerometer
    reading against the guidance model's own predicted acceleration for
    the commanded thrust pitch. A transient sensor fault shows up as a
    sudden, large residual -- standard "innovation/residual monitoring"
    FDI. Requires `consecutive_samples_required` consecutive out-of-bound
    residuals before declaring a fault, so a single-sample glitch does
    not immediately trip guidance into a degraded mode.
    """
    def __init__(self, residual_threshold_m_s2: float = 5.0,
                 consecutive_samples_required: int = 3):
        self.threshold = residual_threshold_m_s2
        self.required = consecutive_samples_required
        self._consecutive_count = 0

    def check(self, measured_accel: float, predicted_accel: float) -> bool:
        residual = abs(measured_accel - predicted_accel)
        if residual > self.threshold:
            self._consecutive_count += 1
        else:
            self._consecutive_count = 0
        return self._consecutive_count >= self.required

    def reset(self):
        self._consecutive_count = 0


def _tangential_accel(r, v, gamma, m, thrust_n, theta, mu=MU_EARTH):
    """True along-track (dv/dt) specific-force component an accelerometer would sense."""
    g = mu / r ** 2
    chi = theta - gamma
    return thrust_n * np.cos(chi) / m - g * np.sin(gamma)


def nominal_cutoff_mass_kg(r0_m, v0_m_s, gamma0_rad, m0_kg, thrust_n, isp_s, target: PEGTarget,
                           dry_mass_fractions=(0.375, 0.2, 0.3, 0.1, 0.4, 0.5, 0.05, 0.25)):
    """
    Propellant a FAULT-FREE closed-loop PEG burn would actually use to reach
    the target from this start state. Used as the shared propellant-exhaustion
    stopping mass for the fault-comparison sims below, so that in the
    zero-fault limit BOTH architectures land on the target.

    The numerical predictor-corrector (guidance/peg.py) is a local
    Levenberg-Marquardt solve and can land in a bad basin for some
    dry-mass floors even when a nearby floor converges cleanly; this
    tries a small set of candidate floors and keeps whichever gives the
    smallest actual insertion residual (checked directly, not via the
    solver's own `converged` flag, which is stricter than "good enough").
    """
    from guidance.peg import simulate_peg_guided_burn
    best = None
    for frac in dry_mass_fractions:
        nominal = simulate_peg_guided_burn(r0_m, v0_m_s, gamma0_rad, m0_kg, thrust_n, isp_s,
                                           target, m_dry_kg=frac * m0_kg)
        e = nominal["insertion_error"]
        resid = abs(e["radius_error_m"]) / 1000.0 + abs(e["velocity_error_m_s"])
        if best is None or resid < best[0]:
            best = (resid, nominal["final_mass_kg"])
        if resid < 0.1:
            break
    return best[1]


def _run_arbitration(
    r0_m, v0_m_s, gamma0_rad, m0_kg, m_dry_kg, thrust_n, isp_s, target: PEGTarget,
    anomaly_start_s: float, fault_bias_m_s2: float, fault_duration_s: float,
    architecture: str, dt: float = 0.05, guidance_cycle_s: float = 2.0,
    noise_sigma_m_s2: float = 0.05, seed: int = 11,
):
    """Shared simulation core for both architectures; only differs after a flagged sample."""
    mdot = thrust_n / (isp_s * G0)
    rng = np.random.default_rng(seed)
    detector = AccelerometerAnomalyDetector()

    r, v, gamma, m = r0_m, v0_m_s, gamma0_rad, m0_kg
    v_est = v0_m_s               # navigation's own along-track velocity estimate
    t = 0.0
    permanently_frozen = False
    frozen_theta = None
    guess = None
    solve_now = True
    theta = np.radians(90.0 - 0.0)

    t_hist, v_hist, r_hist, gamma_hist, flag_hist = [], [], [], [], []
    n_flagged = 0

    while m > m_dry_kg:
        est_state = (r, v_est, gamma, m)     # r, gamma, m assumed known; v is the faulted channel
        if solve_now and not permanently_frozen:
            sol = solve_peg_predictor_corrector(est_state, thrust_n, mdot, target,
                                                m_dry_kg, guess=guess)
            guess = (sol["A"], sol["B"], max(sol["T"] - guidance_cycle_s, 0.5))
            cycle_t0 = t
        theta = float(np.arctan(sol["A"] + sol["B"] * (t - cycle_t0))) if not permanently_frozen else frozen_theta

        # true dynamics step
        state_true = rk4_step((r, v, gamma, m), thrust_n, mdot, theta, dt)
        r_new, v_new, gamma_new, m_new = state_true
        a_true = _tangential_accel(r, v, gamma, m, thrust_n, theta)

        in_fault = anomaly_start_s <= t < anomaly_start_s + fault_duration_s
        measured = a_true + rng.normal(0.0, noise_sigma_m_s2) + (fault_bias_m_s2 if in_fault else 0.0)
        predicted = _tangential_accel(r, v_est, gamma, m, thrust_n, theta)
        flagged = detector.check(measured, predicted)
        n_flagged += int(flagged)

        if architecture == "fallback":
            if flagged and not permanently_frozen:
                permanently_frozen = True
                frozen_theta = theta
            if not permanently_frozen:
                v_est += measured * dt
            # once frozen: guidance stops updating its estimate AND stops re-solving
        else:  # graceful
            if flagged:
                v_est += predicted * dt     # dead-reckon through the bad sample only
            else:
                v_est += measured * dt

        r, v, gamma, m = r_new, v_new, gamma_new, m_new
        t += dt
        solve_now = (not permanently_frozen) and ((t - cycle_t0) >= guidance_cycle_s - 1e-9)

        t_hist.append(t); v_hist.append(v); r_hist.append(r); gamma_hist.append(gamma)
        flag_hist.append(flagged)

    orbit = orbit_from_state(r, v, gamma)
    return {
        "t": np.array(t_hist), "velocity_m_s": np.array(v_hist),
        "radius_m": np.array(r_hist), "gamma_rad": np.array(gamma_hist),
        "flagged": np.array(flag_hist), "n_samples_flagged": n_flagged,
        "permanently_frozen": permanently_frozen,
        "final_velocity_m_s": v, "final_radius_m": r, "final_gamma_rad": gamma,
        "burn_time_s": t, "orbit": orbit,
        "velocity_error_m_s": v - target.v_t,
        "radius_error_m": r - target.r_t,
        "flight_path_angle_error_deg": float(np.degrees(gamma - target.gamma_t)),
    }


def simulate_sslv_style_full_open_loop_fallback(
    r0_m, v0_m_s, gamma0_rad, m0_kg, m_dry_kg, thrust_n, isp_s, target: PEGTarget,
    anomaly_start_s: float, fault_bias_m_s2: float = 40.0, fault_duration_s: float = 2.0,
    dt: float = 0.05,
):
    """SSLV-D1-structure: once the fault is DETECTED, freeze steering forever and stop
    using any further accelerometer update, for the rest of the burn."""
    return _run_arbitration(r0_m, v0_m_s, gamma0_rad, m0_kg, m_dry_kg, thrust_n, isp_s,
                            target, anomaly_start_s, fault_bias_m_s2, fault_duration_s,
                            architecture="fallback", dt=dt)


def simulate_graceful_degradation_guidance(
    r0_m, v0_m_s, gamma0_rad, m0_kg, m_dry_kg, thrust_n, isp_s, target: PEGTarget,
    anomaly_start_s: float, detector: AccelerometerAnomalyDetector = None,
    fault_bias_m_s2: float = 40.0, fault_duration_s: float = 2.0, dt: float = 0.05,
):
    """Isolate only the flagged samples (dead-reckon through them using the guidance
    model's own predicted accel); PEG keeps re-solving every cycle for the whole burn."""
    return _run_arbitration(r0_m, v0_m_s, gamma0_rad, m0_kg, m_dry_kg, thrust_n, isp_s,
                            target, anomaly_start_s, fault_bias_m_s2, fault_duration_s,
                            architecture="graceful", dt=dt)


def compare_sslv_failure_mode_vs_graceful_degradation(
    r0_m, v0_m_s, gamma0_rad, m0_kg, thrust_n, isp_s, target: PEGTarget,
    anomaly_start_s: float = 5.0, fault_bias_m_s2: float = 40.0, fault_duration_s: float = 2.0,
    m_dry_kg: float = None,
):
    """
    Direct, quantified comparison under the IDENTICAL injected fault and
    propellant budget.  `m_dry_kg` defaults to `nominal_cutoff_mass_kg`
    (the mass a fault-free burn would need) so the zero-fault baseline is
    "hit the target", isolating the fault's effect rather than an
    arbitrary stopping point.
    """
    if m_dry_kg is None:
        m_dry_kg = nominal_cutoff_mass_kg(r0_m, v0_m_s, gamma0_rad, m0_kg, thrust_n, isp_s, target)
    fallback = simulate_sslv_style_full_open_loop_fallback(
        r0_m, v0_m_s, gamma0_rad, m0_kg, m_dry_kg, thrust_n, isp_s, target,
        anomaly_start_s, fault_bias_m_s2, fault_duration_s)
    graceful = simulate_graceful_degradation_guidance(
        r0_m, v0_m_s, gamma0_rad, m0_kg, m_dry_kg, thrust_n, isp_s, target,
        anomaly_start_s, fault_bias_m_s2=fault_bias_m_s2, fault_duration_s=fault_duration_s)
    return {
        "sslv_style_fallback": fallback,
        "graceful_degradation": graceful,
        "velocity_error_improvement_m_s": abs(fallback["velocity_error_m_s"]) - abs(graceful["velocity_error_m_s"]),
        "radius_error_improvement_m": abs(fallback["radius_error_m"]) - abs(graceful["radius_error_m"]),
    }
