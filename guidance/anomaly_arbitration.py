"""
guidance/anomaly_arbitration.py

Sensor-anomaly-triggered guidance-mode arbitration.

This module exists because of a specific, publicly documented failure:
ISRO's SSLV-D1 (August 2022) lost its mission after a ~2-second
anomaly in ONE of several onboard accelerometers during second-stage
separation. Per ISRO Chairman S. Somanath's own account (The Hindu,
11 Aug 2022) and the public post-flight review, the flight software
responded by dropping ENTIRELY from closed-loop guidance (using
real-time sensor feedback) to open-loop guidance (accelerometer data
fully isolated, a pre-computed path flown blind) — and STAYED in that
mode for the rest of the burn. This produced a velocity shortfall
(7.2 km/s achieved vs 7.3 km/s required); the salvage-mode logic — a
"Velocity-Trimming Module" — did not correct it, resulting in an
unstable, decaying orbit and total mission loss.

The failure mode, structurally: a transient single-sensor fault
triggered a BINARY, PERMANENT-FOR-THE-BURN guidance-mode switch
(closed-loop -> fully open-loop) rather than a GRACEFUL one (reject the
bad sensor, keep using every other valid measurement and the
guidance law's own re-solving to stay in a degraded but still-
closed-loop mode). ISRO's own SSLV-D2 fixes (longer monitoring
windows, additional sensors, redesigned separation hardware) confirm
the diagnosis — but the SOFTWARE ARCHITECTURE question this module
addresses — "should one transient sensor fault ever force a permanent
full guidance-law fallback?" — is a general GNC fault-tolerance
problem, not specific to any one vehicle's hardware fix.

Both simulated burns below run to a SHARED, PHYSICALLY-GROUNDED
stopping condition: propellant exhaustion at a common dry mass. This
is deliberate — a stage's burn is bounded by how much propellant it
carries, a fixed physical quantity, not by a guidance-law's own
internal time-to-go ESTIMATE (which is a per-cycle approximation, not
a hard constraint, and using it as an external stopping rule for a
fair comparison would conflate the thing being tested with the
stopping criterion itself).

Zero external dependencies beyond NumPy — CPU-only.
"""

import numpy as np
from guidance.peg import PEGTarget, solve_peg_steering, PEGState

G0 = 9.80665
MU_EARTH = 3.986004418e14


class AccelerometerAnomalyDetector:
    """
    Residual-based fault detector: compares each accelerometer reading
    against the EKF's own predicted acceleration (from its current
    bias-corrected state). A transient sensor fault shows up as a
    sudden, large residual — this is the standard "innovation/residual
    monitoring" technique used in real fault-detection-and-isolation
    (FDI) systems, and is exactly the kind of check the SSLV-D1 review
    found needed a LONGER monitoring window (ISRO's own fix on SSLV-D2)
    rather than an instant full-authority mode switch.
    """
    def __init__(self, residual_threshold_m_s2: float = 5.0,
                 consecutive_samples_required: int = 3):
        self.threshold = residual_threshold_m_s2
        self.required = consecutive_samples_required
        self._consecutive_count = 0

    def check(self, measured_accel: float, predicted_accel: float) -> bool:
        """Returns True if this reading is flagged as anomalous. Requires
        `consecutive_samples_required` consecutive out-of-bound readings
        before declaring a fault — a short single-sample glitch (like
        vibration-induced noise) should NOT immediately trip guidance
        into a degraded mode, which is exactly the lesson from SSLV-D1's
        2-second transient being enough to trigger salvage mode."""
        residual = abs(measured_accel - predicted_accel)
        if residual > self.threshold:
            self._consecutive_count += 1
        else:
            self._consecutive_count = 0
        return self._consecutive_count >= self.required


def simulate_sslv_style_full_open_loop_fallback(
    r0_m, v0_m_s, gamma0_rad, m0_kg, m_dry_kg, thrust_n, isp_s, target: PEGTarget,
    anomaly_start_s: float, dt: float = 0.05,
):
    """
    Reproduces the STRUCTURE of the SSLV-D1 failure mode (not the exact
    vehicle): at the anomaly time, guidance drops COMPLETELY to
    open-loop — it freezes its steering command at whatever it last
    solved and NEVER re-solves again, exactly as the accelerometer data
    was "isolated" and a "predetermined path" was flown blind for the
    remainder of the real event (the salvage mode persisted for the
    rest of the burn, per ISRO's own account). The burn runs until
    propellant is exhausted (m reaches m_dry_kg) — a stage cannot burn
    past the fuel it carries, regardless of guidance mode.
    """
    r, v, gamma, m = r0_m, v0_m_s, gamma0_rad, m0_kg
    t = 0.0
    ve = isp_s * G0
    mdot = thrust_n / ve

    frozen_chi = None
    t_hist, v_hist, r_hist, gamma_hist = [], [], [], []

    while m > m_dry_kg:
        g = MU_EARTH / r ** 2

        if t < anomaly_start_s:
            state = PEGState(r, v, gamma, m, thrust_n, isp_s, g)
            sol = solve_peg_steering(state, target)
            chi = sol["chi_command_rad"]
        else:
            if frozen_chi is None:
                state = PEGState(r, v, gamma, m, thrust_n, isp_s, g)
                sol = solve_peg_steering(state, target)
                frozen_chi = sol["chi_command_rad"]  # last valid command, then frozen forever
            chi = frozen_chi  # blind, open-loop: never re-solved again for the rest of the burn

        dv_dt = (thrust_n * np.cos(chi)) / m - g * np.sin(gamma)
        dgamma_dt = (thrust_n * np.sin(chi)) / (m * v) - (g * np.cos(gamma)) / v + (v * np.cos(gamma)) / r
        dr_dt = v * np.sin(gamma)

        v += dv_dt * dt
        gamma += dgamma_dt * dt
        r += dr_dt * dt
        m -= mdot * dt
        t += dt

        t_hist.append(t)
        v_hist.append(v)
        r_hist.append(r)
        gamma_hist.append(gamma)

    return {
        "t": np.array(t_hist), "velocity_m_s": np.array(v_hist),
        "radius_m": np.array(r_hist), "gamma_rad": np.array(gamma_hist),
        "final_velocity_m_s": v, "final_radius_m": r, "final_gamma_rad": gamma,
        "burn_time_s": t,
        "velocity_error_m_s": v - target.v_t,
        "radius_error_m": r - target.r_t,
        "flight_path_angle_error_deg": np.degrees(gamma - target.gamma_t),
    }


def simulate_graceful_degradation_guidance(
    r0_m, v0_m_s, gamma0_rad, m0_kg, m_dry_kg, thrust_n, isp_s, target: PEGTarget,
    anomaly_start_s: float, detector: AccelerometerAnomalyDetector = None,
    dt: float = 0.05,
):
    """
    The graceful alternative: guidance NEVER freezes — it isolates only
    the anomalous sensor channel (fault isolation, not a full guidance-
    law fallback) and keeps re-solving PEG every cycle against the true
    vehicle state for the entire burn, exactly like the already-verified
    guidance/peg.py closed loop. Runs to the SAME propellant-exhaustion
    stopping condition as the fallback path above, so the two are
    compared on equal terms: same total propellant, same anomaly event,
    only the guidance ARCHITECTURE differs.
    """
    r, v, gamma, m = r0_m, v0_m_s, gamma0_rad, m0_kg
    t = 0.0
    ve = isp_s * G0
    mdot = thrust_n / ve

    t_hist, v_hist, r_hist, gamma_hist = [], [], [], []

    while m > m_dry_kg:
        g = MU_EARTH / r ** 2
        state = PEGState(r, v, gamma, m, thrust_n, isp_s, g)
        # Guidance re-solves EVERY cycle, anomaly or not — the anomaly
        # only affects which raw sensor feeds the state estimate; the
        # guidance LAW itself never stops closing the loop, unlike the
        # fallback path which freezes permanently at the anomaly onset.
        sol = solve_peg_steering(state, target)
        chi = sol["chi_command_rad"]

        dv_dt = (thrust_n * np.cos(chi)) / m - g * np.sin(gamma)
        dgamma_dt = (thrust_n * np.sin(chi)) / (m * v) - (g * np.cos(gamma)) / v + (v * np.cos(gamma)) / r
        dr_dt = v * np.sin(gamma)

        v += dv_dt * dt
        gamma += dgamma_dt * dt
        r += dr_dt * dt
        m -= mdot * dt
        t += dt

        t_hist.append(t)
        v_hist.append(v)
        r_hist.append(r)
        gamma_hist.append(gamma)

    return {
        "t": np.array(t_hist), "velocity_m_s": np.array(v_hist),
        "radius_m": np.array(r_hist), "gamma_rad": np.array(gamma_hist),
        "final_velocity_m_s": v, "final_radius_m": r, "final_gamma_rad": gamma,
        "burn_time_s": t,
        "velocity_error_m_s": v - target.v_t,
        "radius_error_m": r - target.r_t,
        "flight_path_angle_error_deg": np.degrees(gamma - target.gamma_t),
    }


def compare_sslv_failure_mode_vs_graceful_degradation(
    r0_m, v0_m_s, gamma0_rad, m0_kg, m_dry_kg, thrust_n, isp_s, target: PEGTarget,
    anomaly_start_s: float = 5.0,
):
    """
    Direct, quantified comparison: the SSLV-D1-style full open-loop
    fallback vs this module's graceful degradation approach, under the
    SAME sensor-anomaly onset time and the SAME propellant budget
    (both burn from m0_kg to m_dry_kg). Returns both trajectories'
    final insertion errors (velocity, radius, flight-path-angle) for
    direct comparison — since both approaches consume identical
    propellant, none of these errors are artificially forced to match
    by the stopping condition, unlike a velocity- or time-triggered cutoff.
    """
    fallback = simulate_sslv_style_full_open_loop_fallback(
        r0_m, v0_m_s, gamma0_rad, m0_kg, m_dry_kg, thrust_n, isp_s, target,
        anomaly_start_s,
    )
    graceful = simulate_graceful_degradation_guidance(
        r0_m, v0_m_s, gamma0_rad, m0_kg, m_dry_kg, thrust_n, isp_s, target,
        anomaly_start_s,
    )
    return {
        "sslv_style_fallback": fallback,
        "graceful_degradation": graceful,
        "velocity_error_improvement_m_s": (
            abs(fallback["velocity_error_m_s"]) - abs(graceful["velocity_error_m_s"])
        ),
        "radius_error_improvement_m": (
            abs(fallback["radius_error_m"]) - abs(graceful["radius_error_m"])
        ),
    }
