"""
guidance/powered_descent_guidance.py

Closed-loop powered-descent guidance via ZEM/ZEV (Zero-Effort-Miss /
Zero-Effort-Velocity) -- the real, published minimum-energy guidance
law behind the Apollo Lunar Module's descent guidance and the class of
algorithms studied for propulsive planetary landing and terrestrial
rocket recovery (a close relative of the "SCvx" family used in
guidance/trajectory_optimization.py, but here solved in closed form
rather than iteratively, because the minimum-EFFORT double-integrator
boundary-value problem this models has an exact analytic solution).

Derivation (double integrator ẍ = a, minimize integral of a^2 dt,
boundary conditions x(0)=x0, v(0)=v0, x(T)=0, v(T)=0): the optimal
control is affine in time, a(t) = c1 + c2*t, and solving the two
boundary conditions for c1, c2 gives, AT THE CURRENT INSTANT (t=0):

    a_cmd = -6*x0/T_go^2 - 4*v0/T_go

where x0 is the current position error (current - target) and v0 is
the current velocity error. This is re-solved every guidance cycle
with the current T_go and current errors (same re-solve-every-cycle
architecture as guidance/peg.py), which is what makes it closed-loop
and dispersion-rejecting rather than a fixed open-loop profile. This
exact formula is verified against a direct double-integrator
simulation in tests/test_reusability.py.

For an actual thrust command, gravity must be added back: if a_cmd is
the TOTAL acceleration needed (thrust + gravity) to null the boundary
conditions, then the THRUST acceleration to command is
    a_thrust = a_cmd - g_vector.

Zero external dependencies beyond NumPy - CPU-only.
"""

import numpy as np


def zem_zev_acceleration_command(current_pos: np.ndarray, current_vel: np.ndarray,
                                 target_pos: np.ndarray, target_vel: np.ndarray,
                                 t_go_s: float) -> np.ndarray:
    """
    Minimum-energy acceleration command (vectorized over any number of
    axes) to drive position and velocity to the target at time t_go_s.
    This is the TOTAL acceleration (thrust + gravity); subtract the
    local gravity vector to get the thrust-only acceleration to command.
    """
    t_go_s = max(t_go_s, 0.05)   # guard against divide-by-zero as t_go -> 0
    x0 = np.asarray(current_pos, dtype=float) - np.asarray(target_pos, dtype=float)
    v0 = np.asarray(current_vel, dtype=float) - np.asarray(target_vel, dtype=float)
    return -6.0 * x0 / t_go_s ** 2 - 4.0 * v0 / t_go_s


def zem(current_pos, current_vel, target_pos, t_go_s: float) -> np.ndarray:
    """Zero-Effort-Miss: predicted position error at T_go if no further acceleration is applied."""
    x0 = np.asarray(current_pos, dtype=float) - np.asarray(target_pos, dtype=float)
    v0 = np.asarray(current_vel, dtype=float)
    return x0 + v0 * t_go_s


def zev(current_vel, target_vel) -> np.ndarray:
    """Zero-Effort-Velocity: current velocity error (no further acceleration applied changes nothing)."""
    return np.asarray(current_vel, dtype=float) - np.asarray(target_vel, dtype=float)


def suicide_burn_ignition_altitude_m(descent_speed_m_s: float, max_deceleration_m_s2: float,
                                     safety_margin_m: float = 50.0) -> float:
    """
    Classic "hoverslam"/suicide-burn ignition-altitude kinematics: the
    altitude at which the engine must ignite to decelerate from the
    current descent speed to zero, given the vehicle's maximum
    available deceleration (thrust/mass - g, at full throttle), using
    v^2 = 2*a*d -> d = v^2/(2a). A real flight computer adds margin
    for engine spool-up lag and guidance dispersion (safety_margin_m);
    igniting exactly at the kinematic minimum leaves zero margin for
    error, which is why every real suicide-burn implementation adds one.
    """
    if max_deceleration_m_s2 <= 0:
        return float("inf")
    return (descent_speed_m_s ** 2) / (2.0 * max_deceleration_m_s2) + safety_margin_m


def estimate_time_to_go_s(remaining_distance_m: float, closing_speed_m_s: float,
                         t_go_floor_s: float = 3.0) -> float:
    """
    Kinematic time-to-go re-estimate, using the AVERAGE-VELOCITY form
    T_go = 2*distance/closing_speed (i.e. assuming closing speed decays
    roughly linearly from its current value toward zero over the
    remaining distance, so average speed ~= closing_speed/2) rather
    than the naive constant-velocity form (distance/closing_speed).
    The naive form badly UNDERESTIMATES T_go whenever the vehicle is
    about to decelerate hard (which is exactly the powered-descent
    case): it assumes the vehicle keeps falling at its current, not-
    yet-slowed speed the whole way down, which demands far more thrust
    than is actually available or needed and can saturate the engine
    (verified against a direct feasibility check in
    tests/test_reusability.py). Floored at t_go_floor_s to avoid the
    classical ZEM/ZEV terminal singularity as T->0 (the closed-form
    law's gains, 6/T^2 and 4/T, blow up otherwise).
    """
    if abs(closing_speed_m_s) < 1e-6:
        return t_go_floor_s * 10.0   # not closing at all yet -> treat as far off
    return max(2.0 * remaining_distance_m / abs(closing_speed_m_s), t_go_floor_s)


def simulate_powered_landing(
    r0_m: np.ndarray, v0_m_s: np.ndarray, m0_kg: float, m_dry_kg: float,
    max_thrust_n: float, min_throttle_frac: float, isp_s: float,
    target_pos_m: np.ndarray, target_vel_m_s: np.ndarray,
    gravity_m_s2: np.ndarray, altitude_axis: int = -1,
    t_go_floor_s: float = 3.0, dt: float = 0.02, guidance_cycle_s: float = 0.1,
    max_sim_time_s: float = 120.0,
):
    """
    Closed-loop ZEM/ZEV powered-descent burn (2D or 3D — any dimension
    current_pos/vel support), re-solving the guidance law every
    guidance_cycle_s using a kinematically RE-ESTIMATED time-to-go
    (estimate_time_to_go_s) rather than a fixed countdown clock — this
    avoids the classical ZEM/ZEV terminal-singularity blow-up (see that
    function's docstring) and matches how real implementations handle
    it. Touchdown is ALTITUDE-triggered (target reached along
    `altitude_axis`), not time-triggered.

    Thrust is clamped to [min_throttle_frac, 1.0] * max_thrust_n, a
    real deep-throttle engine constraint — a fixed-thrust engine CANNOT
    fly ZEM/ZEV exactly, since the commanded acceleration continuously
    varies, which is exactly why real reusable-rocket engines (Merlin,
    BE-3) are designed for deep throttling.

    Returns the full time series plus final touchdown state (position,
    velocity, mass) and constraint-violation flags.
    """
    g0 = 9.80665
    pos = np.asarray(r0_m, dtype=float).copy()
    vel = np.asarray(v0_m_s, dtype=float).copy()
    target_pos_m = np.asarray(target_pos_m, dtype=float)
    target_vel_m_s = np.asarray(target_vel_m_s, dtype=float)
    gravity_m_s2 = np.asarray(gravity_m_s2, dtype=float)
    mass = m0_kg
    t = 0.0

    t_hist, pos_hist, vel_hist, mass_hist, throttle_hist, t_go_hist = [], [], [], [], [], []
    thrust_saturated_low = False
    thrust_saturated_high = False
    propellant_exhausted = False

    while t < max_sim_time_s and mass > m_dry_kg:
        remaining_alt = pos[altitude_axis] - target_pos_m[altitude_axis]
        if remaining_alt <= 0:
            break
        closing_speed = vel[altitude_axis] - target_vel_m_s[altitude_axis]
        t_go = estimate_time_to_go_s(remaining_alt, closing_speed, t_go_floor_s)

        a_total_cmd = zem_zev_acceleration_command(pos, vel, target_pos_m, target_vel_m_s, t_go)
        a_thrust_cmd = a_total_cmd - gravity_m_s2
        thrust_accel_mag_needed = np.linalg.norm(a_thrust_cmd)
        thrust_n_needed = thrust_accel_mag_needed * mass

        throttle_frac = np.clip(thrust_n_needed / max_thrust_n, min_throttle_frac, 1.0)
        if thrust_n_needed / max_thrust_n < min_throttle_frac:
            thrust_saturated_low = True
        if thrust_n_needed / max_thrust_n > 1.0:
            thrust_saturated_high = True
        actual_thrust_n = throttle_frac * max_thrust_n
        direction = a_thrust_cmd / max(thrust_accel_mag_needed, 1e-9)
        thrust_accel_actual = direction * (actual_thrust_n / mass)

        n_sub = max(1, int(round(guidance_cycle_s / dt)))
        for _ in range(n_sub):
            if mass <= m_dry_kg:
                propellant_exhausted = True
                break
            total_accel = thrust_accel_actual + gravity_m_s2
            vel = vel + total_accel * dt
            pos = pos + vel * dt
            mass = mass - (actual_thrust_n / (isp_s * g0)) * dt
            t += dt
            t_hist.append(t); pos_hist.append(pos.copy()); vel_hist.append(vel.copy())
            mass_hist.append(mass); throttle_hist.append(throttle_frac); t_go_hist.append(t_go)
            if pos[altitude_axis] - target_pos_m[altitude_axis] <= 0:
                break
        if propellant_exhausted or pos[altitude_axis] - target_pos_m[altitude_axis] <= 0:
            break

    return {
        "t": np.array(t_hist), "position_m": np.array(pos_hist), "velocity_m_s": np.array(vel_hist),
        "mass_kg": np.array(mass_hist), "throttle_frac": np.array(throttle_hist),
        "t_go_estimate_s": np.array(t_go_hist),
        "final_position_m": pos, "final_velocity_m_s": vel, "final_mass_kg": mass,
        "touchdown_speed_m_s": float(np.linalg.norm(vel)),
        "miss_distance_m": float(np.linalg.norm(pos - target_pos_m)),
        "thrust_saturated_low": thrust_saturated_low, "thrust_saturated_high": thrust_saturated_high,
        "propellant_exhausted": propellant_exhausted,
    }
