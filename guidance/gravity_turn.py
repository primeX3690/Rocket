"""
guidance/gravity_turn.py

2D point-mass ascent trajectory simulator with gravity-turn guidance —
the classical open-loop ascent guidance law used by real launch vehicles
(PSLV, Falcon 9, Soyuz all fly a gravity-turn-based first-stage profile).

Gravity-turn logic:
    Phase 1 (vertical rise): fly straight up until a small "kick" pitch
        angle is applied at a chosen altitude/velocity.
    Phase 2 (gravity turn): after the kick, the vehicle flies at zero
        angle-of-attack — meaning gravity alone rotates the velocity
        vector towards horizontal; no active steering torque is needed,
        which minimizes structural bending loads. This is *why* real
        rockets use this law instead of a manually-flown pitch program.

Frame: 2D, flat-Earth-with-curvature-ignored point mass in a vertical
plane (downrange x, altitude y). Good enough for ascent-phase GNC design
and max-Q / staging-event analysis; full 3D + Earth rotation is a later
extension, not needed for this module's purpose.

Zero external dependencies beyond NumPy — CPU-only.
"""

import numpy as np
from guidance.atmosphere import density, dynamic_pressure

G0 = 9.80665
R_EARTH = 6378137.0


def local_gravity(altitude_m: float) -> float:
    """Inverse-square gravity fall-off with altitude (not flat g0)."""
    r = R_EARTH + altitude_m
    return G0 * (R_EARTH / r) ** 2


class VehicleState:
    """Mutable simulation state for the ascent integrator."""
    def __init__(self, mass_kg, altitude_m=0.0, downrange_m=0.0,
                 vx=0.0, vy=0.0, pitch_rad=np.pi / 2):
        self.mass = mass_kg
        self.altitude = altitude_m
        self.downrange = downrange_m
        self.vx = vx           # horizontal velocity, m/s
        self.vy = vy           # vertical velocity, m/s
        self.pitch = pitch_rad  # flight-path/velocity-vector angle from horizontal
                                 # (pi/2 = straight up)


def gravity_turn_pitch_command(state: VehicleState, kick_altitude_m: float,
                                kick_angle_rad: float, kick_applied: bool):
    """
    Returns the guidance decision for this timestep:
      - whether to apply the kick pitch-over now
      - the commanded flight-path angle during the vertical-rise phase

    After the kick is applied, this module does NOT further command pitch —
    that's the defining feature of a gravity turn: guidance goes passive
    and lets gravity + zero-AoA aerodynamics rotate the vehicle. The
    integrator (below) handles that physics directly.
    """
    if not kick_applied and state.altitude >= kick_altitude_m:
        return True, kick_angle_rad
    return kick_applied, None


def simulate_ascent(
    m0_kg: float,
    m_dry_kg: float,
    thrust_n: float,
    isp_s: float,
    drag_coeff: float,
    ref_area_m2: float,
    kick_altitude_m: float = 1000.0,
    kick_angle_deg: float = 1.0,
    dt: float = 0.05,
    t_max: float = 200.0,
):
    """
    Integrate a full gravity-turn ascent burn (single stage, point mass,
    2D vertical plane) using explicit Euler-Cromer integration (stable
    and simple — sufficient timestep resolution at dt=0.05s for this
    class of problem, verified via convergence check in tests).

    Returns a dict of time-series arrays plus summary stats (max-Q,
    burnout velocity/altitude/mass, time to kick, etc.)
    """
    mdot = thrust_n / (isp_s * G0)   # constant mass flow assumption

    state = VehicleState(mass_kg=m0_kg, pitch_rad=np.pi / 2)
    kick_applied = False

    t_hist, alt_hist, dr_hist = [], [], []
    v_hist, q_hist, mass_hist, pitch_hist = [], [], [], []

    t = 0.0
    max_q = 0.0
    max_q_time = 0.0
    kick_time = None

    while t < t_max and state.mass > m_dry_kg and state.altitude >= -1e-6:
        speed = np.sqrt(state.vx ** 2 + state.vy ** 2)

        # --- guidance decision ---
        kick_applied_new, kick_angle = gravity_turn_pitch_command(
            state, kick_altitude_m, np.radians(90.0 - kick_angle_deg), kick_applied
        )
        if kick_applied_new and not kick_applied:
            kick_time = t
            # Apply the kick: rotate velocity vector by kick_angle_deg from vertical,
            # keeping speed the same (a small instantaneous pitch-over impulse,
            # standard idealization for gravity-turn initiation)
            kick_rad = np.radians(kick_angle_deg)
            state.vx = speed * np.sin(kick_rad) if speed > 0 else 0.1  # small kick velocity if static
            state.vy = speed * np.cos(kick_rad) if speed > 0 else 0.0
            if speed == 0:
                # still on the pad / just past it with negligible speed —
                # give a nominal small horizontal seed velocity so the
                # zero-AoA gravity turn has something to rotate
                state.vx = 0.1
        kick_applied = kick_applied_new

        # --- forces ---
        g = local_gravity(state.altitude)
        rho = density(max(state.altitude, 0.0))
        q = dynamic_pressure(max(state.altitude, 0.0), speed)
        drag = 0.5 * rho * speed ** 2 * drag_coeff * ref_area_m2

        if speed > 1e-6:
            drag_x = -drag * (state.vx / speed)
            drag_y = -drag * (state.vy / speed)
        else:
            drag_x = drag_y = 0.0

        if not kick_applied:
            # vertical rise phase: thrust straight up, zero AoA trivially satisfied
            thrust_x, thrust_y = 0.0, thrust_n
        else:
            # gravity-turn phase: thrust aligned with current velocity vector
            # (zero angle-of-attack assumption — the defining gravity-turn law)
            if speed > 1e-6:
                thrust_x = thrust_n * (state.vx / speed)
                thrust_y = thrust_n * (state.vy / speed)
            else:
                thrust_x, thrust_y = 0.0, thrust_n

        ax = (thrust_x + drag_x) / state.mass
        ay = (thrust_y + drag_y) / state.mass - g

        # --- integrate (semi-implicit Euler) ---
        state.vx += ax * dt
        state.vy += ay * dt
        state.downrange += state.vx * dt
        state.altitude += state.vy * dt
        state.mass -= mdot * dt

        if q > max_q:
            max_q = q
            max_q_time = t

        t += dt
        t_hist.append(t)
        alt_hist.append(state.altitude)
        dr_hist.append(state.downrange)
        v_hist.append(np.sqrt(state.vx ** 2 + state.vy ** 2))
        q_hist.append(q)
        mass_hist.append(state.mass)
        pitch_hist.append(np.degrees(np.arctan2(state.vy, state.vx)) if speed > 1e-6 else 90.0)

    return {
        "t": np.array(t_hist),
        "altitude_m": np.array(alt_hist),
        "downrange_m": np.array(dr_hist),
        "speed_m_s": np.array(v_hist),
        "dynamic_pressure_pa": np.array(q_hist),
        "mass_kg": np.array(mass_hist),
        "pitch_deg": np.array(pitch_hist),
        "max_q_pa": max_q,
        "max_q_time_s": max_q_time,
        "kick_time_s": kick_time,
        "burnout_altitude_m": alt_hist[-1] if alt_hist else 0.0,
        "burnout_speed_m_s": v_hist[-1] if v_hist else 0.0,
        "burnout_mass_kg": mass_hist[-1] if mass_hist else m0_kg,
        "burn_time_s": t_hist[-1] if t_hist else 0.0,
    }
