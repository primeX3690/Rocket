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

Frame: 2D point mass in a vertical plane on a SPHERICAL Earth. State is
(altitude y, tangential inertial velocity vx, radial velocity vy); the
curvature terms (+vx^2/r radial, -vx*vy/r tangential) are included, so
the (r = R+y, v = |(vx,vy)|, gamma = atan2(vy, vx)) conversion used to hand
the state to PEG is exact in this model.  Optional extras:
  * Earth rotation (launch latitude + azimuth): vehicle starts with the
    pad's inertial velocity; the atmosphere co-rotates, so drag and the
    zero-angle-of-attack gravity turn use AIR-RELATIVE velocity.
  * Mach-dependent drag coefficient (transonic drag rise).
Not modelled: winds, lift/angle-of-attack, 3D out-of-plane motion.

Zero external dependencies beyond NumPy — CPU-only.
"""

import numpy as np
from guidance.atmosphere import density, dynamic_pressure, temperature

G0 = 9.80665
R_EARTH = 6378137.0


OMEGA_EARTH = 7.2921159e-5   # rad/s
GAMMA_AIR = 1.4
R_AIR_SPECIFIC = 287.05287

# Generic slender-launcher drag-rise multipliers on the subsonic Cd
# (transonic peak ~1.7-1.8x, decaying through supersonic).  Shape is a
# textbook-style approximation, NOT wind-tunnel data for a specific vehicle.
_MACH_TABLE = np.array([0.0, 0.8, 1.0, 1.2, 1.6, 2.5, 4.0, 8.0])
_CD_FACTOR = np.array([1.0, 1.0, 1.6, 1.75, 1.5, 1.15, 0.95, 0.85])


def speed_of_sound(altitude_m: float) -> float:
    return float(np.sqrt(GAMMA_AIR * R_AIR_SPECIFIC * temperature(max(altitude_m, 0.0))))


def cd_mach_factor(mach: float) -> float:
    """Multiplier applied to the subsonic drag coefficient at a given Mach number."""
    return float(np.interp(mach, _MACH_TABLE, _CD_FACTOR))


def pad_rotation_speed(latitude_deg: float, azimuth_deg: float = 90.0) -> float:
    """Inertial speed of the launch pad along the launch azimuth (m/s)."""
    return OMEGA_EARTH * R_EARTH * np.cos(np.radians(latitude_deg)) * np.sin(np.radians(azimuth_deg))



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
    mach_dependent_drag: bool = False,
    launch_latitude_deg: float = None,
    launch_azimuth_deg: float = 90.0,
    curvature: bool = True,
):
    """
    Integrate a full gravity-turn ascent burn (single stage, point mass,
    2D vertical plane, spherical Earth) using semi-implicit Euler at
    dt=0.05 s (convergence-checked in tests).

    mach_dependent_drag: scale `drag_coeff` (subsonic value) by the Mach
        drag-rise table.
    launch_latitude_deg: if given, include Earth rotation (pad inertial
        speed along `launch_azimuth_deg`); drag/AoA use air-relative velocity.
    curvature: include the spherical-Earth terms (default True).

    Returns time series (`vx`/`vy` are INERTIAL tangential/radial velocity)
    plus summary stats (max-Q, burnout state, kick time...).
    """
    mdot = thrust_n / (isp_s * G0)
    v_pad = 0.0 if launch_latitude_deg is None else pad_rotation_speed(launch_latitude_deg, launch_azimuth_deg)

    state = VehicleState(mass_kg=m0_kg, pitch_rad=np.pi / 2, vx=v_pad, vy=0.0)
    kick_applied = False

    t_hist, alt_hist, dr_hist = [], [], []
    v_hist, q_hist, mass_hist, pitch_hist = [], [], [], []
    vx_hist, vy_hist, mach_hist = [], [], []

    t = 0.0
    max_q, max_q_time, kick_time = 0.0, 0.0, None

    while t < t_max and state.mass > m_dry_kg and state.altitude >= -1e-6:
        r = R_EARTH + state.altitude
        # air-relative velocity (atmosphere co-rotates with Earth)
        vrot_local = v_pad * r / R_EARTH
        vxr, vyr = state.vx - vrot_local, state.vy
        speed = np.sqrt(vxr ** 2 + vyr ** 2)              # air-relative speed

        kick_applied_new, _ = gravity_turn_pitch_command(
            state, kick_altitude_m, np.radians(90.0 - kick_angle_deg), kick_applied
        )
        if kick_applied_new and not kick_applied:
            kick_time = t
            kick_rad = np.radians(kick_angle_deg)
            if speed > 1e-6:
                state.vx = vrot_local + speed * np.sin(kick_rad)
                state.vy = speed * np.cos(kick_rad)
            else:
                state.vx = vrot_local + 0.1
            vxr, vyr = state.vx - vrot_local, state.vy
            speed = np.sqrt(vxr ** 2 + vyr ** 2)
        kick_applied = kick_applied_new

        g = local_gravity(state.altitude)
        h_eval = max(state.altitude, 0.0)
        rho = density(h_eval)
        q = dynamic_pressure(h_eval, speed)
        mach = speed / speed_of_sound(h_eval)
        cd = drag_coeff * (cd_mach_factor(mach) if mach_dependent_drag else 1.0)
        drag = 0.5 * rho * speed ** 2 * cd * ref_area_m2

        if speed > 1e-6:
            drag_x, drag_y = -drag * vxr / speed, -drag * vyr / speed
        else:
            drag_x = drag_y = 0.0

        if not kick_applied or speed <= 1e-6:
            thrust_x, thrust_y = 0.0, thrust_n
        else:
            thrust_x, thrust_y = thrust_n * vxr / speed, thrust_n * vyr / speed

        ax = (thrust_x + drag_x) / state.mass
        ay = (thrust_y + drag_y) / state.mass - g
        if curvature:
            ax -= state.vx * state.vy / r
            ay += state.vx ** 2 / r

        state.vx += ax * dt
        state.vy += ay * dt
        state.downrange += state.vx * (R_EARTH / r) * dt
        state.altitude += state.vy * dt
        state.mass -= mdot * dt

        if q > max_q:
            max_q, max_q_time = q, t

        t += dt
        t_hist.append(t)
        alt_hist.append(state.altitude)
        dr_hist.append(state.downrange)
        v_hist.append(np.sqrt(state.vx ** 2 + state.vy ** 2))
        vx_hist.append(state.vx)
        vy_hist.append(state.vy)
        q_hist.append(q)
        mach_hist.append(mach)
        mass_hist.append(state.mass)
        pitch_hist.append(np.degrees(np.arctan2(state.vy, state.vx)) if v_hist[-1] > 1e-6 else 90.0)

    return {
        "t": np.array(t_hist),
        "altitude_m": np.array(alt_hist),
        "downrange_m": np.array(dr_hist),
        "speed_m_s": np.array(v_hist),
        "vx_m_s": np.array(vx_hist),
        "vy_m_s": np.array(vy_hist),
        "mach": np.array(mach_hist),
        "dynamic_pressure_pa": np.array(q_hist),
        "mass_kg": np.array(mass_hist),
        "pitch_deg": np.array(pitch_hist),
        "max_q_pa": max_q,
        "max_q_time_s": max_q_time,
        "kick_time_s": kick_time,
        "burnout_altitude_m": alt_hist[-1] if alt_hist else 0.0,
        "burnout_speed_m_s": v_hist[-1] if v_hist else 0.0,
        "burnout_vx_m_s": vx_hist[-1] if vx_hist else 0.0,
        "burnout_vy_m_s": vy_hist[-1] if vy_hist else 0.0,
        "burnout_mass_kg": mass_hist[-1] if mass_hist else m0_kg,
        "burn_time_s": t_hist[-1] if t_hist else 0.0,
    }
