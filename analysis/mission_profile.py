"""
analysis/mission_profile.py

End-to-end two-stage ascent-to-orbit: stage-1 open-loop gravity-turn
(guidance/gravity_turn.py) -> ballistic coast (separation / ignition
delay) -> stage-2 closed-loop PEG insertion burn (guidance/peg.py).

Frame handoff: the stage-1 integrator works on a spherical Earth with
tangential/radial inertial velocity (vx, vy), so the conversion to PEG's
polar state  r = R + h,  v = |(vx, vy)|,  gamma = atan2(vy, vx)  is exact
in this model.  Optional Earth rotation and Mach-dependent drag are passed
through to stage 1.

Plausibility (NOT telemetry match) is checked against generic public
order-of-magnitude values in data/reference_missions.py; the vehicle in
data/reference_vehicle.py is a generic small two-stage launcher, not a
model of any specific rocket.

Zero external dependencies beyond NumPy - CPU-only.
"""

import numpy as np
from guidance.gravity_turn import simulate_ascent
from guidance.peg import PEGTarget, simulate_peg_guided_burn, rk4_step
from guidance.orbit_utils import orbit_from_state

R_EARTH = 6378137.0
MU_EARTH = 3.986004418e14
G0 = 9.80665


def _flat_frame_to_polar(altitude_m, vx, vy):
    """
    Conversion from the stage-1 integrator's (altitude, tangential vx,
    radial vy) to PEG's polar frame (radius, speed, flight-path angle).
    Exact for the spherical-Earth point-mass model in gravity_turn.py.
    """
    r = R_EARTH + altitude_m
    v = np.sqrt(vx ** 2 + vy ** 2)
    gamma = np.arctan2(vy, vx) if v > 1e-6 else np.radians(90.0)
    return r, v, gamma


def simulate_coast_phase(altitude0_m, downrange0_m, vx0, vy0, duration_s, dt=0.1):
    """
    Ballistic coast on a spherical Earth (inverse-square gravity, RK4,
    no drag - staging happens above the sensible atmosphere).
    Returns (altitude, downrange, vx, vy) at the end of the coast.
    """
    r, v, gamma = _flat_frame_to_polar(altitude0_m, vx0, vy0)
    downrange = downrange0_m
    n = max(1, int(round(duration_s / dt))) if duration_s > 0 else 0
    h = duration_s / n if n else 0.0
    state = (r, v, gamma, 1.0)
    for _ in range(n):
        vx_a = state[1] * np.cos(state[2])
        state = rk4_step(state, 0.0, 0.0, 0.0, h, MU_EARTH)
        vx_b = state[1] * np.cos(state[2])
        downrange += 0.5 * (vx_a + vx_b) * (R_EARTH / state[0]) * h
    r, v, gamma, _ = state
    return r - R_EARTH, downrange, v * np.cos(gamma), v * np.sin(gamma)


def simulate_two_stage_mission(
    # Stage 1 (ascent, gravity-turn)
    stage1_m0_kg, stage1_m_dry_kg, stage1_thrust_n, stage1_isp_s,
    stage1_drag_coeff, stage1_ref_area_m2,
    stage1_kick_altitude_m, stage1_kick_angle_deg,
    # Coast
    coast_duration_s,
    # Stage 2 (orbit insertion, PEG)
    stage2_m0_kg, stage2_thrust_n, stage2_isp_s,
    target: PEGTarget,
    dt_stage1=0.05, dt_stage2=0.05,
    stage2_m_dry_kg=None,
    launch_latitude_deg=None, launch_azimuth_deg=90.0,
    mach_dependent_drag=True,
):
    """
    Run the complete mission.  `stage2_m0_kg` is given separately from
    stage 1's burnout mass because stage-1 structure is jettisoned.

    stage2_m_dry_kg: stage-2 burnout mass floor (default 20 % of stage2_m0).
    Returns the combined trajectory plus mission-level stats including the
    achieved orbit (perigee/apogee).
    """
    stage1 = simulate_ascent(
        m0_kg=stage1_m0_kg, m_dry_kg=stage1_m_dry_kg,
        thrust_n=stage1_thrust_n, isp_s=stage1_isp_s,
        drag_coeff=stage1_drag_coeff, ref_area_m2=stage1_ref_area_m2,
        kick_altitude_m=stage1_kick_altitude_m, kick_angle_deg=stage1_kick_angle_deg,
        dt=dt_stage1, t_max=400.0, mach_dependent_drag=mach_dependent_drag,
        launch_latitude_deg=launch_latitude_deg, launch_azimuth_deg=launch_azimuth_deg,
    )

    s1_alt = stage1["burnout_altitude_m"]
    s1_dr = stage1["downrange_m"][-1] if len(stage1["downrange_m"]) else 0.0
    vx0, vy0 = stage1["burnout_vx_m_s"], stage1["burnout_vy_m_s"]

    alt_c, dr_c, vx_c, vy_c = simulate_coast_phase(s1_alt, s1_dr, vx0, vy0, coast_duration_s)
    r2, v2, gamma2 = _flat_frame_to_polar(alt_c, vx_c, vy_c)

    if stage2_m_dry_kg is None:
        stage2_m_dry_kg = 0.2 * stage2_m0_kg

    stage2 = simulate_peg_guided_burn(
        r0_m=r2, v0_m_s=v2, gamma0_rad=gamma2, m0_kg=stage2_m0_kg,
        thrust_n=stage2_thrust_n, isp_s=stage2_isp_s, target=target,
        dt_integrate=dt_stage2, m_dry_kg=stage2_m_dry_kg, max_time_s=900.0,
    )

    total_mission_time_s = stage1["burn_time_s"] + coast_duration_s + stage2["burn_time_s"]
    return {
        "stage1": stage1,
        "coast_end_altitude_m": alt_c,
        "coast_end_downrange_m": dr_c,
        "stage2_ignition_state": (r2, v2, gamma2),
        "stage2": stage2,
        "total_mission_time_s": total_mission_time_s,
        "final_altitude_km": (stage2["final_radius_m"] - R_EARTH) / 1000.0,
        "final_velocity_m_s": stage2["final_velocity_m_s"],
        "insertion_error": stage2["insertion_error"],
        "orbit": stage2["orbit"],
        "stage1_burnout_altitude_km": s1_alt / 1000.0,
        "stage1_burnout_speed_m_s": stage1["burnout_speed_m_s"],
        "stage1_burn_time_s": stage1["burn_time_s"],
        "stage1_max_q_pa": stage1["max_q_pa"],
        "stage1_max_mach": float(np.max(stage1["mach"])) if len(stage1["mach"]) else 0.0,
    }
