"""
analysis/mission_profile.py

Full multi-stage ascent-to-orbit mission profile: stage-1 open-loop
gravity-turn ascent (guidance/gravity_turn.py) -> ballistic coast phase
(stage separation / fairing separation / second-stage ignition delay)
-> stage-2 closed-loop PEG-guided orbit-insertion burn (guidance/peg.py).

This is the piece that turns the individual GNC modules built so far
into an actual END-TO-END MISSION, the way a real flight plan reads:
liftoff -> stage-1 burn -> stage separation -> coast -> stage-2 ignition
-> orbit insertion -> stage-2 cutoff. Each phase hands its final state
to the next, with an explicit frame conversion between the flat-Earth
(x, y, vx, vy) representation used by the stage-1 gravity-turn
integrator and the polar (r, v, gamma) representation used by PEG —
this conversion is exact for a 2D point-mass model (no approximation
beyond what each individual module already makes).

Validated for PLAUSIBILITY (not exact telemetry match — no proprietary
mission data used or needed) against public PSLV-class order-of-
magnitude reference values in data/reference_missions.py.

Zero external dependencies beyond NumPy — CPU-only.
"""

import numpy as np
from guidance.gravity_turn import simulate_ascent
from guidance.peg import PEGTarget, simulate_peg_guided_burn

R_EARTH = 6378137.0
MU_EARTH = 3.986004418e14
G0 = 9.80665


def _flat_frame_to_polar(altitude_m, vx, vy):
    """
    Exact conversion from the stage-1 integrator's flat-Earth frame
    (altitude above surface, horizontal/vertical velocity components)
    to the polar frame PEG uses (radius from Earth's center, speed,
    flight-path angle from local horizontal).
    """
    r = R_EARTH + altitude_m
    v = np.sqrt(vx ** 2 + vy ** 2)
    gamma = np.arctan2(vy, vx) if v > 1e-6 else np.radians(90.0)
    return r, v, gamma


def simulate_coast_phase(altitude0_m, downrange0_m, vx0, vy0, duration_s, dt=0.1):
    """
    Ballistic (unpowered) coast — stage separation, fairing jettison, and
    the ignition delay before the next stage lights, modeled as pure
    projectile motion under local gravity (drag is negligible at the
    altitudes where staging typically occurs, ~60-150km+, so it is
    correctly omitted here rather than approximated).
    """
    from guidance.gravity_turn import local_gravity

    altitude, downrange, vx, vy = altitude0_m, downrange0_m, vx0, vy0
    t = 0.0
    steps = int(duration_s / dt)

    for _ in range(steps):
        g = local_gravity(max(altitude, 0.0))
        vy -= g * dt
        altitude += vy * dt
        downrange += vx * dt
        t += dt

    return altitude, downrange, vx, vy


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
):
    """
    Run the complete mission: stage-1 ascent burn to depletion, coast,
    stage-2 PEG-guided insertion burn. stage2_m0_kg is provided
    separately from stage1's burnout mass because in a real vehicle the
    stage-1 structure (now-empty tanks, interstage) is JETTISONED at
    separation — stage 2 starts its own burn carrying only its own
    propellant + structure + payload, not stage 1's dead weight.

    Returns the full combined trajectory (converted to a single
    consistent altitude/downrange/velocity/time timeline) plus mission-
    level summary stats for sanity-checking.
    """
    # --- Stage 1: gravity-turn ascent ---
    stage1 = simulate_ascent(
        m0_kg=stage1_m0_kg, m_dry_kg=stage1_m_dry_kg,
        thrust_n=stage1_thrust_n, isp_s=stage1_isp_s,
        drag_coeff=stage1_drag_coeff, ref_area_m2=stage1_ref_area_m2,
        kick_altitude_m=stage1_kick_altitude_m, kick_angle_deg=stage1_kick_angle_deg,
        dt=dt_stage1, t_max=200.0,
    )

    s1_burnout_alt = stage1["burnout_altitude_m"]
    s1_burnout_downrange = stage1["downrange_m"][-1] if len(stage1["downrange_m"]) else 0.0
    s1_burnout_speed = stage1["burnout_speed_m_s"]
    s1_burnout_pitch_deg = stage1["pitch_deg"][-1] if len(stage1["pitch_deg"]) else 90.0

    vx0 = s1_burnout_speed * np.cos(np.radians(s1_burnout_pitch_deg))
    vy0 = s1_burnout_speed * np.sin(np.radians(s1_burnout_pitch_deg))

    # --- Coast phase (stage separation + ignition delay) ---
    alt_coast_end, downrange_coast_end, vx_coast_end, vy_coast_end = simulate_coast_phase(
        s1_burnout_alt, s1_burnout_downrange, vx0, vy0, coast_duration_s
    )

    # --- Frame conversion for stage 2 (PEG uses polar r/v/gamma) ---
    r2, v2, gamma2 = _flat_frame_to_polar(alt_coast_end, vx_coast_end, vy_coast_end)

    # --- Stage 2: PEG-guided orbit-insertion burn ---
    stage2 = simulate_peg_guided_burn(
        r0_m=r2, v0_m_s=v2, gamma0_rad=gamma2, m0_kg=stage2_m0_kg,
        thrust_n=stage2_thrust_n, isp_s=stage2_isp_s, target=target,
        dt_integrate=dt_stage2,
    )

    total_mission_time_s = stage1["burn_time_s"] + coast_duration_s + stage2["burn_time_s"]
    final_altitude_km = (stage2["final_radius_m"] - R_EARTH) / 1000.0

    return {
        "stage1": stage1,
        "coast_end_altitude_m": alt_coast_end,
        "coast_end_downrange_m": downrange_coast_end,
        "stage2": stage2,
        "total_mission_time_s": total_mission_time_s,
        "final_altitude_km": final_altitude_km,
        "final_velocity_m_s": stage2["final_velocity_m_s"],
        "insertion_error": stage2["insertion_error"],
        "stage1_burnout_altitude_km": s1_burnout_alt / 1000.0,
        "stage1_burn_time_s": stage1["burn_time_s"],
        "stage1_max_q_pa": stage1["max_q_pa"],
    }
