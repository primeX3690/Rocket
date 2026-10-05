"""
data/reference_vehicle.py

A generic small two-stage launch vehicle used as the DEFAULT example
throughout this repo (app.py, tests, README examples).  Parameters are
representative order-of-magnitude figures for a PSLV-class small
satellite launcher — not a model of any specific real vehicle, and not
sourced from any proprietary data.  Chosen so the full mission
(gravity-turn ascent -> coast -> PEG insertion) actually converges to
the target orbit; see analysis/mission_profile.py.
"""

from guidance.peg import PEGTarget
import numpy as np

R_EARTH = 6378137.0
MU_EARTH = 3.986004418e14

REFERENCE_VEHICLE = {
    "stage1_m0_kg": 60000.0,
    "stage1_m_dry_kg": 12000.0,
    "stage1_thrust_n": 850000.0,
    "stage1_isp_s": 285.0,
    "stage1_drag_coeff": 0.3,
    "stage1_ref_area_m2": 2.0,
    "stage1_kick_altitude_m": 1000.0,
    "stage1_kick_angle_deg": 6.0,
    "coast_duration_s": 15.0,
    "stage2_m0_kg": 15000.0,
    "stage2_m_dry_kg": 2500.0,
    "stage2_thrust_n": 220000.0,
    "stage2_isp_s": 340.0,
    "launch_latitude_deg": 13.7,     # Sriharikota-class low-latitude coastal pad
    "launch_azimuth_deg": 90.0,
    "target_altitude_km": 300.0,
}


def reference_target(altitude_km: float = None) -> PEGTarget:
    alt_km = REFERENCE_VEHICLE["target_altitude_km"] if altitude_km is None else altitude_km
    r = R_EARTH + alt_km * 1000.0
    return PEGTarget(r, np.sqrt(MU_EARTH / r), 0.0)


def run_reference_mission(**overrides):
    """Run simulate_two_stage_mission with REFERENCE_VEHICLE defaults, any overridden by kwargs."""
    from analysis.mission_profile import simulate_two_stage_mission
    p = dict(REFERENCE_VEHICLE)
    p.update(overrides)
    target = overrides.get("target", reference_target(p["target_altitude_km"]))
    return simulate_two_stage_mission(
        p["stage1_m0_kg"], p["stage1_m_dry_kg"], p["stage1_thrust_n"], p["stage1_isp_s"],
        p["stage1_drag_coeff"], p["stage1_ref_area_m2"],
        p["stage1_kick_altitude_m"], p["stage1_kick_angle_deg"],
        p["coast_duration_s"],
        p["stage2_m0_kg"], p["stage2_thrust_n"], p["stage2_isp_s"], target,
        stage2_m_dry_kg=p["stage2_m_dry_kg"],
        launch_latitude_deg=p["launch_latitude_deg"], launch_azimuth_deg=p["launch_azimuth_deg"],
    )
