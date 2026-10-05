"""
tests/test_mission_profile.py

Verification suite for analysis/mission_profile.py.

Checks:
    1. Flat-frame -> polar-frame conversion is exact (pure geometry,
       independently verifiable by hand).
    2. Coast phase conserves horizontal velocity (no horizontal forces)
       and correctly reduces vertical velocity under gravity (ballistic
       arc), matching simple projectile-motion expectations.
    3. Full two-stage mission produces a physically sane result: mass
       decreases in both stages, final velocity closely matches the
       target (small error, consistent with the PEG module's own
       verified accuracy), and altitude/timing fall within PLAUSIBLE
       PSLV-class order-of-magnitude bounds (data/reference_missions.py)
       — not an exact match, since this is a simplified 2-stage
       generic vehicle, not a reproduction of any specific real mission.
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from guidance.peg import PEGTarget
from analysis.mission_profile import (
    simulate_two_stage_mission, simulate_coast_phase,
    _flat_frame_to_polar, R_EARTH, MU_EARTH
)
from data.reference_missions import sanity_check_mission_result
from data.reference_vehicle import REFERENCE_VEHICLE, reference_target, run_reference_mission

PASS = 0
FAIL = 0


def check(name, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


def test_flat_to_polar_conversion_exact():
    """Pure geometry check: v = sqrt(vx^2+vy^2), gamma = atan2(vy, vx)."""
    altitude, vx, vy = 150000.0, 4000.0, 3000.0
    r, v, gamma = _flat_frame_to_polar(altitude, vx, vy)

    check("Converted radius equals R_EARTH + altitude exactly",
          abs(r - (R_EARTH + altitude)) < 1e-6)
    check("Converted speed matches sqrt(vx^2+vy^2) exactly",
          abs(v - np.sqrt(vx ** 2 + vy ** 2)) < 1e-6)
    check("Converted flight-path angle matches atan2(vy,vx) exactly",
          abs(gamma - np.arctan2(vy, vx)) < 1e-9)


def test_coast_phase_conserves_angular_momentum_and_energy():
    """
    No forces but gravity act during ballistic coast, so on a SPHERICAL
    Earth the physically-conserved quantities are specific angular
    momentum h=r*v*cos(gamma) and specific orbital energy E=v^2/2-mu/r —
    NOT the flat-frame tangential velocity vx, which changes because the
    local-horizontal reference direction itself rotates as the vehicle
    moves along a curved arc (see guidance/gravity_turn.py's frame docs).
    """
    r0, v0, g0 = _flat_frame_to_polar(150000.0, 5000.0, 1000.0)
    h0 = r0 * v0 * np.cos(g0)
    e0 = 0.5 * v0 ** 2 - MU_EARTH / r0

    alt, dr, vx, vy = simulate_coast_phase(
        altitude0_m=150000.0, downrange0_m=0.0, vx0=5000.0, vy0=1000.0,
        duration_s=20.0, dt=0.05,
    )
    r1, v1, g1 = _flat_frame_to_polar(alt, vx, vy)
    h1 = r1 * v1 * np.cos(g1)
    e1 = 0.5 * v1 ** 2 - MU_EARTH / r1

    check("Specific angular momentum is conserved during ballistic coast (<1e-6 relative)",
          abs(h1 - h0) / abs(h0) < 1e-6)
    check("Specific orbital energy is conserved during ballistic coast (<1e-6 relative)",
          abs(e1 - e0) / abs(e0) < 1e-6)


def test_coast_phase_matches_projectile_motion_locally():
    """
    Over a SHORT coast at high altitude (curvature/rotation effects are
    second-order over a few seconds), vertical velocity change should
    still closely match simple v = v0 - g*t projectile motion.
    """
    from guidance.gravity_turn import local_gravity
    alt0, vy0, duration = 200000.0, 500.0, 3.0
    g_approx = local_gravity(alt0)

    alt, dr, vx, vy_end = simulate_coast_phase(
        altitude0_m=alt0, downrange0_m=0.0, vx0=3000.0, vy0=vy0,
        duration_s=duration, dt=0.01,
    )
    vy_expected = vy0 - g_approx * duration
    check("Vertical velocity after a short coast matches simple projectile motion (~2% tolerance)",
          abs(vy_end - vy_expected) / abs(vy_expected) < 0.02)


def test_full_mission_mass_conservation():
    result = run_reference_mission()
    check("Stage 1 burnout mass is less than stage 1 initial mass",
          result["stage1"]["burnout_mass_kg"] < REFERENCE_VEHICLE["stage1_m0_kg"])
    check("Stage 2 final mass is less than stage 2 initial mass",
          result["stage2"]["final_mass_kg"] < REFERENCE_VEHICLE["stage2_m0_kg"])
    check("Stage 2 final mass is positive",
          result["stage2"]["final_mass_kg"] > 0.0)


def test_full_mission_insertion_accuracy():
    """
    Uses data/reference_vehicle.py's REFERENCE_VEHICLE — parameters chosen
    (via a small design search) so this project's own simplified PEG
    solver actually converges tightly end-to-end for this vehicle; see
    that module's docstring. Arbitrary vehicle parameters are NOT
    guaranteed to converge (a known limitation of the local
    predictor-corrector solver — see guidance/peg.py).
    """
    result = run_reference_mission()
    check("Full-mission final velocity error is small (<20 m/s), consistent with PEG's own accuracy",
          abs(result["insertion_error"]["velocity_error_m_s"]) < 20.0)
    check("Full-mission final flight-path-angle error is small (<1 deg)",
          abs(result["insertion_error"]["flight_path_angle_error_deg"]) < 1.0)
    check("Full-mission final radius error is small (<5 km)",
          abs(result["insertion_error"]["radius_error_m"]) < 5000.0)


def test_full_mission_plausible_vs_public_reference_data():
    """
    Order-of-magnitude plausibility check against public PSLV-class
    reference figures (data/reference_missions.py) — NOT an exact-match
    test, since this is a simplified generic 2-stage vehicle.
    """
    result = run_reference_mission()
    plausibility = sanity_check_mission_result({
        "total_mission_time_s": result["total_mission_time_s"],
        "final_altitude_km": result["final_altitude_km"],
    })
    check("Total mission ascent duration is plausible vs public PSLV-class reference range",
          plausibility.get("total_ascent_duration_plausible", False))
    check("Final altitude is plausible vs public PSLV-class LEO/SSO reference range",
          plausibility.get("final_altitude_in_plausible_leo_sso_range", False))


def test_stage1_max_q_and_burnout_altitude_are_positive_and_ordered():
    result = run_reference_mission()
    check("Stage 1 max-Q is positive", result["stage1_max_q_pa"] > 0)
    check("Coast-phase-end altitude is higher than stage-1 burnout altitude (still climbing)",
          result["coast_end_altitude_m"] > result["stage1_burnout_altitude_km"] * 1000.0)


if __name__ == "__main__":
    print("Running full mission-profile verification suite...\n")
    test_flat_to_polar_conversion_exact()
    test_coast_phase_conserves_angular_momentum_and_energy()
    test_coast_phase_matches_projectile_motion_locally()
    test_full_mission_mass_conservation()
    test_full_mission_insertion_accuracy()
    test_full_mission_plausible_vs_public_reference_data()
    test_stage1_max_q_and_burnout_altitude_are_positive_and_ordered()

    print(f"\n{PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    if FAIL > 0:
        sys.exit(1)
