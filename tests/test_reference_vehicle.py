"""
tests/test_reference_vehicle.py

Verification suite for data/reference_vehicle.py — the generic
two-stage vehicle used as the default example across this repo,
chosen so the full mission actually converges end-to-end.
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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


def test_reference_mission_converges_tightly():
    result = run_reference_mission()
    e = result["insertion_error"]
    check("Reference mission solver reports converged",
          result["stage2"]["solver_converged"])
    check("Reference mission radius error is tiny (<100 m)",
          abs(e["radius_error_m"]) < 100.0)
    check("Reference mission velocity error is tiny (<1 m/s)",
          abs(e["velocity_error_m_s"]) < 1.0)
    check("Reference mission achieves a near-circular orbit (eccentricity < 1e-3)",
          result["orbit"]["e"] < 1e-3)


def test_reference_mission_altitude_matches_target():
    result = run_reference_mission()
    check("Final altitude matches REFERENCE_VEHICLE's target_altitude_km (within 1 km)",
          abs(result["final_altitude_km"] - REFERENCE_VEHICLE["target_altitude_km"]) < 1.0)


def test_reference_target_is_circular():
    target = reference_target(300.0)
    check("reference_target's velocity matches circular speed at that altitude",
          abs(target.gamma_t) < 1e-12)


def test_overrides_are_respected():
    result = run_reference_mission(coast_duration_s=0.0)
    check("Overriding coast_duration_s changes the mission (different total time)",
          result["total_mission_time_s"] != run_reference_mission()["total_mission_time_s"])


if __name__ == "__main__":
    print("Running reference-vehicle verification suite...\n")
    test_reference_mission_converges_tightly()
    test_reference_mission_altitude_matches_target()
    test_reference_target_is_circular()
    test_overrides_are_respected()

    print(f"\n{PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    if FAIL > 0:
        sys.exit(1)
