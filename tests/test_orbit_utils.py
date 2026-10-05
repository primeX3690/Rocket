"""
tests/test_orbit_utils.py

Verification suite for guidance/orbit_utils.py.
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from guidance.orbit_utils import orbit_from_state, circular_speed, orbit_error_km, MU_EARTH, R_EARTH

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


def test_circular_orbit_gives_equal_perigee_apogee():
    r = R_EARTH + 300000.0
    v = circular_speed(r)
    o = orbit_from_state(r, v, 0.0)
    check("Circular-orbit insertion is bound", o["is_bound"])
    check("Circular-orbit eccentricity is ~0", o["e"] < 1e-6)
    check("Circular-orbit perigee ≈ apogee altitude (within 1 m)",
          abs(o["perigee_alt_km"] - o["apogee_alt_km"]) < 0.001)
    check("Circular-orbit altitude matches the target altitude (within 1 m)",
          abs(o["perigee_alt_km"] - 300.0) < 0.001)


def test_elliptical_orbit_has_higher_apogee_than_perigee():
    r = R_EARTH + 300000.0
    v = circular_speed(r) * 1.05   # slightly faster than circular -> raises apogee
    o = orbit_from_state(r, v, 0.0)
    check("Faster-than-circular insertion is still bound", o["is_bound"])
    check("Apogee is higher than perigee for a faster-than-circular insertion",
          o["apogee_alt_km"] > o["perigee_alt_km"])
    check("Perigee altitude matches the insertion point (within 1 km)",
          abs(o["perigee_alt_km"] - 300.0) < 1.0)


def test_escape_velocity_is_unbound():
    r = R_EARTH + 300000.0
    v_esc = np.sqrt(2 * MU_EARTH / r) * 1.01
    o = orbit_from_state(r, v_esc, 0.0)
    check("Just-above-escape-velocity insertion is correctly flagged unbound",
          not o["is_bound"])
    check("Unbound perigee/apogee altitudes are NaN", np.isnan(o["perigee_alt_km"]))


def test_orbit_error_km_matches_manual_computation():
    r = R_EARTH + 305000.0   # 5 km high
    v = circular_speed(R_EARTH + 300000.0)  # circular speed for the TARGET, not current r
    err = orbit_error_km((r, v, 0.0), target_alt_km=300.0)
    o = orbit_from_state(r, v, 0.0)
    check("orbit_error_km's perigee error matches a direct orbit_from_state computation",
          abs(err["perigee_error_km"] - (o["perigee_alt_km"] - 300.0)) < 1e-9)


if __name__ == "__main__":
    print("Running orbit-utils verification suite...\n")
    test_circular_orbit_gives_equal_perigee_apogee()
    test_elliptical_orbit_has_higher_apogee_than_perigee()
    test_escape_velocity_is_unbound()
    test_orbit_error_km_matches_manual_computation()

    print(f"\n{PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    if FAIL > 0:
        sys.exit(1)
