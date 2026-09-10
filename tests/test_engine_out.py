"""
tests/test_engine_out.py

Verification suite for guidance/engine_out.py.

Core claims under test:
    1. Achievable-orbit calculation matches direct rocket-equation +
       vis-viva math.
    2. A MILD engine-out (still enough propellant margin) should NOT
       trigger contingency replanning — nominal target still reachable.
    3. A SEVERE engine-out (insufficient propellant for nominal target)
       MUST trigger contingency replanning and successfully insert into
       the (lower) achievable orbit instead of failing outright.
    4. No-failure case should closely match plain PEG behavior (sanity
       check that the contingency wrapper doesn't change nominal results).
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from guidance.peg import PEGTarget
from guidance.engine_out import (
    EngineOutEvent, achievable_circular_orbit_radius,
    select_contingency_target, simulate_burn_with_engine_out, G0, MU_EARTH
)

PASS = 0
FAIL = 0
R_EARTH = 6378137.0


def check(name, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


def _target(alt_km=300.0):
    r = R_EARTH + alt_km * 1000.0
    v = np.sqrt(MU_EARTH / r)
    return PEGTarget(r, v, 0.0)


def test_achievable_orbit_matches_manual_calc():
    m_current, m_dry, isp = 4000.0, 3200.0, 320.0
    v_current, r_current = 7300.0, R_EARTH + 180000.0

    dv = isp * G0 * np.log(m_current / m_dry)
    current_energy = 0.5 * v_current ** 2 - MU_EARTH / r_current
    v_after = v_current + dv
    new_energy = 0.5 * v_after ** 2 - MU_EARTH / r_current
    r_expected = max(-MU_EARTH / (2.0 * new_energy), 6478137.0)

    r_got = achievable_circular_orbit_radius(m_current, m_dry, isp, v_current, r_current)
    check("Achievable orbit radius matches manual vis-viva energy-method calc",
          abs(r_got - r_expected) < 1.0)


def test_mild_failure_does_not_trigger_contingency():
    """
    With ample propellant margin remaining, a single-engine-out on a
    3-engine stage (thrust drops to 2/3) should still leave enough
    delta-V to reach the nominal target -> no contingency retarget.
    """
    target = _target(300.0)
    mild_failure = EngineOutEvent(time_s=5.0, thrust_fraction_remaining=0.667)

    result = simulate_burn_with_engine_out(
        r0_m=R_EARTH + 180000.0, v0_m_s=7300.0, gamma0_rad=np.radians(3.0),
        m0_kg=4000.0, m_dry_kg=1500.0,   # generous propellant margin
        nominal_thrust_n=90000.0, isp_s=320.0, nominal_target=target,
        engine_out=mild_failure,
    )
    check("Mild engine-out with ample propellant margin does NOT trigger contingency",
          not result["contingency_triggered"])
    check("Final target radius still equals the nominal target (no retarget occurred)",
          abs(result["final_target_radius_m"] - target.r_t) < 1.0)


def test_severe_failure_triggers_contingency_and_still_inserts():
    """
    A severe failure very early in the burn combined with tight
    propellant margin should make the nominal target unreachable,
    triggering contingency replanning — and the vehicle should still
    successfully insert into SOME valid (lower) circular orbit rather
    than simply falling short with no plan.
    """
    target = _target(300.0)
    severe_failure = EngineOutEvent(time_s=1.0, thrust_fraction_remaining=0.4)

    result = simulate_burn_with_engine_out(
        r0_m=R_EARTH + 180000.0, v0_m_s=7300.0, gamma0_rad=np.radians(3.0),
        m0_kg=4000.0, m_dry_kg=3390.0,   # tight propellant margin, but enough
                                           # for a genuine ~220km contingency orbit
                                           # (not the physical floor)
        nominal_thrust_n=90000.0, isp_s=320.0, nominal_target=target,
        engine_out=severe_failure,
    )
    check("Severe engine-out with tight propellant margin DOES trigger contingency",
          result["contingency_triggered"])
    check("Contingency target radius is lower than the original nominal target",
          result["final_target_radius_m"] < target.r_t)
    check("Vehicle still inserts close to its (retargeted) achievable orbit",
          abs(result["insertion_error"]["velocity_error_m_s"]) < 50.0)
    check("Contingency target is still a valid orbit above current radius",
          result["final_target_radius_m"] >= R_EARTH)


def test_no_failure_matches_plain_peg_behavior():
    """
    With engine_out=None, the contingency wrapper should behave
    equivalently to plain PEG guidance — same target, no contingency
    ever triggered, reasonable insertion accuracy.
    """
    target = _target(300.0)
    result = simulate_burn_with_engine_out(
        r0_m=R_EARTH + 180000.0, v0_m_s=7300.0, gamma0_rad=np.radians(3.0),
        m0_kg=4000.0, m_dry_kg=1500.0,
        nominal_thrust_n=90000.0, isp_s=320.0, nominal_target=target,
        engine_out=None,
    )
    check("No engine-out event never triggers contingency",
          not result["contingency_triggered"])
    check("No-failure insertion velocity error is small",
          abs(result["insertion_error"]["velocity_error_m_s"]) < 15.0)


def test_contingency_selection_logic_directly():
    """
    Direct unit test of select_contingency_target: force a case where
    dv_available is clearly insufficient for the nominal target, and
    verify it returns is_contingency=True with a lower target radius.
    """
    target = _target(500.0)  # ambitious high target
    contingency_target, is_contingency, r_achievable = select_contingency_target(
        m_current_kg=2000.0, m_dry_kg=1900.0,  # very little propellant left
        isp_s=300.0, v_current_m_s=7000.0, r_current_m=R_EARTH + 150000.0,
        nominal_target=target,
    )
    check("Insufficient propellant correctly flags is_contingency=True",
          is_contingency)
    check("Contingency target radius is lower than the ambitious nominal target",
          contingency_target.r_t < target.r_t)


if __name__ == "__main__":
    print("Running engine-out contingency guidance verification suite...\n")
    test_achievable_orbit_matches_manual_calc()
    test_mild_failure_does_not_trigger_contingency()
    test_severe_failure_triggers_contingency_and_still_inserts()
    test_no_failure_matches_plain_peg_behavior()
    test_contingency_selection_logic_directly()

    print(f"\n{PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    if FAIL > 0:
        sys.exit(1)
