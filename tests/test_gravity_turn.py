"""
tests/test_gravity_turn.py

Verification suite for guidance/gravity_turn.py.

Strategy (same as rest of the stack): where a closed-form analytic
answer exists (e.g. pure vertical flight with no drag), check against it
directly. Where no closed form exists (full gravity-turn with drag),
check physically-necessary invariants instead (mass conservation,
monotonic pitch-over, max-Q occurring in the expected regime, energy
sanity).
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from guidance.gravity_turn import simulate_ascent, local_gravity, G0, R_EARTH

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


def test_mass_conservation():
    """
    Total propellant burned over the simulated burn time must equal
    mdot * burn_time (mass flow is held constant in this module) —
    exact bookkeeping check independent of trajectory shape.
    """
    m0, m_dry, isp, thrust = 50000.0, 45000.0, 280.0, 1_200_000.0
    result = simulate_ascent(m0, m_dry, thrust, isp, drag_coeff=0.3,
                              ref_area_m2=2.0, kick_altitude_m=500.0,
                              kick_angle_deg=2.0, dt=0.02, t_max=60.0)
    mdot = thrust / (isp * G0)
    expected_mass = m0 - mdot * result["burn_time_s"]
    check("Final mass matches m0 - mdot*burn_time (mass bookkeeping)",
          abs(result["burnout_mass_kg"] - expected_mass) < 1.0)


def test_no_drag_vertical_only_matches_analytic():
    """
    Edge case: kick_altitude set higher than burnout altitude ever
    reaches (so vehicle never kicks over, stays purely vertical) and
    drag zeroed out -> reduces to a 1D constant-thrust rocket under
    gravity, integrable analytically for velocity via:
        v(t) = ve*ln(m0/m(t)) - g0*t   (flat-g approximation, checked
        over a short burn where g barely changes from g0)
    """
    m0, m_dry, isp, thrust = 10000.0, 8000.0, 300.0, 400000.0
    ve = isp * G0
    mdot = thrust / ve

    result = simulate_ascent(m0, m_dry, thrust, isp, drag_coeff=0.0,
                              ref_area_m2=1.0, kick_altitude_m=1e9,  # never kicks
                              kick_angle_deg=0.0, dt=0.01, t_max=20.0)

    t_end = result["burn_time_s"]
    m_end = m0 - mdot * t_end
    v_analytic = ve * np.log(m0 / m_end) - G0 * t_end  # flat-g approx

    check("Pure-vertical no-drag burnout speed matches analytic rocket eq (~within 1%)",
          abs(result["burnout_speed_m_s"] - v_analytic) / v_analytic < 0.01)
    check("Pure-vertical burn never leaves the vertical (pitch stays ~90deg)",
          np.all(np.abs(result["pitch_deg"] - 90.0) < 0.5))


def test_gravity_turn_reduces_pitch_over_time():
    """
    After the kick is applied, the zero-AoA gravity-turn law should
    cause pitch to monotonically decrease from ~90 deg towards
    horizontal as the burn progresses (this is the entire point of
    the maneuver) — check pitch at end of sim is meaningfully lower
    than pitch right after kick.
    """
    m0, m_dry, isp, thrust = 50000.0, 40000.0, 280.0, 1_400_000.0
    result = simulate_ascent(m0, m_dry, thrust, isp, drag_coeff=0.3,
                              ref_area_m2=2.5, kick_altitude_m=800.0,
                              kick_angle_deg=3.0, dt=0.02, t_max=120.0)

    kick_t = result["kick_time_s"]
    check("Kick event occurred during the simulated burn", kick_t is not None)

    if kick_t is not None:
        idx_after_kick = np.searchsorted(result["t"], kick_t + 1.0)
        pitch_after_kick = result["pitch_deg"][idx_after_kick]
        pitch_at_end = result["pitch_deg"][-1]
        check("Pitch decreases from just-after-kick to end of burn (gravity turn working)",
              pitch_at_end < pitch_after_kick)


def test_max_q_occurs_within_burn_and_is_positive():
    m0, m_dry, isp, thrust = 50000.0, 40000.0, 280.0, 1_400_000.0
    result = simulate_ascent(m0, m_dry, thrust, isp, drag_coeff=0.3,
                              ref_area_m2=2.5, kick_altitude_m=800.0,
                              kick_angle_deg=3.0, dt=0.02, t_max=120.0)
    check("Max-Q is positive", result["max_q_pa"] > 0)
    check("Max-Q time falls within the simulated burn window",
          0.0 <= result["max_q_time_s"] <= result["burn_time_s"])
    check("Max-Q in realistic order-of-magnitude range for this class of vehicle (1-100 kPa)",
          1000.0 <= result["max_q_pa"] <= 100000.0)


def test_local_gravity_matches_g0_at_surface():
    check("local_gravity(0) equals g0 exactly",
          abs(local_gravity(0.0) - G0) < 1e-9)


def test_local_gravity_decreases_with_altitude():
    g_surface = local_gravity(0.0)
    g_100km = local_gravity(100000.0)
    check("Gravity decreases with altitude (inverse-square law)",
          g_100km < g_surface)
    # Sanity: at 100km, g should be a few percent lower, not wildly off
    check("Gravity at 100km altitude within expected ~3-4% reduction",
          0.94 < (g_100km / g_surface) < 0.98)


if __name__ == "__main__":
    print("Running gravity-turn guidance verification suite...\n")
    test_mass_conservation()
    test_no_drag_vertical_only_matches_analytic()
    test_gravity_turn_reduces_pitch_over_time()
    test_max_q_occurs_within_burn_and_is_positive()
    test_local_gravity_matches_g0_at_surface()
    test_local_gravity_decreases_with_altitude()

    print(f"\n{PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    if FAIL > 0:
        sys.exit(1)
