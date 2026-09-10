"""
tests/test_propulsion.py

Analytic verification suite for propulsion/rocket_equation.py.
Every check compares against a hand-computable or textbook-known value —
same verification philosophy as astroevo / safeevo.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from propulsion.rocket_equation import (
    delta_v, mass_ratio_for_dv, propellant_mass_for_dv,
    exhaust_velocity, thrust, isp_from_thrust,
    Stage, stage_dv_budget, required_leo_dv, G0
)

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


def test_tsiolkovsky_known_case():
    """
    Classic textbook check: Isp=300s, mass ratio=10 -> dV should be
    ln(10) * 300 * 9.80665 = 6772.4 m/s (Sutton reference case).
    """
    dv = delta_v(isp_s=300.0, m0=10000.0, mf=1000.0)
    expected = np.log(10) * 300.0 * G0
    check("Tsiolkovsky known case (Isp=300s, MR=10)",
          abs(dv - expected) < 1e-6)
    check("Tsiolkovsky value in expected ballpark (~6774 m/s)",
          abs(dv - 6774.2) < 1.0)


def test_inverse_consistency():
    """propellant_mass_for_dv should round-trip through delta_v exactly."""
    isp = 311.0
    dv_target = 4200.0
    mf_dry = 850.0
    mp = propellant_mass_for_dv(isp, dv_target, mf_dry)
    m0 = mf_dry + mp
    dv_recovered = delta_v(isp, m0, mf_dry)
    check("propellant_mass_for_dv round-trips through delta_v",
          abs(dv_recovered - dv_target) < 1e-6)


def test_exhaust_velocity():
    """Isp=450s (vacuum hydrolox-class) -> ve = 450*9.80665 = 4413 m/s."""
    ve = exhaust_velocity(450.0)
    check("Exhaust velocity for Isp=450s ~4413 m/s",
          abs(ve - 4413.0) < 1.0)


def test_thrust_vacuum_matches_mdot_ve_when_pa_zero():
    """In vacuum (pa=0), thrust reduces to mdot*ve + pe*Ae exactly."""
    mdot, ve, pe, ae = 250.0, 3000.0, 5000.0, 0.5
    f_vac = thrust(mdot, ve, pe, pa=0.0, ae=ae)
    expected = mdot * ve + pe * ae
    check("Vacuum thrust reduces to mdot*ve + pe*Ae",
          abs(f_vac - expected) < 1e-6)


def test_isp_from_thrust_recovers_input():
    """isp_from_thrust should invert F = mdot*Isp*g0 exactly for pe=pa case."""
    mdot = 100.0
    isp_in = 320.0
    ve = exhaust_velocity(isp_in)
    f = thrust(mdot, ve, pe=0.0, pa=0.0, ae=0.0)  # no pressure term
    isp_out = isp_from_thrust(f, mdot)
    check("isp_from_thrust inverts thrust equation (pressure term=0)",
          abs(isp_out - isp_in) < 1e-6)


def test_two_stage_budget_matches_manual_calc():
    """
    Two-stage vehicle, hand-computed reference:
    Stage 2 (upper): m_prop=3000, m_struct=500, isp=340
        payload=200
        m0 = 3000+500+200 = 3700, mf = 500+200 = 700
        dv2 = 340*9.80665*ln(3700/700)
    Stage 1 (lower): m_prop=20000, m_struct=2000, isp=280
        mass_above = stage2.total(3500) + payload(200) = 3700
        m0 = 20000+2000+3700 = 25700, mf = 2000+3700 = 5700
        dv1 = 280*9.80665*ln(25700/5700)
    """
    s1 = Stage("Stage-1", m_prop=20000, m_struct=2000, isp_s=280.0)
    s2 = Stage("Stage-2", m_prop=3000, m_struct=500, isp_s=340.0)
    result = stage_dv_budget([s1, s2], payload_mass=200.0)

    dv2_expected = 340.0 * G0 * np.log(3700.0 / 700.0)
    dv1_expected = 280.0 * G0 * np.log(25700.0 / 5700.0)
    total_expected = dv1_expected + dv2_expected

    dv1_got = result["stages"][0]["dv_m_s"]
    dv2_got = result["stages"][1]["dv_m_s"]

    check("Stage 1 dV matches manual calc", abs(dv1_got - dv1_expected) < 1e-6)
    check("Stage 2 dV matches manual calc", abs(dv2_got - dv2_expected) < 1e-6)
    check("Total dV matches sum of stages",
          abs(result["total_dv_m_s"] - total_expected) < 1e-6)


def test_leo_dv_ballpark():
    """
    300km circular LEO orbital velocity should be ~7726 m/s
    (textbook value), plus ~1500 m/s losses -> total ~9200-9300 m/s,
    consistent with real-world PSLV/Falcon 9 LEO delta-V budgets
    (typically cited 9,300-9,700 m/s).
    """
    dv = required_leo_dv(target_altitude_km=300.0, losses_gravity_drag_m_s=1500.0)
    check("300km circular orbital velocity component ~7726 m/s",
          abs((dv - 1500.0) - 7726.0) < 20.0)
    check("Total LEO dV budget in realistic 9000-9700 m/s range",
          9000.0 <= dv <= 9700.0)


def test_mass_ratio_inverse_of_delta_v():
    """mass_ratio_for_dv and delta_v should be exact inverses."""
    isp = 300.0
    mr_in = 8.5
    dv = np.log(mr_in) * isp * G0
    mr_out = mass_ratio_for_dv(isp, dv)
    check("mass_ratio_for_dv inverts delta_v exactly",
          abs(mr_out - mr_in) < 1e-9)


if __name__ == "__main__":
    print("Running propulsion verification suite...\n")
    test_tsiolkovsky_known_case()
    test_inverse_consistency()
    test_exhaust_velocity()
    test_thrust_vacuum_matches_mdot_ve_when_pa_zero()
    test_isp_from_thrust_recovers_input()
    test_two_stage_budget_matches_manual_calc()
    test_leo_dv_ballpark()
    test_mass_ratio_inverse_of_delta_v()

    print(f"\n{PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    if FAIL > 0:
        sys.exit(1)
