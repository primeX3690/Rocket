"""
tests/test_atmosphere.py

Verification against published US Standard Atmosphere 1976 reference
table values at well-known altitudes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from guidance.atmosphere import temperature, pressure, density, dynamic_pressure

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


def test_sea_level():
    # Reference: T=288.15K, P=101325Pa, rho=1.225 kg/m^3
    check("Sea level temperature = 288.15K", abs(temperature(0.0) - 288.15) < 0.01)
    check("Sea level pressure = 101325 Pa", abs(pressure(0.0) - 101325.0) < 1.0)
    check("Sea level density ~1.225 kg/m^3", abs(density(0.0) - 1.225) < 0.001)


def test_11km_tropopause():
    # Reference standard atmosphere table: T=216.65K, P=22632 Pa, rho~0.3639 kg/m^3
    check("11km temperature = 216.65K", abs(temperature(11000.0) - 216.65) < 0.01)
    check("11km pressure ~22632 Pa", abs(pressure(11000.0) - 22632.06) < 5.0)
    check("11km density ~0.364 kg/m^3", abs(density(11000.0) - 0.3639) < 0.005)


def test_25km_stratosphere():
    # Reference: P~2549 Pa (published table value at 25km)
    p25 = pressure(25000.0)
    check("25km pressure in expected ~2400-2700 Pa range", 2400.0 <= p25 <= 2700.0)


def test_density_decreases_monotonically():
    alts = [0, 5000, 11000, 20000, 32000, 47000]
    densities = [density(a) for a in alts]
    check("Density decreases monotonically with altitude",
          all(densities[i] > densities[i + 1] for i in range(len(densities) - 1)))


def test_dynamic_pressure_matches_manual():
    h, v = 10000.0, 300.0
    rho = density(h)
    expected_q = 0.5 * rho * v ** 2
    q = dynamic_pressure(h, v)
    check("Dynamic pressure matches 0.5*rho*v^2 manual calc",
          abs(q - expected_q) < 1e-9)


def test_max_q_realistic_ballpark():
    """
    Real rockets hit max-Q typically around 10-14km altitude at
    ~400-500 m/s, giving q in the 30-45 kPa range. Sanity check that
    our model produces a physically plausible max-Q in that zone.
    """
    import numpy as np
    alts = np.linspace(0, 20000, 200)
    # crude velocity profile approximation just for sanity-checking q shape
    v_profile = 500.0 * (alts / 20000.0)
    qs = [dynamic_pressure(a, v) for a, v in zip(alts, v_profile)]
    max_q = max(qs)
    check("Max dynamic pressure in plausible 10-60 kPa range for this profile",
          10000.0 <= max_q <= 60000.0)


if __name__ == "__main__":
    print("Running atmosphere model verification suite...\n")
    test_sea_level()
    test_11km_tropopause()
    test_25km_stratosphere()
    test_density_decreases_monotonically()
    test_dynamic_pressure_matches_manual()
    test_max_q_realistic_ballpark()

    print(f"\n{PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    if FAIL > 0:
        sys.exit(1)
