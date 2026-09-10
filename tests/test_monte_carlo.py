"""
tests/test_monte_carlo.py

Verification suite for analysis/monte_carlo.py.

Central claim under test: closed-loop PEG guidance should show
significantly tighter (lower standard deviation) insertion-velocity
error distribution than naive open-loop (fixed steering, fixed burn
time) guidance, when both are subjected to the SAME randomized
dispersions. This is the quantitative proof of PEG's value, not just
an assertion.
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from guidance.peg import PEGTarget
from analysis.monte_carlo import DispersionModel, run_monte_carlo, compare_open_vs_closed_loop_dispersion

PASS = 0
FAIL = 0
MU_EARTH = 3.986004418e14
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


def test_monte_carlo_runs_expected_trial_count():
    target = _target()
    result = run_monte_carlo(
        nominal_r0_m=R_EARTH + 180000.0, nominal_v0_m_s=7300.0,
        nominal_gamma0_rad=np.radians(3.0), nominal_m0_kg=4000.0,
        thrust_n=90000.0, nominal_isp_s=320.0, target=target,
        n_trials=50,
    )
    check("Monte Carlo returns exactly n_trials results",
          len(result["velocity_error_m_s"]) == 50)
    check("Success rate is between 0 and 1",
          0.0 <= result["success_rate"] <= 1.0)


def test_monte_carlo_reproducible_with_fixed_seed():
    """Same seed should give bit-identical results — critical for any
    certification-style analysis (reproducibility requirement)."""
    target = _target()
    kwargs = dict(
        nominal_r0_m=R_EARTH + 180000.0, nominal_v0_m_s=7300.0,
        nominal_gamma0_rad=np.radians(3.0), nominal_m0_kg=4000.0,
        thrust_n=90000.0, nominal_isp_s=320.0, target=target,
        n_trials=30, seed=999,
    )
    r1 = run_monte_carlo(**kwargs)
    r2 = run_monte_carlo(**kwargs)
    check("Same seed produces identical velocity error arrays (reproducibility)",
          np.allclose(r1["velocity_error_m_s"], r2["velocity_error_m_s"]))


def test_higher_dispersion_gives_wider_error_spread():
    """
    Sanity check on the dispersion model itself: doubling the input
    sigma values should produce a measurably wider output error
    distribution (std of insertion velocity error should increase).
    """
    target = _target()
    tight = DispersionModel(sigma_v0_m_s=5.0, sigma_gamma0_deg=0.1,
                             sigma_r0_m=500.0, sigma_mass_frac=0.003, sigma_isp_frac=0.003)
    loose = DispersionModel(sigma_v0_m_s=40.0, sigma_gamma0_deg=1.0,
                             sigma_r0_m=4000.0, sigma_mass_frac=0.03, sigma_isp_frac=0.02)

    kwargs = dict(
        nominal_r0_m=R_EARTH + 180000.0, nominal_v0_m_s=7300.0,
        nominal_gamma0_rad=np.radians(3.0), nominal_m0_kg=4000.0,
        thrust_n=90000.0, nominal_isp_s=320.0, target=target,
        n_trials=200, seed=42,
    )

    r_tight = run_monte_carlo(dispersion=tight, **kwargs)
    r_loose = run_monte_carlo(dispersion=loose, **kwargs)

    check("Wider input dispersion produces wider output velocity-error spread",
          r_loose["velocity_error_std"] > r_tight["velocity_error_std"])


def test_closed_loop_beats_open_loop_under_dispersion():
    """
    THE key differentiator claim: under identical randomized dispersion,
    closed-loop PEG should produce a materially tighter (lower std-dev)
    insertion-velocity-error distribution than naive open-loop guidance
    with a fixed steering angle and fixed burn duration.
    """
    target = _target()
    comparison = compare_open_vs_closed_loop_dispersion(
        nominal_r0_m=R_EARTH + 180000.0, nominal_v0_m_s=7300.0,
        nominal_gamma0_rad=np.radians(3.0), nominal_m0_kg=4000.0,
        thrust_n=90000.0, nominal_isp_s=320.0, target=target,
        fixed_chi_deg=-33.3,   # matches PEG's own first-cycle nominal command
        n_trials=200,
    )
    check("Closed-loop PEG has lower velocity-error std-dev than open-loop under same dispersion",
          comparison["closed_loop_std"] < comparison["open_loop_std"])
    check("Improvement factor (open_loop_std / closed_loop_std) is meaningfully > 1",
          comparison["improvement_factor"] > 1.2)


if __name__ == "__main__":
    print("Running Monte Carlo dispersion analysis verification suite...\n")
    test_monte_carlo_runs_expected_trial_count()
    test_monte_carlo_reproducible_with_fixed_seed()
    test_higher_dispersion_gives_wider_error_spread()
    test_closed_loop_beats_open_loop_under_dispersion()

    print(f"\n{PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    if FAIL > 0:
        sys.exit(1)
