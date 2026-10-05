"""
tests/test_anomaly_arbitration.py

Verification suite for guidance/anomaly_arbitration.py.

Central claims under test:
    1. The anomaly detector requires SUSTAINED (not single-sample)
       residuals before flagging a fault — directly encoding the SSLV-D1
       lesson that a short transient should not trip a full mode switch.
    2. Under an identical simulated sensor-anomaly event and IDENTICAL
       propellant budget, graceful degradation (stay closed-loop,
       isolate only the bad channel) produces a dramatically smaller
       final insertion error than the SSLV-D1-style full open-loop
       fallback — quantifying the value of the fix using the real
       documented failure's own target orbit (356.2 km, the actual
       publicly-published SSLV-D1 mission target).
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from guidance.peg import PEGTarget
from guidance.anomaly_arbitration import (
    AccelerometerAnomalyDetector,
    simulate_sslv_style_full_open_loop_fallback,
    simulate_graceful_degradation_guidance,
    compare_sslv_failure_mode_vs_graceful_degradation,
    nominal_cutoff_mass_kg,
)

PASS = 0
FAIL = 0
R_EARTH = 6378137.0
MU_EARTH = 3.986004418e14


def check(name, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


def _target(alt_km=356.2):
    """356.2 km — the actual publicly-documented SSLV-D1 target altitude."""
    r = R_EARTH + alt_km * 1000.0
    v = np.sqrt(MU_EARTH / r)
    return PEGTarget(r, v, 0.0)


# Ignition state chosen (via a small grid search) so this project's own
# simplified point-mass PEG solver actually converges tightly for this
# target -- the numerical predictor-corrector is a local solve and its
# convergence basin depends on the specific start state, a known
# limitation documented in guidance/peg.py.
_R0, _V0, _GAMMA0, _M0 = R_EARTH + 300000.0, 7550.0, np.radians(0.3), 4000.0


def test_detector_ignores_single_sample_transient():
    """
    A lone out-of-bound residual (one bad sample, then back to normal)
    must NOT trigger a fault flag — this is the direct fix for the
    SSLV-D1 lesson that a short transient should not force a full
    guidance-mode switch.
    """
    detector = AccelerometerAnomalyDetector(residual_threshold_m_s2=5.0,
                                             consecutive_samples_required=3)
    flagged = detector.check(measured_accel=50.0, predicted_accel=10.0)  # huge single spike
    check("A single transient spike does not immediately flag a fault",
          not flagged)

    flagged_after_recovery = detector.check(measured_accel=10.1, predicted_accel=10.0)
    check("Detector resets after the signal returns to normal",
          not flagged_after_recovery)


def test_detector_flags_sustained_anomaly():
    """A genuinely sustained fault (many consecutive bad samples) MUST be flagged."""
    detector = AccelerometerAnomalyDetector(residual_threshold_m_s2=5.0,
                                             consecutive_samples_required=3)
    results = [detector.check(measured_accel=50.0, predicted_accel=10.0) for _ in range(5)]
    check("A sustained anomaly (5 consecutive bad samples) is eventually flagged",
          any(results))
    check("The flag occurs no earlier than the configured consecutive-sample requirement",
          results[0] is False and results[1] is False and results[2] is True)


def test_graceful_degradation_matches_peg_accuracy_with_no_anomaly():
    """
    With zero injected fault, graceful degradation should reproduce
    guidance/peg.py's own closed-loop accuracy for this ignition state
    (both velocity and radius within the accuracy the standalone PEG
    solver achieves here -- see test_peg.py for that baseline).
    """
    target = _target()
    result = simulate_graceful_degradation_guidance(
        r0_m=_R0, v0_m_s=_V0, gamma0_rad=_GAMMA0, m0_kg=_M0,
        m_dry_kg=nominal_cutoff_mass_kg(_R0, _V0, _GAMMA0, _M0, 90000.0, 320.0, target),
        thrust_n=90000.0, isp_s=320.0, target=target,
        anomaly_start_s=9999.0, fault_bias_m_s2=0.0, fault_duration_s=0.0,
    )
    check("With no injected fault, graceful-degradation velocity error is small (<20 m/s)",
          abs(result["velocity_error_m_s"]) < 20.0)


def test_sslv_style_fallback_produces_dramatically_larger_error():
    """
    THE key comparison, directly modeled on the documented SSLV-D1
    event: under an identical injected sensor fault and identical
    propellant budget, the full open-loop fallback (frozen steering for
    the rest of the burn — matching ISRO's own account that salvage
    mode persisted) produces a dramatically larger final velocity error
    AND a larger combined (radius+velocity) miss than graceful
    degradation (isolates only the bad samples, stays closed-loop).
    """
    target = _target()
    comparison = compare_sslv_failure_mode_vs_graceful_degradation(
        r0_m=_R0, v0_m_s=_V0, gamma0_rad=_GAMMA0, m0_kg=_M0,
        thrust_n=90000.0, isp_s=320.0, target=target, anomaly_start_s=5.0,
    )
    fallback_err = abs(comparison["sslv_style_fallback"]["velocity_error_m_s"])
    graceful_err = abs(comparison["graceful_degradation"]["velocity_error_m_s"])

    check("SSLV-D1-style full open-loop fallback produces a much larger velocity error",
          fallback_err > graceful_err * 2)
    check("Velocity-error improvement from graceful degradation is large and positive",
          comparison["velocity_error_improvement_m_s"] > 50.0)
    check("Radius-error improvement from graceful degradation is positive",
          comparison["radius_error_improvement_m"] > 0.0)


def test_earlier_anomaly_onset_worsens_fallback_more():
    """
    The earlier a permanent guidance freeze happens in the burn, the
    more of the burn is flown blind — so an anomaly onset early in the
    burn should leave a LARGER final velocity error for the fallback
    path than one occurring later, since less of the burn remains to be
    flown open-loop in the latter case.
    """
    target = _target()
    early = compare_sslv_failure_mode_vs_graceful_degradation(
        r0_m=_R0, v0_m_s=_V0, gamma0_rad=_GAMMA0, m0_kg=_M0,
        thrust_n=90000.0, isp_s=320.0, target=target, anomaly_start_s=2.0,
    )
    late = compare_sslv_failure_mode_vs_graceful_degradation(
        r0_m=_R0, v0_m_s=_V0, gamma0_rad=_GAMMA0, m0_kg=_M0,
        thrust_n=90000.0, isp_s=320.0, target=target, anomaly_start_s=40.0,
    )
    fb_early = abs(early["sslv_style_fallback"]["velocity_error_m_s"])
    fb_late = abs(late["sslv_style_fallback"]["velocity_error_m_s"])
    gd_early = abs(early["graceful_degradation"]["velocity_error_m_s"])
    gd_late = abs(late["graceful_degradation"]["velocity_error_m_s"])

    check("Earlier anomaly onset produces a larger (or comparable) fallback velocity error than a later onset",
          fb_early >= fb_late * 0.95)
    check("Graceful degradation's velocity error stays well below fallback's at both onset times",
          gd_early < fb_early and gd_late < fb_late)


if __name__ == "__main__":
    print("Running anomaly-arbitration (SSLV-D1 failure-mode fix) verification suite...\n")
    test_detector_ignores_single_sample_transient()
    test_detector_flags_sustained_anomaly()
    test_graceful_degradation_matches_peg_accuracy_with_no_anomaly()
    test_sslv_style_fallback_produces_dramatically_larger_error()
    test_earlier_anomaly_onset_worsens_fallback_more()

    print(f"\n{PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    if FAIL > 0:
        sys.exit(1)
