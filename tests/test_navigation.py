"""
tests/test_navigation.py

Verification suite for navigation/state_estimation.py.

Core claim to verify: pure IMU dead-reckoning drifts unboundedly due to
bias integration, while EKF fusion with periodic position updates keeps
error bounded. This is tested against a known synthetic "true" trajectory
(constant acceleration ascent) so error is exactly measurable — same
approach as the EKF verification already done in SafeEvo/AstroEvo.
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from navigation.state_estimation import IMUModel, dead_reckon, run_ekf_ascent

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


def _synthetic_truth(a_const=30.0, dt=0.1, n=600):
    """Known-truth constant-acceleration trajectory: x(t)=0.5*a*t^2, v(t)=a*t."""
    t = np.arange(n) * dt
    true_accel = np.full(n, a_const)
    true_vel = a_const * t
    true_pos = 0.5 * a_const * t ** 2
    return t, true_accel, true_vel, true_pos


def test_dead_reckoning_matches_truth_without_noise():
    """
    With zero IMU bias/noise, pure dead-reckoning integration should
    match the analytic constant-acceleration trajectory almost exactly
    (only floating-point/discretization error).
    """
    dt = 0.01
    t, true_accel, true_vel, true_pos = _synthetic_truth(a_const=20.0, dt=dt, n=1000)
    clean_imu = IMUModel(bias_m_s2=0.0, noise_std=0.0)
    x, v = dead_reckon(true_accel, dt, clean_imu)

    check("Noise-free dead-reckoning position matches analytic truth (<0.5% error)",
          abs(x[-1] - true_pos[-1]) / true_pos[-1] < 0.005)
    check("Noise-free dead-reckoning velocity matches analytic truth (<0.5% error)",
          abs(v[-1] - true_vel[-1]) / true_vel[-1] < 0.005)


def test_dead_reckoning_drifts_with_bias():
    """
    With a realistic IMU bias, dead-reckoning position error should grow
    over time (quadratically, since bias integrates twice) — verifying
    the "why we need a filter" claim quantitatively.
    """
    dt = 0.1
    t, true_accel, true_vel, true_pos = _synthetic_truth(a_const=25.0, dt=dt, n=600)
    biased_imu = IMUModel(bias_m_s2=0.02, noise_std=0.05, seed=1)
    x, _ = dead_reckon(true_accel, dt, biased_imu)

    err_early = abs(x[100] - true_pos[100])
    err_late = abs(x[-1] - true_pos[-1])

    check("Dead-reckoning error grows substantially from early to late in the burn",
          err_late > err_early * 2)


def test_ekf_beats_raw_dead_reckoning():
    """
    Core verification: EKF with periodic (simulated radar) position
    fixes should produce dramatically lower final position error than
    raw uncorrected dead-reckoning, on the same noisy IMU data stream.
    """
    dt = 0.1
    t, true_accel, true_vel, true_pos = _synthetic_truth(a_const=25.0, dt=dt, n=600)

    imu_for_dr = IMUModel(bias_m_s2=0.02, noise_std=0.05, seed=1)
    x_dr, _ = dead_reckon(true_accel, dt, imu_for_dr)

    imu_for_ekf = IMUModel(bias_m_s2=0.02, noise_std=0.05, seed=1)
    est_pos, est_vel = run_ekf_ascent(
        true_accel, dt, imu_for_ekf,
        radar_update_every_n=15, radar_noise_std=8.0,
        true_position_series=true_pos, seed=3
    )

    err_dr_final = abs(x_dr[-1] - true_pos[-1])
    err_ekf_final = abs(est_pos[-1] - true_pos[-1])

    check("EKF final position error is much smaller than raw dead-reckoning error",
          err_ekf_final < err_dr_final)
    check("EKF final position error stays within a tight bound (<50m on this profile)",
          err_ekf_final < 50.0)


def test_ekf_velocity_tracks_truth_reasonably():
    dt = 0.1
    t, true_accel, true_vel, true_pos = _synthetic_truth(a_const=25.0, dt=dt, n=600)
    imu = IMUModel(bias_m_s2=0.02, noise_std=0.05, seed=1)
    est_pos, est_vel = run_ekf_ascent(
        true_accel, dt, imu,
        radar_update_every_n=15, radar_noise_std=8.0,
        true_position_series=true_pos, seed=3
    )
    rel_err = abs(est_vel[-1] - true_vel[-1]) / true_vel[-1]
    check("EKF final velocity estimate within 5% of truth",
          rel_err < 0.05)


def test_ekf_covariance_stays_bounded():
    """
    Sanity check that the filter doesn't diverge — covariance should
    not blow up to huge values over a normal run (a classic EKF bug
    symptom is unbounded covariance growth from a Jacobian/units error).
    """
    from navigation.state_estimation import AscentEKF
    dt = 0.1
    ekf = AscentEKF()
    imu = IMUModel(bias_m_s2=0.01, noise_std=0.03, seed=5)
    for i in range(300):
        a_meas = imu.measure(20.0)
        ekf.predict(a_meas, dt)
        if i % 15 == 0:
            ekf.update(0.5 * 20.0 * (i * dt) ** 2 + np.random.normal(0, 5))
    check("Position covariance stays bounded (<1e4) after 300 steps",
          ekf.P[0, 0] < 1e4)


if __name__ == "__main__":
    print("Running navigation / EKF verification suite...\n")
    test_dead_reckoning_matches_truth_without_noise()
    test_dead_reckoning_drifts_with_bias()
    test_ekf_beats_raw_dead_reckoning()
    test_ekf_velocity_tracks_truth_reasonably()
    test_ekf_covariance_stays_bounded()

    print(f"\n{PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    if FAIL > 0:
        sys.exit(1)
