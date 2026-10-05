"""
tests/test_navigation_advanced.py

Verification suite for navigation/sensor_realism.py,
navigation/multi_rate_fusion.py, and navigation/gnss_integrity.py.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from navigation.sensor_realism import (
    BiasInstabilityModel, RandomWalkAccumulator, scale_factor_error,
    misalignment_matrix, apply_full_imu_error_model,
)
from navigation.attitude_ekf import (
    StrapdownAttitudeEKF, quat_mult, quat_from_rotvec, quat_normalize, quat_to_rotmatrix,
)
from navigation.multi_rate_fusion import SensorSchedule, MultiRateFusionRunner
from navigation.gnss_integrity import GNSSIntegrityMonitor, navigate_with_gnss_integrity

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


def test_bias_instability_is_bounded_not_unbounded():
    b = BiasInstabilityModel(0.01, correlation_time_s=20.0, seed=1)
    vals = [b.step(0.1) for _ in range(3000)]
    check("Bias-instability (Gauss-Markov) process stays bounded near its configured magnitude",
          abs(np.std(vals) - 0.01) < 0.01)
    check("Bias-instability values never blow up to unbounded magnitude",
          max(abs(v) for v in vals) < 0.01 * 10)


def test_random_walk_grows_as_sqrt_time():
    finals = []
    for trial in range(40):
        rw = RandomWalkAccumulator(0.002, seed=trial)
        for _ in range(1000):
            rw.step(0.05)
        finals.append(rw.accumulated)
    empirical_std = np.std(finals)
    theoretical_std = RandomWalkAccumulator(0.002).expected_std_after(50.0)
    check("Random-walk empirical std after T=50s is within 2x of the sqrt(t) theoretical prediction",
          0.5 * theoretical_std < empirical_std < 2.0 * theoretical_std)


def test_scale_factor_and_misalignment_are_correct_linear_algebra():
    check("A 1% (10000 ppm) scale factor scales the true value by exactly 1.01",
          np.allclose(scale_factor_error(np.array([2.0]), 10000), [2.02]))
    M = misalignment_matrix(0.0, 0.0, 0.0)
    check("Zero misalignment tilt angles give exactly the identity matrix",
          np.allclose(M, np.eye(3)))
    M2 = misalignment_matrix(0.01, 0.0, 0.0)
    check("Nonzero misalignment tilt produces a non-identity matrix",
          not np.allclose(M2, np.eye(3)))


def test_full_imu_error_model_composes_all_sources():
    rng = np.random.default_rng(0)
    b = BiasInstabilityModel(0.0, seed=1)   # zero magnitude -> isolate other sources
    rw = RandomWalkAccumulator(0.0, seed=2)
    M = np.eye(3)
    measured_no_noise = apply_full_imu_error_model(
        np.array([1.0, 0.0, 0.0]), b, rw, scale_factor_ppm=0, misalignment=M,
        white_noise_std=0.0, dt=0.1, rng=rng,
    )
    check("With every error source zeroed, the measurement exactly matches the true value",
          np.allclose(measured_no_noise, [1.0, 0.0, 0.0]))

    measured_scaled = apply_full_imu_error_model(
        np.array([1.0, 0.0, 0.0]), BiasInstabilityModel(0.0), RandomWalkAccumulator(0.0),
        scale_factor_ppm=50000, misalignment=M, white_noise_std=0.0, dt=0.1,
    )
    check("A 5% scale factor error alone shows up correctly in the composed measurement",
          abs(measured_scaled[0] - 1.05) < 1e-9)


def test_sensor_schedule_fires_at_correct_rate():
    sched = SensorSchedule("test", rate_hz=10.0)
    fire_times = []
    t = 0.0
    for _ in range(1000):
        if sched.due(t):
            fire_times.append(t)
            sched.mark_fired(t)
        t += 0.01
    check("A 10 Hz schedule fires ~100 times over 10 simulated seconds",
          95 <= len(fire_times) <= 105)


def test_multi_rate_runner_fires_each_sensor_at_its_own_rate():
    ekf = StrapdownAttitudeEKF()
    runner = MultiRateFusionRunner(ekf, imu_rate_hz=100.0, gnss_rate_hz=10.0, star_tracker_rate_hz=1.0)
    for _ in range(300):
        runner.step(0.01, lambda t: np.zeros(3), lambda t: np.array([0, 0, 9.80665]),
                   gnss_fn=lambda t: np.zeros(3), star_tracker_fn=lambda t: np.zeros(3))
    counts = {name: sum(1 for _, n in runner.update_log if n == name)
             for name in ("imu", "gnss", "star_tracker")}
    check("IMU (100 Hz) fires ~300 times over 3 s", 295 <= counts["imu"] <= 305)
    check("GNSS (10 Hz) fires ~30 times over 3 s", 28 <= counts["gnss"] <= 32)
    check("Star tracker (1 Hz) fires ~3 times over 3 s", 2 <= counts["star_tracker"] <= 4)


def test_multi_rate_fusion_tracks_a_maneuvering_vehicle():
    rng = np.random.default_rng(4)
    ekf = StrapdownAttitudeEKF()
    runner = MultiRateFusionRunner(ekf, imu_rate_hz=200.0, gnss_rate_hz=10.0)
    true_p, true_v, true_q = np.zeros(3), np.zeros(3), np.array([1.0, 0, 0, 0])
    true_omega = np.array([0, 0, np.radians(3.0)])
    true_accel = np.array([0, 0, 9.80665 + 10.0])
    for _ in range(500):
        dt = 0.02
        true_q = quat_normalize(quat_mult(true_q, quat_from_rotvec(true_omega * dt)))
        R = quat_to_rotmatrix(true_q)
        a_n = R @ true_accel + np.array([0, 0, -9.80665])
        true_v = true_v + a_n * dt
        true_p = true_p + true_v * dt
        runner.step(dt, lambda t: true_omega + rng.normal(0, 0.001, 3),
                   lambda t: true_accel + rng.normal(0, 0.05, 3),
                   gnss_fn=lambda t: true_p + rng.normal(0, 3.0, 3))
    check("Multi-rate-fused position error stays bounded (<10 m) over a 10 s maneuver",
          np.linalg.norm(ekf.position - true_p) < 10.0)


def test_gnss_integrity_accepts_nominal_fix():
    ekf = StrapdownAttitudeEKF()
    ekf.p = np.array([1000.0, 0.0, 0.0])
    mon = GNSSIntegrityMonitor(max_plausible_accel_m_s2=50.0)
    _, status = navigate_with_gnss_integrity(
        ekf, mon, t_s=10.0, gnss_position_or_none=np.array([1005.0, 2.0, 0.0]),
        last_valid_fix_t_s=9.9, dt_since_last_fix_s=0.1,
    )
    check("A GNSS fix close to the INS estimate is accepted as nominal",
          status["status"] == "nominal" and status["used_fix"])


def test_gnss_integrity_rejects_spoofed_fix():
    ekf = StrapdownAttitudeEKF()
    ekf.p = np.array([1000.0, 0.0, 0.0])
    ekf.v = np.array([50.0, 0.0, 0.0])
    mon = GNSSIntegrityMonitor(max_plausible_accel_m_s2=50.0)
    _, status = navigate_with_gnss_integrity(
        ekf, mon, t_s=10.1, gnss_position_or_none=np.array([51000.0, 0.0, 0.0]),
        last_valid_fix_t_s=10.0, dt_since_last_fix_s=0.1,
    )
    check("A GNSS fix implying an impossible 50 km jump in 0.1 s is rejected as spoofing",
          status["status"] == "spoofing_rejected" and not status["used_fix"])


def test_gnss_integrity_declares_jamming_after_timeout():
    ekf = StrapdownAttitudeEKF()
    mon = GNSSIntegrityMonitor(jamming_timeout_s=2.0)
    _, status = navigate_with_gnss_integrity(
        ekf, mon, t_s=15.0, gnss_position_or_none=None,
        last_valid_fix_t_s=10.0, dt_since_last_fix_s=0.0,
    )
    check("No fix for longer than the jamming timeout is declared as jamming",
          status["status"] == "jamming")
    _, status2 = navigate_with_gnss_integrity(
        ekf, mon, t_s=10.5, gnss_position_or_none=None,
        last_valid_fix_t_s=10.0, dt_since_last_fix_s=0.0,
    )
    check("A brief gap shorter than the jamming timeout is NOT yet declared as jamming",
          status2["status"] == "no_fix_yet")


if __name__ == "__main__":
    print("Running advanced navigation (sensor realism / multi-rate fusion / GNSS integrity) suite...\n")
    test_bias_instability_is_bounded_not_unbounded()
    test_random_walk_grows_as_sqrt_time()
    test_scale_factor_and_misalignment_are_correct_linear_algebra()
    test_full_imu_error_model_composes_all_sources()
    test_sensor_schedule_fires_at_correct_rate()
    test_multi_rate_runner_fires_each_sensor_at_its_own_rate()
    test_multi_rate_fusion_tracks_a_maneuvering_vehicle()
    test_gnss_integrity_accepts_nominal_fix()
    test_gnss_integrity_rejects_spoofed_fix()
    test_gnss_integrity_declares_jamming_after_timeout()

    print(f"\n{PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    if FAIL > 0:
        sys.exit(1)
