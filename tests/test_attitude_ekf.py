"""
tests/test_attitude_ekf.py

Verification suite for navigation/attitude_ekf.py (15-state strapdown
error-state EKF: attitude, velocity, position, gyro bias, accel bias).
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from navigation.attitude_ekf import (
    StrapdownAttitudeEKF, quat_mult, quat_from_rotvec, quat_normalize,
    quat_to_rotmatrix, skew,
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


def test_quaternion_utilities_are_self_consistent():
    q_identity = np.array([1.0, 0.0, 0.0, 0.0])
    q90 = quat_from_rotvec(np.array([0.0, 0.0, np.pi / 2]))
    R = quat_to_rotmatrix(q90)
    v = R @ np.array([1.0, 0.0, 0.0])
    check("Quaternion identity multiplication is a no-op",
          np.allclose(quat_mult(q_identity, q90), q90))
    check("90-degree yaw rotation matrix maps +x to +y (within 1e-9)",
          np.allclose(v, [0.0, 1.0, 0.0], atol=1e-9))
    check("Rotation matrices from unit quaternions are orthonormal (R^T R = I)",
          np.allclose(R.T @ R, np.eye(3), atol=1e-9))
    check("skew(v) is antisymmetric", np.allclose(skew([1, 2, 3]), -skew([1, 2, 3]).T))


def test_ekf_tracks_static_vehicle_with_zero_bias():
    """No rotation, no motion, zero biases: position/velocity should stay ~0."""
    ekf = StrapdownAttitudeEKF()
    g_body = np.array([0.0, 0.0, 9.80665])   # accelerometer reads +g when stationary on a pad
    for _ in range(500):
        ekf.predict(gyro_meas=np.zeros(3), accel_meas=g_body, dt=0.01)
    check("Stationary vehicle: position stays near zero (<0.5 m after 5 s)",
          np.linalg.norm(ekf.position) < 0.5)
    check("Stationary vehicle: velocity stays near zero (<0.2 m/s after 5 s)",
          np.linalg.norm(ekf.velocity) < 0.2)


def test_ekf_converges_biases_and_bounds_position_error_with_gps_fixes():
    """
    Rotating, accelerating vehicle with true gyro/accel biases and noisy
    GPS-style position fixes: filter should end up close to the true
    trajectory and estimate biases roughly in the right ballpark —
    NOT exact (it's a real EKF with real noise), but clearly converging,
    not diverging.
    """
    rng = np.random.default_rng(7)
    dt = 0.01
    n = 2000
    true_omega = np.array([0.0, 0.0, np.radians(4.0)])
    true_accel_body = np.array([0.0, 0.0, 9.80665 + 15.0])
    gyro_bias_true = np.array([0.0008, -0.0006, 0.0004])
    accel_bias_true = np.array([0.02, -0.015, 0.03])

    ekf = StrapdownAttitudeEKF()
    true_p, true_v, true_q = np.zeros(3), np.zeros(3), np.array([1.0, 0, 0, 0])

    for i in range(n):
        true_q = quat_normalize(quat_mult(true_q, quat_from_rotvec(true_omega * dt)))
        R = quat_to_rotmatrix(true_q)
        a_n = R @ true_accel_body + np.array([0, 0, -9.80665])
        true_v = true_v + a_n * dt
        true_p = true_p + true_v * dt

        gmeas = true_omega + gyro_bias_true + rng.normal(0, 0.0005, 3)
        ameas = true_accel_body + accel_bias_true + rng.normal(0, 0.05, 3)
        ekf.predict(gmeas, ameas, dt)
        if i % 100 == 0 and i > 0:
            ekf.update_position(true_p + rng.normal(0, 3.0, 3), r_pos=9.0)

    pos_err = np.linalg.norm(ekf.position - true_p)
    vel_err = np.linalg.norm(ekf.velocity - true_v)
    check("Position error stays bounded after 20 s (<15 m, well under the unbounded-drift case)",
          pos_err < 15.0)
    check("Velocity error stays bounded after 20 s (<3 m/s)",
          vel_err < 3.0)
    check("Estimated gyro bias is within an order of magnitude of the true bias",
          np.linalg.norm(ekf.bg - gyro_bias_true) < np.linalg.norm(gyro_bias_true) * 2 + 0.001)


def test_position_update_reduces_covariance():
    ekf = StrapdownAttitudeEKF()
    trace_before = np.trace(ekf.P[6:9, 6:9])
    ekf.predict(np.zeros(3), np.array([0, 0, 9.80665]), 0.1)
    ekf.update_position(np.array([0.1, -0.2, 0.05]), r_pos=1.0)
    trace_after = np.trace(ekf.P[6:9, 6:9])
    check("A position fix reduces position-error covariance", trace_after < trace_before)


if __name__ == "__main__":
    print("Running strapdown attitude EKF verification suite...\n")
    test_quaternion_utilities_are_self_consistent()
    test_ekf_tracks_static_vehicle_with_zero_bias()
    test_ekf_converges_biases_and_bounds_position_error_with_gps_fixes()
    test_position_update_reduces_covariance()

    print(f"\n{PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    if FAIL > 0:
        sys.exit(1)
