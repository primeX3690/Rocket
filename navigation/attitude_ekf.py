"""
navigation/attitude_ekf.py

3-axis strapdown INS: quaternion attitude propagated from a 3-axis
gyroscope, fused with a 3-axis accelerometer and periodic GPS-style
position fixes in a 15-state error-state Extended Kalman Filter (ESKF) --
the standard architecture used on real strapdown INS (attitude, velocity,
position, gyro bias, accel bias), as opposed to the single-axis
position/velocity/bias filter in state_estimation.py (kept there as the
simpler reference implementation for a single ascent axis).

Error state (all IN THE NAVIGATION FRAME, a locally-flat frame with
Z up, unless noted): [dtheta(3) attitude error (rotation vector),
dv(3), dp(3), d(gyro_bias)(3), d(accel_bias)(3)] -- 15 states.
Nominal state carried separately: quaternion q (body->nav), velocity v,
position p, gyro bias bg, accel bias ba.

Simplification: gravity is modeled as a constant nav-frame vector
(flat-Earth), matching the single-axis filter's scope (ascent-phase,
tens-of-seconds to few-minutes navigation windows) -- for the full
multi-hundred-km ascent gravity model see dynamics/six_dof.py and
guidance/peg.py, which use inverse-square gravity in the polar frame.

Zero external dependencies beyond NumPy - CPU-only.
"""

import numpy as np

GRAVITY_NAV = np.array([0.0, 0.0, -9.80665])


def quat_normalize(q):
    return q / np.linalg.norm(q)


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_from_rotvec(rv):
    """Small-angle (exact for any magnitude) quaternion for a rotation vector."""
    angle = np.linalg.norm(rv)
    if angle < 1e-12:
        return np.array([1.0, 0.5 * rv[0], 0.5 * rv[1], 0.5 * rv[2]])
    axis = rv / angle
    return np.array([np.cos(angle / 2.0), *(axis * np.sin(angle / 2.0))])


def quat_to_rotmatrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class StrapdownAttitudeEKF:
    """
    15-state error-state EKF: attitude (quaternion, gyro-propagated),
    velocity, position (accelerometer-propagated in nav frame), gyro
    bias, accel bias -- fused with periodic position fixes (GPS/radar).
    """
    def __init__(self, q0=None, v0=None, p0=None, bg0=None, ba0=None,
                 gyro_noise_psd=1e-6, gyro_bias_psd=1e-9,
                 accel_noise_psd=1e-4, accel_bias_psd=1e-8,
                 gravity_nav=GRAVITY_NAV):
        self.q = np.array([1.0, 0.0, 0.0, 0.0]) if q0 is None else quat_normalize(np.array(q0, dtype=float))
        self.v = np.zeros(3) if v0 is None else np.array(v0, dtype=float)
        self.p = np.zeros(3) if p0 is None else np.array(p0, dtype=float)
        self.bg = np.zeros(3) if bg0 is None else np.array(bg0, dtype=float)
        self.ba = np.zeros(3) if ba0 is None else np.array(ba0, dtype=float)
        self.g_nav = np.array(gravity_nav, dtype=float)

        self.P = np.diag([1e-4] * 3 + [1.0] * 3 + [10.0] * 3 + [1e-6] * 3 + [1e-4] * 3)
        self.q_gyro, self.q_gyro_bias = gyro_noise_psd, gyro_bias_psd
        self.q_accel, self.q_accel_bias = accel_noise_psd, accel_bias_psd

    def predict(self, gyro_meas, accel_meas, dt):
        """gyro_meas, accel_meas: 3-vectors, body frame."""
        gyro_meas, accel_meas = np.asarray(gyro_meas, float), np.asarray(accel_meas, float)
        omega = gyro_meas - self.bg
        a_b = accel_meas - self.ba

        self.q = quat_normalize(quat_mult(self.q, quat_from_rotvec(omega * dt)))
        R = quat_to_rotmatrix(self.q)
        a_n = R @ a_b + self.g_nav

        self.p = self.p + self.v * dt + 0.5 * a_n * dt ** 2
        self.v = self.v + a_n * dt

        F = np.eye(15)
        F[0:3, 0:3] = np.eye(3) - skew(omega) * dt
        F[0:3, 9:12] = -np.eye(3) * dt
        F[3:6, 0:3] = -R @ skew(a_b) * dt
        F[3:6, 12:15] = -R * dt
        F[6:9, 3:6] = np.eye(3) * dt

        Q = np.zeros((15, 15))
        Q[0:3, 0:3] = np.eye(3) * self.q_gyro * dt
        Q[3:6, 3:6] = np.eye(3) * self.q_accel * dt
        Q[9:12, 9:12] = np.eye(3) * self.q_gyro_bias * dt
        Q[12:15, 12:15] = np.eye(3) * self.q_accel_bias * dt

        self.P = F @ self.P @ F.T + Q

    def update_position(self, pos_meas, r_pos=25.0):
        """Fuse a 3-axis GPS/radar-style position fix (nav frame, metres)."""
        pos_meas = np.asarray(pos_meas, float)
        H = np.zeros((3, 15))
        H[:, 6:9] = np.eye(3)
        R = np.eye(3) * r_pos

        y = pos_meas - self.p
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ y

        self._inject(dx)
        IKH = np.eye(15) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

    def _inject(self, dx):
        self.q = quat_normalize(quat_mult(self.q, quat_from_rotvec(dx[0:3])))
        self.v = self.v + dx[3:6]
        self.p = self.p + dx[6:9]
        self.bg = self.bg + dx[9:12]
        self.ba = self.ba + dx[12:15]

    @property
    def position(self):
        return self.p.copy()

    @property
    def velocity(self):
        return self.v.copy()

    @property
    def euler_deg(self):
        """Roll/pitch/yaw (deg) from the current attitude quaternion, for inspection."""
        w, x, y, z = self.q
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return np.degrees([roll, pitch, yaw])
