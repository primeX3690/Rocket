"""
navigation/state_estimation.py

Launch-vehicle navigation: IMU-based dead-reckoning propagation fused
with an Extended Kalman Filter (EKF) for state estimation during
powered ascent.

Real rockets never have direct access to "true" position/velocity —
only noisy accelerometer + gyro (IMU) measurements, sometimes fused
with GPS (when available) or ground radar tracking. This module models
that reality:

    1. IMUModel — simulates noisy accelerometer readings (specific force)
       given the true trajectory (bias + noise, standard IMU error model)
    2. dead_reckon() — pure integration of noisy IMU data (accumulates
       drift over time — this is *why* a filter is needed)
    3. AscentEKF — fuses IMU propagation with periodic noisy position
       "truth" updates (representing ground radar / GPS fixes) to bound
       the drift

Same verification discipline as the rest of the stack: checked against
a known-truth synthetic trajectory so estimation error is directly
measurable.
"""

import numpy as np


class IMUModel:
    """
    Simple single-axis (can be called per-axis) accelerometer model:
        measured = true_accel + bias + noise

    bias_m_s2:  constant accelerometer bias (typical MEMS IMU: 1e-3 to 1e-2 m/s^2)
    noise_std:  zero-mean Gaussian measurement noise std dev, m/s^2
    """
    def __init__(self, bias_m_s2=0.005, noise_std=0.02, seed=42):
        self.bias = bias_m_s2
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

    def measure(self, true_accel: float) -> float:
        return true_accel + self.bias + self.rng.normal(0.0, self.noise_std)


def dead_reckon(true_accel_series: np.ndarray, dt: float, imu: IMUModel,
                 v0: float = 0.0, x0: float = 0.0):
    """
    Pure IMU dead-reckoning: integrate noisy accelerometer readings twice
    to get velocity and position, with no correction. Demonstrates why
    raw IMU integration alone is unusable for anything beyond a few tens
    of seconds — bias integrates into velocity error linearly and
    position error quadratically.
    """
    n = len(true_accel_series)
    v = np.zeros(n)
    x = np.zeros(n)
    v_prev, x_prev = v0, x0

    for i, a_true in enumerate(true_accel_series):
        a_meas = imu.measure(a_true)
        v_new = v_prev + a_meas * dt
        x_new = x_prev + v_prev * dt + 0.5 * a_meas * dt ** 2
        v[i] = v_new
        x[i] = x_new
        v_prev, x_prev = v_new, x_new

    return x, v


class AscentEKF:
    """
    Extended Kalman Filter for 1D (per-axis) ascent state estimation.

    State vector: [position, velocity, accel_bias]
    Process model: constant-bias IMU propagation (standard INS error-state
                    formulation, simplified to a direct-state filter here
                    for clarity — same structure used in SafeEvo's EKF).

    Prediction step uses IMU acceleration input.
    Update step fuses an external noisy position measurement (proxy for
    ground radar track / GPS fix, applied at a lower rate than IMU, as
    is realistic — radar updates are typically ~1-10 Hz vs IMU ~100-1000 Hz).
    """
    def __init__(self, x0=0.0, v0=0.0, bias0=0.0,
                 process_noise_accel=0.01, process_noise_bias=1e-5,
                 measurement_noise_pos=5.0):
        self.state = np.array([x0, v0, bias0])  # [pos, vel, bias]
        self.P = np.diag([10.0, 5.0, 0.01])       # initial covariance (uncertain start)

        self.q_accel = process_noise_accel
        self.q_bias = process_noise_bias
        self.r_pos = measurement_noise_pos

    def predict(self, a_meas: float, dt: float):
        x, v, b = self.state
        a_corrected = a_meas - b   # remove estimated bias from measurement

        x_new = x + v * dt + 0.5 * a_corrected * dt ** 2
        v_new = v + a_corrected * dt
        b_new = b  # random-walk bias, no deterministic change

        self.state = np.array([x_new, v_new, b_new])

        # State transition Jacobian
        F = np.array([
            [1.0, dt, -0.5 * dt ** 2],
            [0.0, 1.0, -dt],
            [0.0, 0.0, 1.0],
        ])

        Q = np.diag([
            0.25 * self.q_accel * dt ** 4,
            self.q_accel * dt ** 2,
            self.q_bias * dt,
        ])

        self.P = F @ self.P @ F.T + Q

    def update(self, pos_measurement: float):
        """Fuse a noisy external position measurement (radar/GPS-style fix)."""
        H = np.array([[1.0, 0.0, 0.0]])  # measures position directly
        R = np.array([[self.r_pos]])

        y = pos_measurement - (H @ self.state)[0]           # innovation
        S = (H @ self.P @ H.T)[0, 0] + R[0, 0]                # innovation covariance
        K = (self.P @ H.T).flatten() / S                       # Kalman gain

        self.state = self.state + K * y
        self.P = (np.eye(3) - np.outer(K, H)) @ self.P

    @property
    def position(self):
        return self.state[0]

    @property
    def velocity(self):
        return self.state[1]

    @property
    def estimated_bias(self):
        return self.state[2]


def run_ekf_ascent(true_accel_series: np.ndarray, dt: float, imu: IMUModel,
                    radar_update_every_n: int = 20, radar_noise_std: float = 5.0,
                    true_position_series: np.ndarray = None, seed=7):
    """
    Full pipeline: propagate the EKF through a synthetic ascent using
    noisy IMU data, fusing a simulated radar/GPS position fix every
    `radar_update_every_n` steps. Returns filtered position/velocity
    time series plus (if true_position_series given) the estimation
    error for verification.
    """
    rng = np.random.default_rng(seed)
    ekf = AscentEKF()
    n = len(true_accel_series)

    est_pos = np.zeros(n)
    est_vel = np.zeros(n)

    for i, a_true in enumerate(true_accel_series):
        a_meas = imu.measure(a_true)
        ekf.predict(a_meas, dt)

        if true_position_series is not None and i % radar_update_every_n == 0:
            noisy_pos = true_position_series[i] + rng.normal(0.0, radar_noise_std)
            ekf.update(noisy_pos)

        est_pos[i] = ekf.position
        est_vel[i] = ekf.velocity

    return est_pos, est_vel
