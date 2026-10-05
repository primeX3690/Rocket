"""
navigation/multi_rate_fusion.py

Multi-rate asynchronous sensor fusion scheduler, wrapping
navigation/attitude_ekf.py's StrapdownAttitudeEKF: a real flight
computer's sensors don't share one clock -- a typical stack is IMU at
~1000 Hz (drives every predict step), GNSS at ~10 Hz, and a star
tracker/sun sensor at ~1 Hz, each arriving on its own schedule and each
needing its OWN measurement-update call at its OWN rate, not a single
fixed-rate loop. This scheduler drives the EKF's existing predict/update
API at each sensor's correct rate against a shared simulation clock.

Zero external dependencies beyond NumPy - CPU-only.
"""

import numpy as np
from navigation.attitude_ekf import StrapdownAttitudeEKF


class SensorSchedule:
    """One async sensor: fires every 1/rate_hz seconds of sim time."""
    def __init__(self, name: str, rate_hz: float):
        self.name = name
        self.period_s = 1.0 / rate_hz
        self.next_fire_s = 0.0

    def due(self, t_s: float) -> bool:
        return t_s + 1e-12 >= self.next_fire_s

    def mark_fired(self, t_s: float):
        self.next_fire_s += self.period_s
        if self.next_fire_s < t_s:      # catch up if we fell behind (e.g. a slow outer loop)
            self.next_fire_s = t_s + self.period_s


class MultiRateFusionRunner:
    """
    Drives a StrapdownAttitudeEKF from three independent async sources:
    IMU (gyro+accel, drives predict()), GNSS position fixes
    (update_position()), and a star-tracker-style ABSOLUTE ATTITUDE fix
    (modeled here as a very-low-noise position-equivalent update using
    the EKF's own update_position on a synthetic "attitude-derived"
    pseudo-position channel is NOT physically meaningful, so instead
    this directly injects a small attitude-error correction, matching
    how a real star-tracker measurement update actually differs from a
    GNSS one: it corrects ATTITUDE error state directly, not position).
    """
    def __init__(self, ekf: StrapdownAttitudeEKF, imu_rate_hz: float = 1000.0,
                gnss_rate_hz: float = 10.0, star_tracker_rate_hz: float = 1.0):
        self.ekf = ekf
        self.imu = SensorSchedule("imu", imu_rate_hz)
        self.gnss = SensorSchedule("gnss", gnss_rate_hz)
        self.star_tracker = SensorSchedule("star_tracker", star_tracker_rate_hz)
        self.t = 0.0
        self.update_log = []

    def star_tracker_update(self, true_attitude_error_rotvec, r_att=1e-6):
        """
        Direct small-angle attitude-error correction (star trackers give
        near-truth absolute attitude, ~arcsecond-class, hence the tiny
        R). Uses the EKF's own attitude error-state injection machinery.
        """
        from navigation.attitude_ekf import quat_mult, quat_from_rotvec, quat_normalize
        H_gain = 1.0 / (1.0 + r_att)   # simple scalar Kalman-style blend toward the true attitude
        correction = np.asarray(true_attitude_error_rotvec, dtype=float) * H_gain
        self.ekf.q = quat_normalize(quat_mult(self.ekf.q, quat_from_rotvec(correction)))

    def step(self, dt_s: float, gyro_meas_fn, accel_meas_fn, gnss_fn=None, star_tracker_fn=None):
        """
        Advance the shared clock by dt_s, firing each due sensor at its
        own rate. gyro_meas_fn/accel_meas_fn(t)->3-vec are called at
        the IMU rate; gnss_fn(t)->3-vec position and star_tracker_fn(t)
        ->3-vec attitude-error-rotvec are called at their own rates
        (only if provided).
        """
        t_end = self.t + dt_s
        while self.t < t_end - 1e-12:
            step_dt = min(self.imu.period_s, t_end - self.t)
            if self.imu.due(self.t):
                self.ekf.predict(gyro_meas_fn(self.t), accel_meas_fn(self.t), step_dt)
                self.imu.mark_fired(self.t)
                self.update_log.append((self.t, "imu"))
            if gnss_fn is not None and self.gnss.due(self.t):
                self.ekf.update_position(gnss_fn(self.t))
                self.gnss.mark_fired(self.t)
                self.update_log.append((self.t, "gnss"))
            if star_tracker_fn is not None and self.star_tracker.due(self.t):
                self.star_tracker_update(star_tracker_fn(self.t))
                self.star_tracker.mark_fired(self.t)
                self.update_log.append((self.t, "star_tracker"))
            self.t += step_dt
        return self.ekf
