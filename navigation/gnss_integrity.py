"""
navigation/gnss_integrity.py

GNSS integrity monitoring: detects jamming (signal loss -> no fixes
arrive) and spoofing (fixes keep arriving but jump implausibly, i.e.
disagree with the vehicle's own inertial dead-reckoning by more than
physically possible) and falls back to inertial-only navigation --
letting a bad or missing GNSS fix corrupt the state estimate is worse
than a somewhat-larger free-inertial drift.

This is receiver-independent (autonomous) integrity monitoring: it
works from consistency with the vehicle's own INS propagation, not
from GNSS-internal signals (RAIM, C/N0, correlator-peak shape) a real
receiver chip would also use and this project has no access to.

Zero external dependencies beyond NumPy - CPU-only.
"""

import numpy as np


class GNSSIntegrityMonitor:
    """
    Flags a GNSS fix as untrustworthy if it disagrees with the current
    INS position estimate by more than what's physically plausible
    given the elapsed time and a generous max-vehicle-acceleration
    bound, OR if fixes stop arriving altogether (jamming).
    """
    def __init__(self, max_plausible_accel_m_s2: float = 100.0, jamming_timeout_s: float = 2.0):
        self.max_accel = max_plausible_accel_m_s2
        self.jamming_timeout = jamming_timeout_s
        self.last_fix_time_s = 0.0
        self.consecutive_rejects = 0

    def check_fix(self, gnss_position: np.ndarray, ins_position: np.ndarray,
                 ins_velocity: np.ndarray, dt_since_last_fix_s: float) -> dict:
        """
        Returns {"trust": bool, "reason": str}. A fix is rejected if the
        implied displacement from the INS-predicted position exceeds
        what max_plausible_accel_m_s2 could produce in dt_since_last_fix_s.
        """
        gnss_position = np.asarray(gnss_position, dtype=float)
        ins_position = np.asarray(ins_position, dtype=float)
        residual = np.linalg.norm(gnss_position - ins_position)
        max_plausible_residual = 0.5 * self.max_accel * dt_since_last_fix_s ** 2 + \
            np.linalg.norm(ins_velocity) * dt_since_last_fix_s * 0.1 + 50.0   # +margin for INS's own drift
        if residual > max_plausible_residual:
            self.consecutive_rejects += 1
            return {"trust": False, "reason": "spoofing_suspected", "residual_m": residual,
                    "max_plausible_residual_m": max_plausible_residual}
        self.consecutive_rejects = 0
        self.last_fix_time_s = self.last_fix_time_s + dt_since_last_fix_s
        return {"trust": True, "reason": "nominal", "residual_m": residual,
                "max_plausible_residual_m": max_plausible_residual}

    def jamming_detected(self, t_s: float, last_valid_fix_t_s: float) -> bool:
        """No trusted fix for longer than jamming_timeout_s -> declare jamming."""
        return (t_s - last_valid_fix_t_s) > self.jamming_timeout


def navigate_with_gnss_integrity(ekf, monitor: GNSSIntegrityMonitor, t_s: float,
                                 gnss_position_or_none, last_valid_fix_t_s: float,
                                 dt_since_last_fix_s: float):
    """
    One integrity-gated navigation-update decision: accepts a trusted
    fix, rejects (and logs) a spoofed/implausible one, or declares
    jamming and coasts on inertial-only propagation -- the EKF's own
    predict() (already called elsewhere per IMU cycle) is the fallback;
    this function only decides whether to ALSO call update_position().

    Returns (new_last_valid_fix_t_s, status_dict).
    """
    if gnss_position_or_none is None:
        jammed = monitor.jamming_detected(t_s, last_valid_fix_t_s)
        return last_valid_fix_t_s, {"status": "jamming" if jammed else "no_fix_yet", "used_fix": False}

    result = monitor.check_fix(gnss_position_or_none, ekf.position, ekf.velocity, dt_since_last_fix_s)
    if result["trust"]:
        ekf.update_position(gnss_position_or_none)
        return t_s, {"status": "nominal", "used_fix": True, **result}
    return last_valid_fix_t_s, {"status": "spoofing_rejected", "used_fix": False, **result}
