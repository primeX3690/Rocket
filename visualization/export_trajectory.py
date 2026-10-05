"""
visualization/export_trajectory.py

Exports a quaternion attitude history (from dynamics/six_dof.py's
simulate_six_dof_free_flight or navigation/attitude_ekf.py's
StrapdownAttitudeEKF) to the JSON format visualization/web/
attitude_viewer.html's file-load control expects: {"t": [...],
"quaternion": [[w,x,y,z], ...]}, so the real-time 3D viewer genuinely
renders THIS project's own simulated/estimated attitude, not just its
own built-in demo tumble.

Zero external dependencies beyond Python's stdlib `json` - CPU-only.
"""

import json
import numpy as np


def export_quaternion_trajectory_json(t: np.ndarray, quaternion_wxyz: np.ndarray, output_path: str):
    """
    t: 1D array of times (s). quaternion_wxyz: (N,4) array, each row [w,x,y,z]
    (this project's convention throughout dynamics/six_dof.py and
    navigation/attitude_ekf.py). Writes a JSON file attitude_viewer.html
    can load directly via its file-input control.
    """
    t = np.asarray(t, dtype=float)
    quaternion_wxyz = np.asarray(quaternion_wxyz, dtype=float)
    if quaternion_wxyz.shape[0] != t.shape[0] or quaternion_wxyz.shape[1] != 4:
        raise ValueError(f"Expected quaternion array of shape (len(t), 4), got {quaternion_wxyz.shape} "
                        f"for {t.shape[0]} time samples.")
    payload = {"t": t.tolist(), "quaternion": quaternion_wxyz.tolist()}
    with open(output_path, "w") as f:
        json.dump(payload, f)
    return output_path


def export_six_dof_result(six_dof_result: dict, output_path: str):
    """Convenience wrapper for dynamics/six_dof.py's simulate_six_dof_free_flight output dict."""
    return export_quaternion_trajectory_json(
        six_dof_result["t"], six_dof_result["quaternion"], output_path,
    )
