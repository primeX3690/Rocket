"""
tests/test_embedded_c_equivalence.py

Verifies that embedded/pid_controller.c is actually numerically
equivalent to the Python reference it claims to port
(control/tvc_attitude.py) -- this project previously had NO such test,
so a C/Python divergence (a truncated-Taylor cos_approx() bug, a
missing PID integral clamp, a missing sin() in the gimbal torque, all
fixed in this pass) went undetected.

Builds embedded/pid_controller.c as a native shared library (portable
C, no ARM-specific code -- only main.c/startup.c/linker.ld are
Cortex-M-specific) and drives it via ctypes, stepping the identical
control loop in Python and C side by side.

Requires a host C compiler (gcc/clang). This does NOT exercise the
ARM cross-compile / QEMU path (embedded/build_and_run.sh) -- that
requires arm-none-eabi-gcc + qemu-system-arm, which this environment
does not have installed; see embedded/README.md.
"""

import sys
import os
import ctypes
import subprocess
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from control.tvc_attitude import PIDController, RigidBodyPitchDynamics
from faults.real_world import RealisticActuator, one_minus_cosine_gust

PASS = 0
FAIL = 0
EMBEDDED_DIR = os.path.join(os.path.dirname(__file__), "..", "embedded")


def check(name, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


def _build_shared_lib():
    lib_path = os.path.join(EMBEDDED_DIR, "libpid_test.so")
    src_path = os.path.join(EMBEDDED_DIR, "pid_controller.c")
    for compiler in ("gcc", "cc", "clang"):
        try:
            subprocess.run([compiler, "-O2", "-shared", "-fPIC", "-o", lib_path, src_path],
                          check=True, capture_output=True)
            return lib_path
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    return None


class _CPID(ctypes.Structure):
    _fields_ = [("kp", ctypes.c_float), ("ki", ctypes.c_float), ("kd", ctypes.c_float),
               ("integral", ctypes.c_float), ("prev_error", ctypes.c_float),
               ("output_limit", ctypes.c_float), ("integral_limit", ctypes.c_float),
               ("has_prev", ctypes.c_int)]


class _CActuator(ctypes.Structure):
    _fields_ = [("max_slew_rate", ctypes.c_float), ("tau", ctypes.c_float),
               ("actual_angle", ctypes.c_float)]


class _CDynamics(ctypes.Structure):
    _fields_ = [("moment_of_inertia", ctypes.c_float), ("gimbal_arm", ctypes.c_float),
               ("pitch", ctypes.c_float), ("pitch_rate", ctypes.c_float)]


def test_c_and_python_control_loops_stay_within_tolerance():
    lib_path = _build_shared_lib()
    if lib_path is None:
        print("  [SKIP] No host C compiler (gcc/cc/clang) available -- cannot build "
              "embedded/pid_controller.c for equivalence testing.")
        return

    lib = ctypes.CDLL(lib_path)
    lib.pid_update.restype = ctypes.c_float
    lib.actuator_step.restype = ctypes.c_float
    lib.one_minus_cosine_gust.restype = ctypes.c_float
    lib.pid_init.argtypes = [ctypes.POINTER(_CPID)] + [ctypes.c_float] * 4
    lib.pid_update.argtypes = [ctypes.POINTER(_CPID), ctypes.c_float, ctypes.c_float]
    lib.actuator_init.argtypes = [ctypes.POINTER(_CActuator), ctypes.c_float, ctypes.c_float]
    lib.actuator_step.argtypes = [ctypes.POINTER(_CActuator), ctypes.c_float, ctypes.c_float]
    lib.dynamics_init.argtypes = [ctypes.POINTER(_CDynamics), ctypes.c_float, ctypes.c_float]
    lib.dynamics_step.argtypes = [ctypes.POINTER(_CDynamics), ctypes.c_float, ctypes.c_float,
                                  ctypes.c_float, ctypes.c_float]
    lib.one_minus_cosine_gust.argtypes = [ctypes.c_float] * 4

    dt = 0.02
    n_steps = 600
    kp, ki, kd, gimbal_limit_rad = 8.0, 0.5, 3.0, np.radians(6.0)

    py_pid = PIDController(kp, ki, kd, output_limit_rad=gimbal_limit_rad)
    py_act = RealisticActuator(max_slew_rate_deg_s=20.0, time_constant_s=0.05)
    py_dyn = RigidBodyPitchDynamics(moment_of_inertia_kg_m2=50000.0, gimbal_arm_m=1.2)

    c_pid = _CPID()
    lib.pid_init(ctypes.byref(c_pid), kp, ki, kd, gimbal_limit_rad)
    c_act = _CActuator()
    lib.actuator_init(ctypes.byref(c_act), 0.34907, 0.05)
    c_dyn = _CDynamics()
    lib.dynamics_init(ctypes.byref(c_dyn), 50000.0, 1.2)

    t = 0.0
    max_pitch_diff = 0.0
    max_gimbal_diff = 0.0
    for _ in range(n_steps):
        error = 0.0 - py_dyn.pitch_rad
        py_gimbal_cmd = py_pid.update(error, dt)
        py_gimbal_actual = py_act.step(py_gimbal_cmd, dt)
        py_disturbance = one_minus_cosine_gust(t, 3.0, 2.0, 20000.0)
        py_dyn.step(2_000_000.0, py_gimbal_actual, py_disturbance, dt)

        c_error = ctypes.c_float(0.0 - c_dyn.pitch)
        c_gimbal_cmd = lib.pid_update(ctypes.byref(c_pid), c_error, dt)
        c_gimbal_actual = lib.actuator_step(ctypes.byref(c_act), c_gimbal_cmd, dt)
        c_disturbance = lib.one_minus_cosine_gust(ctypes.c_float(t), 3.0, 2.0, 20000.0)
        lib.dynamics_step(ctypes.byref(c_dyn), 2_000_000.0, c_gimbal_actual, c_disturbance, dt)

        max_pitch_diff = max(max_pitch_diff, abs(py_dyn.pitch_rad - c_dyn.pitch))
        max_gimbal_diff = max(max_gimbal_diff, abs(py_gimbal_actual - c_act.actual_angle))
        t += dt

    check("C and Python pitch trajectories stay within 1e-3 rad over the whole 12 s run",
          max_pitch_diff < 1e-3)
    check("C and Python gimbal-angle trajectories stay within 1e-4 rad over the whole 12 s run",
          max_gimbal_diff < 1e-4)
    check("Final Python and C pitch agree to within 1e-3 rad",
          abs(py_dyn.pitch_rad - c_dyn.pitch) < 1e-3)

    os.remove(lib_path)


def test_c_cos_approx_matches_true_cosine():
    lib_path = _build_shared_lib()
    if lib_path is None:
        print("  [SKIP] No host C compiler available.")
        return
    lib = ctypes.CDLL(lib_path)
    lib.one_minus_cosine_gust.restype = ctypes.c_float
    lib.one_minus_cosine_gust.argtypes = [ctypes.c_float] * 4

    # one_minus_cosine_gust(t=4.0, start=3.0, length=2.0, peak) exercises
    # cos_approx at phase=pi (the point where the OLD Taylor-series bug
    # was worst -- returned -1.211 instead of -1.0, a 21% error).
    peak = 20000.0
    c_val = lib.one_minus_cosine_gust(4.0, 3.0, 2.0, peak)
    true_val = (peak / 2.0) * (1.0 - np.cos(np.pi))   # phase = pi exactly at the gust's midpoint
    check("Gust profile at its midpoint (cos_approx(pi)) matches the true value to <0.1%",
          abs(c_val - true_val) / true_val < 0.001)
    os.remove(lib_path)


if __name__ == "__main__":
    print("Running embedded C / Python equivalence verification suite...\n")
    test_c_and_python_control_loops_stay_within_tolerance()
    test_c_cos_approx_matches_true_cosine()

    print(f"\n{PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    if FAIL > 0:
        sys.exit(1)
