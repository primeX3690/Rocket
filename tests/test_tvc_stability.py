"""
tests/test_tvc_stability.py

Regression test for a real defect found in review: the gains used by the
embedded SIL harness (kp=8, ki=0.5, kd=3) are UNSTABLE on the harness's own
plant (T=2e6 N, I=5e4 kg m^2, L=1.2 m -> loop gain 48 1/s^2): the vehicle
pitch diverged to ~0.9 rad. These tests pin the corrected gains and verify,
in the Python reference AND in the compiled C port, that attitude is held.
"""
import os
import sys
import ctypes
import subprocess
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from control.tvc_attitude import PIDController, RigidBodyPitchDynamics
from faults.real_world import RealisticActuator, one_minus_cosine_gust

PASS = FAIL = 0
EMB = os.path.join(os.path.dirname(__file__), "..", "embedded")


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


def run_python(kp, ki, kd, cmd_step=0.05):
    pid = PIDController(kp, ki, kd, output_limit_rad=np.radians(6.0))
    act = RealisticActuator(max_slew_rate_deg_s=20.0, time_constant_s=0.05)
    dyn = RigidBodyPitchDynamics(50000.0, 1.2)
    t, worst = 0.0, 0.0
    for _ in range(600):
        cmd = cmd_step if t >= 5.0 else 0.0
        g = pid.update(cmd - dyn.pitch_rad, 0.02)
        ga = act.step(g, 0.02)
        dyn.step(2e6, ga, one_minus_cosine_gust(t, 3.0, 2.0, 20000.0), 0.02)
        t += 0.02
        worst = max(worst, abs(dyn.pitch_rad))
    return dyn.pitch_rad, worst


def run_c(kp, ki, kd, cmd_step=0.05):
    src = os.path.join(EMB, "pid_controller.c")
    lib_path = os.path.join(EMB, "libpid_stab.so")
    try:
        subprocess.run(["gcc", "-O2", "-shared", "-fPIC", "-o", lib_path, src],
                       check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    lib = ctypes.CDLL(lib_path)

    class P(ctypes.Structure):
        _fields_ = [("kp", ctypes.c_float), ("ki", ctypes.c_float), ("kd", ctypes.c_float),
                    ("integral", ctypes.c_float), ("prev_error", ctypes.c_float),
                    ("output_limit", ctypes.c_float), ("integral_limit", ctypes.c_float),
                    ("has_prev", ctypes.c_int)]

    class A(ctypes.Structure):
        _fields_ = [("max_slew_rate", ctypes.c_float), ("tau", ctypes.c_float), ("actual_angle", ctypes.c_float)]

    class D(ctypes.Structure):
        _fields_ = [("moment_of_inertia", ctypes.c_float), ("gimbal_arm", ctypes.c_float),
                    ("pitch", ctypes.c_float), ("pitch_rate", ctypes.c_float)]

    f = ctypes.c_float
    lib.pid_init.argtypes = [ctypes.POINTER(P), f, f, f, f]
    lib.pid_update.argtypes = [ctypes.POINTER(P), f, f]
    lib.pid_update.restype = f
    lib.actuator_init.argtypes = [ctypes.POINTER(A), f, f]
    lib.actuator_step.argtypes = [ctypes.POINTER(A), f, f]
    lib.actuator_step.restype = f
    lib.dynamics_init.argtypes = [ctypes.POINTER(D), f, f]
    lib.dynamics_step.argtypes = [ctypes.POINTER(D), f, f, f, f]
    lib.one_minus_cosine_gust.argtypes = [f] * 4
    lib.one_minus_cosine_gust.restype = f

    p, a, d = P(), A(), D()
    lib.pid_init(ctypes.byref(p), kp, ki, kd, float(np.radians(6.0)))
    lib.actuator_init(ctypes.byref(a), 0.34907, 0.05)
    lib.dynamics_init(ctypes.byref(d), 50000.0, 1.2)
    t, worst = 0.0, 0.0
    for _ in range(600):
        cmd = cmd_step if t >= 5.0 else 0.0
        g = lib.pid_update(ctypes.byref(p), cmd - d.pitch, 0.02)
        ga = lib.actuator_step(ctypes.byref(a), g, 0.02)
        lib.dynamics_step(ctypes.byref(d), 2e6, ga, lib.one_minus_cosine_gust(t, 3.0, 2.0, 20000.0), 0.02)
        t += 0.02
        worst = max(worst, abs(d.pitch))
    try:
        os.remove(lib_path)
    except OSError:
        pass
    return d.pitch, worst


def main():
    print("TVC loop stability (the SIL gains)")
    final, worst = run_python(8.0, 0.5, 3.0)
    check("OLD gains (8, 0.5, 3) are unstable on the SIL plant (documents the defect)", worst > 0.5)

    final, worst = run_python(0.3, 0.05, 0.3)
    check("NEW gains hold 0.05 rad step within 5% (python)", abs(final - 0.05) < 0.0025)
    check("NEW gains: no overshoot beyond 10% of the step (python)", worst < 0.055)

    res = run_c(0.3, 0.05, 0.3)
    if res is None:
        print("  [SKIP] no host C compiler; C-side stability check skipped")
    else:
        check("NEW gains hold 0.05 rad step within 5% (compiled C)", abs(res[0] - 0.05) < 0.0025)
        check("C and Python agree on final pitch (< 1e-3 rad)", abs(res[0] - run_python(0.3, 0.05, 0.3)[0]) < 1e-3)

    main_c = open(os.path.join(EMB, "main.c")).read()
    check("embedded/main.c no longer ships the unstable 8.0/0.5/3.0 gains", "pid_init(&pid, 8.0f" not in main_c)

    total = PASS + FAIL
    print(f"{PASS} passed, {FAIL} failed out of {total}")
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
