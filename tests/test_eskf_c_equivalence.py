"""
tests/test_eskf_c_equivalence.py

Proves flight_software/fsw_eskf.c (float32, libm-free) is a faithful port of
navigation/attitude_ekf.py (float64): both filters are fed byte-identical
inputs and their full state and covariance are compared. Also covers the
behaviours the C port ADDS (NIS gating, singular-S rejection, health check,
accelerometer leveling) and verifies the flight objects have no libm
dependency (nm -u).
"""
import ctypes
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from navigation.attitude_ekf import (StrapdownAttitudeEKF, quat_mult, quat_from_rotvec,
                                     quat_normalize, quat_to_rotmatrix)

FSW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "flight_software")
PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


def build_lib():
    lib = os.path.join(FSW, "build", "libfsw_eskf.so")
    os.makedirs(os.path.dirname(lib), exist_ok=True)
    srcs = [os.path.join(FSW, "fsw_eskf.c"), os.path.join(FSW, "fsw_math.c")]
    for cc in ("gcc", "cc", "clang"):
        try:
            subprocess.run([cc, "-std=c99", "-O2", "-shared", "-fPIC", "-I", FSW, "-o", lib] + srcs,
                           check=True, capture_output=True)
            return lib
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    return None


class Cfg(ctypes.Structure):
    _fields_ = [("gyro_noise_psd", ctypes.c_float), ("gyro_bias_psd", ctypes.c_float),
                ("accel_noise_psd", ctypes.c_float), ("accel_bias_psd", ctypes.c_float),
                ("gravity_nav", ctypes.c_float * 3), ("nis_gate", ctypes.c_float),
                ("max_consecutive_rejects", ctypes.c_uint8), ("gravity_model", ctypes.c_uint8),
                ("mu", ctypes.c_double), ("j2", ctypes.c_double), ("r_eq", ctypes.c_double),
                ("polar_axis", ctypes.c_double * 3)]


class CEskf:
    def __init__(self, lib, cfg=None, q0=None, p0=None, bg0=None, v0=None):
        self.lib = lib
        f3 = ctypes.c_float * 3
        f4 = ctypes.c_float * 4
        fp = ctypes.POINTER(ctypes.c_float)
        lib.fsw_eskf_sizeof.restype = ctypes.c_uint
        lib.fsw_eskf_default_cfg.argtypes = [ctypes.POINTER(Cfg)]
        dp = ctypes.POINTER(ctypes.c_double)
        lib.fsw_eskf_init.argtypes = [ctypes.c_void_p, ctypes.POINTER(Cfg), fp, dp, dp, fp, fp]
        lib.fsw_eskf_predict.argtypes = [ctypes.c_void_p, f3, f3, ctypes.c_float]
        lib.fsw_eskf_update_position.argtypes = [ctypes.c_void_p, ctypes.c_double * 3, ctypes.c_float]
        lib.fsw_eskf_update_position.restype = ctypes.c_int
        lib.fsw_eskf_healthy.argtypes = [ctypes.c_void_p]
        lib.fsw_eskf_align_accel.argtypes = [ctypes.c_void_p, f3]
        lib.fsw_eskf_get.argtypes = [ctypes.c_void_p, f4, f3, f3, f3, f3]
        lib.fsw_eskf_get_P.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
        lib.fsw_eskf_set_P_diag.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float]
        lib.fsw_eskf_pitch.argtypes = [ctypes.c_void_p]
        lib.fsw_eskf_pitch.restype = ctypes.c_float
        self.buf = ctypes.create_string_buffer(lib.fsw_eskf_sizeof())
        self.cfg = Cfg()
        lib.fsw_eskf_default_cfg(ctypes.byref(self.cfg))
        if cfg:
            for k, v in cfg.items():
                setattr(self.cfg, k, v)
        qa = (ctypes.c_float * 4)(*q0) if q0 is not None else None
        pa = (ctypes.c_double * 3)(*p0) if p0 is not None else None
        va = (ctypes.c_double * 3)(*v0) if v0 is not None else None
        ba = (ctypes.c_float * 3)(*bg0) if bg0 is not None else None
        lib.fsw_eskf_init(self.buf, ctypes.byref(self.cfg),
                          ctypes.cast(qa, fp) if qa else None, ctypes.cast(va, dp) if va else None,
                          ctypes.cast(pa, dp) if pa else None,
                          ctypes.cast(ba, fp) if ba else None, None)

    def predict(self, g, a, dt):
        self.lib.fsw_eskf_predict(self.buf, (ctypes.c_float * 3)(*g), (ctypes.c_float * 3)(*a), dt)

    def update(self, pos, r):
        return self.lib.fsw_eskf_update_position(self.buf, (ctypes.c_double * 3)(*pos), r)

    def state(self):
        q, v, p, bg, ba = ((ctypes.c_float * 4)(), (ctypes.c_float * 3)(), (ctypes.c_float * 3)(),
                           (ctypes.c_float * 3)(), (ctypes.c_float * 3)())
        self.lib.fsw_eskf_get(self.buf, q, v, p, bg, ba)
        return [np.array(x, dtype=float) for x in (q, v, p, bg, ba)]

    def P(self):
        out = (ctypes.c_float * 225)()
        self.lib.fsw_eskf_get_P(self.buf, out)
        return np.array(out, dtype=float).reshape(15, 15)

    def pv(self):
        v, p = (ctypes.c_double * 3)(), (ctypes.c_double * 3)()
        self.lib.fsw_eskf_get_pv.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        self.lib.fsw_eskf_get_pv(self.buf, v, p)
        return np.array(p), np.array(v)

    def healthy(self):
        return bool(self.lib.fsw_eskf_healthy(self.buf))


def f32(x):
    return np.asarray(x, dtype=np.float32).astype(np.float64)


def test_trajectory_equivalence(lib):
    print("Trajectory equivalence (the Python suite's own scenario, 2000 steps + GPS fixes)")
    rng = np.random.default_rng(7)
    dt = 0.01
    true_omega = np.array([0.0, 0.0, np.radians(4.0)])
    true_accel_body = np.array([0.0, 0.0, 9.80665 + 15.0])
    gyro_bias_true = np.array([0.0008, -0.0006, 0.0004])
    accel_bias_true = np.array([0.02, -0.015, 0.03])

    py = StrapdownAttitudeEKF()
    c = CEskf(lib)
    true_p, true_v, true_q = np.zeros(3), np.zeros(3), np.array([1.0, 0, 0, 0])
    max_q = max_p = max_v = max_bg = max_ba = max_dP = 0.0
    n_upd = 0
    for i in range(2000):
        true_q = quat_normalize(quat_mult(true_q, quat_from_rotvec(true_omega * dt)))
        a_n = quat_to_rotmatrix(true_q) @ true_accel_body + np.array([0, 0, -9.80665])
        true_v = true_v + a_n * dt
        true_p = true_p + true_v * dt
        g = f32(true_omega + gyro_bias_true + rng.normal(0, 0.0005, 3))
        a = f32(true_accel_body + accel_bias_true + rng.normal(0, 0.05, 3))
        py.predict(g, a, np.float32(dt))
        c.predict(g, a, dt)
        if i % 100 == 0 and i > 0:
            fix = f32(true_p + rng.normal(0, 3.0, 3))
            py.update_position(fix, r_pos=9.0)
            n_upd += c.update(fix, 9.0)
        if i % 50 == 0 or i == 1999:
            q, v, p, bg, ba = c.state()
            sign = 1.0 if np.dot(q, py.q) >= 0 else -1.0
            max_q = max(max_q, np.max(np.abs(sign * q - py.q)))
            max_p = max(max_p, np.max(np.abs(p - py.p)))
            max_v = max(max_v, np.max(np.abs(v - py.v)))
            max_bg = max(max_bg, np.max(np.abs(bg - py.bg)))
            max_ba = max(max_ba, np.max(np.abs(ba - py.ba)))
            Pc = c.P()
            d = np.abs(np.diag(Pc) - np.diag(py.P)) / (np.abs(np.diag(py.P)) + 1e-9)
            max_dP = max(max_dP, d.max())
    print(f"     max |dq|={max_q:.2e}  |dp|={max_p:.2e} m  |dv|={max_v:.2e} m/s  "
          f"|dbg|={max_bg:.2e}  |dba|={max_ba:.2e}  max rel dP_diag={max_dP:.2e}")
    check("all 19 position fixes accepted by the C filter", n_upd == 19)
    # float32 over 2000 renormalised steps: ~1.6e-5 measured (~0.001 deg), 4 orders below gyro noise
    check("quaternion matches Python (max abs diff < 5e-5, i.e. < 0.006 deg)", max_q < 5e-5)
    check("position matches Python (< 0.05 m over 20 s)", max_p < 0.05)
    check("velocity matches Python (< 0.01 m/s)", max_v < 0.01)
    check("gyro bias estimate matches Python (< 1e-5 rad/s)", max_bg < 1e-5)
    check("accel bias estimate matches Python (< 2e-3 m/s^2)", max_ba < 2e-3)
    check("covariance diagonal matches Python (rel < 2e-2)", max_dP < 2e-2)
    Pc = c.P()
    check("C covariance stays symmetric", np.max(np.abs(Pc - Pc.T)) < 1e-6 * max(1.0, np.max(np.abs(Pc))))
    check("C covariance diagonal stays positive", np.all(np.diag(Pc) > 0))
    check("C filter reports healthy at the end", c.healthy())
    check("final position error vs truth < 15 m (same bound as the Python test)",
          np.linalg.norm(c.state()[2] - true_p) < 15.0)


def test_large_angle_branch(lib):
    print("Large rotation per step (exercises the non-series sin/cos branch)")
    py = StrapdownAttitudeEKF()
    c = CEskf(lib)
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(40):
        g = f32(rng.normal(0, 2.0, 3))
        a = f32(np.array([0, 0, 9.80665]) + rng.normal(0, 0.3, 3))
        py.predict(g, a, np.float32(0.4))          # |omega*dt| ~ 1.4 rad
        c.predict(g, a, 0.4)
        q = c.state()[0]
        sign = 1.0 if np.dot(q, py.q) >= 0 else -1.0
        worst = max(worst, np.max(np.abs(sign * q - py.q)))
    print(f"     max |dq| after 40 large steps = {worst:.2e}")
    check("quaternion tracks Python through 1+ rad steps (< 1e-4)", worst < 1e-4)
    q = c.state()[0]
    check("quaternion stays unit-norm", abs(np.linalg.norm(q) - 1.0) < 1e-5)


def test_initial_conditions(lib):
    print("Initial conditions")
    q0 = quat_from_rotvec(np.array([0.1, -0.2, 0.3]))
    py = StrapdownAttitudeEKF(q0=q0, p0=[1, 2, 3], bg0=[0.001, 0, -0.001])
    c = CEskf(lib, q0=f32(q0), p0=[1, 2, 3], bg0=[0.001, 0, -0.001])
    q, v, p, bg, ba = c.state()
    check("non-default q0/p0/bg0 are honoured identically", np.allclose(q, py.q, atol=1e-6) and
          np.allclose(p, [1, 2, 3]) and np.allclose(bg, [0.001, 0, -0.001], atol=1e-7))
    check("initial covariance equals the Python initial covariance", np.allclose(np.diag(c.P()), np.diag(py.P)))


def test_gating_and_robustness(lib):
    print("Behaviour the port adds: gating, singular S, bad input, health")
    c = CEskf(lib, cfg={"nis_gate": 11.34, "max_consecutive_rejects": 3})
    for _ in range(100):
        c.predict([0, 0, 0], [0, 0, 9.80665], 0.02)
    check("a good fix is accepted with the gate on", c.update([1.0, -1.0, 0.5], 9.0) == 1)
    p_before = c.state()[2].copy()
    ok = c.update([500.0, 0.0, 0.0], 9.0)
    check("a 500 m outlier is rejected by the NIS gate", ok == 0 and np.allclose(c.state()[2], p_before))
    c.update([500.0, 0.0, 0.0], 9.0)
    c.update([500.0, 0.0, 0.0], 9.0)
    ok = c.update([500.0, 0.0, 0.0], 9.0)
    check("after max_consecutive_rejects the next fix is accepted (no permanent lock-out)", ok == 1)
    check("NaN position fix rejected", c.update([float("nan"), 0, 0], 9.0) == 0)
    check("Inf position fix rejected", c.update([float("inf"), 0, 0], 9.0) == 0)
    check("non-positive measurement variance rejected", c.update([0, 0, 0], 0.0) == 0)
    check("state still healthy after rejected inputs", c.healthy())

    c2 = CEskf(lib)
    c2.predict([0, 0, 0], [0, 0, 9.80665], 0.02)
    for i in range(6, 9):
        c2.lib.fsw_eskf_set_P_diag(c2.buf, i, 0.0)    # floor handles it only after a predict; force S ~ r only
    check("update still works with tiny position covariance (R keeps S invertible)", c2.update([0.1, 0, 0], 1.0) == 1)

    c3 = CEskf(lib)
    c3.lib.fsw_eskf_set_P_diag(c3.buf, 4, float("nan"))
    check("NaN in covariance is detected by the health check", not c3.healthy())
    c4 = CEskf(lib)
    c4.lib.fsw_eskf_set_P_diag(c4.buf, 0, -1.0)
    check("negative covariance diagonal is detected", not c4.healthy())
    c5 = CEskf(lib)
    c5.predict([float("nan"), 0, 0], [0, 0, 9.8], 0.02)
    check("NaN gyro input is caught by the health check (voter should have removed it)", not c5.healthy())


def test_alignment(lib):
    print("Static accelerometer leveling")
    ok_all = True
    worst = 0.0
    for roll_deg, pitch_deg in [(0, 0), (0, 5), (3, 0), (-4, 7), (10, -12), (1, 1)]:
        ph, th = np.radians(roll_deg), np.radians(pitch_deg)
        # R = Ry(th) Rx(ph); specific force at rest in body = R^T (0,0,g)
        Rx = np.array([[1, 0, 0], [0, np.cos(ph), -np.sin(ph)], [0, np.sin(ph), np.cos(ph)]])
        Ry = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
        f = (Ry @ Rx).T @ np.array([0, 0, 9.80665])
        c = CEskf(lib)
        ok = c.lib.fsw_eskf_align_accel(c.buf, (ctypes.c_float * 3)(*f))
        q = c.state()[0]
        Rc = quat_to_rotmatrix(q)
        err = np.max(np.abs(Rc - Ry @ Rx))
        worst = max(worst, err)
        ok_all &= bool(ok)
    print(f"     max |R_c - R_true| over 6 tilts = {worst:.2e}")
    check("alignment accepted for valid static specific force", ok_all)
    check("leveling recovers roll/pitch (rotation-matrix error < 1e-4)", worst < 1e-4)
    c = CEskf(lib)
    bad = c.lib.fsw_eskf_align_accel(c.buf, (ctypes.c_float * 3)(0, 0, 25.0))
    check("alignment refused while accelerating (|f| far from g); q unchanged", bad == 0 and np.allclose(c.state()[0], [1, 0, 0, 0]))
    c2 = CEskf(lib)
    pitch = c2.lib.fsw_eskf_pitch(c2.buf)
    check("pitch of identity attitude is 0", abs(pitch) < 1e-7)


def test_pitch_full_range(lib):
    print("Pitch extraction through and beyond 90 deg (regression: asin() form is singular and folds back)")
    worst_pure = worst_tilted = 0.0
    folded = []
    for phi in np.linspace(-3.0, 3.0, 25):
        q = np.array([np.cos(phi / 2), 0.0, np.sin(phi / 2), 0.0])
        c = CEskf(lib, q0=f32(q))
        worst_pure = max(worst_pure, abs(c.lib.fsw_eskf_pitch(c.buf) - phi))
        # the old formula, evaluated in float64: asin(2(wy - zx)) -> cannot exceed +-pi/2
        old = np.arcsin(np.clip(2 * (q[0] * q[2] - q[3] * q[1]), -1, 1))
        if abs(phi) > 1.7:
            folded.append(abs(old - phi))
        # small roll and yaw on top: result must equal atan2(R02, R22) of the same attitude
        qt = quat_mult(quat_mult(q, quat_from_rotvec(np.array([0.05, 0, 0]))), quat_from_rotvec(np.array([0, 0, 0.05])))
        c2 = CEskf(lib, q0=f32(qt))
        R = quat_to_rotmatrix(c2.state()[0])
        worst_tilted = max(worst_tilted, abs(c2.lib.fsw_eskf_pitch(c2.buf) - np.arctan2(R[0, 2], R[2, 2])))
    print(f"     max |pitch - truth| over -3..+3 rad: {worst_pure:.2e} rad; with 0.05 rad roll+yaw vs atan2(R02,R22): {worst_tilted:.2e}")
    check("pitch error < 1e-4 rad everywhere on (-3, +3) rad, i.e. through +-90 deg", worst_pure < 1e-4)
    check("with roll/yaw present, pitch equals atan2(R02, R22) of the attitude (< 1e-4)", worst_tilted < 1e-4)
    check("the asin() form really is wrong beyond 1.7 rad (documents what the regression guards against)", min(folded) > 0.2)


def test_no_libm():
    print("libm independence of the flight objects")
    objs = ["fsw_math", "fsw_eskf", "fsw_fdir", "fsw_state", "fsw_telemetry", "fsw_watchdog", "fsw_crc"]   # fsw_peg/fsw_guidance/fsw_core: see test_peg_c_equivalence.py (libm confinement)
    outdir = os.path.join(FSW, "build", "objs")
    os.makedirs(outdir, exist_ok=True)
    banned = {"sinf", "cosf", "sqrtf", "atan2f", "atanf", "asinf", "acosf", "tanf", "fabsf", "floorf",
              "powf", "expf", "logf", "sin", "cos", "sqrt", "atan2", "fabs", "pow", "exp", "log", "isnan", "isinf"}
    leaked = set()
    built = True
    for o in objs:
        try:
            r = subprocess.run(["gcc", "-std=c99", "-O2", "-c", "-I", FSW, os.path.join(FSW, o + ".c"),
                                "-o", os.path.join(outdir, o + ".o")], capture_output=True)
            built &= (r.returncode == 0)
            nm = subprocess.run(["nm", "-u", os.path.join(outdir, o + ".o")], capture_output=True, text=True)
            for line in nm.stdout.splitlines():
                sym = line.split()[-1] if line.split() else ""
                if sym in banned:
                    leaked.add(sym)
        except FileNotFoundError:
            print("  [SKIP] gcc/nm not available")
            return
    check("all flight objects compile standalone", built)
    check(f"no libm symbols referenced (found: {sorted(leaked) or 'none'})", not leaked)


def main():
    lib_path = build_lib()
    if lib_path is None:
        print("  [SKIP] no host C compiler; ESKF equivalence skipped")
        print("0 passed, 0 failed out of 0")
        return 0
    lib = ctypes.CDLL(lib_path)
    test_trajectory_equivalence(lib)
    test_large_angle_branch(lib)
    test_initial_conditions(lib)
    test_gating_and_robustness(lib)
    test_alignment(lib)
    test_pitch_full_range(lib)
    test_no_libm()
    print(f"{PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
