"""
tests/test_eci_nav.py

Independent verification (numpy / scipy, different formulations from the C) of the orbit-scale navigation
pieces: fsw_frames.c (ECEF <-> pad-aligned inertial frame, Earth rotation, launch azimuth, polar quantities),
the central + J2 gravity model in fsw_eskf.c, long-horizon propagation against a DOP853 reference, the
gravity-gradient block of the covariance propagation against a finite-difference Jacobian, and the
decorrelating reset used for hand-offs.
"""
import ctypes
import os
import subprocess
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_eskf_c_equivalence import CEskf, Cfg  # noqa: E402

FSW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "flight_software")
MU, J2, REQ, OMEGA = 3.986004418e14, 1.08262668e-3, 6378137.0, 7.2921150e-5
PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


class Site(ctypes.Structure):
    _fields_ = [(n, ctypes.c_double) for n in ("lat", "lon", "r_pad", "az", "omega_e")] + \
               [("bx", ctypes.c_double * 3), ("by", ctypes.c_double * 3), ("bz", ctypes.c_double * 3), ("axis", ctypes.c_double * 3)]


def load():
    lib_path = os.path.join(FSW, "build", "libfsw_eci.so")
    os.makedirs(os.path.dirname(lib_path), exist_ok=True)
    srcs = [os.path.join(FSW, f) for f in ("fsw_eskf.c", "fsw_math.c", "fsw_frames.c")]
    for cc in ("gcc", "cc", "clang"):
        try:
            subprocess.run([cc, "-std=c99", "-O2", "-shared", "-fPIC", "-I", FSW, "-o", lib_path] + srcs + ["-lm"],
                           check=True, capture_output=True)
            break
        except FileNotFoundError:
            continue
        except subprocess.CalledProcessError as e:      # compiler present, code does not build: FAILURE, not skip
            raise RuntimeError("ECI navigation sources failed to build:\n" + e.stderr.decode()[-1500:])
    else:
        return None
    lib = ctypes.CDLL(lib_path)
    d3 = ctypes.c_double * 3
    d = ctypes.c_double
    lib.fsw_site_init.argtypes = [ctypes.POINTER(Site), d, d, d, d, d]
    lib.fsw_site_pad_state.argtypes = [ctypes.POINTER(Site), d3, d3]
    lib.fsw_ecef_to_inertial.argtypes = [ctypes.POINTER(Site), d3, d, d3]
    lib.fsw_inertial_to_ecef.argtypes = [ctypes.POINTER(Site), d3, d, d3]
    lib.fsw_polar_from_inertial.argtypes = [d3, d3] + [ctypes.POINTER(d)] * 4
    lib.fsw_eskf_gravity.argtypes = [ctypes.c_void_p, d3, d3]
    lib.fsw_eskf_reset_pv.argtypes = [ctypes.c_void_p, d3, d3, ctypes.c_float, ctypes.c_float]
    lib.fsw_eskf_set_P_diag.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float]
    return lib


def basis(lat, lon, az):
    E = np.array([-np.sin(lon), np.cos(lon), 0.0])
    N = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    U = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    x = np.sin(az) * E + np.cos(az) * N
    z = U
    y = np.cross(z, x)
    return np.vstack([x, y, z]), U


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def make_site(lib, lat_deg, lon_deg, az_deg, r_pad=REQ + 10.0):
    s = Site()
    lib.fsw_site_init(ctypes.byref(s), np.radians(lat_deg), np.radians(lon_deg), r_pad, np.radians(az_deg), OMEGA)
    return s


def test_frames(lib):
    print("Frames: ECEF <-> pad-aligned inertial (numpy reference built independently)")
    rng = np.random.default_rng(5)
    worst_t0 = worst_rot = worst_rt = worst_v = 0.0
    for lat, lon, az in [(13.72, 80.23, 90.0), (28.5, -80.6, 90.0), (-5.2, -52.7, 70.0), (60.0, 20.0, 20.0)]:
        s = make_site(lib, lat, lon, az)
        M, U = basis(np.radians(lat), np.radians(lon), np.radians(az))
        r_pad = REQ + 10.0
        ecef_pad = r_pad * U
        out = (ctypes.c_double * 3)()
        lib.fsw_ecef_to_inertial(ctypes.byref(s), (ctypes.c_double * 3)(*ecef_pad), 0.0, out)
        worst_t0 = max(worst_t0, np.max(np.abs(np.array(out) - [0, 0, r_pad])))
        for _ in range(6):
            e = rng.normal(0, 1, 3) * 3e6 + ecef_pad
            t = rng.uniform(0, 20000.0)
            ref = M @ (rz(OMEGA * t) @ e)                       # rotate in ECEF about the spin axis, then project
            lib.fsw_ecef_to_inertial(ctypes.byref(s), (ctypes.c_double * 3)(*e), t, out)
            worst_rot = max(worst_rot, np.max(np.abs(np.array(out) - ref)))
            back = (ctypes.c_double * 3)()
            lib.fsw_inertial_to_ecef(ctypes.byref(s), out, t, back)
            worst_rt = max(worst_rt, np.max(np.abs(np.array(back) - e)))
        p0, v0 = (ctypes.c_double * 3)(), (ctypes.c_double * 3)()
        lib.fsw_site_pad_state(ctypes.byref(s), p0, v0)
        v_ref = M @ np.cross([0, 0, OMEGA], ecef_pad)
        worst_v = max(worst_v, np.max(np.abs(np.array(v0) - v_ref)))
        if az == 90.0 and lat == 13.72:
            check("due-east launch: pad velocity is purely +X (downrange) with magnitude Omega*r*cos(lat) ~ 451.8 m/s",
                  abs(v0[0] - OMEGA * r_pad * np.cos(np.radians(lat))) < 1e-6 and abs(v0[1]) < 1e-6 and abs(v0[2]) < 1e-6)
    print(f"     pad position error {worst_t0:.1e} m, rotation vs numpy {worst_rot:.1e} m, round trip {worst_rt:.1e} m, pad velocity {worst_v:.1e} m/s")
    check("the pad's ECEF position maps to (0, 0, r_pad) at t_L for 4 sites / azimuths (< 1e-6 m)", worst_t0 < 1e-6)
    check("ECEF -> inertial matches M * Rz(Omega t) * ecef over 24 random points and times (< 1e-6 m)", worst_rot < 1e-6)
    check("inertial -> ECEF round trip (< 1e-6 m)", worst_rt < 1e-6)
    check("pad inertial velocity equals M (omega x r) for every azimuth (< 1e-9 m/s)", worst_v < 1e-9)
    s = make_site(lib, 13.72, 80.23, 90.0)
    e = (ctypes.c_double * 3)(*(np.array([4.0e6, 3.0e6, 3.5e6])))
    o1, o2 = (ctypes.c_double * 3)(), (ctypes.c_double * 3)()
    lib.fsw_ecef_to_inertial(ctypes.byref(s), e, 0.0, o1)
    lib.fsw_ecef_to_inertial(ctypes.byref(s), e, 2 * np.pi / OMEGA, o2)
    check("after one sidereal day (2 pi / Omega) the same ECEF point returns to the same inertial vector (< 1e-5 m)", np.max(np.abs(np.array(o1) - np.array(o2))) < 1e-5)
    t = 10.0
    lib.fsw_ecef_to_inertial(ctypes.byref(s), e, t, o2)
    check("Earth rotation: a point at the equatorial radius moves 465 m/s -> ~4.65 km in 10 s",
          abs(np.linalg.norm(np.array(o2) - np.array(o1)) - OMEGA * t * np.hypot(4.0e6, 3.0e6)) < 5.0)

    print("Polar quantities")
    worst = 0.0
    for _ in range(50):
        p = rng.normal(0, 1, 3) * 1e6 + [0, 0, 6.6e6]
        v = rng.normal(0, 1, 3) * 3e3
        r_, sp_, g_, ps_ = ctypes.c_double(), ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
        lib.fsw_polar_from_inertial((ctypes.c_double * 3)(*p), (ctypes.c_double * 3)(*v), r_, sp_, g_, ps_)
        ref = (np.linalg.norm(p), np.linalg.norm(v), np.arcsin(p @ v / (np.linalg.norm(p) * np.linalg.norm(v))), np.arctan2(p[0], p[2]))
        worst = max(worst, abs(r_.value - ref[0]), abs(sp_.value - ref[1]), abs(g_.value - ref[2]) * 1e6, abs(ps_.value - ref[3]) * 1e6)
    check("r, speed, flight-path angle, psi match numpy on 50 random states (< 1e-6 in m, m/s, urad)", worst < 1e-6)


def num_gravity(p, model, k_axis):
    r = np.linalg.norm(p)
    g = -MU * p / r ** 3
    if model == 2:
        # independent formulation: work in the polar-axis frame (z' = k) with the textbook component formula
        k = np.array(k_axis) / np.linalg.norm(k_axis)
        e1 = np.cross(k, [1.0, 0.0, 0.0])
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(k, [0.0, 1.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(k, e1)
        x, y, z = p @ e1, p @ e2, p @ k
        f = -1.5 * J2 * MU * REQ ** 2 / r ** 5
        a = f * np.array([x * (1 - 5 * z * z / r ** 2), y * (1 - 5 * z * z / r ** 2), z * (3 - 5 * z * z / r ** 2)])
        g = g + a[0] * e1 + a[1] * e2 + a[2] * k
    return g


def eskf_cfg(model, axis=(0, 0, 1.0)):
    return {"gravity_model": model, "mu": MU, "j2": J2, "r_eq": REQ, "polar_axis": (ctypes.c_double * 3)(*axis)}


def test_gravity(lib):
    print("Gravity models (central, central + J2) vs an independent formulation")
    axis = (0.0, np.cos(np.radians(13.72)), np.sin(np.radians(13.72)))
    rng = np.random.default_rng(11)
    worst_c = worst_j = 0.0
    for _ in range(8):
        p = rng.normal(0, 1, 3)
        p = p / np.linalg.norm(p) * rng.uniform(6.4e6, 7.5e6)
        for model in (1, 2):
            e = CEskf(lib, cfg=eskf_cfg(model, axis))
            g = (ctypes.c_double * 3)()
            lib.fsw_eskf_gravity(e.buf, (ctypes.c_double * 3)(*p), g)
            ref = num_gravity(p, model, axis)
            err = np.linalg.norm(np.array(g) - ref) / np.linalg.norm(ref)
            if model == 1:
                worst_c = max(worst_c, err)
            else:
                worst_j = max(worst_j, err)
    print(f"     relative error: central {worst_c:.1e}, central+J2 {worst_j:.1e}")
    check("point-mass gravity matches (relative error < 1e-12)", worst_c < 1e-12)
    check("point-mass + J2 matches the independent polar-axis formulation (relative error < 1e-12)", worst_j < 1e-12)
    p = np.array([0.0, 0.0, REQ + 200e3])
    e1_, e2_ = CEskf(lib, cfg=eskf_cfg(1, axis)), CEskf(lib, cfg=eskf_cfg(2, axis))
    g1, g2 = (ctypes.c_double * 3)(), (ctypes.c_double * 3)()
    lib.fsw_eskf_gravity(e1_.buf, (ctypes.c_double * 3)(*p), g1)
    lib.fsw_eskf_gravity(e2_.buf, (ctypes.c_double * 3)(*p), g2)
    dj = np.linalg.norm(np.array(g2) - np.array(g1))
    print(f"     J2 acceleration at 200 km above the pad: {dj:.4f} m/s^2")
    check("J2 acceleration at 200 km is 0.008 - 0.03 m/s^2 (the number quoted in the docs, ~0.014)", 0.008 < dj < 0.03)
    ef = CEskf(lib)
    gf = (ctypes.c_double * 3)()
    lib.fsw_eskf_gravity(ef.buf, (ctypes.c_double * 3)(1e6, 2e6, 3e6), gf)
    check("FLAT model returns the configured constant vector regardless of position", np.allclose(np.array(gf), [0, 0, -9.80665], atol=1e-6))
    ei = CEskf(lib, cfg=eskf_cfg(2, axis))
    gi = (ctypes.c_double * 3)()
    lib.fsw_eskf_gravity(ei.buf, (ctypes.c_double * 3)(0.0, 0.0, 0.0), gi)
    check("inside the Earth / uninitialised position: gravity is zero, not NaN/Inf", np.all(np.isfinite(np.array(gi))) and np.allclose(np.array(gi), 0))


def reference_orbit(p0, v0, T, model, axis):
    def f(t, y):
        return np.concatenate([y[3:], num_gravity(y[:3], model, axis)])
    sol = solve_ivp(f, (0, T), np.concatenate([p0, v0]), method="DOP853", rtol=1e-13, atol=1e-9)
    return sol.y[:3, -1], sol.y[3:, -1]


def test_propagation(lib):
    print("Long-horizon propagation (free fall, specific force = 0) vs DOP853 reference")
    axis = (0.0, np.cos(np.radians(13.72)), np.sin(np.radians(13.72)))
    r0 = REQ + 400e3
    p0 = np.array([0.0, 0.0, r0])
    v0 = np.array([np.sqrt(MU / r0), 0.0, 0.0])
    T, dt = 600.0, 0.02
    for model, name in ((1, "central"), (2, "central + J2")):
        e = CEskf(lib, cfg=eskf_cfg(model, axis), p0=p0, v0=v0)
        g0, a0 = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        for _ in range(int(T / dt)):
            e.predict(g0, a0, dt)
        p, v = e.pv()
        pr, vr = reference_orbit(p0, v0, T, model, axis)
        ep, ev = np.linalg.norm(p - pr), np.linalg.norm(v - vr)
        E0 = 0.5 * v0 @ v0 - MU / r0
        E1 = 0.5 * v @ v - MU / np.linalg.norm(p)
        print(f"     {name:13s}: 600 s ({int(T/dt)} steps) position error {ep:.3e} m, velocity error {ev:.3e} m/s, specific-energy drift {abs(E1 - E0):.2e} J/kg")
        check(f"{name}: position within 0.5 m and velocity within 5e-4 m/s of DOP853 after 600 s (30 000 filter steps)", ep < 0.5 and ev < 5e-4)
    e = CEskf(lib, cfg=eskf_cfg(1, axis), p0=p0, v0=v0)
    for _ in range(int(600.0 / dt)):
        e.predict([0, 0, 0], [0, 0, 0], dt)
    p, v = e.pv()
    h0, h1 = np.linalg.norm(np.cross(p0, v0)), np.linalg.norm(np.cross(p, v))
    check("central gravity: angular momentum conserved over 600 s to < 1e-9 relative", abs(h1 - h0) / h0 < 1e-9)
    check("filter stays healthy after 30 000 predictions in ECI", e.healthy())


def test_covariance_gradient(lib):
    print("Covariance propagation: gravity-gradient block vs finite-difference Jacobian (500 steps along the C filter's own path)")
    r0 = REQ + 200e3
    p0 = np.array([1.0e6, -2.0e5, np.sqrt(r0 ** 2 - 1.0e6 ** 2 - 2.0e5 ** 2)])
    dt, n = 0.02, 500
    e = CEskf(lib, cfg=eskf_cfg(1), p0=p0, v0=[7000.0, 100.0, 50.0])
    Pn = e.P().astype(np.float64)
    Pg = Pn.copy()
    Q = np.zeros((15, 15))
    Q[0:3, 0:3] = np.eye(3) * 1e-6 * dt
    Q[3:6, 3:6] = np.eye(3) * 1e-4 * dt
    Q[9:12, 9:12] = np.eye(3) * 1e-9 * dt
    Q[12:15, 12:15] = np.eye(3) * 1e-8 * dt
    h = 1000.0
    for _ in range(n):
        p_now, _v = e.pv()                       # F is built from the state at the START of the step
        G = np.zeros((3, 3))
        for j in range(3):
            d = np.zeros(3); d[j] = h
            G[:, j] = (num_gravity(p_now + d, 1, None) - num_gravity(p_now - d, 1, None)) / (2 * h)
        F = np.eye(15)
        F[6:9, 3:6] = np.eye(3) * dt
        F[0:3, 9:12] = -np.eye(3) * dt
        F[3:6, 12:15] = -np.eye(3) * dt
        Fg = F.copy()
        Fg[3:6, 6:9] = G * dt
        Pn = F @ Pn @ F.T + Q
        Pg = Fg @ Pg @ Fg.T + Q
        e.predict([0, 0, 0], [0, 0, 0], dt)
    P1 = e.P()
    sd = np.sqrt(np.outer(np.diag(Pg), np.diag(Pg)) + 1e-30)
    dev = np.max(np.abs(P1 - Pg) / sd)
    dev_nog = np.max(np.abs(P1 - Pn) / sd)
    print(f"     C vs numpy WITH gravity gradient: {dev:.2e} | C vs numpy WITHOUT it: {dev_nog:.2e} (correlation-normalised, max over 225 entries)")
    check("500 steps: all covariance entries match numpy F P F^T + Q with a finite-difference gravity gradient (< 2e-3)", dev < 2e-3)
    check("... and the same comparison WITHOUT the gradient term is clearly worse (> 10x): the block is present and correct", dev_nog > 10.0 * dev)
    check("covariance stays symmetric", np.max(np.abs(P1 - P1.T)) < 1e-6 * np.max(np.abs(P1)))


def test_reset_pv(lib):
    print("Hand-off reset: decorrelation keeps P positive semi-definite")
    e = CEskf(lib, cfg=eskf_cfg(1), p0=[0, 0, REQ + 1e5], v0=[7000.0, 0, 0])
    for i in range(300):
        e.predict([0.01, 0.02, -0.01], [1.0, 0.5, 20.0], 0.02)       # thrusting: builds attitude <-> velocity cross terms
    P = e.P()
    cross_before = np.max(np.abs(P[3:9, :3]))
    e.lib.fsw_eskf_reset_pv(e.buf, (ctypes.c_double * 3)(0.0, 0.0, REQ + 2e5), (ctypes.c_double * 3)(7500.0, 0, 0), 5.0, 0.05)
    P2 = e.P()
    others = [i for i in range(15) if i < 3 or i >= 9]
    cross_after = np.max(np.abs(P2[3:9][:, others]))
    eig_min = np.min(np.linalg.eigvalsh(0.5 * (P2 + P2.T)) / np.max(np.diag(P2)))
    p, v = e.pv()
    print(f"     cross terms before reset {cross_before:.2e}, after {cross_after:.1e}; min normalised eigenvalue {eig_min:.1e}")
    check("reset sets the nominal state and the position/velocity variances", np.allclose(p, [0, 0, REQ + 2e5]) and np.allclose(v, [7500, 0, 0])
          and abs(P2[6, 6] - 25.0) < 1e-4 and abs(P2[3, 3] - 0.0025) < 1e-7)
    check("cross-covariances between {velocity, position} and the other states are zeroed", cross_after == 0.0)
    check("covariance is positive semi-definite after the reset", eig_min > -1e-6)
    e2 = CEskf(lib, cfg=eskf_cfg(1), p0=[0, 0, REQ + 1e5], v0=[7000.0, 0, 0])
    for i in range(300):
        e2.predict([0.01, 0.02, -0.01], [1.0, 0.5, 20.0], 0.02)
    for k, val in zip(range(3, 9), (0.0025,) * 3 + (25.0,) * 3):
        e2.lib.fsw_eskf_set_P_diag(e2.buf, k, val)             # the WRONG way: diagonal only, old cross terms stay
    Pw = e2.P()
    eig_w = np.min(np.linalg.eigvalsh(0.5 * (Pw + Pw.T)) / np.max(np.diag(Pw)))
    print(f"     diagonal-only edit (the pitfall): min normalised eigenvalue {eig_w:.1e}")
    check("documenting the pitfall: editing only the diagonal leaves P NOT positive semi-definite", eig_w < -1e-8)


def main():
    lib = load()
    if lib is None:
        print("  [SKIP] no host C compiler; ECI navigation tests skipped")
        print("0 passed, 0 failed out of 0")
        return 0
    test_frames(lib)
    test_gravity(lib)
    test_propagation(lib)
    test_covariance_gradient(lib)
    test_reset_pv(lib)
    print(f"{PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
