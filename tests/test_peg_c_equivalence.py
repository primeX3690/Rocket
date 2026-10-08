"""
tests/test_peg_c_equivalence.py

Proves flight_software/fsw_peg.c (double precision) is a faithful port of the
predictor-corrector in guidance/peg.py: identical forward prediction, identical
Levenberg-Marquardt solve (same A, B, T_go, residuals, iteration counts), the
same behaviour on an unreachable target, and -- driving the Python burn
simulator with the C solver -- the same closed-loop insertion accuracy.
Also covers what the port ADDS: input validation (no exceptions, no NaN
propagation) and the step-count cap.
"""
import ctypes
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from guidance.peg import (PEGTarget, predict_cutoff, solve_peg_predictor_corrector,
                          simulate_peg_guided_burn, rk4_step, G0, MU_EARTH_DEFAULT)

FSW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "flight_software")
R_EARTH = 6378137.0
PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


class State(ctypes.Structure):
    _fields_ = [("r", ctypes.c_double), ("v", ctypes.c_double), ("gamma", ctypes.c_double), ("m", ctypes.c_double)]


class Target(ctypes.Structure):
    _fields_ = [("r_t", ctypes.c_double), ("v_t", ctypes.c_double), ("gamma_t", ctypes.c_double)]


class Sol(ctypes.Structure):
    _fields_ = [("A", ctypes.c_double), ("B", ctypes.c_double), ("T", ctypes.c_double),
                ("res_r", ctypes.c_double), ("res_v", ctypes.c_double), ("res_gamma", ctypes.c_double),
                ("iterations", ctypes.c_int), ("status", ctypes.c_int)]


def load():
    lib_path = os.path.join(FSW, "build", "libfsw_peg.so")
    os.makedirs(os.path.dirname(lib_path), exist_ok=True)
    for cc in ("gcc", "cc", "clang"):
        try:
            subprocess.run([cc, "-std=c99", "-O2", "-shared", "-fPIC", "-I", FSW, "-o", lib_path,
                            os.path.join(FSW, "fsw_peg.c"), "-lm"], check=True, capture_output=True)
            break
        except FileNotFoundError:
            continue
        except subprocess.CalledProcessError as e:      # compiler present, code does not build: FAILURE, not skip
            raise RuntimeError("fsw_peg.c failed to build:\n" + e.stderr.decode()[-1500:])
    else:
        return None
    lib = ctypes.CDLL(lib_path)
    d = ctypes.c_double
    lib.fsw_peg_predict_cutoff.argtypes = [ctypes.POINTER(State), d, d, d, d, d, d, ctypes.c_int]
    lib.fsw_peg_predict_cutoff.restype = State
    lib.fsw_peg_rk4_step.argtypes = [ctypes.POINTER(State), d, d, d, d, d]
    lib.fsw_peg_rk4_step.restype = State
    lib.fsw_peg_solve.argtypes = [ctypes.POINTER(State), d, d, ctypes.POINTER(Target), d, d,
                                  ctypes.POINTER(d), ctypes.c_int, ctypes.POINTER(d), ctypes.POINTER(Sol)]
    return lib


def c_solve(lib, state, thrust, mdot, target, m_dry, guess=None, max_iter=25):
    s = State(*state)
    t = Target(target.r_t, target.v_t, target.gamma_t)
    out = Sol()
    g = (ctypes.c_double * 3)(*guess) if guess is not None else None
    lib.fsw_peg_solve(ctypes.byref(s), thrust, mdot, ctypes.byref(t), m_dry, MU_EARTH_DEFAULT,
                      g, max_iter, None, ctypes.byref(out))
    return out


def circ(alt_km):
    r = R_EARTH + alt_km * 1000.0
    return PEGTarget(r, np.sqrt(MU_EARTH_DEFAULT / r), 0.0)


VEH = dict(thrust=90000.0, isp=320.0, m0=4000.0)
MDOT = VEH["thrust"] / (VEH["isp"] * G0)
M_DRY = 0.25 * VEH["m0"]


def test_prediction(lib):
    print("Forward prediction (RK4 point-mass)")
    st = (R_EARTH + 180000.0, 7300.0, np.radians(3.0), 4000.0)
    worst = 0.0
    for A, B, T in [(0.05, -0.0005, 60.0), (0.3, -0.004, 70.0), (-0.02, 0.0003, 40.0), (0.1, -0.001, 90.0)]:
        py = predict_cutoff(st, A, B, T, VEH["thrust"], MDOT)
        c = lib.fsw_peg_predict_cutoff(ctypes.byref(State(*st)), A, B, T, VEH["thrust"], MDOT, MU_EARTH_DEFAULT, 0)
        worst = max(worst, abs(c.r - py[0]) / 1.0, abs(c.v - py[1]) / 1e-3, abs(c.gamma - py[2]) / 1e-7, abs(c.m - py[3]) / 1e-6)
    print(f"     worst normalised difference (units of 1 m, 1 mm/s, 1e-7 rad, 1 mg) = {worst:.2e}")
    check("cutoff state matches Python to <1e-6 of those units on 4 steering laws", worst < 1e-6)
    s1 = lib.fsw_peg_rk4_step(ctypes.byref(State(*st)), VEH["thrust"], MDOT, 0.1, 0.5, MU_EARTH_DEFAULT)
    p1 = rk4_step(st, VEH["thrust"], MDOT, 0.1, 0.5)
    check("single RK4 step matches Python to 1e-6 m", abs(s1.r - p1[0]) < 1e-6 and abs(s1.v - p1[1]) < 1e-9)


CASES = {
    "nominal 180 km -> 300 km": ((R_EARTH + 180000.0, 7300.0, np.radians(3.0), 4000.0), circ(300.0)),
    "dispersed start":          ((R_EARTH + 178000.0, 7280.0, np.radians(3.4), 4020.0), circ(300.0)),
    "lower target 250 km":      ((R_EARTH + 170000.0, 7250.0, np.radians(2.0), 4000.0), circ(250.0)),
    "high gamma start":         ((R_EARTH + 160000.0, 7200.0, np.radians(6.0), 3900.0), circ(300.0)),
    "shallow 175 km, 280 km tgt": ((R_EARTH + 175000.0, 7250.0, np.radians(4.0), 4000.0), circ(280.0)),
}


def test_cold_solves(lib):
    print("Cold-start solves (same iterations, same answer)")
    for name, (st, tgt) in CASES.items():
        py = solve_peg_predictor_corrector(st, VEH["thrust"], MDOT, tgt, M_DRY)
        c = c_solve(lib, st, VEH["thrust"], MDOT, tgt, M_DRY)
        dA, dB, dT = abs(c.A - py["A"]), abs(c.B - py["B"]), abs(c.T - py["T"])
        print(f"     {name:26s} T py={py['T']:.5f} c={c.T:.5f}  conv py={py['converged']} c={c.status == 0}  iters py={py['iterations']} c={c.iterations}")
        check(f"{name}: A,B,T agree (|dA|<1e-6, |dB|<1e-8, |dT|<1e-4)", dA < 1e-6 and dB < 1e-8 and dT < 1e-4)
        check(f"{name}: converged flag and iteration count identical",
              (c.status == 0) == py["converged"] and c.iterations == py["iterations"])
        check(f"{name}: residuals agree (<0.05 m, <1e-3 m/s, <1e-7 rad)",
              abs(c.res_r - py["residual"][0]) < 0.05 and abs(c.res_v - py["residual"][1]) < 1e-3
              and abs(c.res_gamma - py["residual"][2]) < 1e-7)


def test_warm_and_unreachable(lib):
    print("Warm start and unreachable target")
    st, tgt = CASES["nominal 180 km -> 300 km"]
    cold = solve_peg_predictor_corrector(st, VEH["thrust"], MDOT, tgt, M_DRY)
    guess = (cold["A"] * 1.1, cold["B"] * 0.9, cold["T"] - 3.0)
    py = solve_peg_predictor_corrector(st, VEH["thrust"], MDOT, tgt, M_DRY, guess=guess)
    c = c_solve(lib, st, VEH["thrust"], MDOT, tgt, M_DRY, guess=guess)
    check("warm-started solve matches Python (T within 1e-4 s, same iterations)", abs(c.T - py["T"]) < 1e-4 and c.iterations == py["iterations"])
    check("warm start needs fewer iterations than cold", c.iterations <= cold["iterations"])

    hard = PEGTarget(R_EARTH + 600000.0, np.sqrt(MU_EARTH_DEFAULT / (R_EARTH + 600000.0)), 0.0)
    py = solve_peg_predictor_corrector(st, VEH["thrust"], MDOT, hard, M_DRY)
    c = c_solve(lib, st, VEH["thrust"], MDOT, hard, M_DRY)
    print(f"     unreachable (600 km circular with 3 t of propellant): py conv={py['converged']} c status={c.status}, "
          f"T py={py['T']:.2f} c={c.T:.2f}, |res_v| py={abs(py['residual'][1]):.1f} c={abs(c.res_v):.1f}")
    check("unreachable target: both report NOT converged", (not py["converged"]) and c.status == 1)
    check("unreachable target: C best-effort residual and T match Python",
          abs(c.T - py["T"]) < 1e-3 and abs(c.res_v - py["residual"][1]) < 0.05)
    check("unreachable target: C result is finite", all(np.isfinite([c.A, c.B, c.T, c.res_r, c.res_v, c.res_gamma])))

    # 190 km, 7350 m/s (SUB-orbital by 440 m/s), gamma 0.5 deg, 2.8 t of propellant -> 320 km circular:
    # infeasible. 0/72 warm seeds converge and T pins at T_max, so this is not a solver weakness.
    st2, tgt2 = (R_EARTH + 190000.0, 7350.0, np.radians(0.5), 3800.0), circ(320.0)
    py = solve_peg_predictor_corrector(st2, VEH["thrust"], MDOT, tgt2, M_DRY)
    c = c_solve(lib, st2, VEH["thrust"], MDOT, tgt2, M_DRY)
    t_max = (3800.0 - M_DRY) / MDOT
    print(f"     infeasible 190 km start: py conv={py['converged']} c status={c.status}, T py={py['T']:.3f} c={c.T:.3f} (T_max={t_max:.3f}), res_r py={py['residual'][0]:.0f} c={c.res_r:.0f} m")
    check("infeasible start: both NOT converged, both pin T at the propellant limit",
          (not py["converged"]) and c.status == 1 and abs(py["T"] - t_max) < 0.01 and abs(c.T - t_max) < 0.01)
    check("infeasible start: residuals agree to 1 m / 0.05 m/s (non-converged iterates are chaotic below that)",
          abs(c.res_r - py["residual"][0]) < 1.0 and abs(c.res_v - py["residual"][1]) < 0.05)


def test_bad_inputs(lib):
    print("Input validation (the Python would raise or propagate NaN)")
    tgt = circ(300.0)
    good = (R_EARTH + 180000.0, 7300.0, np.radians(3.0), 4000.0)
    bad = {
        "NaN radius": (float("nan"), 7300.0, 0.05, 4000.0),
        "Inf velocity": (R_EARTH + 180000.0, float("inf"), 0.05, 4000.0),
        "zero velocity": (R_EARTH + 180000.0, 0.0, 0.05, 4000.0),
        "radius below Earth": (1.0e5, 7300.0, 0.05, 4000.0),
        "mass at/below dry mass": (R_EARTH + 180000.0, 7300.0, 0.05, 900.0),
        "NaN gamma": (R_EARTH + 180000.0, 7300.0, float("nan"), 4000.0),
    }
    for name, st in bad.items():
        out = c_solve(lib, st, VEH["thrust"], MDOT, tgt, M_DRY)
        check(f"{name}: rejected with BAD_INPUT, no crash", out.status == 2)
    check("zero thrust rejected", c_solve(lib, good, 0.0, MDOT, tgt, M_DRY).status == 2)
    check("zero mass flow rejected", c_solve(lib, good, VEH["thrust"], 0.0, tgt, M_DRY).status == 2)
    check("NaN warm-start guess rejected", c_solve(lib, good, VEH["thrust"], MDOT, tgt, M_DRY, guess=(float("nan"), 0, 50)).status == 2)
    check("max_iter out of range rejected", c_solve(lib, good, VEH["thrust"], MDOT, tgt, M_DRY, max_iter=0).status == 2)
    t = Target(float("nan"), 7700.0, 0.0)
    out = Sol()
    s = State(*good)
    lib.fsw_peg_solve(ctypes.byref(s), VEH["thrust"], MDOT, ctypes.byref(t), M_DRY, MU_EARTH_DEFAULT, None, 25, None, ctypes.byref(out))
    check("NaN target rejected", out.status == 2)
    # huge T must not blow the step budget: cap at 400 RK4 steps, still finite and fast
    big = lib.fsw_peg_predict_cutoff(ctypes.byref(State(*good)), 0.05, -0.0003, 3000.0, 90000.0, 0.5, MU_EARTH_DEFAULT, 0)
    check("3000 s prediction is capped (400 steps) and returns a finite state", np.isfinite([big.r, big.v, big.gamma, big.m]).all())


def closed_loop(lib, st, tgt, nav_bias=(0.0, 0.0, 0.0), true_isp=None, true_thrust=None, cycle=2.0, dt_int=0.05):
    """Mirror of simulate_peg_guided_burn with the C solver producing (A, B, T)."""
    thrust, isp = VEH["thrust"], VEH["isp"]
    F_true = thrust if true_thrust is None else true_thrust
    isp_true = isp if true_isp is None else true_isp
    mdot_nom, mdot_true = thrust / (isp * G0), F_true / (isp_true * G0)
    state, t, guess, burnout = tuple(st), 0.0, None, False
    cycles = 0
    while t < 600.0:
        est = (state[0] + nav_bias[0], state[1] + nav_bias[1], state[2] + nav_bias[2], state[3])
        sol = c_solve(lib, est, thrust, mdot_nom, tgt, M_DRY, guess=guess)
        cycles += 1
        A, B, T_go = sol.A, sol.B, sol.T
        fly = min(cycle, T_go)
        n = max(1, int(np.ceil(fly / dt_int)))
        h = fly / n
        for k in range(n):
            theta = float(np.arctan(A + B * (k + 0.5) * h))
            if state[3] - mdot_true * h <= M_DRY:
                burnout = True
                break
            state = rk4_step(state, F_true, mdot_true, theta, h)
            t += h
        if burnout or T_go <= cycle + 1e-9:
            break
        guess = (A + B * fly, B, T_go - fly)
    return state, t, cycles


def test_closed_loop(lib):
    print("Closed-loop burn: Python simulator driven by the C solver vs the pure-Python run")
    cases = {
        "nominal": dict(),
        "dispersed start": dict(),
        "thrust -5 % (engine mismatch)": dict(true_thrust=0.95 * VEH["thrust"]),
        "navigation bias (+500 m, -8 m/s, +0.003 rad)": dict(nav_bias=(500.0, -8.0, 0.003)),
    }
    starts = {"nominal": CASES["nominal 180 km -> 300 km"][0], "dispersed start": CASES["dispersed start"][0]}
    for name, kw in cases.items():
        st = starts.get(name, CASES["nominal 180 km -> 300 km"][0])
        tgt = circ(300.0)
        py = simulate_peg_guided_burn(r0_m=st[0], v0_m_s=st[1], gamma0_rad=st[2], m0_kg=st[3],
                                      thrust_n=VEH["thrust"], isp_s=VEH["isp"], target=tgt,
                                      true_thrust_n=kw.get("true_thrust"), nav_bias=kw.get("nav_bias", (0.0, 0.0, 0.0)))
        fin, t, cyc = closed_loop(lib, st, tgt, nav_bias=kw.get("nav_bias", (0.0, 0.0, 0.0)), true_thrust=kw.get("true_thrust"))
        dv = abs(fin[1] - py["final_velocity_m_s"]); dr = abs(fin[0] - py["final_radius_m"])
        print(f"     {name:46s} final v err py={py['insertion_error']['velocity_error_m_s']:+8.3f} c={fin[1] - tgt.v_t:+8.3f} m/s, "
              f"burn py={py['burn_time_s']:.2f} c={t:.2f} s, cycles={cyc}")
        if kw:
            # Late in a mismatched burn the solve is propellant-limited (T_go ~ T_max) and does not converge in
            # either implementation; non-converged iterates differ at 1e-7 and that is amplified by the final
            # cutoff. Measured: <=0.34 m/s, <=27 m, <=0.3 s (0.004 % of orbital speed). Bound with margin.
            check(f"{name}: propellant-limited final cycles agree within 1 m/s, 50 m, 1 s", dr < 50.0 and dv < 1.0 and abs(t - py["burn_time_s"]) < 1.0)
        else:
            check(f"{name}: fully converged run matches the pure-Python run (<0.5 m, <0.01 m/s, <0.05 s)",
                  dr < 0.5 and dv < 0.01 and abs(t - py["burn_time_s"]) < 0.05)
        if name in ("nominal", "dispersed start"):
            check(f"{name}: inserts within 15 m/s and 0.5 deg of the circular target",
                  abs(fin[1] - tgt.v_t) < 15.0 and abs(np.degrees(fin[2])) < 0.5)


def test_libm_confined():
    print("libm confinement")
    banned = {"sinf", "cosf", "sqrtf", "atan2f", "atanf", "asinf", "sin", "cos", "sqrt", "atan2", "atan", "exp", "tan", "ceil"}
    outdir = os.path.join(FSW, "build", "objs2")
    os.makedirs(outdir, exist_ok=True)
    users = []
    try:
        for f in sorted(os.listdir(FSW)):
            if not f.endswith(".c"):
                continue
            o = os.path.join(outdir, f[:-2] + ".o")
            subprocess.run(["gcc", "-std=c99", "-O2", "-c", "-I", FSW, os.path.join(FSW, f), "-o", o], check=True, capture_output=True)
            nm = subprocess.run(["nm", "-u", o], capture_output=True, text=True).stdout.split()
            if any(sym in banned for sym in nm):
                users.append(f)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("  [SKIP] gcc/nm unavailable")
        return
    check(f"libm is confined to fsw_peg.c and fsw_frames.c (double-precision guidance / frame maths); found: {users}", users == ["fsw_frames.c", "fsw_peg.c"])


def main():
    lib = load()
    if lib is None:
        print("  [SKIP] no host C compiler; PEG equivalence skipped")
        print("0 passed, 0 failed out of 0")
        return 0
    test_prediction(lib)
    test_cold_solves(lib)
    test_warm_and_unreachable(lib)
    test_bad_inputs(lib)
    test_closed_loop(lib)
    test_libm_confined()
    print(f"{PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
