"""
tests/test_visualization.py

Verification suite for visualization/plots.py.

Since visual correctness can't be asserted numerically, these tests
verify the things that CAN be checked automatically: every plotting
function runs without raising an exception on realistic input data
(sourced from the actual simulation functions elsewhere in this
project, not synthetic placeholder data), produces a file at the
expected path, and that file is a non-trivial, valid PNG (correct
magic bytes, non-tiny file size — catching the common failure mode of
silently saving a blank/corrupt image).
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "visualization"))

from guidance.gravity_turn import simulate_ascent
from guidance.peg import PEGTarget, simulate_peg_guided_burn
from analysis.monte_carlo import compare_open_vs_closed_loop_dispersion
from dynamics.six_dof import SixDOFState, InertiaTensor, simulate_six_dof_free_flight
from faults.real_world import RealisticActuator, simulate_realistic_attitude_hold
from visualization.plots import (
    plot_ascent_profile, plot_peg_insertion, plot_monte_carlo_comparison,
    plot_six_dof_3d_trajectory, plot_realistic_vs_ideal_attitude
)

PASS = 0
FAIL = 0
R_EARTH = 6378137.0
MU_EARTH = 3.986004418e14

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "visualization", "output")
os.makedirs(OUT_DIR, exist_ok=True)


def check(name, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


def _is_valid_nontrivial_png(path, min_bytes=5000):
    if not os.path.exists(path):
        return False
    size = os.path.getsize(path)
    if size < min_bytes:
        return False
    with open(path, "rb") as f:
        header = f.read(8)
    return header == b"\x89PNG\r\n\x1a\n"


def test_ascent_profile_plot():
    result = simulate_ascent(m0_kg=50000, m_dry_kg=6000, thrust_n=1_200_000, isp_s=280,
                              drag_coeff=0.3, ref_area_m2=2.5,
                              kick_altitude_m=1000, kick_angle_deg=2.0)
    path = os.path.join(OUT_DIR, "test_ascent_profile.png")
    plot_ascent_profile(result, path)
    check("Ascent profile plot produces a valid, non-trivial PNG file",
          _is_valid_nontrivial_png(path))


def test_peg_insertion_plot():
    target_r = R_EARTH + 300000
    target_v = np.sqrt(MU_EARTH / target_r)
    target = PEGTarget(target_r, target_v, 0.0)
    result = simulate_peg_guided_burn(r0_m=R_EARTH + 180000, v0_m_s=7300.0,
                                       gamma0_rad=np.radians(3.0), m0_kg=4000.0,
                                       thrust_n=90000.0, isp_s=320.0, target=target)
    path = os.path.join(OUT_DIR, "test_peg_insertion.png")
    plot_peg_insertion(result, target_r, path)
    check("PEG insertion plot produces a valid, non-trivial PNG file",
          _is_valid_nontrivial_png(path))


def test_monte_carlo_comparison_plot():
    target_r = R_EARTH + 300000
    target_v = np.sqrt(MU_EARTH / target_r)
    target = PEGTarget(target_r, target_v, 0.0)
    comparison = compare_open_vs_closed_loop_dispersion(
        nominal_r0_m=R_EARTH + 180000, nominal_v0_m_s=7300.0,
        nominal_gamma0_rad=np.radians(3.0), nominal_m0_kg=4000.0,
        thrust_n=90000.0, nominal_isp_s=320.0, target=target,
        fixed_chi_deg=-33.3, n_trials=50,
    )
    path = os.path.join(OUT_DIR, "test_monte_carlo.png")
    plot_monte_carlo_comparison(comparison, path)
    check("Monte Carlo comparison plot produces a valid, non-trivial PNG file",
          _is_valid_nontrivial_png(path))


def test_six_dof_3d_plot():
    state = SixDOFState(position=[0, 0, 0], velocity=[0, 0, 0], quaternion=[1, 0, 0, 0],
                         angular_velocity=[0, 0, 0], mass=8000.0)
    inertia = InertiaTensor(100.0, 150.0, 120.0)
    thrust_func = lambda t, s: np.array([15000.0 * np.sin(t * 0.15), 8000.0 * np.cos(t * 0.1), 180000.0])
    torque_func = lambda t, s: np.array([0.0, 0.0, 0.0])
    result = simulate_six_dof_free_flight(state, inertia, thrust_func, torque_func,
                                           mdot_kg_s=25.0, gravity_inertial=np.array([0, 0, -9.807]),
                                           dt=0.05, t_max=15.0)
    path = os.path.join(OUT_DIR, "test_six_dof_3d.png")
    plot_six_dof_3d_trajectory(result, path)
    check("6DOF 3D trajectory plot produces a valid, non-trivial PNG file",
          _is_valid_nontrivial_png(path))


def test_realistic_vs_ideal_plot():
    thrust, I, arm = 1_200_000.0, 60000.0, 3.5
    K = thrust * arm / I
    wn, zeta = 2.0, 0.9
    kp, kd = wn ** 2 / K, 2 * zeta * wn / K
    ideal = RealisticActuator(max_slew_rate_deg_s=1e6, time_constant_s=1e-6)
    realistic = RealisticActuator(max_slew_rate_deg_s=15.0, time_constant_s=0.08)
    ideal_result = simulate_realistic_attitude_hold(0.0, thrust, I, arm, kp, 0.0, kd,
                                                      actuator=ideal, gust_peak_torque_nm=25000.0)
    realistic_result = simulate_realistic_attitude_hold(0.0, thrust, I, arm, kp, 0.0, kd,
                                                          actuator=realistic, gust_peak_torque_nm=25000.0)
    path = os.path.join(OUT_DIR, "test_realistic_vs_ideal.png")
    plot_realistic_vs_ideal_attitude(ideal_result, realistic_result, path)
    check("Realistic-vs-ideal attitude plot produces a valid, non-trivial PNG file",
          _is_valid_nontrivial_png(path))


if __name__ == "__main__":
    print("Running visualization verification suite...\n")
    test_ascent_profile_plot()
    test_peg_insertion_plot()
    test_monte_carlo_comparison_plot()
    test_six_dof_3d_plot()
    test_realistic_vs_ideal_plot()

    print(f"\n{PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    if FAIL > 0:
        sys.exit(1)
