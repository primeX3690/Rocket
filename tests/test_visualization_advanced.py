"""
tests/test_visualization_advanced.py

Verification suite for analysis/dispersion_analysis.py (CEP, dispersion
ellipses, sensitivity data) and visualization/export_trajectory.py (the
JSON exporter feeding visualization/web/attitude_viewer.html).
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.dispersion_analysis import circular_error_probable, dispersion_ellipse, sensitivity_heatmap_data
from visualization.export_trajectory import export_quaternion_trajectory_json, export_six_dof_result
from dynamics.six_dof import SixDOFState, InertiaTensor, simulate_six_dof_free_flight

PASS = 0
FAIL = 0


def check(name, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


def test_cep_increases_with_percentile():
    rng = np.random.default_rng(2)
    x, y = rng.normal(0, 40, 3000), rng.normal(0, 40, 3000)
    cep50 = circular_error_probable(x, y, 50.0)
    cep90 = circular_error_probable(x, y, 90.0)
    check("CEP90 (90% containment radius) is larger than CEP50",
          cep90 > cep50 > 0)


def test_cep_scales_with_dispersion_magnitude():
    rng = np.random.default_rng(3)
    x_tight, y_tight = rng.normal(0, 10, 3000), rng.normal(0, 10, 3000)
    x_wide, y_wide = rng.normal(0, 100, 3000), rng.normal(0, 100, 3000)
    check("A 10x wider scatter gives an approximately proportionally larger CEP",
          circular_error_probable(x_wide, y_wide) > 5 * circular_error_probable(x_tight, y_tight))


def test_dispersion_ellipse_aligns_with_larger_spread_axis():
    rng = np.random.default_rng(5)
    x, y = rng.normal(0, 100, 3000), rng.normal(0, 20, 3000)   # x has 5x the spread of y
    ellipse = dispersion_ellipse(x, y, confidence=0.95)
    check("Dispersion ellipse's semi-major axis is larger than its semi-minor axis when spread is unequal",
          ellipse["semi_major_m"] > ellipse["semi_minor_m"])
    check("Dispersion ellipse center is near the scatter's true mean",
          abs(ellipse["center"][0]) < 10 and abs(ellipse["center"][1]) < 5)


def test_dispersion_ellipse_boundary_has_expected_point_count():
    x, y = np.random.default_rng(6).normal(0, 30, 500), np.random.default_rng(7).normal(0, 30, 500)
    ellipse = dispersion_ellipse(x, y)
    check("Dispersion ellipse boundary is returned as a closed set of points",
          len(ellipse["boundary_points"]) == 100)


def test_sensitivity_heatmap_detects_correlation_sign():
    rng = np.random.default_rng(8)
    n = 2000
    isp_error = rng.normal(0, 10, n)
    mc_result = {
        "radius_error_m": 500 * isp_error + rng.normal(0, 50, n),      # strongly positively correlated
        "velocity_error_m_s": -20 * isp_error + rng.normal(0, 2, n),   # strongly negatively correlated
    }
    heat = sensitivity_heatmap_data(mc_result, {"isp_error": isp_error})
    check("Sensitivity analysis correctly finds a strong POSITIVE correlation where one exists",
          heat["correlation_matrix"][0, 0] > 0.8)
    check("Sensitivity analysis correctly finds a strong NEGATIVE correlation where one exists",
          heat["correlation_matrix"][0, 1] < -0.8)


def test_sensitivity_heatmap_near_zero_for_unrelated_param():
    rng = np.random.default_rng(9)
    n = 2000
    unrelated = rng.normal(0, 1, n)
    mc_result = {"radius_error_m": rng.normal(0, 100, n), "velocity_error_m_s": rng.normal(0, 5, n)}
    heat = sensitivity_heatmap_data(mc_result, {"unrelated": unrelated})
    check("An unrelated parameter shows near-zero correlation with both outputs",
          abs(heat["correlation_matrix"][0, 0]) < 0.15 and abs(heat["correlation_matrix"][0, 1]) < 0.15)


def test_export_trajectory_json_round_trips():
    import json
    t = np.linspace(0, 5, 50)
    q = np.tile([1.0, 0.0, 0.0, 0.0], (50, 1))
    path = "/tmp/test_export_traj.json"
    export_quaternion_trajectory_json(t, q, path)
    data = json.load(open(path))
    check("Exported JSON has the expected 't' and 'quaternion' fields with correct lengths",
          len(data["t"]) == 50 and len(data["quaternion"]) == 50 and len(data["quaternion"][0]) == 4)
    os.remove(path)


def test_export_rejects_mismatched_shapes():
    caught = False
    try:
        export_quaternion_trajectory_json(np.linspace(0, 5, 50), np.zeros((49, 4)), "/tmp/bad.json")
    except ValueError:
        caught = True
    check("Exporting with mismatched t/quaternion lengths raises a clear ValueError",
          caught)


def test_export_six_dof_result_produces_normalized_quaternions():
    import json
    st = SixDOFState([0, 0, 0], [0, 0, 0], [1, 0, 0, 0], [0.1, 0.05, 0.02], 2000.0)
    inertia = InertiaTensor(800, 800, 120)
    result = simulate_six_dof_free_flight(st, inertia, lambda t, s: np.array([0, 0, 25000.0]),
                                          lambda t, s: np.zeros(3), 8.0, dt=0.02, t_max=3.0)
    path = "/tmp/test_sixdof_export.json"
    export_six_dof_result(result, path)
    data = json.load(open(path))
    q_sample = data["quaternion"][100]
    norm = sum(x ** 2 for x in q_sample) ** 0.5
    check("Exported six-DOF quaternions remain unit-norm (well-formed rotations)",
          abs(norm - 1.0) < 1e-6)
    os.remove(path)


if __name__ == "__main__":
    print("Running dispersion-analysis / trajectory-export verification suite...\n")
    test_cep_increases_with_percentile()
    test_cep_scales_with_dispersion_magnitude()
    test_dispersion_ellipse_aligns_with_larger_spread_axis()
    test_dispersion_ellipse_boundary_has_expected_point_count()
    test_sensitivity_heatmap_detects_correlation_sign()
    test_sensitivity_heatmap_near_zero_for_unrelated_param()
    test_export_trajectory_json_round_trips()
    test_export_rejects_mismatched_shapes()
    test_export_six_dof_result_produces_normalized_quaternions()

    print(f"\n{PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    if FAIL > 0:
        sys.exit(1)
