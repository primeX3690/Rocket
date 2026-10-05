"""
tests/test_environment.py

Verification suite for environment/wind_models.py, environment/aero_center.py,
dynamics/mass_properties.py, and dynamics/slosh.py.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment.wind_models import (
    log_law_wind_speed, jet_stream_bump, mean_wind_profile, DrydenTurbulence, wind_relative_velocity,
)
from environment.aero_center import center_of_gravity_m, center_of_pressure_m, static_margin_calibers, stability_history
from dynamics.mass_properties import full_inertia_tensor, inertia_history, parallel_axis_shift
from dynamics.slosh import SloshPendulum

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


def test_wind_profile_increases_with_altitude_in_boundary_layer():
    v_low = log_law_wind_speed(10.0, 8.0)
    v_high = log_law_wind_speed(500.0, 8.0)
    check("Log-law wind speed increases with altitude within the boundary layer",
          v_high > v_low)
    check("Log-law wind speed at reference altitude equals the reference speed",
          abs(log_law_wind_speed(10.0, 8.0, reference_altitude_m=10.0) - 8.0) < 1e-9)


def test_jet_stream_bump_peaks_near_tropopause():
    check("Jet-stream contribution peaks near its configured peak altitude",
          jet_stream_bump(11000.0) > jet_stream_bump(5000.0) and
          jet_stream_bump(11000.0) > jet_stream_bump(20000.0))


def test_dryden_turbulence_is_zero_mean_and_bounded():
    t = DrydenTurbulence(turbulence_intensity_m_s=2.0, seed=3)
    vals = [t.step(0.02) for _ in range(5000)]
    check("Dryden turbulence has near-zero long-run mean",
          abs(np.mean(vals)) < 0.5)
    check("Dryden turbulence standard deviation is close to the configured intensity",
          abs(np.std(vals) - 2.0) < 1.0)


def test_wind_relative_velocity_subtracts_wind():
    rel = wind_relative_velocity(np.array([50.0, 0.0, 0.0]), altitude_m=0.001,
                                 surface_wind_speed_m_s=0.0, include_jet_stream=False)
    check("With zero wind, air-relative velocity equals inertial velocity",
          np.allclose(rel, [50.0, 0.0, 0.0], atol=0.1))


def test_cg_shifts_from_propellant_to_dry_centroid_as_fuel_depletes():
    cg_full = center_of_gravity_m(10000, 3000, prop_centroid_m=6.0, dry_centroid_m=9.0)
    cg_empty = center_of_gravity_m(0, 3000, prop_centroid_m=6.0, dry_centroid_m=9.0)
    check("Cg with full tank is closer to propellant centroid than dry centroid",
          abs(cg_full - 6.0) < abs(cg_full - 9.0))
    check("Cg with empty tank equals the dry-structure centroid exactly",
          abs(cg_empty - 9.0) < 1e-9)


def test_cp_shifts_aft_through_transonic():
    cp_sub = center_of_pressure_m(15.0, 1.5, 3.0, mach=0.3)
    cp_trans = center_of_pressure_m(15.0, 1.5, 3.0, mach=1.1)
    cp_super = center_of_pressure_m(15.0, 1.5, 3.0, mach=4.0)
    check("Cp moves aft (increases) going from subsonic to supersonic",
          cp_sub < cp_trans < cp_super)


def test_static_margin_sign_convention():
    margin_stable = static_margin_calibers(cg_m=5.0, cp_m=8.0, body_diameter_m=1.0)
    margin_unstable = static_margin_calibers(cg_m=8.0, cp_m=5.0, body_diameter_m=1.0)
    check("Cp aft of Cg gives a positive (stable) static margin",
          margin_stable > 0)
    check("Cp forward of Cg gives a negative (unstable) static margin",
          margin_unstable < 0)


def test_stability_history_runs_over_a_burn():
    prop_series = np.linspace(10000, 0, 20)
    mach_series = np.linspace(0.1, 3.0, 20)
    hist = stability_history(prop_series, 3000, 6.0, 9.0, 15.0, 1.5, 3.0, mach_series)
    check("Stability history returns one Cg/Cp/margin value per input sample",
          len(hist["static_margin_calibers"]) == 20)


def test_inertia_tensor_symmetric_case_has_no_cross_terms():
    inertia, I_full, cg = full_inertia_tensor(10000, 0.75, 6.0, 3.0, 3000, 0.75, 4.0, 8.0)
    check("Axisymmetric mass distribution (zero tank offset) has zero I_xy",
          abs(I_full[0, 1]) < 1e-9)
    check("Axisymmetric mass distribution has zero I_xz",
          abs(I_full[0, 2]) < 1e-9)


def test_inertia_tensor_offset_case_has_cross_terms():
    _, I_full, _ = full_inertia_tensor(10000, 0.75, 6.0, 3.0, 3000, 0.75, 4.0, 8.0,
                                       tank_lateral_offset_m=0.1)
    check("A laterally-offset tank produces a nonzero I_xy cross term",
          abs(I_full[0, 1]) > 1e-6)


def test_inertia_decreases_as_propellant_depletes():
    hist = inertia_history(np.linspace(10000, 0, 10), 0.75, 6.0, 3.0, 3000, 0.75, 4.0, 8.0)
    check("Pitch/yaw inertia (Iyy) decreases monotonically as propellant depletes",
          np.all(np.diff(hist["iyy"]) <= 0))


def test_parallel_axis_shift_matches_manual_formula():
    I_local = np.diag([10.0, 20.0, 20.0])
    d = np.array([1.0, 0.0, 0.0])
    shifted = parallel_axis_shift(I_local, 5.0, d)
    # Shift along x should only add to Iyy and Izz (m*d^2), not Ixx.
    check("Parallel-axis shift along x leaves Ixx unchanged",
          abs(shifted[0, 0] - 10.0) < 1e-9)
    check("Parallel-axis shift along x adds m*d^2 to Iyy",
          abs(shifted[1, 1] - (20.0 + 5.0 * 1.0)) < 1e-9)


def test_slosh_pendulum_responds_to_lateral_forcing():
    p = SloshPendulum(slosh_mass_kg=200.0, tank_radius_m=0.75, fill_fraction=0.5, damping_ratio=0.02)
    thetas = []
    for i in range(1000):
        p.step(lateral_accel_m_s2=2.0, axial_accel_m_s2=15.0, dt=0.01)
        thetas.append(p.theta)
    check("Slosh pendulum angle responds (moves away from zero) under sustained lateral forcing",
          max(abs(t) for t in thetas) > 1e-4)
    check("Slosh reaction torque is nonzero once the pendulum has deflected",
          abs(p.reaction_torque_nm(15.0, moment_arm_m=2.0)) > 0.0)


def test_slosh_length_shrinks_as_tank_drains():
    p_full = SloshPendulum(200.0, 0.75, fill_fraction=0.9)
    p_empty = SloshPendulum(200.0, 0.75, fill_fraction=0.1)
    check("Slosh equivalent-pendulum length differs between a nearly-full and nearly-empty tank",
          abs(p_full.length - p_empty.length) > 1e-6)


if __name__ == "__main__":
    print("Running environment/mass-properties/slosh verification suite...\n")
    test_wind_profile_increases_with_altitude_in_boundary_layer()
    test_jet_stream_bump_peaks_near_tropopause()
    test_dryden_turbulence_is_zero_mean_and_bounded()
    test_wind_relative_velocity_subtracts_wind()
    test_cg_shifts_from_propellant_to_dry_centroid_as_fuel_depletes()
    test_cp_shifts_aft_through_transonic()
    test_static_margin_sign_convention()
    test_stability_history_runs_over_a_burn()
    test_inertia_tensor_symmetric_case_has_no_cross_terms()
    test_inertia_tensor_offset_case_has_cross_terms()
    test_inertia_decreases_as_propellant_depletes()
    test_parallel_axis_shift_matches_manual_formula()
    test_slosh_pendulum_responds_to_lateral_forcing()
    test_slosh_length_shrinks_as_tank_drains()

    print(f"\n{PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    if FAIL > 0:
        sys.exit(1)
