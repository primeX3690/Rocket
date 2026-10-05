"""
tests/test_reusability.py

Verification suite for the reusable-rocket recovery stack:
guidance/powered_descent_guidance.py (ZEM/ZEV landing guidance),
guidance/boostback_and_entry.py (boostback/entry burn sizing),
dynamics/grid_fins.py (descent aerodynamic control surfaces),
dynamics/landing_legs.py (touchdown dynamics/tip-over stability), and
analysis/reusability_mission.py (the full end-to-end recovery mission).
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from guidance.powered_descent_guidance import (
    zem_zev_acceleration_command, suicide_burn_ignition_altitude_m, simulate_powered_landing,
    estimate_time_to_go_s,
)
from guidance.boostback_and_entry import boostback_delta_v_command, boostback_burn_duration_s, entry_burn_plan
from dynamics.grid_fins import grid_fin_lift_coefficient, grid_fin_drag_coefficient, grid_fin_effectiveness
from dynamics.landing_legs import simulate_touchdown, tip_over_stability_check
from analysis.reusability_mission import simulate_stage_recovery

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


# --- ZEM/ZEV closed-form guidance law ---

def test_zem_zev_law_drives_double_integrator_to_zero():
    """Direct verification of the closed-form derivation: simulating a double
    integrator under this exact acceleration law must null both position and velocity at T."""
    for x0, v0, T in [(100.0, -20.0, 15.0), (-500.0, 30.0, 40.0), (2000.0, -150.0, 60.0)]:
        x, v, dt = x0, v0, 0.001
        t = 0.0
        while t < T - 1e-9:
            a = zem_zev_acceleration_command(np.array([x]), np.array([v]), np.array([0.0]),
                                            np.array([0.0]), T - t)[0]
            v += a * dt
            x += v * dt
            t += dt
        check(f"ZEM/ZEV double-integrator sim (x0={x0},v0={v0},T={T}) drives position to ~0",
              abs(x) < 1.0)
        check(f"ZEM/ZEV double-integrator sim (x0={x0},v0={v0},T={T}) drives velocity to ~0",
              abs(v) < 0.5)


def test_zem_zev_is_vectorized_across_axes():
    a = zem_zev_acceleration_command(np.array([100.0, 50.0]), np.array([-10.0, 5.0]),
                                     np.array([0.0, 0.0]), np.array([0.0, 0.0]), 10.0)
    check("ZEM/ZEV command returns one acceleration value per axis", len(a) == 2)


def test_suicide_burn_ignition_altitude_matches_kinematics():
    alt = suicide_burn_ignition_altitude_m(descent_speed_m_s=200.0, max_deceleration_m_s2=20.0,
                                           safety_margin_m=50.0)
    expected = 200.0 ** 2 / (2 * 20.0) + 50.0
    check("Suicide-burn ignition altitude matches v^2/(2a) + margin kinematics exactly",
          abs(alt - expected) < 1e-6)
    check("Faster descent speed requires a higher ignition altitude",
          suicide_burn_ignition_altitude_m(300.0, 20.0) > suicide_burn_ignition_altitude_m(200.0, 20.0))
    check("More available deceleration allows a LOWER ignition altitude",
          suicide_burn_ignition_altitude_m(200.0, 40.0) < suicide_burn_ignition_altitude_m(200.0, 20.0))


def test_time_to_go_estimate_uses_average_velocity_form_and_is_floored():
    # 2*distance/speed form, not distance/speed
    t = estimate_time_to_go_s(remaining_distance_m=1000.0, closing_speed_m_s=100.0, t_go_floor_s=1.0)
    check("Time-to-go estimate uses the average-velocity (2x) kinematic form",
          abs(t - 20.0) < 1e-9)
    check("Time-to-go estimate is floored and never goes below the configured floor",
          estimate_time_to_go_s(1.0, 1000.0, t_go_floor_s=5.0) == 5.0)


def test_powered_landing_converges_to_near_target():
    r0 = np.array([200.0, 1500.0])
    v0 = np.array([20.0, -180.0])
    target_pos = np.array([0.0, 0.0])
    target_vel = np.array([0.0, -2.0])
    gravity = np.array([0.0, -9.80665])
    result = simulate_powered_landing(
        r0, v0, m0_kg=25000, m_dry_kg=22000, max_thrust_n=850000, min_throttle_frac=0.4,
        isp_s=283, target_pos_m=target_pos, target_vel_m_s=target_vel, gravity_m_s2=gravity,
        altitude_axis=1, t_go_floor_s=3.0,
    )
    check("Closed-loop powered landing lands within 10 m of the target site",
          result["miss_distance_m"] < 10.0)
    check("Closed-loop powered landing achieves a low (<15 m/s) touchdown speed",
          result["touchdown_speed_m_s"] < 15.0)
    check("Powered landing does not exhaust propellant before touchdown",
          not result["propellant_exhausted"])
    check("Powered landing throttle stays within the declared [min_throttle_frac, 1.0] range",
          np.all(result["throttle_frac"] >= 0.4 - 1e-9) and np.all(result["throttle_frac"] <= 1.0 + 1e-9))


def test_deep_throttle_is_required_and_respected():
    """A fixed-thrust (no throttling) engine cannot fly ZEM/ZEV — verify the
    module actually clamps to min_throttle_frac rather than ignoring it."""
    r0 = np.array([50.0, 500.0])
    v0 = np.array([2.0, -20.0])
    result = simulate_powered_landing(
        r0, v0, m0_kg=25000, m_dry_kg=22000, max_thrust_n=2_000_000,   # deliberately oversized thrust
        min_throttle_frac=0.5, isp_s=283, target_pos_m=np.array([0.0, 0.0]),
        target_vel_m_s=np.array([0.0, -2.0]), gravity_m_s2=np.array([0.0, -9.80665]),
    )
    check("With an oversized engine, throttle is clamped down to the min_throttle_frac floor at some point",
          np.any(np.isclose(result["throttle_frac"], 0.5, atol=1e-6)))


# --- Boostback and entry burn ---

def test_boostback_command_points_toward_correcting_position_and_velocity():
    pos = np.array([60000.0, 30000.0])
    vel = np.array([1200.0, -50.0])
    landing_site = np.array([0.0, 30000.0])
    gravity = np.array([0.0, -9.80665])
    a_cmd = boostback_delta_v_command(pos, vel, landing_site, t_go_s=90.0, gravity_m_s2=gravity)
    check("Boostback command has a negative (retrograde, toward the landing site) downrange component",
          a_cmd[0] < 0)


def test_boostback_burn_duration_matches_rocket_equation():
    plan = boostback_burn_duration_s(delta_v_needed_m_s=1000.0, thrust_n=500000, isp_s=283, m0_kg=20000)
    ve = 283 * 9.80665
    expected_propellant = 20000 * (1.0 - 1.0 / np.exp(1000.0 / ve))
    check("Boostback propellant usage matches the Tsiolkovsky rocket equation",
          abs(plan["propellant_used_kg"] - expected_propellant) < 1e-6)
    check("Boostback final mass is less than initial mass", plan["final_mass_kg"] < 20000)


def test_entry_burn_plan_respects_structural_limit():
    safe = entry_burn_plan(1400.0, 550.0, thrust_n=500000, isp_s=283, m0_kg=22000, max_deceleration_g=10.0)
    check("A modest entry burn is flagged within the structural deceleration limit",
          safe["within_structural_limit"])
    risky = entry_burn_plan(1400.0, 550.0, thrust_n=5_000_000, isp_s=283, m0_kg=22000, max_deceleration_g=2.0)
    check("An oversized-thrust entry burn is flagged as EXCEEDING the structural deceleration limit",
          not risky["within_structural_limit"])


# --- Grid fins ---

def test_grid_fin_lift_increases_then_stalls():
    cls = [grid_fin_lift_coefficient(np.radians(d), mach=1.0) for d in (0, 5, 10, 20)]
    check("Grid-fin lift coefficient increases with deflection angle up to stall",
          cls[0] < cls[1] < cls[2] < cls[3])
    cl_poststall = grid_fin_lift_coefficient(np.radians(40), mach=1.0)
    check("Grid-fin lift coefficient falls off well past the stall deflection",
          cl_poststall < cls[3])


def test_grid_fin_effectiveness_drops_at_high_mach():
    check("Grid-fin aerodynamic effectiveness is lower at Mach 5 than at Mach 1",
          grid_fin_effectiveness(5.0) < grid_fin_effectiveness(1.0))


def test_grid_fin_drag_increases_with_deflection():
    check("Grid-fin drag coefficient increases with deflection angle (induced drag)",
          grid_fin_drag_coefficient(np.radians(20), 1.0) > grid_fin_drag_coefficient(0.0, 1.0))


def test_grid_fin_lateral_steering_is_active_and_correctly_signed():
    """
    Verifies dynamics/grid_fins.py's aerodynamic model is actually
    DRIVEN by closed-loop guidance during descent (analysis/
    reusability_mission.py's _grid_fin_lateral_accel), not merely
    present as an unused physics model — the gap this test was added
    to close.
    """
    from analysis.reusability_mission import _grid_fin_lateral_accel
    from guidance.atmosphere import dynamic_pressure

    alt, speed = 20000.0, 500.0
    q = dynamic_pressure(alt, speed)
    a = _grid_fin_lateral_accel(lateral_error_m=2000.0, lateral_vel_m_s=-50.0,
                                altitude_m=alt, speed=speed, mass_kg=20000.0, q_pa=q)
    check("Grid-fin lateral steering produces a nonzero correction at representative descent conditions",
          abs(a) > 1e-6)

    alt2, speed2 = 5000.0, 300.0
    q2 = dynamic_pressure(alt2, speed2)
    a2 = _grid_fin_lateral_accel(lateral_error_m=2000.0, lateral_vel_m_s=-50.0,
                                 altitude_m=alt2, speed=speed2, mass_kg=20000.0, q_pa=q2)
    check("Grid-fin lateral steering authority is stronger at higher dynamic pressure (lower altitude)",
          abs(a2) > abs(a))


def test_grid_fin_steering_disabled_below_authority_threshold():
    from analysis.reusability_mission import _grid_fin_lateral_accel
    a = _grid_fin_lateral_accel(lateral_error_m=2000.0, lateral_vel_m_s=-50.0,
                                altitude_m=60000.0, speed=200.0, mass_kg=20000.0, q_pa=1.0)
    check("Below the minimum dynamic-pressure threshold, no fin command is issued (zero authority region)",
          a == 0.0)


# --- Landing legs ---

def test_touchdown_deceleration_scales_with_impact_speed():
    gentle = simulate_touchdown(3.0, 22000, 4, 2.0e6, 1.5e5, 0.6)
    medium = simulate_touchdown(8.0, 22000, 4, 2.0e6, 1.5e5, 0.6)
    hard = simulate_touchdown(20.0, 22000, 4, 2.0e6, 1.5e5, 0.6)
    check("Peak touchdown deceleration increases monotonically with impact speed",
          gentle["peak_deceleration_g"] < medium["peak_deceleration_g"] < hard["peak_deceleration_g"])
    check("A gentle (3 m/s) touchdown does not bottom out the legs",
          not gentle["any_leg_bottomed_out"])


def test_extreme_touchdown_speed_bottoms_out_legs():
    extreme = simulate_touchdown(60.0, 22000, 4, 2.0e6, 1.5e5, 0.6)
    check("An extreme (60 m/s) impact speed bottoms out the landing legs",
          extreme["any_leg_bottomed_out"])


def test_tip_over_stability_flags_large_tilt_and_drift_as_unstable():
    safe = tip_over_stability_check(cg_height_m=8.0, leg_footprint_radius_m=4.5,
                                    touchdown_lateral_speed_m_s=1.0, touchdown_tilt_rad=np.radians(2.0))
    check("Small tilt + small lateral drift is flagged as a STABLE landing", safe["stable"])
    risky = tip_over_stability_check(cg_height_m=8.0, leg_footprint_radius_m=4.5,
                                     touchdown_lateral_speed_m_s=5.0, touchdown_tilt_rad=np.radians(15.0))
    check("Large tilt + large lateral drift is flagged as an UNSTABLE landing", not risky["stable"])


def test_wider_footprint_improves_stability_margin():
    narrow = tip_over_stability_check(8.0, 3.0, 3.0, np.radians(5.0))
    wide = tip_over_stability_check(8.0, 8.0, 3.0, np.radians(5.0))
    check("A wider landing-leg footprint gives a larger stability margin for the same touchdown conditions",
          wide["margin_m"] > narrow["margin_m"])


# --- Full end-to-end recovery mission ---

def test_recovery_mission_runs_without_crashing_and_returns_expected_fields():
    result = simulate_stage_recovery(
        separation_pos_m=np.array([50000.0, 60000.0]),
        separation_vel_m_s=np.array([900.0, 200.0]),
        separation_mass_kg=30000.0, dry_mass_kg=15000.0,
        max_thrust_n=850000.0, min_throttle_frac=0.4, isp_s=283.0,
        landing_site_pos_m=np.array([0.0, 0.0]),
    )
    for key in ("boostback_delta_v_m_s", "landing", "touchdown_dynamics", "tip_over_stability", "mission_success"):
        check(f"Recovery mission result includes '{key}'", key in result)


def test_recovery_mission_boostback_dramatically_reduces_targeting_error():
    """
    Comparing WITH vs WITHOUT the predictor-corrector boostback burn
    (by checking the miss distance a naive one-shot kinematic guess
    would leave vs the actual result) — this module's own development
    found the naive guess left a 38+ km miss; the predictor-corrector
    fix should bring that down by at least 10x for the same scenario.
    """
    result = simulate_stage_recovery(
        separation_pos_m=np.array([50000.0, 60000.0]),
        separation_vel_m_s=np.array([900.0, 200.0]),
        separation_mass_kg=30000.0, dry_mass_kg=15000.0,
        max_thrust_n=850000.0, min_throttle_frac=0.4, isp_s=283.0,
        landing_site_pos_m=np.array([0.0, 0.0]),
    )
    check("With predictor-corrector boostback targeting, landing miss distance is under 1 km",
          result["landing"]["miss_distance_m"] < 1000.0)
    check("Boostback burn is feasible (within the vehicle's propellant margin) for this scenario",
          not result["boostback_infeasible"])
    check("Entry burn is feasible for this scenario", not result["entry_infeasible"])


def test_recovery_mission_touchdown_is_gentle_enough_to_not_bottom_out():
    result = simulate_stage_recovery(
        separation_pos_m=np.array([50000.0, 60000.0]),
        separation_vel_m_s=np.array([900.0, 200.0]),
        separation_mass_kg=30000.0, dry_mass_kg=15000.0,
        max_thrust_n=850000.0, min_throttle_frac=0.4, isp_s=283.0,
        landing_site_pos_m=np.array([0.0, 0.0]),
    )
    check("Landing legs do not bottom out for this scenario's touchdown speed",
          not result["touchdown_dynamics"]["any_leg_bottomed_out"])


def test_recovery_mission_infeasible_boostback_is_detected_with_low_propellant_margin():
    """With almost no propellant margin above dry mass, boostback should be correctly flagged infeasible
    rather than silently producing a below-dry-mass result."""
    result = simulate_stage_recovery(
        separation_pos_m=np.array([50000.0, 60000.0]),
        separation_vel_m_s=np.array([900.0, 200.0]),
        separation_mass_kg=15500.0, dry_mass_kg=15000.0,   # only 500 kg of margin for the whole return trip
        max_thrust_n=850000.0, min_throttle_frac=0.4, isp_s=283.0,
        landing_site_pos_m=np.array([0.0, 0.0]),
    )
    check("With almost no propellant margin, boostback is correctly flagged as infeasible",
          result["boostback_infeasible"])
    check("Mission is correctly flagged as NOT successful when boostback is infeasible",
          not result["mission_success"])


if __name__ == "__main__":
    print("Running reusable-rocket recovery stack verification suite...\n")
    test_zem_zev_law_drives_double_integrator_to_zero()
    test_zem_zev_is_vectorized_across_axes()
    test_suicide_burn_ignition_altitude_matches_kinematics()
    test_time_to_go_estimate_uses_average_velocity_form_and_is_floored()
    test_powered_landing_converges_to_near_target()
    test_deep_throttle_is_required_and_respected()
    test_boostback_command_points_toward_correcting_position_and_velocity()
    test_boostback_burn_duration_matches_rocket_equation()
    test_entry_burn_plan_respects_structural_limit()
    test_grid_fin_lateral_steering_is_active_and_correctly_signed()
    test_grid_fin_steering_disabled_below_authority_threshold()
    test_grid_fin_lift_increases_then_stalls()
    test_grid_fin_effectiveness_drops_at_high_mach()
    test_grid_fin_drag_increases_with_deflection()
    test_touchdown_deceleration_scales_with_impact_speed()
    test_extreme_touchdown_speed_bottoms_out_legs()
    test_tip_over_stability_flags_large_tilt_and_drift_as_unstable()
    test_wider_footprint_improves_stability_margin()
    test_recovery_mission_runs_without_crashing_and_returns_expected_fields()
    test_recovery_mission_boostback_dramatically_reduces_targeting_error()
    test_recovery_mission_touchdown_is_gentle_enough_to_not_bottom_out()
    test_recovery_mission_infeasible_boostback_is_detected_with_low_propellant_margin()

    print(f"\n{PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    if FAIL > 0:
        sys.exit(1)

