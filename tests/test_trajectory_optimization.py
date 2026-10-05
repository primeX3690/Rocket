"""
tests/test_trajectory_optimization.py

Verification suite for guidance/trajectory_optimization.py (successive-
convexification-style ascent pitch-profile optimizer) and
guidance/staging_events.py (stage separation / fairing jettison events).
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from guidance.trajectory_optimization import optimize_ascent_pitch_profile
from guidance.staging_events import (
    SeparationEvent, apply_separation_event, fairing_jettison_event, stage_separation_event,
)

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


def test_optimizer_respects_max_q_constraint():
    result = optimize_ascent_pitch_profile(
        m0_kg=60000, m_dry_kg=12000, thrust_n=850000, isp_s=285,
        drag_coeff=0.3, ref_area_m2=2.0, max_q_limit_pa=45000, max_pitch_rate_deg_s=3.0,
        n_segments=6, t_max=100.0, dt=0.5, max_outer_iterations=3,
    )
    check("Optimized trajectory's max dynamic pressure is within the constraint (allowing 1% solver slack)",
          result["max_q_pa"] <= result["max_q_limit_pa"] * 1.01)
    check("Optimizer reports max_q_satisfied=True",
          result["max_q_satisfied"])


def test_optimizer_respects_pitch_rate_constraint():
    result = optimize_ascent_pitch_profile(
        m0_kg=60000, m_dry_kg=12000, thrust_n=850000, isp_s=285,
        drag_coeff=0.3, ref_area_m2=2.0, max_q_limit_pa=60000, max_pitch_rate_deg_s=1.0,
        n_segments=6, t_max=100.0, dt=0.5, max_outer_iterations=3,
    )
    rates = np.abs(np.diff(result["knot_pitch_deg"])) / np.diff(result["knot_times_s"])
    check("Optimized knot-to-knot pitch rates stay within the configured max-pitch-rate limit (10% slack)",
          np.all(rates <= 1.0 * 1.10))


def test_tighter_max_q_constraint_reduces_final_speed():
    """A tighter dynamic-pressure ceiling should force a more lofted (slower-turning) trajectory,
    reducing achievable burnout speed for the same propellant -- a real, physically expected trade-off."""
    loose = optimize_ascent_pitch_profile(
        m0_kg=60000, m_dry_kg=12000, thrust_n=850000, isp_s=285,
        drag_coeff=0.3, ref_area_m2=2.0, max_q_limit_pa=90000, max_pitch_rate_deg_s=3.0,
        n_segments=6, t_max=100.0, dt=0.5, max_outer_iterations=3,
    )
    tight = optimize_ascent_pitch_profile(
        m0_kg=60000, m_dry_kg=12000, thrust_n=850000, isp_s=285,
        drag_coeff=0.3, ref_area_m2=2.0, max_q_limit_pa=25000, max_pitch_rate_deg_s=3.0,
        n_segments=6, t_max=100.0, dt=0.5, max_outer_iterations=3,
    )
    check("A tighter max-Q constraint results in lower (or equal) achieved burnout speed",
          tight["final_speed_m_s"] <= loose["final_speed_m_s"] + 5.0)


def test_stage_separation_event_gives_remaining_stack_forward_delta_v():
    event = stage_separation_event(spent_stage_dry_mass_kg=12000, separation_spring_impulse_ns=8000,
                                   remaining_stack_mass_kg=15000)
    check("Stage separation imparts a positive forward delta-v to the remaining stack",
          event.separation_delta_v_m_s > 0)
    check("A lighter remaining stack gets MORE delta-v from the same impulse (Newton's 3rd law)",
          stage_separation_event(12000, 8000, 5000).separation_delta_v_m_s >
          stage_separation_event(12000, 8000, 15000).separation_delta_v_m_s)


def test_apply_separation_event_reduces_mass_and_adds_tipoff_rate():
    event = SeparationEvent("test_event", jettisoned_mass_kg=5000, tipoff_rate_deg_s=2.0,
                            separation_delta_v_m_s=0.5)
    post = apply_separation_event(20000.0, np.array([0.0, 0.0, 0.0]), 7000.0, event)
    check("Post-separation mass equals pre-separation mass minus jettisoned mass",
          abs(post["mass_kg"] - 15000.0) < 1e-6)
    check("Post-separation angular rate is nonzero (tip-off applied)",
          np.linalg.norm(post["angular_rate_rad_s"]) > 0)
    check("Post-separation velocity includes the separation delta-v",
          abs(post["velocity_m_s"] - 7000.5) < 1e-6)


def test_fairing_jettison_gated_by_dynamic_pressure():
    _, safe_low_q = fairing_jettison_event(500, 110000, dynamic_pressure_limit_pa=1135,
                                           current_dynamic_pressure_pa=800)
    _, safe_high_q = fairing_jettison_event(500, 60000, dynamic_pressure_limit_pa=1135,
                                            current_dynamic_pressure_pa=5000)
    check("Fairing jettison is flagged safe when dynamic pressure is below the limit",
          safe_low_q)
    check("Fairing jettison is flagged UNSAFE when dynamic pressure exceeds the limit",
          not safe_high_q)


if __name__ == "__main__":
    print("Running trajectory-optimization / staging-events verification suite...\n")
    test_optimizer_respects_max_q_constraint()
    test_optimizer_respects_pitch_rate_constraint()
    test_tighter_max_q_constraint_reduces_final_speed()
    test_stage_separation_event_gives_remaining_stack_forward_delta_v()
    test_apply_separation_event_reduces_mass_and_adds_tipoff_rate()
    test_fairing_jettison_gated_by_dynamic_pressure()

    print(f"\n{PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    if FAIL > 0:
        sys.exit(1)
