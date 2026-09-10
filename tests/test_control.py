"""
tests/test_control.py

Verification suite for control/tvc_attitude.py.

Checks: gimbal torque physics against manual calc, PID output saturation
respects the physical gimbal limit, closed-loop system nulls an initial
attitude error (stabilizes), and rejects a step disturbance (wind gust)
without exceeding the gimbal limit.
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from control.tvc_attitude import (
    RigidBodyPitchDynamics, PIDController, simulate_attitude_hold
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


def test_gimbal_torque_matches_manual_calc():
    """torque = F * sin(delta) * L, checked directly."""
    dyn = RigidBodyPitchDynamics(moment_of_inertia_kg_m2=50000.0, gimbal_arm_m=3.0)
    thrust = 1_000_000.0
    delta = np.radians(4.0)
    expected_torque = thrust * np.sin(delta) * 3.0
    expected_alpha = expected_torque / 50000.0

    dyn.step(thrust, delta, disturbance_torque_nm=0.0, dt=0.01)
    # after one step: pitch_rate = alpha * dt
    check("Angular acceleration matches torque/I manual calc",
          abs(dyn.pitch_rate - expected_alpha * 0.01) < 1e-9)


def test_pid_output_saturates_at_gimbal_limit():
    """A huge error should saturate PID output at exactly the output_limit."""
    pid = PIDController(kp=100.0, ki=0.0, kd=0.0, output_limit_rad=np.radians(6.0))
    out = pid.update(error=np.radians(90.0), dt=0.02)  # deliberately huge error
    check("PID output saturates at the configured gimbal limit",
          abs(out - np.radians(6.0)) < 1e-9)


def test_pid_zero_error_gives_zero_output():
    pid = PIDController(kp=50.0, ki=1.0, kd=5.0, output_limit_rad=np.radians(6.0))
    out = pid.update(error=0.0, dt=0.02)
    check("PID output is zero for zero error (no integral wind-up yet)",
          abs(out) < 1e-9)


def _analytic_gains(thrust_n, arm_m, moment_of_inertia, wn, zeta):
    """
    Analytic PD gain design for this plant. Small-angle gimbal torque
    makes pitch dynamics a double integrator: theta'' = K * delta, with
    K = F*L/I. For a desired closed-loop natural frequency wn and
    damping ratio zeta:
        kp = wn^2 / K
        kd = 2*zeta*wn / K
    This is the standard double-integrator pole-placement result and is
    what a real TVC autopilot gain schedule is derived from (scaled per
    flight phase as F, L, I change with propellant burn-off) — not a
    guessed constant.
    """
    K = thrust_n * arm_m / moment_of_inertia
    return wn ** 2 / K, 2 * zeta * wn / K


def test_closed_loop_nulls_initial_attitude_error():
    """
    Starting with a 3-degree initial pitch error and zero disturbance,
    a PD gain set derived analytically from the plant's double-integrator
    characteristic (K = F*L/I) should drive the error to ~zero and
    stay settled well within the simulation window.
    """
    thrust, arm, I = 1_200_000.0, 3.5, 60000.0
    kp, kd = _analytic_gains(thrust, arm, I, wn=2.0, zeta=0.9)

    result = simulate_attitude_hold(
        pitch_command_rad=0.0,
        thrust_n=thrust,
        moment_of_inertia=I,
        gimbal_arm_m=arm,
        kp=kp, ki=0.0, kd=kd,
        gimbal_limit_deg=6.0,
        initial_pitch_error_rad=np.radians(3.0),
        dt=0.01, t_max=10.0,
    )
    check("Final pitch error is small (<0.2 deg) after 10s settling",
          abs(result["final_pitch_error_deg"]) < 0.2)
    check("Controller reports settled=True",
          result["settled"])
    check("Gimbal command never exceeds the physical limit",
          result["max_gimbal_deg"] <= 6.0 + 1e-6)


def test_closed_loop_rejects_step_disturbance():
    """
    With zero initial error but a constant wind-gust disturbance torque
    applied throughout, the closed-loop system (PD design + small
    integral term for zero steady-state error) should settle to a small
    steady-state error, demonstrating disturbance rejection.
    """
    thrust, arm, I = 1_200_000.0, 3.5, 60000.0
    kp, kd = _analytic_gains(thrust, arm, I, wn=2.0, zeta=0.9)
    ki = kp * 0.5  # integral action tuned to null steady-state disturbance offset within the settle window

    disturbance = lambda t: 20000.0  # constant 20 kNm disturbance torque (wind gust proxy)

    result = simulate_attitude_hold(
        pitch_command_rad=0.0,
        thrust_n=thrust,
        moment_of_inertia=I,
        gimbal_arm_m=arm,
        kp=kp, ki=ki, kd=kd,
        gimbal_limit_deg=6.0,
        disturbance_profile=disturbance,
        initial_pitch_error_rad=0.0,
        dt=0.01, t_max=10.0,
    )
    check("System rejects constant disturbance to a small steady-state error (<1 deg)",
          abs(result["final_pitch_error_deg"]) < 1.0)
    check("Gimbal stays within physical limit under disturbance",
          result["max_gimbal_deg"] <= 6.0 + 1e-6)


def test_unstable_gains_are_detectable():
    """
    Sanity check on the test harness itself: deliberately bad (zero)
    gains should NOT settle, confirming the settle check is meaningful
    and not a tautology.
    """
    result = simulate_attitude_hold(
        pitch_command_rad=0.0,
        thrust_n=1_200_000.0,
        moment_of_inertia=60000.0,
        gimbal_arm_m=3.5,
        kp=0.0, ki=0.0, kd=0.0,   # no control effort at all
        gimbal_limit_deg=6.0,
        initial_pitch_error_rad=np.radians(3.0),
        dt=0.02, t_max=10.0,
    )
    check("With zero gains (no control), system does NOT settle (confirms test is meaningful)",
          not result["settled"])


if __name__ == "__main__":
    print("Running TVC / attitude control verification suite...\n")
    test_gimbal_torque_matches_manual_calc()
    test_pid_output_saturates_at_gimbal_limit()
    test_pid_zero_error_gives_zero_output()
    test_closed_loop_nulls_initial_attitude_error()
    test_closed_loop_rejects_step_disturbance()
    test_unstable_gains_are_detectable()

    print(f"\n{PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    if FAIL > 0:
        sys.exit(1)
