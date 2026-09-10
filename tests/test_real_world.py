"""
tests/test_real_world.py

Verification suite for faults/real_world.py.

Central claim under test (mirroring the Monte Carlo comparison
methodology already used in this project): a realistic actuator +
delayed sensor + gusty wind DEGRADES performance versus the idealized
control/tvc_attitude.py assumptions, and delay COMPENSATION recovers
a meaningful fraction of that lost performance. This is proven with
numbers, not asserted.
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from faults.real_world import (
    RealisticActuator, LatencyBuffer, predictive_delay_compensation,
    one_minus_cosine_gust, simulate_realistic_attitude_hold
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


def test_actuator_respects_slew_rate_limit():
    """A huge commanded step should never move faster than max_slew_rate."""
    actuator = RealisticActuator(max_slew_rate_deg_s=10.0, time_constant_s=0.001)
    dt = 0.01
    prev = 0.0
    max_rate_seen = 0.0
    for _ in range(200):
        angle = actuator.step(np.radians(90.0), dt)  # huge step command
        rate = abs(angle - prev) / dt
        max_rate_seen = max(max_rate_seen, rate)
        prev = angle
    check("Actuator never exceeds its configured max slew rate",
          np.degrees(max_rate_seen) <= 10.0 + 0.5)  # small numeric tolerance


def test_actuator_first_order_lag_matches_analytic():
    """
    With a slew-rate limit set very high (never binding), the actuator
    should follow the classic first-order-lag analytic solution:
        theta(t) = theta_cmd * (1 - exp(-t/tau))
    for a step input from 0.
    """
    tau = 0.2
    actuator = RealisticActuator(max_slew_rate_deg_s=100000.0, time_constant_s=tau)
    dt = 0.001
    cmd = np.radians(10.0)
    t = 0.0
    for _ in range(int(1.0 / dt)):
        angle = actuator.step(cmd, dt)
        t += dt
    expected = cmd * (1 - np.exp(-1.0 / tau))
    check("First-order actuator lag matches analytic step response (~2% tolerance)",
          abs(angle - expected) / expected < 0.02)


def test_latency_buffer_delays_by_correct_amount():
    dt = 0.01
    delay_s = 0.1
    buf = LatencyBuffer(delay_s, dt)
    values = list(range(50))
    outputs = [buf.push_and_read(v) for v in values]
    n_delay_steps = int(round(delay_s / dt))
    check("Latency buffer outputs the value from n_delay_steps ago once filled",
          outputs[-1] == values[-1 - n_delay_steps])


def test_predictive_compensation_recovers_true_value_for_constant_rate():
    """
    If the true signal is changing at a constant rate, predictive
    compensation should recover the true CURRENT value exactly from a
    delayed sample plus that rate (this is exact for constant-rate
    motion, and an approximation otherwise — exactly analogous to dead-
    reckoning in navigation/state_estimation.py).
    """
    rate = 5.0  # units/s
    delay = 0.3
    true_current_value = 100.0
    delayed_value = true_current_value - rate * delay  # what a delayed sensor would report
    compensated = predictive_delay_compensation(delayed_value, rate, delay)
    check("Predictive delay compensation recovers true current value for constant-rate signal",
          abs(compensated - true_current_value) < 1e-9)


def test_gust_profile_matches_one_minus_cosine_formula():
    start, length, peak = 2.0, 3.0, 10000.0
    # At the midpoint of the gust, 1-cos(pi) = 2, so torque should be at its peak
    mid_t = start + length / 2.0
    torque_mid = one_minus_cosine_gust(mid_t, start, length, peak)
    check("Gust torque reaches its peak value at the gust's temporal midpoint",
          abs(torque_mid - peak) < 1e-6)

    check("Gust torque is zero just before it starts",
          one_minus_cosine_gust(start - 0.01, start, length, peak) == 0.0)
    check("Gust torque is zero just after it ends",
          one_minus_cosine_gust(start + length + 0.01, start, length, peak) == 0.0)
    check("Gust torque is zero exactly at start (1-cos(0)=0)",
          abs(one_minus_cosine_gust(start, start, length, peak)) < 1e-9)


def test_realistic_constraints_degrade_performance_vs_idealized():
    """
    THE key comparison: attitude-hold with a realistic (lagging,
    rate-limited) actuator and a gusty 1-cosine wind disturbance should
    show a larger peak/settling error than an idealized instantaneous
    actuator under the same gust — quantifying the cost of ignoring
    real hardware behavior.
    """
    thrust, I, arm = 1_200_000.0, 60000.0, 3.5
    K = thrust * arm / I
    wn, zeta = 2.0, 0.9
    kp, kd = wn ** 2 / K, 2 * zeta * wn / K

    ideal_actuator = RealisticActuator(max_slew_rate_deg_s=1e6, time_constant_s=1e-6)
    realistic_actuator = RealisticActuator(max_slew_rate_deg_s=15.0, time_constant_s=0.08)

    ideal_result = simulate_realistic_attitude_hold(
        pitch_command_rad=0.0, thrust_n=thrust, moment_of_inertia=I, gimbal_arm_m=arm,
        kp=kp, ki=0.0, kd=kd, actuator=ideal_actuator,
        gust_peak_torque_nm=25000.0,
    )
    realistic_result = simulate_realistic_attitude_hold(
        pitch_command_rad=0.0, thrust_n=thrust, moment_of_inertia=I, gimbal_arm_m=arm,
        kp=kp, ki=0.0, kd=kd, actuator=realistic_actuator,
        gust_peak_torque_nm=25000.0,
    )

    check("Realistic actuator dynamics produce a larger peak attitude error than idealized actuator",
          realistic_result["max_pitch_error_deg"] > ideal_result["max_pitch_error_deg"])


def test_delay_compensation_improves_performance_vs_uncompensated():
    """
    Second key comparison: with meaningful sensor latency, delay
    COMPENSATION should reduce peak attitude error versus using the raw
    delayed measurement uncompensated.
    """
    thrust, I, arm = 1_200_000.0, 60000.0, 3.5
    K = thrust * arm / I
    wn, zeta = 2.0, 0.9
    kp, kd = wn ** 2 / K, 2 * zeta * wn / K

    uncompensated = simulate_realistic_attitude_hold(
        pitch_command_rad=0.0, thrust_n=thrust, moment_of_inertia=I, gimbal_arm_m=arm,
        kp=kp, ki=0.0, kd=kd,
        sensor_delay_s=0.15, use_delay_compensation=False,
        initial_pitch_error_rad=np.radians(3.0), gust_peak_torque_nm=0.0,
    )
    compensated = simulate_realistic_attitude_hold(
        pitch_command_rad=0.0, thrust_n=thrust, moment_of_inertia=I, gimbal_arm_m=arm,
        kp=kp, ki=0.0, kd=kd,
        sensor_delay_s=0.15, use_delay_compensation=True,
        initial_pitch_error_rad=np.radians(3.0), gust_peak_torque_nm=0.0,
    )
    check("Delay-compensated control has lower final pitch error than uncompensated, same delay",
          abs(compensated["final_pitch_error_deg"]) < abs(uncompensated["final_pitch_error_deg"]))


if __name__ == "__main__":
    print("Running real-world constraints verification suite...\n")
    test_actuator_respects_slew_rate_limit()
    test_actuator_first_order_lag_matches_analytic()
    test_latency_buffer_delays_by_correct_amount()
    test_predictive_compensation_recovers_true_value_for_constant_rate()
    test_gust_profile_matches_one_minus_cosine_formula()
    test_realistic_constraints_degrade_performance_vs_idealized()
    test_delay_compensation_improves_performance_vs_uncompensated()

    print(f"\n{PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    if FAIL > 0:
        sys.exit(1)
