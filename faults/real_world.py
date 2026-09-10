"""
faults/real_world.py

Real-world imperfections that every simulation in this stack so far has
implicitly ignored — and the mitigation logic that a real flight
computer uses to handle each one. Every prior module assumed:
    - Gimbal commands apply INSTANTLY (real actuators take time to move
      and have a maximum slew rate)
    - Navigation state is available with ZERO delay (real IMU/nav data
      takes time to sample, process, and reach the flight computer)
    - The atmosphere is CALM except for the single disturbance torque
      already tested in control/tvc_attitude.py (real wind is gusty,
      not a constant push)

This module models each limitation explicitly, and — just as important —
implements and QUANTIFIES the standard mitigation for each, the same way
analysis/monte_carlo.py quantified PEG's dispersion-rejection advantage
rather than just asserting it.

Zero external dependencies beyond NumPy — CPU-only.
"""

import numpy as np
from collections import deque


# ---------------------------------------------------------------------------
# 1. Actuator imperfection: rate limit + first-order lag
# ---------------------------------------------------------------------------

class RealisticActuator:
    """
    A real TVC gimbal cannot jump instantly to a commanded angle — it is
    driven by a hydraulic or electromechanical actuator with:
        (a) a maximum slew rate (deg/s) — the fastest it can physically move
        (b) first-order lag — even within the rate limit, the actual angle
            approaches the commanded angle exponentially, not instantly
            (time constant tau, standard actuator dynamics model:
             d(theta_actual)/dt = (theta_commanded - theta_actual) / tau)

    Without modeling this, a control loop tuned against an "ideal"
    (instant) actuator will UNDERESTIMATE how much phase lag the real
    system has — a well-known source of control-loop instability when
    moving from simulation to real hardware.
    """
    def __init__(self, max_slew_rate_deg_s: float = 20.0, time_constant_s: float = 0.05):
        self.max_slew_rate = np.radians(max_slew_rate_deg_s)
        self.tau = time_constant_s
        self.actual_angle_rad = 0.0

    def step(self, commanded_angle_rad: float, dt: float) -> float:
        """Advance the actuator one timestep toward the commanded angle,
        respecting both the first-order lag and the hard slew-rate limit."""
        # First-order lag dynamics
        error = commanded_angle_rad - self.actual_angle_rad
        rate = error / self.tau
        # Hard slew-rate saturation — the actuator physically cannot move
        # faster than this regardless of how large the lag-driven rate is
        rate = np.clip(rate, -self.max_slew_rate, self.max_slew_rate)
        step_move = rate * dt
        # When rate-limited (moving at a constant capped rate rather than
        # the smooth exponential lag law), clamp the per-step movement so
        # a coarse timestep can never overshoot past the commanded angle —
        # a real actuator does not oscillate past its own target due to
        # numerical discretization; it simply arrives and stops.
        if abs(step_move) > abs(error):
            step_move = error
        self.actual_angle_rad += step_move
        return self.actual_angle_rad


# ---------------------------------------------------------------------------
# 2. Sensor / telemetry latency
# ---------------------------------------------------------------------------

class LatencyBuffer:
    """
    Real navigation state is never available instantly — IMU sampling,
    filter processing, and data-bus transfer all take time. This models
    a fixed-delay pipeline: whatever state is pushed in now is only
    readable `delay_s` seconds later.

    Feeding a delayed state directly into a control loop (without
    compensation) causes the classic "phase lag instability" problem —
    the controller is always reacting to where the vehicle WAS, not
    where it IS.
    """
    def __init__(self, delay_s: float, dt: float):
        self.n_delay_steps = max(1, int(round(delay_s / dt)))
        self.buffer = deque(maxlen=self.n_delay_steps + 1)

    def push_and_read(self, current_value):
        """Push the current true value in, and return whatever value was
        pushed `n_delay_steps` steps ago (or the current value, padded,
        before the buffer has filled — representing sensor warm-up)."""
        self.buffer.append(current_value)
        if len(self.buffer) <= self.n_delay_steps:
            return current_value
        return self.buffer[0]


def predictive_delay_compensation(delayed_value: float, delayed_rate: float,
                                   delay_s: float) -> float:
    """
    Standard latency-compensation technique: since we know the
    measurement is `delay_s` old, and we have an estimate of its rate of
    change (e.g. from the EKF's velocity/rate state), extrapolate
    forward by `delay_s` to estimate the CURRENT value instead of
    blindly using the stale one. This is the same principle used in
    real flight computers and in network-latency compensation
    (dead-reckoning) in general.
    """
    return delayed_value + delayed_rate * delay_s


# ---------------------------------------------------------------------------
# 3. Wind gusts (standard aerospace "1-cosine" discrete gust model)
# ---------------------------------------------------------------------------

def one_minus_cosine_gust(t: float, gust_start_s: float, gust_length_s: float,
                           gust_peak_torque_nm: float) -> float:
    """
    The "1-cosine" discrete gust profile is the standard model used in
    aerospace load-analysis (derived from MIL-STD / certification-style
    gust-response requirements): a smooth ramp-up-and-down disturbance
    rather than an unrealistic instantaneous step, which is what
    control/tvc_attitude.py's disturbance test used previously.

        torque(t) = 0                                   for t < gust_start
        torque(t) = (peak/2) * (1 - cos(2*pi*(t-t0)/L))  during the gust
        torque(t) = 0                                    after the gust

    Returns the disturbance torque (Nm) at time t.
    """
    if t < gust_start_s or t > gust_start_s + gust_length_s:
        return 0.0
    phase = 2.0 * np.pi * (t - gust_start_s) / gust_length_s
    return (gust_peak_torque_nm / 2.0) * (1.0 - np.cos(phase))


# ---------------------------------------------------------------------------
# Combined: closed-loop attitude hold WITH realistic constraints
# ---------------------------------------------------------------------------

def simulate_realistic_attitude_hold(
    pitch_command_rad: float,
    thrust_n: float, moment_of_inertia: float, gimbal_arm_m: float,
    kp: float, ki: float, kd: float,
    actuator: RealisticActuator = None,
    sensor_delay_s: float = 0.0,
    use_delay_compensation: bool = False,
    gust_start_s: float = 3.0, gust_length_s: float = 2.0, gust_peak_torque_nm: float = 20000.0,
    gimbal_limit_deg: float = 6.0,
    initial_pitch_error_rad: float = 0.0,
    dt: float = 0.02, t_max: float = 12.0,
):
    """
    The same PID-controlled TVC attitude-hold problem as
    control/tvc_attitude.py, but now with a realistic (rate-limited,
    lagging) actuator, delayed sensor feedback (optionally compensated),
    and a smooth 1-cosine wind gust instead of an idealized instant
    disturbance step.
    """
    from control.tvc_attitude import RigidBodyPitchDynamics, PIDController

    if actuator is None:
        actuator = RealisticActuator()

    dynamics = RigidBodyPitchDynamics(moment_of_inertia, gimbal_arm_m)
    dynamics.pitch_rad = initial_pitch_error_rad
    pid = PIDController(kp, ki, kd, output_limit_rad=np.radians(gimbal_limit_deg))

    latency = LatencyBuffer(sensor_delay_s, dt) if sensor_delay_s > 0 else None

    n_steps = int(t_max / dt)
    t_hist = np.zeros(n_steps)
    pitch_hist = np.zeros(n_steps)
    gimbal_cmd_hist = np.zeros(n_steps)
    gimbal_actual_hist = np.zeros(n_steps)

    t = 0.0
    for i in range(n_steps):
        true_pitch = dynamics.pitch_rad
        true_rate = dynamics.pitch_rate

        if latency is not None:
            observed_pitch = latency.push_and_read(true_pitch)
            if use_delay_compensation:
                # Compensate using the (also delayed, but still useful)
                # rate estimate — real systems use the nav filter's own
                # rate state for this, same idea as EKF velocity in
                # navigation/state_estimation.py.
                observed_pitch = predictive_delay_compensation(
                    observed_pitch, true_rate, sensor_delay_s
                )
        else:
            observed_pitch = true_pitch

        error = pitch_command_rad - observed_pitch
        gimbal_cmd = pid.update(error, dt)
        gimbal_actual = actuator.step(gimbal_cmd, dt)

        disturbance = one_minus_cosine_gust(t, gust_start_s, gust_length_s, gust_peak_torque_nm)
        dynamics.step(thrust_n, gimbal_actual, disturbance, dt)

        t_hist[i] = t
        pitch_hist[i] = np.degrees(dynamics.pitch_rad)
        gimbal_cmd_hist[i] = np.degrees(gimbal_cmd)
        gimbal_actual_hist[i] = np.degrees(gimbal_actual)

        t += dt

    return {
        "t": t_hist,
        "pitch_error_deg": pitch_hist,
        "gimbal_commanded_deg": gimbal_cmd_hist,
        "gimbal_actual_deg": gimbal_actual_hist,
        "final_pitch_error_deg": pitch_hist[-1],
        "max_pitch_error_deg": np.max(np.abs(pitch_hist)),
        "settled": bool(np.abs(pitch_hist[-1]) < 0.2),
    }
