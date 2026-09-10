"""
control/tvc_attitude.py

Thrust Vector Control (TVC) attitude control loop — the module that
keeps a launch vehicle pointed the way guidance commands, by gimbaling
the engine nozzle to produce a correcting torque.

Physics:
    A gimbaled engine offset by angle `delta` from the vehicle centerline
    produces a torque about the pitch axis:
        torque = F * sin(delta) * L
    where F is thrust and L is the distance from the engine gimbal pivot
    to the vehicle's center of mass (the moment arm).

    This torque changes angular acceleration via the vehicle's moment of
    inertia:
        angular_accel = torque / I

Control:
    A PID controller on pitch-angle error (commanded pitch from guidance
    vs. estimated pitch from navigation) outputs a gimbal deflection
    command, saturated to the physical gimbal limit (real engines
    typically gimbal +/-5 to +/-8 degrees).

This closes the GNC loop: Guidance (gravity_turn.py) commands a pitch
profile -> Navigation (state_estimation.py) estimates current attitude
-> Control (this module) drives gimbal angle to null the error.

Zero external dependencies beyond NumPy — CPU-only.
"""

import numpy as np


class RigidBodyPitchDynamics:
    """
    Single-axis (pitch) rigid-body rotational dynamics for a launch
    vehicle stage, driven by TVC torque and a disturbance torque (wind
    gust / aerodynamic moment, representing an external perturbation the
    controller must reject).
    """
    def __init__(self, moment_of_inertia_kg_m2: float, gimbal_arm_m: float):
        self.I = moment_of_inertia_kg_m2
        self.L = gimbal_arm_m
        self.pitch_rad = 0.0        # attitude error state (0 = perfectly on target)
        self.pitch_rate = 0.0        # rad/s

    def step(self, thrust_n: float, gimbal_deflection_rad: float,
              disturbance_torque_nm: float, dt: float):
        torque = thrust_n * np.sin(gimbal_deflection_rad) * self.L + disturbance_torque_nm
        angular_accel = torque / self.I

        self.pitch_rate += angular_accel * dt
        self.pitch_rad += self.pitch_rate * dt
        return self.pitch_rad, self.pitch_rate


class PIDController:
    """Standard PID with output saturation and anti-windup (clamped integral)."""
    def __init__(self, kp: float, ki: float, kd: float,
                 output_limit_rad: float, integral_limit: float = None):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.output_limit = output_limit_rad
        self.integral_limit = integral_limit if integral_limit is not None else output_limit_rad * 10
        self.integral = 0.0
        self.prev_error = 0.0
        self._first = True

    def update(self, error: float, dt: float) -> float:
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)

        derivative = 0.0 if self._first else (error - self.prev_error) / dt
        self._first = False

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        output = np.clip(output, -self.output_limit, self.output_limit)

        self.prev_error = error
        return output


def simulate_attitude_hold(
    pitch_command_rad: float,
    thrust_n: float,
    moment_of_inertia: float,
    gimbal_arm_m: float,
    kp: float, ki: float, kd: float,
    gimbal_limit_deg: float = 6.0,
    disturbance_profile=None,
    initial_pitch_error_rad: float = 0.0,
    dt: float = 0.02,
    t_max: float = 10.0,
):
    """
    Closed-loop simulation: PID + gimbal-limited TVC holding a commanded
    pitch attitude against a disturbance (e.g. wind gust torque), starting
    from some initial attitude error.

    disturbance_profile: callable(t) -> torque_Nm, or None for zero disturbance.
    """
    dynamics = RigidBodyPitchDynamics(moment_of_inertia, gimbal_arm_m)
    dynamics.pitch_rad = initial_pitch_error_rad  # start with an offset to correct

    pid = PIDController(kp, ki, kd, output_limit_rad=np.radians(gimbal_limit_deg))

    if disturbance_profile is None:
        disturbance_profile = lambda t: 0.0

    n_steps = int(t_max / dt)
    t_hist = np.zeros(n_steps)
    pitch_hist = np.zeros(n_steps)
    gimbal_hist = np.zeros(n_steps)

    t = 0.0
    for i in range(n_steps):
        error = pitch_command_rad - dynamics.pitch_rad  # want pitch_rad -> 0 (on target)
        gimbal_cmd = pid.update(error, dt)

        dist = disturbance_profile(t)
        dynamics.step(thrust_n, gimbal_cmd, dist, dt)

        t_hist[i] = t
        pitch_hist[i] = dynamics.pitch_rad
        gimbal_hist[i] = np.degrees(gimbal_cmd)

        t += dt

    return {
        "t": t_hist,
        "pitch_error_deg": np.degrees(pitch_hist),
        "gimbal_deg": gimbal_hist,
        "final_pitch_error_deg": np.degrees(pitch_hist[-1]),
        "max_gimbal_deg": np.max(np.abs(gimbal_hist)),
        "settled": bool(np.abs(pitch_hist[-1]) < np.radians(0.1)),
    }
