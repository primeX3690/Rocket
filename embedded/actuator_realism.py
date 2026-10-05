"""
embedded/actuator_realism.py

Additional real TVC actuator effects, additive to
faults/real_world.py's RealisticActuator (which already models slew-
rate limiting and first-order actuation lag): a DEADBAND (the actuator
doesn't move at all for small commands -- static friction/stiction in
a real electromechanical or hydraulic gimbal actuator) and BACKLASH
(mechanical gear/linkage play -- reversing direction doesn't
immediately reverse the output, a classic source of limit-cycle
oscillation in a poorly-tuned control loop).

Zero external dependencies beyond NumPy - CPU-only.
"""

import numpy as np


class DeadbandBacklashActuator:
    """
    Wraps a slew-rate/lag model (same interface as
    faults.real_world.RealisticActuator) with a command deadband and
    output backlash. Commands smaller than the deadband are treated as
    zero; the actual output tracks the (dead-banded) command through a
    backlash gap, so reversing direction produces a flat "stuck" region
    before the output starts moving again -- the textbook backlash
    hysteresis loop.
    """
    def __init__(self, max_slew_rate_deg_s: float = 20.0, time_constant_s: float = 0.05,
                deadband_deg: float = 0.05, backlash_deg: float = 0.1):
        self.max_slew_rate = np.radians(max_slew_rate_deg_s)
        self.tau = time_constant_s
        self.deadband = np.radians(deadband_deg)
        self.backlash = np.radians(backlash_deg)
        self.actual_angle_rad = 0.0
        self.backlash_state = 0.0          # position within the backlash gap, [-backlash/2, +backlash/2]
        self.last_command_direction = 0

    def step(self, commanded_angle_rad: float, dt: float) -> float:
        # Deadband: ignore commands too small to overcome static friction.
        if abs(commanded_angle_rad) < self.deadband:
            commanded_angle_rad = 0.0

        # Backlash: the mechanism only starts moving the output once the
        # command has moved far enough (within the gap) in the new direction.
        direction = np.sign(commanded_angle_rad - self.actual_angle_rad) if \
            commanded_angle_rad != self.actual_angle_rad else self.last_command_direction
        if direction != 0 and direction != self.last_command_direction:
            # Direction reversal: must traverse the backlash gap before output engages.
            self.backlash_state = -direction * self.backlash / 2.0
            self.last_command_direction = direction
        effective_command = commanded_angle_rad - self.backlash_state

        # Rate limit + first-order lag, same structure as RealisticActuator.
        delta = effective_command - self.actual_angle_rad
        rate_limited_delta = np.clip(delta, -self.max_slew_rate * dt, self.max_slew_rate * dt)
        alpha = dt / max(self.tau, 1e-6)
        self.actual_angle_rad += min(alpha, 1.0) * rate_limited_delta
        return self.actual_angle_rad
