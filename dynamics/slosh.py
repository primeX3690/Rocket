"""
dynamics/slosh.py

Propellant slosh dynamics via the standard "equivalent mechanical
pendulum" model (the same lumped-parameter approach used throughout the
real aerospace industry -- e.g. Dodge's "The New Dynamic Behavior of
Liquids in Moving Containers" NASA/Southwest Research reference model
-- rather than full CFD, which is far outside a CPU-only point-mass
stack's scope).

A partially-full tank under lateral acceleration is modeled as a
pendulum (mass, length, damping) hinged at a point offset from the tank
centroid; the pendulum's swing couples back into the vehicle as an
additional disturbance force/torque, and CAN destabilize an otherwise
well-tuned attitude control loop if the slosh frequency approaches the
control bandwidth -- exactly the real failure mode (e.g. slosh-induced
POGO-adjacent instabilities) this model exists to let a control
designer check for.

Zero external dependencies beyond NumPy - CPU-only.
"""

import numpy as np

G0 = 9.80665


class SloshPendulum:
    """
    Single equivalent pendulum representing the dominant (first) slosh
    mode of a partially-full tank. `fill_fraction` (0-1) sets the
    pendulum length via a standard cylindrical-tank slosh correlation
    (shorter effective pendulum, i.e. higher slosh frequency, as the
    tank drains and the free-surface radius-to-depth ratio changes).
    """
    def __init__(self, slosh_mass_kg: float, tank_radius_m: float, fill_fraction: float,
                damping_ratio: float = 0.02, hinge_offset_m: float = 0.0):
        self.m = slosh_mass_kg
        self.R = tank_radius_m
        self.fill = np.clip(fill_fraction, 0.05, 0.95)
        self.zeta = damping_ratio
        self.hinge_offset = hinge_offset_m
        # Equivalent pendulum length for the first slosh mode of a
        # cylindrical tank (Dodge's correlation, first Bessel-zero mode):
        # L_eq = R / (1.84 * tanh(1.84 * fill_depth_ratio))
        depth_ratio = self.fill * 2.0
        self.length = tank_radius_m / (1.84 * np.tanh(1.84 * depth_ratio))
        self.theta = 0.0     # pendulum angle from vertical (rad)
        self.theta_dot = 0.0

    def natural_frequency_rad_s(self, axial_accel_m_s2: float = G0) -> float:
        return float(np.sqrt(axial_accel_m_s2 / max(self.length, 1e-6)))

    def step(self, lateral_accel_m_s2: float, axial_accel_m_s2: float, dt: float):
        """
        Advance the pendulum one step under a lateral base acceleration
        (the forcing input, e.g. from vehicle lateral acceleration or
        angular acceleration times moment arm) and the current axial
        (thrust-direction) acceleration, which sets the pendulum's
        effective gravity and hence its natural frequency -- a slosh
        mode's frequency genuinely shifts with axial acceleration
        (higher accel -> stiffer effective restoring force -> higher
        frequency), which is why slosh coupling is worst during
        LOW-acceleration phases (e.g. right after a throttle-down).
        """
        omega_n = np.sqrt(max(axial_accel_m_s2, 0.1) / max(self.length, 1e-6))
        theta_ddot = (-omega_n ** 2 * self.theta - 2 * self.zeta * omega_n * self.theta_dot
                     - lateral_accel_m_s2 / max(self.length, 1e-6))
        self.theta_dot += theta_ddot * dt
        self.theta += self.theta_dot * dt
        return self.theta

    def reaction_force_n(self, axial_accel_m_s2: float) -> float:
        """Lateral force the sloshing mass exerts back on the tank/vehicle structure."""
        return float(self.m * max(axial_accel_m_s2, 0.1) * np.sin(self.theta))

    def reaction_torque_nm(self, axial_accel_m_s2: float, moment_arm_m: float) -> float:
        """Torque about the vehicle Cg from the slosh reaction force acting at the tank's moment arm."""
        return self.reaction_force_n(axial_accel_m_s2) * moment_arm_m


def slosh_disturbance_torque(pendulum: SloshPendulum, lateral_accel_m_s2: float,
                             axial_accel_m_s2: float, moment_arm_m: float, dt: float) -> float:
    """Convenience one-shot: step the pendulum and return the resulting disturbance torque (Nm)."""
    pendulum.step(lateral_accel_m_s2, axial_accel_m_s2, dt)
    return pendulum.reaction_torque_nm(axial_accel_m_s2, moment_arm_m)
