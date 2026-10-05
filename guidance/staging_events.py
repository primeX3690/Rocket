"""
guidance/staging_events.py

Discrete-event modeling for multi-stage separation and fairing jettison
-- previously these mission phases were invisible seams between
analysis/mission_profile.py's stage-1/coast/stage-2 calls (an instant,
disturbance-free mass change). Real separation events inject a real
disturbance (pyrotechnic/pneumatic separation impulse imparts a
tip-off rate) and a real, sudden mass/Cg/inertia discontinuity that a
naive controller can be caught off-guard by if it isn't told about it.

Zero external dependencies beyond NumPy - CPU-only.
"""

import numpy as np


class SeparationEvent:
    """A single staging event: spent-stage jettison or fairing jettison."""
    def __init__(self, name: str, jettisoned_mass_kg: float, tipoff_rate_deg_s: float = 0.0,
                separation_delta_v_m_s: float = 0.0, new_cg_shift_m: float = 0.0):
        self.name = name
        self.jettisoned_mass_kg = jettisoned_mass_kg
        self.tipoff_rate_deg_s = tipoff_rate_deg_s
        self.separation_delta_v_m_s = separation_delta_v_m_s
        self.new_cg_shift_m = new_cg_shift_m


def apply_separation_event(mass_kg: float, angular_rate_rad_s: np.ndarray, velocity_m_s: float,
                           event: SeparationEvent, separation_direction: np.ndarray = None):
    """
    Apply a staging event's discontinuity to the current state:
    - mass drops by the jettisoned mass
    - angular rate gets a tip-off kick (pyro/pneumatic separation is
      never perfectly axial/symmetric -- residual thrust misalignment
      and uneven separation-spring/pusher forces impart a real rate)
    - velocity gets a small separation delta-v (spring-pusher/RCS impulse)

    Returns the post-event state; the CALLER is responsible for feeding
    this into its controller as a real disturbance (e.g. as an impulsive
    torque in dynamics/six_dof.py or a rate-offset in
    control/tvc_attitude.py), which is the actual point of modeling this
    explicitly rather than a silent instantaneous mass change.
    """
    new_mass = max(mass_kg - event.jettisoned_mass_kg, 1e-3)
    if separation_direction is None:
        separation_direction = np.array([1.0, 0.0, 0.0])
    separation_direction = separation_direction / max(np.linalg.norm(separation_direction), 1e-9)

    tipoff_axis = np.array([0.0, 1.0, 0.0])   # pitch-plane tip-off (worst-case common orientation)
    new_angular_rate = np.asarray(angular_rate_rad_s, dtype=float) + tipoff_axis * np.radians(event.tipoff_rate_deg_s)
    new_velocity = velocity_m_s + event.separation_delta_v_m_s

    return {
        "mass_kg": new_mass,
        "angular_rate_rad_s": new_angular_rate,
        "velocity_m_s": new_velocity,
        "event_name": event.name,
    }


def fairing_jettison_event(fairing_mass_kg: float, altitude_m: float, dynamic_pressure_limit_pa: float,
                           current_dynamic_pressure_pa: float, tipoff_rate_deg_s: float = 0.5):
    """
    Standard fairing-jettison safety gate: only jettison once dynamic
    pressure has dropped below a heating/loads-safe threshold (real
    missions jettison the fairing once q is low enough that the exposed
    payload won't see excessive aeroheating or dynamic pressure loads --
    jettisoning too early risks payload damage; too late wastes mass to
    a higher altitude for no benefit).
    """
    safe_to_jettison = current_dynamic_pressure_pa <= dynamic_pressure_limit_pa
    event = SeparationEvent("fairing_jettison", jettisoned_mass_kg=fairing_mass_kg,
                            tipoff_rate_deg_s=tipoff_rate_deg_s if safe_to_jettison else 0.0)
    return event, safe_to_jettison


def stage_separation_event(spent_stage_dry_mass_kg: float, separation_spring_impulse_ns: float,
                           remaining_stack_mass_kg: float, tipoff_rate_deg_s: float = 1.5):
    """
    Standard cold spring/pneumatic-pusher stage separation: the spent
    stage falls away, and the separation impulse imparts a small
    forward delta-v to the REMAINING stack (Newton's third law — the
    springs push both bodies apart, and the lighter/remaining stack
    gets more delta-v per unit impulse than the heavier spent stage).
    """
    delta_v = separation_spring_impulse_ns / max(remaining_stack_mass_kg, 1e-3)
    event = SeparationEvent("stage_separation", jettisoned_mass_kg=spent_stage_dry_mass_kg,
                            tipoff_rate_deg_s=tipoff_rate_deg_s, separation_delta_v_m_s=delta_v)
    return event
