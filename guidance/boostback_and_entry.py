"""
guidance/boostback_and_entry.py

Boostback-burn targeting and entry-burn deceleration planning -- the
two propulsive maneuvers a returning first stage performs between
staging and the atmospheric descent phase (guidance/
powered_descent_guidance.py handles the final landing burn).

- Boostback burn: after staging, the spent stage is still moving
  DOWNRANGE (away from the launch/landing site) at high speed. The
  boostback burn reverses enough of that downrange velocity to put the
  stage on a ballistic path back toward the landing site — this reuses
  the SAME minimum-energy ZEM/ZEV law from powered_descent_guidance.py
  (a real design choice: it is the right tool for "null a position/
  velocity error by a given time" in general, not landing-specific).
- Entry burn: before re-entering the denser atmosphere, a real reusable
  stage fires a short retrograde burn to shed speed and reduce peak
  heating/dynamic pressure during the hypersonic-to-supersonic descent
  — modeled here as a delta-v sized to hit a target entry-interface
  speed, given the vehicle's mass/thrust/Isp (a standard rocket-
  equation delta-v/burn-time sizing, not a full guidance law since a
  simple deceleration-only burn does not need one).

Zero external dependencies beyond NumPy - CPU-only.
"""

import numpy as np
from guidance.powered_descent_guidance import zem_zev_acceleration_command

G0 = 9.80665


def boostback_delta_v_command(current_pos_m: np.ndarray, current_vel_m_s: np.ndarray,
                              landing_site_pos_m: np.ndarray, t_go_s: float,
                              gravity_m_s2: np.ndarray) -> np.ndarray:
    """
    Thrust-acceleration command (same ZEM/ZEV law as the landing-burn
    guidance) to redirect the stage from its post-separation state onto
    a trajectory that reaches the landing site's horizontal position at
    time t_go_s (typically the apex or a chosen point of the resulting
    ballistic arc, chosen by the caller/mission planner — this function
    only computes the acceleration needed for that one targeting
    condition, matching how a real boostback burn is a single
    short targeting burn, not a continuously-guided full descent).
    """
    target_vel = np.zeros_like(np.asarray(current_vel_m_s, dtype=float))
    a_total = zem_zev_acceleration_command(current_pos_m, current_vel_m_s,
                                           landing_site_pos_m, target_vel, t_go_s)
    return a_total - np.asarray(gravity_m_s2, dtype=float)


def boostback_burn_duration_s(delta_v_needed_m_s: float, thrust_n: float,
                              isp_s: float, m0_kg: float) -> dict:
    """
    Given a required delta-v (e.g. the magnitude of a constant boostback
    acceleration command sustained for a nominal duration), the burn
    time and propellant used, via the rocket equation (Tsiolkovsky),
    consistent with propulsion/rocket_equation.py's existing model.
    """
    ve = isp_s * G0
    mass_ratio = np.exp(delta_v_needed_m_s / ve)
    propellant_used_kg = m0_kg * (1.0 - 1.0 / mass_ratio)
    mdot = thrust_n / ve
    burn_time_s = propellant_used_kg / mdot if mdot > 0 else float("inf")
    return {"delta_v_m_s": delta_v_needed_m_s, "propellant_used_kg": propellant_used_kg,
            "burn_time_s": burn_time_s, "final_mass_kg": m0_kg - propellant_used_kg}


def entry_burn_plan(entry_interface_speed_m_s: float, target_post_entry_speed_m_s: float,
                    thrust_n: float, isp_s: float, m0_kg: float,
                    max_deceleration_g: float = 5.0) -> dict:
    """
    Sizes an entry burn: the delta-v needed to slow from the current
    (pre-entry-burn) speed down to a target entry-interface speed, the
    resulting burn duration/propellant (rocket equation), and a check
    against a maximum-deceleration structural/crew-load limit (5g is a
    representative, conservative bound; real vehicles are sized per
    their own structural margins).
    """
    delta_v = max(entry_interface_speed_m_s - target_post_entry_speed_m_s, 0.0)
    plan = boostback_burn_duration_s(delta_v, thrust_n, isp_s, m0_kg)
    achieved_decel_g = (thrust_n / m0_kg) / G0
    plan["achieved_deceleration_g"] = achieved_decel_g
    plan["within_structural_limit"] = achieved_decel_g <= max_deceleration_g
    plan["entry_interface_speed_m_s"] = entry_interface_speed_m_s
    plan["target_post_entry_speed_m_s"] = target_post_entry_speed_m_s
    return plan
