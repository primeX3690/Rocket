"""
dynamics/landing_legs.py

Landing-gear touchdown dynamics and tip-over stability, for checking
whether a powered-descent-guidance solution (guidance/
powered_descent_guidance.py) actually results in a SAFE landing, not
just a low touchdown speed at the right spot -- a vehicle can hit the
right point at low vertical speed and still tip over if it touches
down with too much lateral drift, tilt, or asymmetric leg loading.

Two real, standard models:
1. Spring-damper leg energy absorption: each leg treated as a
   spring-damper (the real mechanism -- crushable aluminum honeycomb or
   hydraulic struts on an actual vehicle -- absorbing touchdown kinetic
   energy so the vehicle doesn't bounce/rebound).
2. Tip-over stability: the classic "stability cone" check used in
   legged-vehicle/lander stability analysis -- a vehicle tips over if
   its center-of-gravity's ground projection (accounting for lateral
   drift and tilt at the instant of touchdown) falls outside the
   support polygon defined by the landing-leg footprint.

Zero external dependencies beyond NumPy - CPU-only.
"""

import numpy as np

G0 = 9.80665


class LandingLeg:
    """
    One leg's spring-damper touchdown model. The leg's compression is
    tied DIRECTLY to how far the vehicle body has penetrated past the
    ground-contact point (not an independently-driven oscillator), and
    the damping term acts on the vehicle's actual closing (descent)
    rate -- this is a standard 1-DOF impact-oscillator model, which is
    what a real strut/honeycomb landing leg physically is.
    """
    def __init__(self, stiffness_n_m: float, damping_ns_m: float, max_stroke_m: float):
        self.k = stiffness_n_m
        self.c = damping_ns_m
        self.max_stroke = max_stroke_m
        self.compression_m = 0.0

    def step(self, vehicle_height_m: float, vehicle_velocity_m_s: float, dt: float) -> float:
        """
        vehicle_height_m: vehicle body height above the ground-contact
        datum (touchdown at height=0; height<0 means the leg has begun
        compressing). vehicle_velocity_m_s: vehicle's vertical velocity
        (negative = descending). Returns this leg's upward reaction
        force (N); zero once the vehicle has left the ground again
        (a leg can only push, never pull).
        """
        target_compression = max(-vehicle_height_m, 0.0)
        self.compression_m = np.clip(target_compression, 0.0, self.max_stroke)
        if target_compression <= 0.0:
            return 0.0
        closing_rate = max(-vehicle_velocity_m_s, 0.0)   # only damps while still closing (descending)
        spring_force = self.k * self.compression_m
        damper_force = self.c * closing_rate
        return spring_force + damper_force

    @property
    def bottomed_out(self) -> bool:
        return self.compression_m >= self.max_stroke * 0.999


def simulate_touchdown(vertical_speed_m_s: float, total_mass_kg: float, n_legs: int,
                       leg_stiffness_n_m: float, leg_damping_ns_m: float, max_stroke_m: float,
                       dt: float = 0.0002, max_sim_time_s: float = 2.0) -> dict:
    """
    Simulates one vehicle mass landing on n_legs identical spring-damper
    legs from a given vertical touchdown speed under gravity, until the
    vehicle's vertical velocity settles near zero or a leg bottoms out.
    Returns peak deceleration (g's -- the number a crew/payload load
    limit would be checked against) and whether any leg bottomed out
    (a bottomed-out leg means the shock absorber ran out of travel and
    further loads pass straight to the airframe -- a real hard-landing
    failure mode).
    """
    legs = [LandingLeg(leg_stiffness_n_m, leg_damping_ns_m, max_stroke_m) for _ in range(n_legs)]
    v = -abs(vertical_speed_m_s)
    z = 0.0   # vehicle body height above the ground-contact datum; touchdown starts exactly at z=0
    t = 0.0
    peak_decel_g = 0.0
    any_bottomed = False
    v_hist, z_hist, t_hist = [], [], []

    settled_count = 0
    while t < max_sim_time_s:
        total_reaction = sum(leg.step(z, v, dt) for leg in legs)
        any_bottomed = any_bottomed or any(leg.bottomed_out for leg in legs)
        net_force = total_reaction - total_mass_kg * G0
        accel = net_force / total_mass_kg
        if total_reaction > 0:
            peak_decel_g = max(peak_decel_g, abs(accel) / G0)
        v += accel * dt
        z += v * dt
        z = max(z, -max_stroke_m)   # cannot penetrate past the legs' physical travel limit
        t += dt
        v_hist.append(v); z_hist.append(z); t_hist.append(t)
        if abs(v) < 0.02 and z <= 0.0:
            settled_count += 1
            if settled_count > 50:   # sustained near-zero velocity, not just a single-step crossing
                break
        else:
            settled_count = 0

    return {
        "t": np.array(t_hist), "velocity_m_s": np.array(v_hist), "height_m": np.array(z_hist),
        "peak_deceleration_g": peak_decel_g, "any_leg_bottomed_out": any_bottomed,
        "final_leg_compressions_m": [leg.compression_m for leg in legs],
        "settled": abs(v) < 0.1,
    }


def tip_over_stability_check(cg_height_m: float, leg_footprint_radius_m: float,
                             touchdown_lateral_speed_m_s: float, touchdown_tilt_rad: float,
                             time_to_arrest_lateral_s: float = 0.5) -> dict:
    """
    Classic tip-over stability check: the vehicle's Cg, projected onto
    the ground accounting for BOTH the static tilt at touchdown and the
    lateral drift accumulated while the legs arrest that drift (a real,
    physical effect -- a vehicle touching down with lateral velocity
    keeps sliding/rocking until friction/leg dynamics stop it), must
    stay within the leg footprint's support radius or the vehicle tips.
    """
    tilt_offset_m = cg_height_m * np.sin(touchdown_tilt_rad)
    lateral_drift_m = abs(touchdown_lateral_speed_m_s) * time_to_arrest_lateral_s
    total_cg_offset_m = tilt_offset_m + lateral_drift_m
    stable = total_cg_offset_m < leg_footprint_radius_m
    margin_m = leg_footprint_radius_m - total_cg_offset_m
    return {
        "stable": bool(stable), "cg_offset_m": total_cg_offset_m,
        "footprint_radius_m": leg_footprint_radius_m, "margin_m": margin_m,
        "tilt_contribution_m": tilt_offset_m, "lateral_drift_contribution_m": lateral_drift_m,
    }
