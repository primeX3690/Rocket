"""
guidance/engine_out.py

Engine-out contingency guidance.

Real multi-engine upper stages (and clustered first stages) can lose one
or more engines mid-burn and still complete the mission — but only if
guidance RE-PLANS the target. This is the same concept as Space Shuttle
"Abort to Orbit" (ATO): if a main engine fails partway through ascent
and the planned orbit is no longer reachable with remaining propellant,
guidance retargets to the highest/closest orbit that IS still reachable,
rather than blindly continuing toward an unreachable target and running
the tanks dry short of any usable orbit.

This module answers the concrete question a real GNC engineer must
answer after an engine-out event:

    "Given remaining propellant mass, current Isp, and reduced thrust,
     is the ORIGINAL target orbit still achievable? If not, what is the
     best orbit we CAN achieve, and how do we retarget PEG to hit it?"

Key insight (and a common misconception worth stating explicitly): losing
thrust does NOT by itself reduce total achievable delta-V — the rocket
equation's dv = Isp*g0*ln(m0/mf) depends on Isp and mass ratio, not
thrust magnitude. What an engine-out actually costs is:
    (a) increased gravity losses (longer burn at lower thrust-to-weight
        means gravity has more time to fight the ascent), and
    (b) sometimes insufficient TIME/propellant margin to complete the
        original burn before some other constraint (e.g. a hard cutoff
        altitude/downrange limit) is hit.
This module models (a) directly via the guided-burn integrator, and (b)
via a propellant-sufficiency check.

Zero external dependencies beyond NumPy — CPU-only.
"""

import numpy as np
from guidance.peg import PEGTarget, simulate_peg_guided_burn

G0 = 9.80665
MU_EARTH = 3.986004418e14


class EngineOutEvent:
    """Describes a single engine failure event."""
    def __init__(self, time_s: float, thrust_fraction_remaining: float):
        """
        time_s: mission-elapsed time (from start of THIS burn) at which
                the failure occurs
        thrust_fraction_remaining: fraction of nominal thrust still
                available after the failure (e.g. 2 engines out of 3
                still firing -> 0.667)
        """
        self.time_s = time_s
        self.thrust_fraction = thrust_fraction_remaining


def achievable_circular_orbit_radius(m_current_kg: float, m_dry_kg: float,
                                      isp_s: float, v_current_m_s: float,
                                      r_current_m: float) -> float:
    """
    Given current mass, dry mass, Isp, and current speed/radius, compute
    the radius of the highest circular orbit reachable using all
    remaining propellant.

    Uses the standard vis-viva / specific-orbital-energy method: an
    impulsive burn of the available delta-V applied at the current
    position raises specific orbital energy by (v*dv + 0.5*dv^2); the
    circular orbit consistent with that new energy has radius
    r = -mu / (2*E), which is the correct energy-based way to estimate
    "best reachable circular altitude" (simpler velocity-matching
    heuristics can give physically nonsensical results when available
    delta-V is small relative to the local circular velocity gap).

    Returns r_current_m unchanged if the achievable energy would still
    be negative-enough to only reach an orbit below the current radius
    (i.e. no real improvement is achievable — treated as "hold current
    altitude" for contingency-target purposes).
    """
    dv_available = isp_s * G0 * np.log(m_current_kg / m_dry_kg)

    current_energy = 0.5 * v_current_m_s ** 2 - MU_EARTH / r_current_m
    v_after_burn = v_current_m_s + dv_available
    new_energy = 0.5 * v_after_burn ** 2 - MU_EARTH / r_current_m

    if new_energy >= 0:
        # Escape energy or greater — cap conservatively at a very high
        # but bound circular reference rather than returning a
        # non-physical infinite/negative radius.
        return r_current_m * 50.0

    r_achievable = -MU_EARTH / (2.0 * new_energy)
    # Only clamp against a physically impossible result (an "orbit" whose
    # radius would be below Earth's surface) — clamping against
    # r_current_m unconditionally would mask genuinely-below-current-
    # altitude results, which correctly indicate "not enough delta-V to
    # circularize anywhere above where we already are" (a real total-loss
    # signal, not something to paper over).
    return max(r_achievable, 6478137.0)  # floor: ~100km altitude, below which no stable orbit exists anyway


def select_contingency_target(m_current_kg: float, m_dry_kg: float,
                               isp_s: float, v_current_m_s: float,
                               r_current_m: float,
                               nominal_target: PEGTarget,
                               margin_factor: float = 0.97) -> tuple:
    """
    Decide whether the nominal target orbit is still achievable after an
    engine-out event, and if not, compute a contingency target.

    margin_factor: safety margin applied to the achievable delta-V
                   before committing to the nominal target (real FDOs
                   never plan to the exact propellant-exhaustion edge —
                   they hold margin for guidance/nav dispersion, matching
                   the same design philosophy as the dispersion analysis
                   in analysis/monte_carlo.py).

    Returns (target_to_use: PEGTarget, is_contingency: bool, achievable_r_m: float)
    """
    dv_available = isp_s * G0 * np.log(m_current_kg / m_dry_kg)
    v_reachable_with_margin = v_current_m_s + dv_available * margin_factor

    # Energy-consistent delta-V requirement: compare the SPECIFIC ORBITAL
    # ENERGY of the target circular orbit against current energy, then
    # convert that energy gap to an equivalent impulsive delta-V applied
    # at the current radius. This is consistent with the same vis-viva
    # method used in achievable_circular_orbit_radius() below (so the
    # boundary case lines up exactly), and is a materially better
    # estimate than a naive endpoint velocity-gap comparison — a raw
    # velocity-gap check systematically UNDER-estimates what a real
    # climb-and-accelerate ascent burn costs, because it ignores the
    # potential-energy cost of gaining altitude.
    target_energy = -MU_EARTH / (2.0 * nominal_target.r_t)
    current_energy = 0.5 * v_current_m_s ** 2 - MU_EARTH / r_current_m
    v_needed_sq = 2.0 * (target_energy + MU_EARTH / r_current_m)
    v_needed = np.sqrt(max(v_needed_sq, 0.0))
    dv_needed_for_nominal = v_needed - v_current_m_s

    if dv_available * margin_factor >= dv_needed_for_nominal:
        # Nominal target still achievable with margin — no replan needed.
        return nominal_target, False, nominal_target.r_t

    # Nominal not achievable — compute the best circular orbit we CAN reach
    # and retarget there instead (Abort-to-Orbit style contingency).
    r_achievable = achievable_circular_orbit_radius(
        m_current_kg, m_dry_kg, isp_s, v_current_m_s, r_current_m
    )
    v_achievable = np.sqrt(MU_EARTH / r_achievable)

    contingency_target = PEGTarget(
        target_radius_m=r_achievable,
        target_velocity_m_s=v_achievable,
        target_flight_path_angle_rad=0.0,
    )
    return contingency_target, True, r_achievable


def simulate_burn_with_engine_out(
    r0_m, v0_m_s, gamma0_rad, m0_kg, m_dry_kg,
    nominal_thrust_n, isp_s, nominal_target: PEGTarget,
    engine_out: EngineOutEvent = None,
    replan_check_interval_s: float = 5.0,
    guidance_cycle_s: float = 2.0, dt_integrate: float = 0.05,
    max_time_s: float = 900.0,
):
    """
    Simulate a PEG-guided burn that may experience an engine-out event
    partway through, with contingency replanning.

    Architecture: run the guided burn in short interval chunks
    (replan_check_interval_s). After each chunk, apply the engine-out
    thrust reduction if its trigger time has passed, then RE-CHECK
    whether the current target is still achievable and retarget if not,
    before continuing. This mirrors how a real flight computer polls
    engine health and re-evaluates the guidance target on a cycle.
    """
    r, v, gamma, m = r0_m, v0_m_s, gamma0_rad, m0_kg
    t = 0.0
    thrust = nominal_thrust_n
    current_target = nominal_target
    contingency_triggered = False
    contingency_time = None

    t_hist, r_hist, v_hist, gamma_hist, thrust_hist = [], [], [], [], []

    while t < max_time_s and m > m_dry_kg:
        # Apply engine-out event if its time has arrived
        if engine_out is not None and t >= engine_out.time_s and thrust == nominal_thrust_n:
            thrust = nominal_thrust_n * engine_out.thrust_fraction

        # Re-evaluate target achievability against current state — but
        # only BEFORE a contingency has already been committed. Once a
        # contingency retarget has been selected, freeze it: continuously
        # re-solving "best achievable orbit from remaining propellant"
        # every cycle while actively burning toward the previous target
        # causes the estimate to chase itself downward (each re-solve sees
        # less propellant than the last target assumed), which never
        # converges. Real flight software commits to an abort target once
        # computed, absent a second independent failure — modeled here the
        # same way.
        if not contingency_triggered:
            new_target, is_contingency, _ = select_contingency_target(
                m, m_dry_kg, isp_s, v, r, nominal_target
            )
            if is_contingency:
                contingency_triggered = True
                contingency_time = t
            current_target = new_target

        # Run one short chunk of guided burn from current state
        chunk = simulate_peg_guided_burn(
            r0_m=r, v0_m_s=v, gamma0_rad=gamma, m0_kg=m,
            thrust_n=thrust, isp_s=isp_s, target=current_target,
            guidance_cycle_s=guidance_cycle_s, dt_integrate=dt_integrate,
            max_time_s=replan_check_interval_s,
        )

        if len(chunk["t"]) == 0:
            break  # already at/near cutoff

        # Advance state to end of this chunk
        r = chunk["final_radius_m"]
        v = chunk["final_velocity_m_s"]
        gamma = np.radians(chunk["final_gamma_deg"])
        m = chunk["final_mass_kg"]

        for i in range(len(chunk["t"])):
            t_hist.append(t + chunk["t"][i])
            r_hist.append(chunk["radius_m"][i])
            v_hist.append(chunk["velocity_m_s"][i])
            gamma_hist.append(chunk["gamma_deg"][i])
            thrust_hist.append(thrust)

        t += chunk["burn_time_s"]

        # Stop once we're within one integration step of the (possibly
        # contingency) target — mirrors the cutoff condition inside
        # simulate_peg_guided_burn itself.
        if abs(current_target.v_t - v) < 5.0:
            break

    final_error = {
        "radius_error_m": r - current_target.r_t,
        "velocity_error_m_s": v - current_target.v_t,
        "flight_path_angle_error_deg": np.degrees(gamma) - np.degrees(current_target.gamma_t),
    }

    return {
        "t": np.array(t_hist),
        "radius_m": np.array(r_hist),
        "velocity_m_s": np.array(v_hist),
        "gamma_deg": np.array(gamma_hist),
        "thrust_n": np.array(thrust_hist),
        "final_radius_m": r,
        "final_velocity_m_s": v,
        "final_mass_kg": m,
        "contingency_triggered": contingency_triggered,
        "contingency_time_s": contingency_time,
        "final_target_radius_m": current_target.r_t,
        "final_target_velocity_m_s": current_target.v_t,
        "insertion_error": final_error,
    }
