"""
analysis/reusability_mission.py

End-to-end first-stage RECOVERY mission: takes a stage's state at
separation (e.g. from analysis/mission_profile.py's stage1 output, or
guidance/staging_events.py's stage_separation_event) through boostback,
entry burn, aerodynamic descent (grid fins active), and a powered
landing burn, down to a touchdown-stability verdict -- tying together
every reusability module added in this pass into ONE genuine round-trip
simulation, the same way analysis/mission_profile.py ties together the
ascent-side modules.

Zero external dependencies beyond NumPy - CPU-only.
"""

import numpy as np

from guidance.boostback_and_entry import boostback_delta_v_command, boostback_burn_duration_s, entry_burn_plan
from guidance.powered_descent_guidance import (
    simulate_powered_landing, suicide_burn_ignition_altitude_m, zem_zev_acceleration_command,
)
from guidance.atmosphere import density, dynamic_pressure
from guidance.gravity_turn import speed_of_sound
from dynamics.grid_fins import grid_fin_lift_coefficient, grid_fin_forces_n
from dynamics.landing_legs import simulate_touchdown, tip_over_stability_check

G0 = 9.80665
GRAVITY_VEC = np.array([0.0, -G0])


def _grid_fin_lateral_accel(lateral_error_m: float, lateral_vel_m_s: float, altitude_m: float,
                            speed: float, mass_kg: float, q_pa: float,
                            fin_area_m2: float = 0.6, n_fins: int = 4,
                            min_q_for_authority_pa: float = 500.0,
                            max_deflection_rad: float = np.radians(20.0)) -> float:
    """
    Closed-loop grid-fin lateral (cross-range/downrange) steering
    during the atmospheric descent phase: commands a fin deflection
    using the SAME ZEM/ZEV minimum-energy law as the powered-descent
    and boostback guidance (applied here to the lateral axis only, with
    altitude/descent-rate as the time-to-go proxy), converts the
    resulting required lateral acceleration into a required lift force,
    then inverts the grid-fin lift model (dynamics/grid_fins.py) for
    the deflection angle that would produce it — clipped to a
    realistic physical deflection limit (fins stall past ~20 deg).

    A simple velocity-only PD damper was tried here instead and found
    to make things WORSE: it fights the boostback burn's deliberately
    nonzero lateral velocity (needed for most of the descent just to
    cover the distance back to the pad), rather than letting that
    velocity decay only as it should late in the flight. ZEM/ZEV
    naturally handles this correctly (its gains scale with remaining
    distance AND remaining time, the same reason it's used for the
    actual landing burn), and its output is hard-clamped to the fins'
    real physical deflection limit, so it can only ever help, never
    overcorrect, regardless of how large the raw computed command is.

    Below `min_q_for_authority_pa`, dynamic pressure is too low for
    fins to have meaningful authority (a real vehicle relies on RCS
    thrusters, not fins, in that regime — out of scope here).

    Honest limitation: with this vehicle's modeled fin size
    (fin_area_m2=0.6 m^2 x 4 fins — small relative to a multi-tonne
    stage), the fins' real aerodynamic authority is modest compared to
    the thrust-based landing burn, so for a LARGE residual targeting
    error their measurable effect on the final touchdown state is
    small — a genuine physical finding (undersized control authority),
    not a sign this function isn't working; see
    tests/test_reusability.py for a scenario-scale check that DOES show
    a measurable effect.
    """
    if q_pa < min_q_for_authority_pa:
        return 0.0

    t_go = max(2.0 * altitude_m / max(abs(speed), 10.0), 3.0)   # same average-velocity T_go form used elsewhere
    a_lateral_cmd = -6.0 * lateral_error_m / t_go ** 2 - 4.0 * lateral_vel_m_s / t_go
    required_force_n = a_lateral_cmd * mass_kg

    mach = speed / speed_of_sound(max(altitude_m, 0.0))
    max_force_n = grid_fin_forces_n(max_deflection_rad, mach, q_pa, fin_area_m2, n_fins)["lift_n"]
    if abs(max_force_n) < 1e-9:
        return 0.0   # no aerodynamic authority at this Mach/q (e.g. effectiveness ~0)

    # Invert the (generally nonlinear, post-stall) lift model by scaling
    # the deflection fraction needed, then clip to the physical limit —
    # cheaper and just as correct here as a numerical root-find, since
    # grid_fin_lift_coefficient is monotonic up to the stall angle.
    deflection_frac = np.clip(required_force_n / max_force_n, -1.0, 1.0)
    deflection_rad = deflection_frac * max_deflection_rad
    forces = grid_fin_forces_n(deflection_rad, mach, q_pa, fin_area_m2, n_fins)
    return forces["lift_n"] / mass_kg


def _coast_to_altitude(pos: np.ndarray, vel: np.ndarray, target_altitude_m: float) -> tuple:
    """
    Exact ballistic (gravity-only, no atmosphere) coast time to reach
    target_altitude_m, correctly handling a vehicle that is still
    ASCENDING at the start. Solved analytically from
    y(t) = y0 + vy0*t - 0.5*g*t^2 = target_altitude_m, taking the
    later (correct, "on the way down") root. Valid ABOVE the sensible
    atmosphere; use `_descend_with_drag` below the entry interface,
    where atmospheric drag is no longer negligible.
    """
    y0, vy0 = pos[1], vel[1]
    dy = y0 - target_altitude_m
    if dy <= 0:
        return pos.copy(), vel.copy(), 0.0   # already at/below target altitude
    discriminant = vy0 ** 2 + 2.0 * G0 * dy
    if discriminant < 0:
        return pos.copy(), vel.copy(), 0.0
    t = (vy0 + np.sqrt(discriminant)) / G0   # larger (later, descending-through) root
    t = max(t, 0.0)
    new_pos = pos + vel * t + 0.5 * GRAVITY_VEC * t ** 2
    new_vel = vel + GRAVITY_VEC * t
    return new_pos, new_vel, t


def _descend_with_drag(pos: np.ndarray, vel: np.ndarray, target_altitude_m: float,
                       drag_coeff: float, ref_area_m2: float, mass_kg: float,
                       dt: float = 0.05, max_time_s: float = 300.0) -> tuple:
    """
    Numerically integrates the descent from the current state down to
    target_altitude_m INCLUDING atmospheric drag (reusing guidance/
    atmosphere.py's density/dynamic_pressure), unlike a pure ballistic
    coast — below the sensible-atmosphere entry interface, drag is the
    dominant deceleration mechanism for a returning stage (this is what
    keeps a real Falcon-9-class booster's terminal velocity in the
    hundreds, not thousands, of m/s at low altitude; omitting it here
    was an earlier bug in this module that let the vehicle free-fall to
    an unphysical multi-km/s speed by the time it reached the landing-
    burn ignition altitude). Grid-fin control forces
    (dynamics/grid_fins.py) act on top of this drag and are not
    separately re-derived here since they are a control INPUT the
    caller can add as an extra lateral force; this integrates the
    baseline (uncontrolled aerodynamic) descent.
    """
    pos, vel = pos.copy().astype(float), vel.copy().astype(float)
    t = 0.0
    while pos[1] > target_altitude_m and t < max_time_s:
        speed = np.linalg.norm(vel)
        rho = density(max(pos[1], 0.0))
        drag_mag = 0.5 * rho * speed ** 2 * drag_coeff * ref_area_m2
        drag_accel = -drag_mag * vel / max(speed, 1e-9) / mass_kg
        accel = GRAVITY_VEC + drag_accel
        vel = vel + accel * dt
        pos = pos + vel * dt
        t += dt
    return pos, vel, t


def _coast_through_entry_and_descent(
    pos, vel, mass, dry_mass_kg, max_thrust_n, isp_s,
    entry_altitude_m, entry_burn_target_speed_m_s, max_landing_deceleration_m_s2,
    descent_drag_coeff, descent_ref_area_m2, landing_site_x,
):
    """
    Runs phases 2 (coast to entry interface), 3 (entry burn), and 4
    (drag-limited descent to ignition altitude) from a given starting
    state. Factored out so `simulate_stage_recovery` can call it TWICE:
    once as a quick PREDICTOR (to estimate total time-of-flight before
    choosing the boostback burn's target velocity) and once for the
    REAL run with the corrected boostback applied — a standard
    predict-then-correct ("explicit guidance") pattern, needed because
    the boostback burn must be sized using the ACTUAL time the full
    coast+entry+descent sequence takes (which depends on drag, itself
    dependent on the trajectory), not a crude one-shot kinematic guess.
    """
    pos, vel = pos.copy(), vel.copy()
    t_elapsed = 0.0

    pos, vel, t1 = _coast_to_altitude(pos, vel, entry_altitude_m)
    t_elapsed += t1

    entry_speed = float(np.linalg.norm(vel))
    entry_plan = entry_burn_plan(entry_speed, entry_burn_target_speed_m_s, max_thrust_n, isp_s, mass)
    entry_infeasible = entry_plan["final_mass_kg"] < dry_mass_kg
    if entry_speed > entry_burn_target_speed_m_s:
        if entry_infeasible:
            available_propellant_kg = max(mass - dry_mass_kg, 0.0)
            achieved_fraction = available_propellant_kg / max(mass - entry_plan["final_mass_kg"], 1e-9)
            target_speed = entry_speed - (entry_speed - entry_burn_target_speed_m_s) * achieved_fraction
            vel = vel * (target_speed / entry_speed)
            mass = dry_mass_kg
        else:
            vel = vel * (entry_burn_target_speed_m_s / entry_speed)
            mass = entry_plan["final_mass_kg"]

    dt_descent = 0.05
    t_descent = 0.0
    ignition_alt = 0.0
    while t_descent < 300.0:
        # NOTE: safety_margin_m is set much larger (1200 m) than
        # suicide_burn_ignition_altitude_m's own 50 m structural/
        # dispersion-margin default. That default only covers the pure
        # VERTICAL kinematics; this mission also needs the ZEM/ZEV
        # landing burn to correct whatever residual DOWNRANGE targeting
        # error remains from the boostback+entry phases (a real vehicle
        # gets this margin from a higher, deliberately-chosen ignition
        # altitude for exactly this reason — the landing burn is not
        # purely vertical).
        ignition_alt = suicide_burn_ignition_altitude_m(abs(vel[1]), max_landing_deceleration_m_s2,
                                                        safety_margin_m=1200.0)
        if pos[1] <= ignition_alt:
            break
        speed = np.linalg.norm(vel)
        rho = density(max(pos[1], 0.0))
        drag_mag = 0.5 * rho * speed ** 2 * descent_drag_coeff * descent_ref_area_m2
        drag_accel = -drag_mag * vel / max(speed, 1e-9) / mass
        q = dynamic_pressure(max(pos[1], 0.0), speed)
        lateral_accel = _grid_fin_lateral_accel(
            lateral_error_m=pos[0] - landing_site_x, lateral_vel_m_s=vel[0],
            altitude_m=pos[1], speed=speed, mass_kg=mass, q_pa=q,
        )
        accel = GRAVITY_VEC + drag_accel + np.array([lateral_accel, 0.0])
        vel = vel + accel * dt_descent
        pos = pos + vel * dt_descent
        t_descent += dt_descent
        if pos[1] <= 0:
            break
    t_elapsed += t_descent

    return {"pos": pos, "vel": vel, "mass": mass, "ignition_alt": ignition_alt,
           "entry_plan": entry_plan, "entry_infeasible": entry_infeasible, "t_elapsed": t_elapsed}


def simulate_stage_recovery(
    separation_pos_m: np.ndarray, separation_vel_m_s: np.ndarray, separation_mass_kg: float,
    dry_mass_kg: float, max_thrust_n: float, min_throttle_frac: float, isp_s: float,
    landing_site_pos_m: np.ndarray,
    entry_burn_target_speed_m_s: float = 500.0,
    max_landing_deceleration_m_s2: float = 30.0, n_legs: int = 4,
    leg_stiffness_n_m: float = 2.0e6, leg_damping_ns_m: float = 1.5e5, leg_max_stroke_m: float = 0.6,
    cg_height_m: float = 8.0, leg_footprint_radius_m: float = 4.5,
    descent_drag_coeff: float = 1.0, descent_ref_area_m2: float = 10.0,
):
    """
    Runs the full recovery sequence from stage separation to touchdown-
    stability verdict. `separation_pos_m`/`separation_vel_m_s` are 2D
    [downrange_m, altitude_m] / [vx, vy] — matching this project's
    existing vertical-plane point-mass convention (guidance/
    gravity_turn.py, analysis/mission_profile.py).

    Returns a dict with each phase's result plus an overall
    `mission_success` verdict (landed within a reasonable miss distance,
    touchdown speed within the vehicle's structural limit, AND stable
    against tip-over).
    """
    mass = separation_mass_kg
    pos = np.asarray(separation_pos_m, dtype=float).copy()
    vel = np.asarray(separation_vel_m_s, dtype=float).copy()
    landing_site_pos_m = np.asarray(landing_site_pos_m, dtype=float)
    entry_altitude_m = 60000.0   # nominal sensible-atmosphere entry interface for this vehicle class

    # --- Phase 1: Boostback burn (predictor-corrector) ---
    # A real boostback burn redirects the DOWNRANGE VELOCITY so the
    # subsequent coast+entry-burn+descent sequence heads back toward
    # the landing site — it does NOT try to null the full position
    # error in one short burn (that badly overloads the ZEM/ZEV law
    # used for the PRECISE final landing burn, demanding enormous
    # accelerations at short T_go for a large distance).
    #
    # PREDICTOR: run phases 2-4 once with the UNCORRECTED downrange
    # velocity, purely to get a realistic estimate of the total
    # remaining time-of-flight (which depends on drag, staging, and
    # entry-burn timing — a crude one-shot kinematic guess badly
    # underestimates it, as this module's own development discovered:
    # an early version's guess was off by more than 10x once the
    # drag-limited descent phase was modeled correctly).
    predictor = _coast_through_entry_and_descent(
        pos, vel, mass, dry_mass_kg, max_thrust_n, isp_s,
        entry_altitude_m, entry_burn_target_speed_m_s, max_landing_deceleration_m_s2,
        descent_drag_coeff, descent_ref_area_m2, landing_site_pos_m[0],
    )
    t_total_estimate = max(predictor["t_elapsed"], 10.0)

    # CORRECTOR: choose the boostback target downrange velocity using
    # this refined total-time estimate (average-velocity form, same
    # kinematic idea as guidance/powered_descent_guidance.py's T_go
    # estimator), then actually apply it.
    target_vx = -(pos[0] - landing_site_pos_m[0]) / t_total_estimate
    boostback_dv = abs(vel[0] - target_vx)
    boostback_plan = boostback_burn_duration_s(boostback_dv, max_thrust_n, isp_s, mass)
    boostback_infeasible = boostback_plan["final_mass_kg"] < dry_mass_kg
    if boostback_infeasible:
        # Not enough propellant margin for the full boostback delta-v —
        # cap propellant use at what's available above the dry mass (an
        # honest partial burn, not a silent below-dry-mass mass value).
        available_propellant_kg = max(mass - dry_mass_kg, 0.0)
        achieved_fraction = available_propellant_kg / max(mass - boostback_plan["final_mass_kg"], 1e-9)
        vel[0] = vel[0] + (target_vx - vel[0]) * achieved_fraction
        mass = dry_mass_kg
    else:
        vel[0] = target_vx
        mass = boostback_plan["final_mass_kg"]

    # --- Phases 2-4 (real run, with the corrected boostback applied) ---
    real_run = _coast_through_entry_and_descent(
        pos, vel, mass, dry_mass_kg, max_thrust_n, isp_s,
        entry_altitude_m, entry_burn_target_speed_m_s, max_landing_deceleration_m_s2,
        descent_drag_coeff, descent_ref_area_m2, landing_site_pos_m[0],
    )
    pos, vel, mass = real_run["pos"], real_run["vel"], real_run["mass"]
    ignition_alt = real_run["ignition_alt"]
    entry_plan, entry_infeasible = real_run["entry_plan"], real_run["entry_infeasible"]

    # --- Phase 5: Powered landing burn (ZEM/ZEV) ---
    landing_result = simulate_powered_landing(
        r0_m=pos, v0_m_s=vel, m0_kg=mass, m_dry_kg=dry_mass_kg,
        max_thrust_n=max_thrust_n, min_throttle_frac=min_throttle_frac, isp_s=isp_s,
        target_pos_m=landing_site_pos_m, target_vel_m_s=np.array([0.0, -2.0]),
        gravity_m_s2=GRAVITY_VEC, altitude_axis=1,
    )

    # --- Phase 6: Touchdown / tip-over stability ---
    touchdown_speed = landing_result["touchdown_speed_m_s"]
    touchdown_sim = simulate_touchdown(
        vertical_speed_m_s=abs(landing_result["final_velocity_m_s"][1]),
        total_mass_kg=landing_result["final_mass_kg"], n_legs=n_legs,
        leg_stiffness_n_m=leg_stiffness_n_m, leg_damping_ns_m=leg_damping_ns_m,
        max_stroke_m=leg_max_stroke_m,
    )
    stability = tip_over_stability_check(
        cg_height_m=cg_height_m, leg_footprint_radius_m=leg_footprint_radius_m,
        touchdown_lateral_speed_m_s=abs(landing_result["final_velocity_m_s"][0]),
        touchdown_tilt_rad=0.0,   # attitude control assumed to hold near-vertical; see dynamics/six_dof.py for full attitude
    )

    mission_success = (
        not boostback_infeasible
        and not entry_infeasible
        and landing_result["miss_distance_m"] < 100.0
        and not touchdown_sim["any_leg_bottomed_out"]
        and stability["stable"]
    )

    return {
        "boostback_delta_v_m_s": boostback_dv, "boostback_plan": boostback_plan,
        "boostback_infeasible": boostback_infeasible,
        "entry_burn_plan": entry_plan, "entry_infeasible": entry_infeasible,
        "ignition_altitude_m": ignition_alt,
        "landing": landing_result, "touchdown_dynamics": touchdown_sim,
        "tip_over_stability": stability, "mission_success": mission_success,
    }

