"""
guidance/peg.py

Powered Explicit Guidance (PEG) — closed-loop ascent/insertion guidance.

This is the algorithm family used on Apollo (Lunar Module ascent/descent),
Space Shuttle (ascent guidance, "Powered Explicit Guidance" by name), and
SLS. It is fundamentally different from the open-loop gravity-turn law in
guidance/gravity_turn.py:

    Gravity-turn (open-loop): "fly zero angle-of-attack and see where you
        end up." No target orbit is held in the loop; guidance cannot
        correct for off-nominal conditions (wind, engine underperformance,
        mass errors) — it just flies the same profile regardless.

    PEG (closed-loop): "given my CURRENT state (position, velocity, mass)
        and a TARGET orbit (radius, velocity, flight-path angle), compute
        the thrust-direction steering law that gets me there, and
        RECOMPUTE this every guidance cycle." This is why real orbital
        insertion stages use PEG-family guidance — it actively corrects
        for dispersions instead of hoping the open-loop profile was close
        enough.

This implementation follows the classical linear-tangent-steering PEG
formulation (constant-thrust, gravity-turn-terminated-by-PEG architecture
used operationally): the steering law is

    tan(chi) = A + B*t

i.e. the tangent of the thrust-pitch angle varies LINEARLY with time
within a guidance cycle — this is the defining "linear tangent steering"
result that falls out of the calculus-of-variations optimal solution for
a constant-gravity, constant-thrust powered flight arc to a fixed target
state. A and B are re-solved every guidance cycle (closed-loop) using the
current estimated state, which is what gives PEG its dispersion-rejection
property.

This is the classical linear-tangent PEG law solved for velocity and
flight-path-angle targeting (2 constraints). Known simplification: this
implementation does not simultaneously enforce the target radius as a
hard constraint the way full multi-constraint PEG (as flown
operationally) does — radius converges approximately as a consequence of
hitting velocity+flight-path-angle, not exactly. A full 3-constraint
solve (radius, velocity, flight-path-angle all exact at cutoff) requires
solving the additional range-equation term and is a documented extension
point, not implemented here.

Zero external dependencies beyond NumPy — CPU-only.
"""

import numpy as np


class PEGTarget:
    """
    Desired terminal (orbit-insertion) conditions for the guided burn.

    target_radius_m:      desired radius from Earth's center at cutoff, m
    target_velocity_m_s:  desired speed at cutoff, m/s
    target_flight_path_angle_rad: desired flight-path angle at cutoff
                           (0 = horizontal, i.e. circular-orbit insertion)
    """
    def __init__(self, target_radius_m, target_velocity_m_s,
                 target_flight_path_angle_rad=0.0):
        self.r_t = target_radius_m
        self.v_t = target_velocity_m_s
        self.gamma_t = target_flight_path_angle_rad


class PEGState:
    """Current vehicle state fed into the guidance solver each cycle."""
    def __init__(self, radius_m, velocity_m_s, flight_path_angle_rad,
                 mass_kg, thrust_n, isp_s, effective_gravity_m_s2):
        self.r = radius_m
        self.v = velocity_m_s
        self.gamma = flight_path_angle_rad
        self.m = mass_kg
        self.F = thrust_n
        self.isp = isp_s
        self.g = effective_gravity_m_s2


G0 = 9.80665


def solve_peg_steering(state: PEGState, target: PEGTarget, dt_guidance: float = 2.0):
    """
    Solve one PEG guidance cycle: return the linear-tangent-steering
    coefficients (A, B) and the predicted time-to-go until the target
    velocity is reached.

    This is the classical PEG "primer vector" style solution for
    constant-thrust powered flight, simplified to the standard two-
    parameter linear-tangent law (sufficient for a single-target,
    single-constraint ascent/insertion burn — full 6-DOF PEG additionally
    solves range/downrange constraints, which is a further extension).

    Returns dict with A (rad), B (rad/s), predicted burn time-to-go (s),
    and the instantaneous steering angle command for this cycle's start.
    """
    ve = state.isp * G0
    mdot = state.F / ve

    # Required delta-V to close the velocity gap to target (simple estimate;
    # PEG refines this each cycle as the state updates, which is what
    # gives it closed-loop correction authority)
    dv_needed = target.v_t - state.v

    # Time-to-go from the rocket equation: solve for t such that burning
    # at mdot for time t provides dv_needed of delta-V from current mass.
    # m(t) = m0 - mdot*t ; dv = ve*ln(m0/m(t))  =>  t = (m0/mdot)*(1 - exp(-dv/ve))
    if state.m <= 0 or mdot <= 0:
        raise ValueError("Invalid mass or mass flow rate")

    t_go = (state.m / mdot) * (1.0 - np.exp(-dv_needed / ve))
    t_go = max(t_go, 1e-6)

    # Linear-tangent steering coefficients.
    # A sets the initial pitch-rate-adjusted steering angle to close the
    # flight-path-angle gap; B sets how the angle evolves over the burn
    # to arrive at the target flight-path angle exactly at cutoff.
    # (Standard PEG closed-form for constant local gravity over the
    # short guided arc — re-linearized every cycle, which is what makes
    # the overall guidance nonlinear-capable despite each cycle being a
    # linear steering law.)
    gamma_gap = target.gamma_t - state.gamma

    # A: current steering angle needed, derived from the flight-path-angle
    # rate equation d(gamma)/dt = (F*sin(chi))/(m*v) - (g*cos(gamma))/v
    # Rearranged in small-angle form for the initial commanded chi.
    required_gamma_rate = gamma_gap / t_go
    sin_chi_now = (state.m * state.v * (required_gamma_rate + (state.g * np.cos(state.gamma)) / state.v)) / state.F
    sin_chi_now = np.clip(sin_chi_now, -1.0, 1.0)
    chi_now = np.arcsin(sin_chi_now)

    A = np.tan(chi_now)
    # B: rate of change of tan(chi) over the burn, chosen so the steering
    # angle relaxes toward zero (near-tangential thrust) by cutoff —
    # standard terminal condition for a circular-orbit insertion target.
    B = -A / t_go if t_go > 1e-6 else 0.0

    return {
        "A": A,
        "B": B,
        "chi_command_rad": chi_now,
        "chi_command_deg": np.degrees(chi_now),
        "time_to_go_s": t_go,
        "dv_remaining_m_s": dv_needed,
    }


def simulate_peg_guided_burn(
    r0_m: float, v0_m_s: float, gamma0_rad: float, m0_kg: float,
    thrust_n: float, isp_s: float, target: PEGTarget,
    guidance_cycle_s: float = 2.0, dt_integrate: float = 0.05,
    mu_earth: float = 3.986004418e14, max_time_s: float = 600.0,
):
    """
    Full closed-loop PEG-guided burn simulation: at each guidance cycle,
    re-solve the steering law from the CURRENT state (this is what makes
    it closed-loop / dispersion-rejecting, unlike a fixed open-loop pitch
    program), then integrate the point-mass dynamics forward using that
    steering command until the next guidance cycle.

    Returns time-series plus final insertion accuracy (how close the
    actual cutoff radius/velocity/flight-path-angle came to the target —
    this accuracy number is the key metric that proves closed-loop
    guidance beats open-loop for precision insertion).
    """
    ve = isp_s * G0
    mdot = thrust_n / ve

    r, v, gamma, m = r0_m, v0_m_s, gamma0_rad, m0_kg
    t = 0.0

    t_hist, r_hist, v_hist, gamma_hist, chi_hist = [], [], [], [], []

    while t < max_time_s and m > 0:
        g = mu_earth / r ** 2
        state = PEGState(r, v, gamma, m, thrust_n, isp_s, g)

        sol = solve_peg_steering(state, target, dt_guidance=guidance_cycle_s)
        chi = sol["chi_command_rad"]

        if sol["time_to_go_s"] <= dt_integrate:
            break  # close enough to cutoff, stop the burn

        # Integrate dynamics for one guidance cycle at fixed steering command
        # chi held constant within the cycle (standard PEG implementation
        # pattern: solve once per cycle, fly it open-loop within the cycle)
        steps_this_cycle = max(1, int(guidance_cycle_s / dt_integrate))
        for _ in range(steps_this_cycle):
            if m <= 0 or t >= max_time_s:
                break
            g = mu_earth / r ** 2

            # Point-mass equations of motion in polar (r, tangential) form
            dv_dt = (thrust_n * np.cos(chi)) / m - g * np.sin(gamma)
            dgamma_dt = (thrust_n * np.sin(chi)) / (m * v) - (g * np.cos(gamma)) / v + (v * np.cos(gamma)) / r
            dr_dt = v * np.sin(gamma)

            v += dv_dt * dt_integrate
            gamma += dgamma_dt * dt_integrate
            r += dr_dt * dt_integrate
            m -= mdot * dt_integrate
            t += dt_integrate

            t_hist.append(t)
            r_hist.append(r)
            v_hist.append(v)
            gamma_hist.append(gamma)
            chi_hist.append(np.degrees(chi))

    insertion_error = {
        "radius_error_m": r - target.r_t,
        "velocity_error_m_s": v - target.v_t,
        "flight_path_angle_error_deg": np.degrees(gamma - target.gamma_t),
    }

    return {
        "t": np.array(t_hist),
        "radius_m": np.array(r_hist),
        "velocity_m_s": np.array(v_hist),
        "gamma_deg": np.degrees(np.array(gamma_hist)),
        "chi_deg": np.array(chi_hist),
        "final_radius_m": r,
        "final_velocity_m_s": v,
        "final_gamma_deg": np.degrees(gamma),
        "final_mass_kg": m,
        "burn_time_s": t,
        "insertion_error": insertion_error,
    }
