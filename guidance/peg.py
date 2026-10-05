"""
guidance/peg.py

Closed-loop ascent/insertion guidance in the Powered-Explicit-Guidance
(PEG) family, plus the shared point-mass burn integrator used by the rest
of the stack (engine-out, anomaly arbitration, Monte Carlo, mission profile).

What is actually implemented (honest summary)
---------------------------------------------
* Steering law: linear-tangent steering,  tan(theta) = A + B*tau,  where
  theta is the thrust pitch measured from the LOCAL HORIZONTAL and tau is
  time since the last guidance solve.
* Solve: a numerical *predictor-corrector*.  Each guidance cycle a
  Levenberg-Marquardt/Newton iteration finds the three unknowns
  (A, B, T_go) such that forward-integrating the point-mass equations of
  motion lands EXACTLY on the target (radius, speed, flight-path angle).
  That is a genuine 3-constraint solve: radius, velocity and flight-path
  angle are all enforced at cutoff.  (The original closed-form 2-constraint
  law is kept as `solve_peg_steering` for reference and for use as an
  initial guess.)
* Re-solved every cycle from the *estimated* state, warm-started from the
  previous solution, which is what gives the dispersion-rejecting
  behaviour.  If the target is unreachable with the remaining propellant
  (engine-out, sensor fault) the solver falls back to damped least squares
  and reports `converged=False` together with the residuals.

Not modelled: Earth oblateness (J2), rotation, aerodynamics above the
sensible atmosphere, thrust-vector-control dynamics (see
analysis/closed_loop_gnc.py and dynamics/six_dof.py for those).

Zero external dependencies beyond NumPy - CPU-only.
"""

import numpy as np
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



MU_EARTH_DEFAULT = 3.986004418e14


# ---------------------------------------------------------------------------
# Shared point-mass dynamics (polar frame: radius, speed, flight-path angle)
# ---------------------------------------------------------------------------

def point_mass_derivs(r, v, gamma, m, thrust_n, mdot, theta, mu=MU_EARTH_DEFAULT):
    """
    Derivatives of (r, v, gamma, m) for thrust pointed at pitch `theta`
    (from local horizontal) under inverse-square gravity.  The angle
    between thrust and velocity is chi = theta - gamma.
    """
    g = mu / r ** 2
    chi = theta - gamma
    dr = v * np.sin(gamma)
    dv = thrust_n * np.cos(chi) / m - g * np.sin(gamma)
    dgam = thrust_n * np.sin(chi) / (m * v) + (v / r - g / v) * np.cos(gamma)
    return dr, dv, dgam, -mdot


def rk4_step(state, thrust_n, mdot, theta, dt, mu=MU_EARTH_DEFAULT):
    """One RK4 step of the point-mass burn with the thrust pitch held at `theta`."""
    r, v, g, m = state

    def f(s):
        return point_mass_derivs(s[0], s[1], s[2], s[3], thrust_n, mdot, theta, mu)

    k1 = f((r, v, g, m))
    k2 = f((r + 0.5 * dt * k1[0], v + 0.5 * dt * k1[1], g + 0.5 * dt * k1[2], m + 0.5 * dt * k1[3]))
    k3 = f((r + 0.5 * dt * k2[0], v + 0.5 * dt * k2[1], g + 0.5 * dt * k2[2], m + 0.5 * dt * k2[3]))
    k4 = f((r + dt * k3[0], v + dt * k3[1], g + dt * k3[2], m + dt * k3[3]))
    return (r + dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
            v + dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
            g + dt / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]),
            m + dt / 6.0 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]))


def predict_cutoff(state, A, B, T, thrust_n, mdot, mu=MU_EARTH_DEFAULT, n_steps=None):
    """Forward-integrate a burn of duration T with tan(theta)=A+B*tau; return cutoff state."""
    n = n_steps if n_steps is not None else max(12, int(np.ceil(T / 1.5)))
    h = T / n
    s = tuple(state)
    for k in range(n):
        theta = np.arctan(A + B * (k + 0.5) * h)
        s = rk4_step(s, thrust_n, mdot, theta, h, mu)
    return s


# ---------------------------------------------------------------------------
# 3-constraint predictor-corrector solve
# ---------------------------------------------------------------------------

_RES_SCALE = np.array([100.0, 1.0, 1.0e-3])     # metres, m/s, rad -> comparable units


def _residual(cut, target):
    return np.array([cut[0] - target.r_t, cut[1] - target.v_t, cut[2] - target.gamma_t])


def solve_peg_predictor_corrector(state, thrust_n, mdot, target, m_dry_kg,
                                  mu=MU_EARTH_DEFAULT, guess=None, max_iter=25,
                                  tol=(2.0, 0.02, 2.0e-5)):
    """
    Solve for (A, B, T_go) so the predicted cutoff state equals the target.

    state:   (r, v, gamma, m) estimated current state
    guess:   optional (A, B, T) warm start (previous cycle's solution)
    tol:     convergence tolerances on |dr| [m], |dv| [m/s], |dgamma| [rad]

    Returns dict(A, B, T, converged, residual (r, v, gamma), iterations).
    """
    r0, v0, g0, m0 = state
    T_max = max(1.0, (m0 - m_dry_kg) / mdot)
    ve = thrust_n / mdot

    if guess is None:
        E0 = 0.5 * v0 ** 2 - mu / r0
        Et = 0.5 * target.v_t ** 2 - mu / target.r_t
        dv_est = max(5.0, (Et - E0) / max(0.5 * (v0 + target.v_t), 1.0))
        T0 = (m0 / mdot) * (1.0 - np.exp(-dv_est / ve))
        x = np.array([np.tan(g0 + 0.05), 0.0, np.clip(T0, 1.0, T_max)])
    else:
        x = np.array(guess, dtype=float)
        x[2] = np.clip(x[2], 0.2, T_max)

    steps = np.array([1.0e-3, 1.0e-4, 0.05])
    lam = 1.0e-2
    res = _residual(predict_cutoff(state, *x, thrust_n, mdot, mu), target)
    cost = np.sum((res / _RES_SCALE) ** 2)
    converged = False
    it = 0

    for it in range(1, max_iter + 1):
        if abs(res[0]) < tol[0] and abs(res[1]) < tol[1] and abs(res[2]) < tol[2]:
            converged = True
            it -= 1
            break
        J = np.empty((3, 3))
        for j in range(3):
            xp = x.copy()
            xp[j] += steps[j]
            xp[2] = min(xp[2], T_max) if j == 2 else xp[2]
            hj = xp[j] - x[j] if xp[j] != x[j] else -steps[j]
            if xp[j] == x[j]:
                xp[j] = x[j] - steps[j]
            rp = _residual(predict_cutoff(state, *xp, thrust_n, mdot, mu), target)
            J[:, j] = (rp - res) / (xp[j] - x[j])
        Js = J / _RES_SCALE[:, None]
        rs = res / _RES_SCALE
        improved = False
        for _ in range(8):
            H = Js.T @ Js + lam * np.diag(np.maximum(np.diag(Js.T @ Js), 1e-12))
            try:
                dx = -np.linalg.solve(H, Js.T @ rs)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            xn = x + dx
            xn[2] = np.clip(xn[2], 0.2, T_max)
            resn = _residual(predict_cutoff(state, *xn, thrust_n, mdot, mu), target)
            costn = np.sum((resn / _RES_SCALE) ** 2)
            if costn < cost:
                x, res, cost = xn, resn, costn
                lam = max(lam / 5.0, 1.0e-9)
                improved = True
                break
            lam *= 5.0
        if not improved:
            break
    else:
        converged = (abs(res[0]) < tol[0] and abs(res[1]) < tol[1] and abs(res[2]) < tol[2])

    if not converged:
        converged = (abs(res[0]) < tol[0] and abs(res[1]) < tol[1] and abs(res[2]) < tol[2])

    return {"A": float(x[0]), "B": float(x[1]), "T": float(x[2]),
            "converged": bool(converged), "residual": tuple(float(q) for q in res),
            "iterations": it}


# ---------------------------------------------------------------------------
# Closed-loop guided burn
# ---------------------------------------------------------------------------

def simulate_peg_guided_burn(
    r0_m: float, v0_m_s: float, gamma0_rad: float, m0_kg: float,
    thrust_n: float, isp_s: float, target: PEGTarget,
    guidance_cycle_s: float = 2.0, dt_integrate: float = 0.05,
    mu_earth: float = MU_EARTH_DEFAULT, max_time_s: float = 600.0,
    m_dry_kg: float = None, true_isp_s: float = None, true_thrust_n: float = None,
    nav_bias: tuple = (0.0, 0.0, 0.0), initial_guess: tuple = None,
):
    """
    Closed-loop PEG burn.  Each guidance cycle the predictor-corrector
    re-solves (A, B, T_go) from the ESTIMATED state (truth + `nav_bias`),
    then the TRUE dynamics are integrated (RK4) using that steering until
    the next cycle.  The burn is cut off when the solved T_go elapses.

    Model mismatch that guidance must reject:
      true_isp_s / true_thrust_n: actual engine (guidance assumes isp_s/thrust_n)
      nav_bias: (dr_m, dv_m_s, dgamma_rad) added to the state guidance sees

    m_dry_kg: propellant floor (default 25 % of m0).  If propellant runs out
    before the solve says cutoff, the burn ends at burnout (`burnout=True`).

    Returns time series and final accuracy including the resulting orbit's
    perigee/apogee (the metric that matters for a launch vehicle).
    """
    from guidance.orbit_utils import orbit_from_state

    if m_dry_kg is None:
        m_dry_kg = 0.25 * m0_kg
    F_true = thrust_n if true_thrust_n is None else true_thrust_n
    isp_true = isp_s if true_isp_s is None else true_isp_s
    mdot_nom = thrust_n / (isp_s * G0)
    mdot_true = F_true / (isp_true * G0)

    state = (r0_m, v0_m_s, gamma0_rad, m0_kg)
    t = 0.0
    guess = initial_guess
    hist = {k: [] for k in ("t", "r", "v", "gamma", "theta", "chi")}
    cycles = []
    burnout = False

    def log(s, theta, tt):
        hist["t"].append(tt); hist["r"].append(s[0]); hist["v"].append(s[1])
        hist["gamma"].append(s[2]); hist["theta"].append(theta)
        hist["chi"].append(theta - s[2])

    while t < max_time_s:
        est = (state[0] + nav_bias[0], state[1] + nav_bias[1], state[2] + nav_bias[2], state[3])
        sol = solve_peg_predictor_corrector(est, thrust_n, mdot_nom, target, m_dry_kg,
                                            mu=mu_earth, guess=guess)
        cycles.append({"t": t, "A": sol["A"], "B": sol["B"], "T_go": sol["T"],
                       "converged": sol["converged"], "iterations": sol["iterations"]})
        A, B, T_go = sol["A"], sol["B"], sol["T"]
        fly = min(guidance_cycle_s, T_go)
        n = max(1, int(np.ceil(fly / dt_integrate)))
        h = fly / n
        for k in range(n):
            theta = float(np.arctan(A + B * (k + 0.5) * h))
            if state[3] - mdot_true * h <= m_dry_kg:
                burnout = True
                break
            log(state, theta, t)
            state = rk4_step(state, F_true, mdot_true, theta, h, mu_earth)
            t += h
        if burnout or T_go <= guidance_cycle_s + 1e-9:
            break
        guess = (A + B * fly, B, T_go - fly)

    r, v, gamma, m = state
    log(state, hist["theta"][-1] if hist["theta"] else gamma, t)
    orbit = orbit_from_state(r, v, gamma, mu_earth)
    insertion_error = {
        "radius_error_m": r - target.r_t,
        "velocity_error_m_s": v - target.v_t,
        "flight_path_angle_error_deg": float(np.degrees(gamma - target.gamma_t)),
    }
    return {
        "t": np.array(hist["t"]),
        "radius_m": np.array(hist["r"]),
        "velocity_m_s": np.array(hist["v"]),
        "gamma_deg": np.degrees(np.array(hist["gamma"])),
        "pitch_deg": np.degrees(np.array(hist["theta"])),
        "chi_deg": np.degrees(np.array(hist["chi"])),
        "final_radius_m": r,
        "final_velocity_m_s": v,
        "final_gamma_deg": float(np.degrees(gamma)),
        "final_mass_kg": m,
        "burn_time_s": t,
        "insertion_error": insertion_error,
        "orbit": orbit,
        "burnout": burnout,
        "cycles": cycles,
        "solver_converged": all(c["converged"] for c in cycles[-2:]),
        "final_state": (r, v, gamma, m),
        "final_guess": (guess if guess is not None else (A, B, T_go)),
    }
