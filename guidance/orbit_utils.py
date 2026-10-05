"""
guidance/orbit_utils.py

Two-body orbit helpers used to judge burn-cutoff quality in the metric
that actually matters for a launch vehicle: *what orbit did we end up in?*

Cutoff radius/velocity/flight-path-angle errors are hard to interpret on
their own (a +10 m/s error at the wrong flight-path angle can be worse
than +30 m/s at the right one).  Converting the cutoff state to
perigee/apogee altitude gives a single physically meaningful number.
"""

import numpy as np

MU_EARTH = 3.986004418e14
R_EARTH = 6378137.0


def orbit_from_state(r_m: float, v_m_s: float, gamma_rad: float,
                     mu: float = MU_EARTH, r_earth_m: float = R_EARTH) -> dict:
    """
    Osculating two-body orbit from a polar state (radius, speed, flight-path
    angle measured from local horizontal).

    Returns semi-major axis, eccentricity, perigee/apogee altitude (km).
    For a non-elliptic (energy >= 0) state the altitudes are NaN and
    `is_bound` is False.
    """
    eps = 0.5 * v_m_s ** 2 - mu / r_m               # specific orbital energy
    h = r_m * v_m_s * np.cos(gamma_rad)             # specific angular momentum
    if eps >= 0.0:
        return {"is_bound": False, "a_m": np.inf, "e": np.nan,
                "perigee_alt_km": np.nan, "apogee_alt_km": np.nan,
                "energy_j_kg": eps, "h_m2_s": h}
    a = -mu / (2.0 * eps)
    e = np.sqrt(max(0.0, 1.0 + 2.0 * eps * h ** 2 / mu ** 2))
    rp, ra = a * (1.0 - e), a * (1.0 + e)
    return {"is_bound": True, "a_m": a, "e": e,
            "perigee_alt_km": (rp - r_earth_m) / 1000.0,
            "apogee_alt_km": (ra - r_earth_m) / 1000.0,
            "energy_j_kg": eps, "h_m2_s": h}


def circular_speed(r_m: float, mu: float = MU_EARTH) -> float:
    return float(np.sqrt(mu / r_m))


def orbit_error_km(final_state: tuple, target_alt_km: float,
                   mu: float = MU_EARTH, r_earth_m: float = R_EARTH) -> dict:
    """
    Perigee/apogee altitude error of a cutoff state relative to a circular
    target orbit of `target_alt_km`.
    """
    o = orbit_from_state(*final_state, mu=mu, r_earth_m=r_earth_m)
    return {"perigee_error_km": o["perigee_alt_km"] - target_alt_km,
            "apogee_error_km": o["apogee_alt_km"] - target_alt_km,
            "eccentricity": o["e"], "is_bound": o["is_bound"]}
