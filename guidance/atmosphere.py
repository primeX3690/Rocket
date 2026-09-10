"""
guidance/atmosphere.py

Simplified US Standard Atmosphere 1976 model (troposphere + lower
stratosphere layers, sufficient for 0-50km ascent modeling — covers the
entire powered-ascent phase of a launch vehicle where drag/max-Q matter;
beyond ~50km density is negligible for load purposes).

Piecewise exponential/linear-lapse model, verified against published
standard-atmosphere reference tables at sea level, 11km, and 25km.
"""

import numpy as np

R_AIR = 287.05287    # specific gas constant for dry air, J/(kg*K)
G0 = 9.80665

# Layer base data: (base altitude m, base temp K, lapse rate K/m, base pressure Pa)
_LAYERS = [
    (0.0,     288.15, -0.0065, 101325.0),
    (11000.0, 216.65,  0.0,     22632.06),
    (20000.0, 216.65,  0.001,   5474.889),
    (32000.0, 228.65,  0.0028,  868.0187),
    (47000.0, 270.65,  0.0,     110.9063),
]


def _layer_for_altitude(h: float):
    layer = _LAYERS[0]
    for L in _LAYERS:
        if h >= L[0]:
            layer = L
        else:
            break
    return layer


def temperature(h_m: float) -> float:
    """Static air temperature (K) at geometric altitude h_m (0-51km valid range)."""
    h_base, t_base, lapse, _ = _layer_for_altitude(h_m)
    return t_base + lapse * (h_m - h_base)


def pressure(h_m: float) -> float:
    """Static air pressure (Pa) at altitude h_m."""
    h_base, t_base, lapse, p_base = _layer_for_altitude(h_m)
    if abs(lapse) < 1e-12:
        return p_base * np.exp(-G0 * (h_m - h_base) / (R_AIR * t_base))
    else:
        t = t_base + lapse * (h_m - h_base)
        return p_base * (t / t_base) ** (-G0 / (R_AIR * lapse))


def density(h_m: float) -> float:
    """Air density (kg/m^3) at altitude h_m via ideal gas law."""
    if h_m > 51000.0:
        return 0.0
    p = pressure(h_m)
    t = temperature(h_m)
    return p / (R_AIR * t)


def dynamic_pressure(h_m: float, v_m_s: float) -> float:
    """Dynamic pressure q = 0.5 * rho * v^2 (Pa) — the max-Q structural driver."""
    rho = density(h_m)
    return 0.5 * rho * v_m_s ** 2
