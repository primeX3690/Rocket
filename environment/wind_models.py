"""
environment/wind_models.py

Wind shear and turbulence models, additive to guidance/atmosphere.py's
US Standard Atmosphere 1976 (which gives density/pressure/temperature
but no wind). Two real, standard models are implemented:

1. Logarithmic wind-shear profile (the standard boundary-layer model
   used in wind engineering / launch-pad ground-wind analysis) for
   mean wind speed as a function of altitude, up to the boundary layer
   top (~2 km); above that, a simple jet-stream bump is layered on.
2. A discrete Dryden turbulence approximation (the standard MIL-F-8785C
   turbulence model used in flight dynamics/handling-qualities work) for
   the fluctuating (gust) component, implemented as a first-order
   shaping filter driven by white noise -- the same structure used in
   full-motion flight simulators, simplified to a single (longitudinal)
   channel for this ascent-focused stack.

NOT implemented (stated plainly): NRLMSISE-00 (needs proprietary/large
tabulated coefficient sets and solar-activity indices this project has
no way to source or validate offline), real historical/live weather
data, or full 3D non-stationary wind fields. This is a standard
ENGINEERING wind model, not a weather-data product.

Zero external dependencies beyond NumPy - CPU-only.
"""

import numpy as np

VON_KARMAN_CONSTANT = 0.4
BOUNDARY_LAYER_TOP_M = 2000.0


def log_law_wind_speed(altitude_m: float, reference_speed_m_s: float,
                       reference_altitude_m: float = 10.0, roughness_length_m: float = 0.03) -> float:
    """
    Mean wind speed via the logarithmic law: v(z) = v_ref * ln(z/z0) / ln(z_ref/z0),
    valid within the atmospheric boundary layer (up to ~BOUNDARY_LAYER_TOP_M).
    roughness_length_m: surface roughness (0.03 m ~ open flat terrain/launch pad).
    Above the boundary layer, wind is held at the boundary-layer-top value
    (real wind above the BL is driven by different, larger-scale dynamics
    this simplified model does not attempt).
    """
    z = np.clip(altitude_m, roughness_length_m * 1.001, BOUNDARY_LAYER_TOP_M)
    return float(reference_speed_m_s * np.log(z / roughness_length_m) /
                np.log(reference_altitude_m / roughness_length_m))


def jet_stream_bump(altitude_m: float, peak_altitude_m: float = 11000.0,
                    peak_speed_m_s: float = 45.0, width_m: float = 4000.0) -> float:
    """Gaussian bump approximating a mid-latitude jet-stream wind-speed peak near the tropopause."""
    return float(peak_speed_m_s * np.exp(-0.5 * ((altitude_m - peak_altitude_m) / width_m) ** 2))


def mean_wind_profile(altitude_m: float, surface_wind_speed_m_s: float = 8.0,
                      include_jet_stream: bool = True) -> float:
    """Total mean wind speed (m/s) at a given altitude: boundary-layer log law + jet-stream bump."""
    bl = log_law_wind_speed(min(altitude_m, BOUNDARY_LAYER_TOP_M), surface_wind_speed_m_s)
    jet = jet_stream_bump(altitude_m) if include_jet_stream else 0.0
    return bl + jet


class DrydenTurbulence:
    """
    Discrete-time single-channel Dryden turbulence filter (MIL-F-8785C
    form): white noise shaped by a first-order low-pass with the
    Dryden longitudinal turbulence time/length scale, producing
    a realistic gust velocity time series (not just uncorrelated noise
    -- real turbulence is spectrally colored, and an uncorrelated-noise
    gust model understates sustained-gust loading).
    """
    def __init__(self, turbulence_intensity_m_s: float = 3.0, length_scale_m: float = 533.0,
                mean_airspeed_m_s: float = 100.0, seed: int = None):
        self.sigma = turbulence_intensity_m_s
        self.L = length_scale_m
        self.V = max(mean_airspeed_m_s, 1.0)
        self.rng = np.random.default_rng(seed)
        self.state = 0.0

    def step(self, dt: float) -> float:
        """Advance the shaping filter one step; returns gust speed (m/s)."""
        tau = self.L / self.V
        alpha = dt / tau
        white = self.rng.normal(0.0, 1.0)
        # Discrete first-order shaping filter: x[k+1] = x[k]*(1-alpha) + sqrt(2*alpha)*sigma*w[k]
        self.state = self.state * (1.0 - alpha) + np.sqrt(max(2.0 * alpha, 0.0)) * self.sigma * white
        return float(self.state)


def wind_relative_velocity(vehicle_velocity_inertial: np.ndarray, altitude_m: float,
                           surface_wind_speed_m_s: float = 8.0, wind_heading_rad: float = 0.0,
                           turbulence: DrydenTurbulence = None, dt: float = 0.02,
                           include_jet_stream: bool = True) -> np.ndarray:
    """
    Air-relative velocity = vehicle velocity - wind velocity, for use in
    place of raw inertial velocity when computing drag/dynamic pressure
    (guidance/atmosphere.py's dynamic_pressure/density assume this is
    already air-relative; an earlier version of the ascent integrator
    used inertial velocity directly, i.e. implicitly assumed zero wind).
    """
    speed = mean_wind_profile(altitude_m, surface_wind_speed_m_s, include_jet_stream=include_jet_stream)
    if turbulence is not None:
        speed += turbulence.step(dt)
    wind_vec = speed * np.array([np.cos(wind_heading_rad), np.sin(wind_heading_rad), 0.0])
    return np.asarray(vehicle_velocity_inertial, dtype=float) - wind_vec
