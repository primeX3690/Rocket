"""
navigation/sensor_realism.py

Advanced IMU error models, additive to navigation/sensor_datasheets.py's
existing white-noise + constant-bias models: bias INSTABILITY (a
slowly-wandering bias, not a fixed constant), angular/velocity random
walk (ARW/VRW -- the integrated effect of high-frequency noise that
shows up as a growing uncertainty even with zero mean noise), constant
scale-factor error, and a fixed (but unknown-to-the-filter) sensor
misalignment matrix -- all standard terms from a real IMU datasheet's
Allan-variance specification, not just "add some gaussian noise".

Zero external dependencies beyond NumPy - CPU-only.
"""

import numpy as np


class BiasInstabilityModel:
    """
    A slowly-varying bias, modeled as a first-order Gauss-Markov
    process (the standard model for Allan-variance "bias instability" --
    NOT a random walk, which grows unboundedly; a Gauss-Markov process
    has a finite correlation time and settles into a bounded random
    walk-like wander, matching real IMU bias behavior).
    """
    def __init__(self, instability_magnitude: float, correlation_time_s: float = 100.0, seed: int = None):
        self.sigma = instability_magnitude
        self.tau = correlation_time_s
        self.rng = np.random.default_rng(seed)
        self.state = self.rng.normal(0.0, instability_magnitude)

    def step(self, dt: float) -> float:
        alpha = np.exp(-dt / self.tau)
        driving_noise_std = self.sigma * np.sqrt(1.0 - alpha ** 2)
        self.state = alpha * self.state + self.rng.normal(0.0, driving_noise_std)
        return self.state


class RandomWalkAccumulator:
    """
    Angular/velocity random walk: the STOCHASTIC ERROR that accumulates
    when integrating high-frequency white sensor noise over time (its
    standard deviation grows as sqrt(t), the classic random-walk
    signature) -- distinct from bias, which is what's left even with
    the noise turned off.
    """
    def __init__(self, random_walk_coefficient: float, seed: int = None):
        """random_walk_coefficient: e.g. deg/sqrt(hr) for ARW, converted by the caller to SI/sqrt(s)."""
        self.coefficient = random_walk_coefficient
        self.rng = np.random.default_rng(seed)
        self.accumulated = 0.0

    def step(self, dt: float) -> float:
        self.accumulated += self.rng.normal(0.0, self.coefficient * np.sqrt(dt))
        return self.accumulated

    def expected_std_after(self, t_s: float) -> float:
        """Theoretical 1-sigma random-walk growth after time t (for validating the simulation)."""
        return self.coefficient * np.sqrt(t_s)


def scale_factor_error(true_value: np.ndarray, scale_factor_ppm: float) -> np.ndarray:
    """Constant multiplicative scale-factor error (parts-per-million), a real, common IMU spec."""
    return np.asarray(true_value, dtype=float) * (1.0 + scale_factor_ppm * 1e-6)


def misalignment_matrix(x_tilt_rad: float, y_tilt_rad: float, z_tilt_rad: float) -> np.ndarray:
    """
    Small-angle sensor-to-body misalignment matrix (the IMU's own axes
    are never PERFECTLY aligned to the vehicle body frame -- a real,
    fixed (but a priori unknown to the filter) manufacturing/mounting
    error, modeled here as a small-angle rotation composed from three
    axis-tilt terms).
    """
    return np.array([
        [1.0, -z_tilt_rad, y_tilt_rad],
        [z_tilt_rad, 1.0, -x_tilt_rad],
        [-y_tilt_rad, x_tilt_rad, 1.0],
    ])


def apply_full_imu_error_model(true_vector: np.ndarray, bias_instability: BiasInstabilityModel,
                               random_walk: RandomWalkAccumulator, scale_factor_ppm: float,
                               misalignment: np.ndarray, white_noise_std: float, dt: float,
                               rng: np.random.Generator = None) -> np.ndarray:
    """
    Compose all error sources into one realistic measurement: true value
    -> misalignment -> scale factor -> + bias instability -> + random-
    walk increment -> + white noise. Order matches how these physically
    compound on a real sensor (misalignment/scale-factor are
    multiplicative on the true signal; bias/noise are additive at the
    output). `random_walk.step(dt)` returns the ACCUMULATED random-walk
    value (a slowly growing offset, not a per-step increment), matching
    how ARW/VRW actually shows up in a navigation solution.
    """
    rng = rng or np.random.default_rng()
    true_vector = np.asarray(true_vector, dtype=float)
    measured = misalignment @ true_vector
    measured = scale_factor_error(measured, scale_factor_ppm)
    bias = bias_instability.step(dt)
    rw = random_walk.step(dt)
    noise = rng.normal(0.0, white_noise_std, size=measured.shape)
    return measured + bias + rw + noise
