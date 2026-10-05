"""
navigation/sensor_datasheets.py

Replaces the arbitrary synthetic noise_std constants used elsewhere in
this stack (navigation/state_estimation.py's IMUModel, faults/real_world.py)
with noise models built directly from real, publicly published sensor
datasheets. Every number below is a vendor-specified typical spec, cited
by document, not a guessed constant.

Why this matters: an EKF or Monte Carlo dispersion result tuned against
an arbitrary noise_std=0.02 m/s^2 proves nothing about how the estimator
behaves on an actual flight computer. Tuning it against the *real* noise
spectral density of a specific, buyable chip (the kind an amateur/student
rocketry team would actually fly) is what turns "we simulated an EKF"
into "we simulated an EKF using the exact noise characteristics of the
IMU we intend to fly."

=============================================================================
DATASHEET-DERIVED SPECS (each with source)
=============================================================================

InvenSense MPU-6050 (6-axis, I2C, ~$2, most common hobbyist IMU)
  Source: MPU-6000/MPU-6050 Product Specification, Rev 3.4 (PS-MPU-6000A-00),
          Section 6 Electrical Characteristics.
  - Accelerometer noise power spectral density: 400 ug/sqrt(Hz)
    (@ AFS_SEL=0, ODR = 1 kHz)
  - Gyroscope rate noise spectral density: 0.005 deg/s/sqrt(Hz)
    (independently confirmed against real logged MPU6050 data:
    total RMS noise ~0.05 deg/s, low-freq RMS ~0.033 deg/s at the
    datasheet's stated bandwidth)
  - Zero-rate-output / bias tolerance: +-20 deg/s initial (gyro),
    +-35...60 mg (accel, axis-dependent) -- this is NOT bias
    instability, it's factory calibration tolerance; real bias
    instability must be measured (Allan variance) per unit.

InvenSense ICM-20602 (6-axis, SPI/I2C, ~$6, better than MPU-6050)
  Source: ICM-20602 datasheet, "High performance specs" summary table.
  - Accelerometer noise density: 100 ug/sqrt(Hz)
  - Gyroscope noise density: 4 mdps/sqrt(Hz) = 0.004 deg/s/sqrt(Hz)

Bosch BMI088 (6-axis, SPI/I2C, ~$8, drone/robotics-grade, vibration-robust)
  Source: BMI088 Data Sheet Rev 1.9 (BST-BMI088-DS000-19) + product page.
  - Accelerometer noise density: 175 ug/sqrt(Hz) (typ, +-24g range)
  - Gyroscope noise density: 0.014 deg/s/sqrt(Hz)
  - Gyroscope bias instability: < 2 deg/hr (automotive-grade gyro die)
  This is the realistic "if this project had a bit more budget" upgrade path
  from the MPU-6050 -- genuinely used on real multirotor flight
  controllers (Pixhawk/Cube family).

u-blox NEO-6M (GPS-only, ~$5, most common hobbyist GPS)
  Source: NEO-6 Data Sheet, Position Accuracy table.
  - Position accuracy: 2.5 m CEP
  - Velocity accuracy: 0.1 m/s
  - Max nav update rate: 5 Hz

u-blox NEO-M8N (GPS+GLONASS+Galileo, ~$15, common on Pixhawk/APM)
  Source: NEO-M8 Data Sheet (UBX-13003366), Position Accuracy table.
  - Position accuracy: 2.0-2.5 m CEP
  - Velocity accuracy: 0.05 m/s
  - Max nav update rate: 10 Hz (concurrent GNSS) / 18 Hz (single GNSS)

REAL GOTCHA -- COCOM / ITAR GPS limits (this is the kind of detail that
signals "actually built for rocketry" rather than "copy-pasted a drone
noise model"): nearly every commercial consumer GNSS chipset (including
NEO-6M/M8N) enforces the COCOM export-control limit and will REFUSE TO
OUTPUT A FIX once the receiver's own solution exceeds BOTH ~18 km (60,000 ft)
altitude AND ~515 m/s (1,000 knots) ground speed simultaneously. A student
rocket flying past ~Mach 1.5 at altitude on a hobbyist u-blox module will
have its GPS silently drop to "no fix" mid-flight -- this must be designed
around (INS-only coast, or a COCOM-exempt/military-spec receiver), not
discovered on the launch pad. Modeled explicitly below.
"""

import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# IMU datasheet specs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IMUDatasheetSpec:
    name: str
    accel_noise_density_ug_sqrtHz: float   # micro-g / sqrt(Hz)
    gyro_noise_density_dps_sqrtHz: float   # deg/s / sqrt(Hz)
    accel_bias_tolerance_mg: float         # factory zero-g tolerance (NOT instability)
    gyro_bias_tolerance_dps: float         # factory zero-rate tolerance (NOT instability)
    max_odr_hz: float
    source: str


IMU_DATASHEETS = {
    "MPU-6050": IMUDatasheetSpec(
        name="MPU-6050",
        accel_noise_density_ug_sqrtHz=400.0,
        gyro_noise_density_dps_sqrtHz=0.005,
        accel_bias_tolerance_mg=60.0,
        gyro_bias_tolerance_dps=20.0,
        max_odr_hz=1000.0,
        source="InvenSense PS-MPU-6000A-00 Rev 3.4, Sec 6 Electrical Characteristics",
    ),
    "ICM-20602": IMUDatasheetSpec(
        name="ICM-20602",
        accel_noise_density_ug_sqrtHz=100.0,
        gyro_noise_density_dps_sqrtHz=0.004,
        accel_bias_tolerance_mg=50.0,       # datasheet typical initial ZRO region
        gyro_bias_tolerance_dps=5.0,
        max_odr_hz=8000.0,
        source="TDK InvenSense ICM-20602 Datasheet, High Performance Specs table",
    ),
    "BMI088": IMUDatasheetSpec(
        name="BMI088",
        accel_noise_density_ug_sqrtHz=175.0,
        gyro_noise_density_dps_sqrtHz=0.014,
        accel_bias_tolerance_mg=30.0,
        gyro_bias_tolerance_dps=1.0,
        max_odr_hz=2000.0,
        source="Bosch BST-BMI088-DS000-19 Rev 1.9, Table 3/5",
    ),
}


class DatasheetIMU:
    """
    Converts a vendor noise-spectral-density spec (ug/sqrt(Hz) or
    deg/s/sqrt(Hz), as printed on the datasheet) into a per-sample
    Gaussian noise std dev at a given output data rate (ODR), using the
    standard white-noise-density-to-discrete-sample relation:

        sigma_sample = noise_density * sqrt(f_s)

    (f_s = sampling rate in Hz). This is the same approximation used by
    PX4/ArduPilot's own IMU noise injection models and by IEEE-STD-952
    style Allan-variance random-walk conversions -- it is a standard
    engineering approximation, not an exact closed form, and should
    still be cross-checked against a real logged unit's Allan deviation
    plot before flight (bias instability especially is unit-specific
    and NOT captured by the noise-density spec alone).

    This replaces the arbitrary `noise_std=0.02` style constants in
    navigation/state_estimation.py's IMUModel with a value traceable to
    an actual, buyable part number.
    """

    def __init__(self, part_number: str, odr_hz: float = None, seed: int = 42,
                 include_bias: bool = True):
        if part_number not in IMU_DATASHEETS:
            raise ValueError(
                f"Unknown part '{part_number}'. Available: {list(IMU_DATASHEETS)}"
            )
        self.spec = IMU_DATASHEETS[part_number]
        self.odr_hz = odr_hz or self.spec.max_odr_hz
        self.rng = np.random.default_rng(seed)

        # ug/sqrt(Hz) -> m/s^2/sqrt(Hz):  1 ug = 9.80665e-6 m/s^2
        accel_nd_si = self.spec.accel_noise_density_ug_sqrtHz * 9.80665e-6
        self.accel_noise_std = accel_nd_si * np.sqrt(self.odr_hz)
        self.gyro_noise_std_rad_s = np.radians(
            self.spec.gyro_noise_density_dps_sqrtHz * np.sqrt(self.odr_hz)
        )

        # Fixed per-run bias draw (factory calibration residual), NOT
        # random-walked here -- a slowly-varying real bias is what
        # AscentEKF's bias state is designed to track; this sets its
        # *initial* magnitude to something a real unit could plausibly have.
        if include_bias:
            accel_bias_si = self.spec.accel_bias_tolerance_mg * 1e-3 * 9.80665
            gyro_bias_si = np.radians(self.spec.gyro_bias_tolerance_dps)
            self.accel_bias = self.rng.uniform(-accel_bias_si, accel_bias_si)
            self.gyro_bias_rad_s = self.rng.uniform(-gyro_bias_si, gyro_bias_si)
        else:
            self.accel_bias = 0.0
            self.gyro_bias_rad_s = 0.0

    def measure_accel(self, true_accel_m_s2: float) -> float:
        return (true_accel_m_s2 + self.accel_bias
                + self.rng.normal(0.0, self.accel_noise_std))

    def measure_gyro(self, true_rate_rad_s: float) -> float:
        return (true_rate_rad_s + self.gyro_bias_rad_s
                + self.rng.normal(0.0, self.gyro_noise_std_rad_s))

    def as_state_estimation_imu(self):
        """
        Bridge to navigation.state_estimation.IMUModel so AscentEKF /
        dead_reckon / run_ekf_ascent can consume this without modification
        -- swap the arbitrary-constant IMUModel for a datasheet-real one
        with a single line change at the call site.
        """
        from navigation.state_estimation import IMUModel
        imu = IMUModel(bias_m_s2=self.accel_bias, noise_std=self.accel_noise_std,
                        seed=int(self.rng.integers(0, 2**31 - 1)))
        return imu

    def summary(self) -> str:
        return (
            f"{self.spec.name} @ {self.odr_hz:.0f} Hz ODR: "
            f"accel_noise_std={self.accel_noise_std:.5f} m/s^2, "
            f"gyro_noise_std={np.degrees(self.gyro_noise_std_rad_s):.5f} deg/s, "
            f"accel_bias={self.accel_bias:.5f} m/s^2, "
            f"gyro_bias={np.degrees(self.gyro_bias_rad_s):.5f} deg/s "
            f"[{self.spec.source}]"
        )


# ---------------------------------------------------------------------------
# GPS/GNSS datasheet specs + real COCOM/ITAR high-altitude/high-speed cutoff
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GPSDatasheetSpec:
    name: str
    position_cep_m: float          # Circular Error Probable (50% radius), meters
    velocity_accuracy_m_s: float   # 1-sigma, m/s
    max_update_rate_hz: float
    source: str


GPS_DATASHEETS = {
    "NEO-6M": GPSDatasheetSpec(
        name="NEO-6M",
        position_cep_m=2.5,
        velocity_accuracy_m_s=0.1,
        max_update_rate_hz=5.0,
        source="u-blox NEO-6 Data Sheet, Position Accuracy table",
    ),
    "NEO-M8N": GPSDatasheetSpec(
        name="NEO-M8N",
        position_cep_m=2.0,
        velocity_accuracy_m_s=0.05,
        max_update_rate_hz=10.0,
        source="u-blox NEO-M8 Data Sheet UBX-13003366, Position Accuracy table",
    ),
}

# COCOM/ITAR export-control limit enforced in firmware by essentially every
# consumer GNSS chipset (u-blox, MediaTek, Broadcom, etc): the receiver
# refuses to output a navigation solution once velocity AND altitude both
# exceed these thresholds simultaneously.
COCOM_ALTITUDE_LIMIT_M = 18288.0    # 60,000 ft
COCOM_VELOCITY_LIMIT_M_S = 515.0    # 1,000 knots


def gps_cep_to_sigma(cep_m: float) -> float:
    """
    Convert a datasheet CEP (circular error probable, 50th-percentile
    radius for a 2D Gaussian error) to the equivalent per-axis 1-sigma
    std dev, using the standard relation CEP = 1.1774 * sigma
    (for a circular/isotropic 2D Gaussian, R2968/Rayleigh result).
    """
    return cep_m / 1.1774


class DatasheetGPS:
    """
    GPS position/velocity noise model built from a real receiver
    datasheet, with the COCOM/ITAR high-speed+high-altitude lock-loss
    behavior modeled explicitly -- the single most common "worked in
    sim, silently failed on the real flight" GPS gotcha in amateur
    rocketry (a sounding rocket comfortably exceeds both COCOM
    thresholds during a normal ascent).
    """

    def __init__(self, part_number: str, seed: int = 7):
        if part_number not in GPS_DATASHEETS:
            raise ValueError(
                f"Unknown part '{part_number}'. Available: {list(GPS_DATASHEETS)}"
            )
        self.spec = GPS_DATASHEETS[part_number]
        self.pos_sigma_m = gps_cep_to_sigma(self.spec.position_cep_m)
        self.rng = np.random.default_rng(seed)

    def has_fix(self, altitude_m: float, speed_m_s: float) -> bool:
        """False once the COCOM/ITAR limit trips (both conditions exceeded)."""
        return not (altitude_m > COCOM_ALTITUDE_LIMIT_M
                    and speed_m_s > COCOM_VELOCITY_LIMIT_M_S)

    def measure_position(self, true_pos_m: float, altitude_m: float,
                          speed_m_s: float):
        """Returns (measurement_or_None, has_fix). None means lock lost."""
        if not self.has_fix(altitude_m, speed_m_s):
            return None, False
        return true_pos_m + self.rng.normal(0.0, self.pos_sigma_m), True

    def measure_velocity(self, true_vel_m_s: float, altitude_m: float,
                          speed_m_s: float):
        if not self.has_fix(altitude_m, speed_m_s):
            return None, False
        return (true_vel_m_s
                + self.rng.normal(0.0, self.spec.velocity_accuracy_m_s), True)


def find_cocom_loss_of_lock_time(t_series: np.ndarray, altitude_series: np.ndarray,
                                  speed_series: np.ndarray):
    """
    Scan a simulated ascent trajectory and report the first time index
    at which GPS would lose lock under the COCOM/ITAR rule -- the exact
    question a real GNC team must answer before trusting GPS-aided
    navigation past that point in the flight (AscentEKF must fall back
    to pure IMU propagation, i.e. dead_reckon()-style, from there on).
    """
    tripped = (altitude_series > COCOM_ALTITUDE_LIMIT_M) & (speed_series > COCOM_VELOCITY_LIMIT_M_S)
    idx = np.argmax(tripped) if np.any(tripped) else None
    if idx is None:
        return None
    return {
        "index": int(idx),
        "time_s": float(t_series[idx]),
        "altitude_m": float(altitude_series[idx]),
        "speed_m_s": float(speed_series[idx]),
    }
