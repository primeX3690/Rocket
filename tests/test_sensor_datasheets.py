"""
tests/test_sensor_datasheets.py

Verifies navigation/sensor_datasheets.py: datasheet-derived noise
matches the printed spec numbers to within statistical sampling error,
the CEP->sigma conversion is correct, and the COCOM/ITAR GPS lock-loss
logic trips at exactly the documented thresholds (not before, not after).
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from navigation.sensor_datasheets import (
    IMU_DATASHEETS, GPS_DATASHEETS, DatasheetIMU, DatasheetGPS,
    gps_cep_to_sigma, find_cocom_loss_of_lock_time,
    COCOM_ALTITUDE_LIMIT_M, COCOM_VELOCITY_LIMIT_M_S,
)

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name}  {detail}")


def test_imu_noise_matches_datasheet_statistically():
    print("\n[IMU noise density -> sampled std dev matches spec]")
    for part in IMU_DATASHEETS:
        imu = DatasheetIMU(part, odr_hz=IMU_DATASHEETS[part].max_odr_hz,
                            seed=123, include_bias=False)
        n = 200_000
        accel_samples = np.array([imu.measure_accel(0.0) for _ in range(n)])
        gyro_samples = np.array([imu.measure_gyro(0.0) for _ in range(n)])

        measured_accel_std = np.std(accel_samples)
        measured_gyro_std = np.std(gyro_samples)

        # within 3% of the analytically expected std (large-n sanity check)
        check(f"{part} accel noise std matches analytic value",
              abs(measured_accel_std - imu.accel_noise_std) / imu.accel_noise_std < 0.03,
              f"measured={measured_accel_std:.6f} expected={imu.accel_noise_std:.6f}")
        check(f"{part} gyro noise std matches analytic value",
              abs(measured_gyro_std - imu.gyro_noise_std_rad_s) / imu.gyro_noise_std_rad_s < 0.03,
              f"measured={measured_gyro_std:.6f} expected={imu.gyro_noise_std_rad_s:.6f}")


def test_relative_imu_quality_ordering():
    # Sanity check against known real-world quality ordering. Note: BMI088's
    # advantage over MPU-6050 is bias-instability/vibration-robustness
    # (automotive-grade gyro die, <2 deg/hr), NOT raw noise density -- its
    # datasheet noise density is actually higher than MPU-6050's. Don't
    # assert the wrong axis of "better."
    print("\n[Relative sensor quality ordering matches known real-world ranking]")
    mpu = IMU_DATASHEETS["MPU-6050"]
    bmi = IMU_DATASHEETS["BMI088"]
    icm = IMU_DATASHEETS["ICM-20602"]
    check("BMI088 gyro bias tolerance is tighter than MPU-6050's (its real advantage)",
          bmi.gyro_bias_tolerance_dps < mpu.gyro_bias_tolerance_dps)
    check("ICM-20602 accel noise density is lowest of the three",
          icm.accel_noise_density_ug_sqrtHz < mpu.accel_noise_density_ug_sqrtHz
          and icm.accel_noise_density_ug_sqrtHz < bmi.accel_noise_density_ug_sqrtHz)


def test_cep_to_sigma_conversion():
    print("\n[CEP -> 1-sigma conversion]")
    # Known relation: CEP = 1.1774 * sigma  =>  sigma = CEP / 1.1774
    sigma = gps_cep_to_sigma(2.5)
    check("2.5 m CEP converts to ~2.123 m sigma",
          abs(sigma - 2.5 / 1.1774) < 1e-9, f"got {sigma}")


def test_gps_position_noise_matches_cep_statistically():
    print("\n[GPS position noise statistically matches CEP spec]")
    for part in GPS_DATASHEETS:
        gps = DatasheetGPS(part, seed=99)
        n = 50_000
        samples = np.array([
            gps.measure_position(0.0, altitude_m=1000.0, speed_m_s=100.0)[0]
            for _ in range(n)
        ])
        # Recover empirical CEP: median absolute value should sit near
        # the theoretical CEP for a zero-mean Gaussian (~0.6745*sigma
        # is the 1D median-absolute-deviation-equivalent; here we just
        # check the reconstructed sigma is close to the design sigma)
        empirical_sigma = np.std(samples)
        check(f"{part} empirical position sigma matches CEP-derived sigma",
              abs(empirical_sigma - gps.pos_sigma_m) / gps.pos_sigma_m < 0.03,
              f"measured={empirical_sigma:.4f} expected={gps.pos_sigma_m:.4f}")


def test_cocom_limit_requires_both_conditions():
    print("\n[COCOM/ITAR limit: must require BOTH altitude AND speed exceeded]")
    gps = DatasheetGPS("NEO-M8N")
    # High altitude alone -> should still have fix (e.g. commercial airliner)
    check("high altitude alone does not lose fix",
          gps.has_fix(altitude_m=20000.0, speed_m_s=250.0) is True)
    # High speed alone (low altitude) -> should still have fix
    check("high speed alone (low altitude) does not lose fix",
          gps.has_fix(altitude_m=5000.0, speed_m_s=600.0) is True)
    # Both exceeded -> lock lost
    check("both exceeded together loses fix",
          gps.has_fix(altitude_m=25000.0, speed_m_s=700.0) is False)
    # Exactly at the boundary (not exceeding) -> should still have fix
    check("exactly at threshold (not exceeding) keeps fix",
          gps.has_fix(altitude_m=COCOM_ALTITUDE_LIMIT_M,
                      speed_m_s=COCOM_VELOCITY_LIMIT_M_S) is True)


def test_cocom_loss_of_lock_finder_on_synthetic_ascent():
    print("\n[COCOM loss-of-lock detector on a synthetic ascent profile]")
    t = np.linspace(0, 120, 1200)
    # Synthetic ascent: altitude and speed both ramp up, crossing both
    # COCOM thresholds somewhere in the middle of the profile.
    altitude = 50.0 * t ** 1.5          # climbs past 18288 m around t ~ 96s
    speed = 8.0 * t                     # climbs past 515 m/s around t ~ 64s

    result = find_cocom_loss_of_lock_time(t, altitude, speed)
    check("loss-of-lock detected (both thresholds are crossed in this profile)",
          result is not None)
    if result is not None:
        check("reported crossing point actually exceeds both thresholds",
              result["altitude_m"] > COCOM_ALTITUDE_LIMIT_M
              and result["speed_m_s"] > COCOM_VELOCITY_LIMIT_M_S)
        # the index right before should NOT have both exceeded
        i = result["index"]
        if i > 0:
            check("the sample immediately before crossing had NOT tripped yet",
                  not (altitude[i - 1] > COCOM_ALTITUDE_LIMIT_M
                       and speed[i - 1] > COCOM_VELOCITY_LIMIT_M_S))

    # Profile that never exceeds both (e.g. a low, slow balloon) -> None
    low_alt = np.full_like(t, 3000.0)
    low_speed = np.full_like(t, 20.0)
    result_none = find_cocom_loss_of_lock_time(t, low_alt, low_speed)
    check("a flight that never exceeds both thresholds never loses lock",
          result_none is None)


def test_unknown_part_raises():
    print("\n[Unknown part number handling]")
    try:
        DatasheetIMU("FAKE-9999")
        check("unknown IMU part raises ValueError", False)
    except ValueError:
        check("unknown IMU part raises ValueError", True)
    try:
        DatasheetGPS("FAKE-9999")
        check("unknown GPS part raises ValueError", False)
    except ValueError:
        check("unknown GPS part raises ValueError", True)


def test_bridge_to_existing_imu_model():
    print("\n[Bridge into existing navigation.state_estimation.IMUModel]")
    imu = DatasheetIMU("BMI088", odr_hz=1000.0, seed=5)
    bridged = imu.as_state_estimation_imu()
    check("bridged IMUModel carries over the datasheet noise_std",
          abs(bridged.noise_std - imu.accel_noise_std) < 1e-12)
    check("bridged IMUModel carries over the datasheet bias",
          abs(bridged.bias - imu.accel_bias) < 1e-12)


if __name__ == "__main__":
    test_imu_noise_matches_datasheet_statistically()
    test_relative_imu_quality_ordering()
    test_cep_to_sigma_conversion()
    test_gps_position_noise_matches_cep_statistically()
    test_cocom_limit_requires_both_conditions()
    test_cocom_loss_of_lock_finder_on_synthetic_ascent()
    test_unknown_part_raises()
    test_bridge_to_existing_imu_model()

    print(f"\n{'='*60}\nRESULTS: {PASS} passed, {FAIL} failed\n{'='*60}")
    if FAIL > 0:
        sys.exit(1)
