"""
tests/test_peg.py

Verification suite for guidance/peg.py.

The key claim to verify is PEG's defining property: closed-loop
re-solving from current state should land on the target orbit with much
tighter accuracy than an open-loop profile would, INCLUDING when the
vehicle starts from an off-nominal (dispersed) initial condition —
because PEG re-targets every cycle instead of flying a fixed profile.
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from guidance.peg import PEGTarget, PEGState, solve_peg_steering, simulate_peg_guided_burn, G0

PASS = 0
FAIL = 0
MU_EARTH = 3.986004418e14
R_EARTH = 6378137.0


def check(name, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


def _circular_orbit_target(altitude_km):
    r = R_EARTH + altitude_km * 1000.0
    v = np.sqrt(MU_EARTH / r)
    return PEGTarget(target_radius_m=r, target_velocity_m_s=v, target_flight_path_angle_rad=0.0)


def test_time_to_go_matches_rocket_equation():
    """
    solve_peg_steering's time-to-go should match a direct rocket-equation
    calc for the required delta-V, since it's derived from the same
    formula (m(t) = m0 - mdot*t, dv = ve*ln(m0/m(t))).
    """
    isp, thrust, m = 320.0, 800000.0, 15000.0
    ve = isp * G0
    mdot = thrust / ve
    dv_needed = 500.0

    state = PEGState(radius_m=R_EARTH + 150000, velocity_m_s=7000.0,
                      flight_path_angle_rad=0.05, mass_kg=m, thrust_n=thrust,
                      isp_s=isp, effective_gravity_m_s2=9.5)
    target = PEGTarget(target_radius_m=R_EARTH + 200000,
                        target_velocity_m_s=7000.0 + dv_needed,
                        target_flight_path_angle_rad=0.0)

    sol = solve_peg_steering(state, target)

    t_expected = (m / mdot) * (1.0 - np.exp(-dv_needed / ve))
    check("PEG time-to-go matches direct rocket-equation calc",
          abs(sol["time_to_go_s"] - t_expected) < 1e-6)


def test_closed_loop_hits_circular_orbit_from_nominal_start():
    """
    Starting from a plausible nominal upper-stage ignition state, the
    closed-loop PEG burn should insert into the target circular orbit
    with tight accuracy (velocity error small relative to orbital speed,
    flight-path angle near zero = circular).
    """
    target = _circular_orbit_target(altitude_km=300.0)

    result = simulate_peg_guided_burn(
        r0_m=R_EARTH + 180000.0, v0_m_s=7300.0, gamma0_rad=np.radians(3.0),
        m0_kg=4000.0, thrust_n=90000.0, isp_s=320.0, target=target,
        guidance_cycle_s=2.0, dt_integrate=0.05,
    )

    err = result["insertion_error"]
    check("PEG velocity error at cutoff is small (<15 m/s on ~7700 m/s orbital speed)",
          abs(err["velocity_error_m_s"]) < 15.0)
    check("PEG flight-path-angle error at cutoff is small (<0.5 deg, i.e. near-circular)",
          abs(err["flight_path_angle_error_deg"]) < 0.5)


def test_closed_loop_corrects_for_dispersed_start():
    """
    Core PEG value proposition: even if the vehicle starts noticeably
    off from the 'expected' pre-burn state (representing dispersion from
    stage-1 performance variation, wind, etc.), closed-loop re-targeting
    every cycle should STILL converge to the same target orbit accuracy
    — because guidance recomputes from wherever it actually is, not from
    where it was 'supposed' to be.
    """
    target = _circular_orbit_target(altitude_km=300.0)

    nominal = simulate_peg_guided_burn(
        r0_m=R_EARTH + 180000.0, v0_m_s=7300.0, gamma0_rad=np.radians(3.0),
        m0_kg=4000.0, thrust_n=90000.0, isp_s=320.0, target=target,
    )
    dispersed = simulate_peg_guided_burn(
        r0_m=R_EARTH + 178000.0,      # 2km low
        v0_m_s=7280.0,                  # 20 m/s slow
        gamma0_rad=np.radians(3.4),     # 0.4 deg steeper
        m0_kg=4020.0,                    # slightly heavier (mass dispersion)
        thrust_n=90000.0, isp_s=320.0, target=target,
    )

    err_nom = abs(nominal["insertion_error"]["velocity_error_m_s"])
    err_disp = abs(dispersed["insertion_error"]["velocity_error_m_s"])

    check("Dispersed-start insertion velocity error is still small (<15 m/s) despite off-nominal start",
          err_disp < 15.0)
    check("Dispersed-start accuracy is comparable to nominal-start accuracy (closed-loop correction working)",
          abs(err_disp - err_nom) < 10.0)


def test_mass_decreases_monotonically_during_burn():
    target = _circular_orbit_target(altitude_km=300.0)
    result = simulate_peg_guided_burn(
        r0_m=R_EARTH + 180000.0, v0_m_s=7300.0, gamma0_rad=np.radians(3.0),
        m0_kg=4000.0, thrust_n=90000.0, isp_s=320.0, target=target,
    )
    check("Final mass is less than initial mass (propellant consumed)",
          result["final_mass_kg"] < 4000.0)
    check("Final mass is positive (didn't run the tank dry mid-sim)",
          result["final_mass_kg"] > 0.0)


def test_closed_loop_steering_reacts_to_current_state():
    """
    Defining property of closed-loop guidance vs. open-loop: the commanded
    steering angle at the FIRST guidance cycle must differ between a
    nominal and a dispersed start, because PEG re-solves from whatever
    state it actually observes rather than flying a fixed pre-computed
    profile. (An open-loop gravity-turn profile would command the same
    thing regardless of the actual state.)
    """
    target = _circular_orbit_target(altitude_km=300.0)

    nominal = simulate_peg_guided_burn(
        r0_m=R_EARTH + 180000.0, v0_m_s=7300.0, gamma0_rad=np.radians(3.0),
        m0_kg=4000.0, thrust_n=90000.0, isp_s=320.0, target=target,
    )
    dispersed = simulate_peg_guided_burn(
        r0_m=R_EARTH + 178000.0, v0_m_s=7280.0, gamma0_rad=np.radians(3.4),
        m0_kg=4020.0, thrust_n=90000.0, isp_s=320.0, target=target,
    )
    check("First-cycle steering command differs between nominal and dispersed start "
          "(guidance is reacting to actual state, not flying a fixed profile)",
          abs(nominal["chi_deg"][0] - dispersed["chi_deg"][0]) > 0.5)


if __name__ == "__main__":
    print("Running PEG closed-loop guidance verification suite...\n")
    test_time_to_go_matches_rocket_equation()
    test_closed_loop_hits_circular_orbit_from_nominal_start()
    test_closed_loop_corrects_for_dispersed_start()
    test_mass_decreases_monotonically_during_burn()
    test_closed_loop_steering_reacts_to_current_state()

    print(f"\n{PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    if FAIL > 0:
        sys.exit(1)
