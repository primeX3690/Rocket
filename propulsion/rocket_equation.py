"""
propulsion/rocket_equation.py

Core propulsion physics for launch-vehicle ascent performance analysis.
Zero external dependencies beyond NumPy — CPU-only, no GPU, no proprietary
libraries (no astropy/poliastro) — consistent with the rest of the stack.

Covers:
    - Tsiolkovsky ideal rocket equation (single burn)
    - Multi-stage delta-V budgeting (series staging)
    - Nozzle performance: thrust, specific impulse, exit velocity,
      pressure-thrust correction (sea-level vs vacuum Isp)
    - Mass-fraction sizing (propellant / structure / payload split)

Every function is independently verifiable against textbook closed-form
values (Sutton's "Rocket Propulsion Elements" conventions used throughout).
"""

import numpy as np

G0 = 9.80665  # standard gravity, m/s^2 (used to convert Isp <-> exit velocity)


# ---------------------------------------------------------------------------
# Single-stage Tsiolkovsky rocket equation
# ---------------------------------------------------------------------------

def exhaust_velocity(isp_s: float) -> float:
    """Effective exhaust velocity c = Isp * g0 (m/s)."""
    return isp_s * G0


def delta_v(isp_s: float, m0: float, mf: float) -> float:
    """
    Tsiolkovsky rocket equation.

    dV = Isp * g0 * ln(m0 / mf)

    m0: initial (wet) mass, kg
    mf: final (dry) mass after burn, kg
    """
    if mf <= 0 or m0 <= 0 or mf > m0:
        raise ValueError("Require 0 < mf <= m0")
    return exhaust_velocity(isp_s) * np.log(m0 / mf)


def mass_ratio_for_dv(isp_s: float, dv: float) -> float:
    """
    Inverse of the rocket equation — given a required delta-V, return the
    required mass ratio m0/mf.
    """
    return np.exp(dv / exhaust_velocity(isp_s))


def propellant_mass_for_dv(isp_s: float, dv: float, mf_dry: float) -> float:
    """
    Given required dV and the dry (post-burn) mass, return propellant mass
    needed: mp = mf_dry * (mass_ratio - 1)
    """
    r = mass_ratio_for_dv(isp_s, dv)
    return mf_dry * (r - 1.0)


# ---------------------------------------------------------------------------
# Nozzle performance
# ---------------------------------------------------------------------------

def thrust(mdot: float, ve: float, pe: float, pa: float, ae: float) -> float:
    """
    Full thrust equation including pressure-thrust term.

        F = mdot * ve + (pe - pa) * Ae

    mdot: propellant mass flow rate, kg/s
    ve:   effective exhaust velocity at nozzle exit, m/s
    pe:   nozzle exit static pressure, Pa
    pa:   ambient (atmospheric) pressure at altitude, Pa
    ae:   nozzle exit area, m^2
    """
    return mdot * ve + (pe - pa) * ae


def isp_from_thrust(f: float, mdot: float) -> float:
    """Specific impulse (s) recovered from thrust and mass flow rate."""
    return f / (mdot * G0)


def sea_level_to_vacuum_isp(isp_vac: float, pe: float, ae: float, mdot: float,
                             pa_sea_level: float = 101325.0) -> float:
    """
    Convert vacuum Isp to sea-level Isp given nozzle geometry, i.e. account
    for the pressure-thrust penalty an over-expanded nozzle pays at sea
    level. Returns sea-level Isp (s).
    """
    ve = isp_vac * G0
    f_vac = mdot * ve + pe * ae          # pa = 0 in vacuum
    f_sl = f_vac - pa_sea_level * ae      # subtract sea-level back-pressure loss
    return isp_from_thrust(f_sl, mdot)


def expansion_ratio(ae: float, at: float) -> float:
    """Nozzle area expansion ratio epsilon = Ae / At."""
    return ae / at


# ---------------------------------------------------------------------------
# Multi-stage delta-V budgeting
# ---------------------------------------------------------------------------

class Stage:
    """
    A single launch-vehicle stage.

    m_prop:      propellant mass, kg
    m_struct:    stage dry/structural mass (tanks, engine, avionics for
                 this stage), kg — everything that gets discarded at
                 staging except the propellant already burned
    isp_s:       stage specific impulse, s (use sea-level for stage 1
                 lower atmosphere portion, vacuum for upper stages —
                 caller's responsibility to pick the right value, or use
                 isp_effective() below for a blended estimate)
    """
    def __init__(self, name: str, m_prop: float, m_struct: float, isp_s: float):
        self.name = name
        self.m_prop = m_prop
        self.m_struct = m_struct
        self.isp_s = isp_s

    @property
    def m_total(self) -> float:
        return self.m_prop + self.m_struct

    @property
    def mass_fraction(self) -> float:
        """Propellant mass fraction of this stage alone (structural efficiency)."""
        return self.m_prop / self.m_total


def stage_dv_budget(stages: list, payload_mass: float) -> dict:
    """
    Compute delta-V contributed by each stage in series (bottom stage
    burns first, discarded, then next stage ignites), plus total mission
    delta-V.

    stages: list of Stage objects, ordered bottom (stage 1) to top (stage N)
    payload_mass: mass carried on top of the final stage, kg

    Returns a dict with per-stage dV, cumulative mass at each burn start/end,
    and total dV.
    """
    # Mass above the current stage at ignition = sum of all stages above it
    # (already-fired stages are dropped) + payload
    results = []
    n = len(stages)
    total_dv = 0.0

    for i, stage in enumerate(stages):
        # mass of everything still attached above this stage (not yet fired/dropped)
        mass_above = payload_mass + sum(s.m_total for s in stages[i + 1:])
        m0 = stage.m_total + mass_above           # wet mass at ignition of this stage
        mf = stage.m_struct + mass_above           # mass after this stage's propellant is spent

        dv_stage = delta_v(stage.isp_s, m0, mf)
        total_dv += dv_stage

        results.append({
            "stage": stage.name,
            "isp_s": stage.isp_s,
            "m0_kg": m0,
            "mf_kg": mf,
            "mass_ratio": m0 / mf,
            "mass_fraction": stage.mass_fraction,
            "dv_m_s": dv_stage,
        })

    return {
        "stages": results,
        "total_dv_m_s": total_dv,
        "payload_mass_kg": payload_mass,
    }


def required_leo_dv(target_altitude_km: float = 300.0,
                     losses_gravity_drag_m_s: float = 1500.0) -> float:
    """
    Rough total delta-V budget required to reach a circular LEO orbit,
    including standard gravity + drag + steering losses.

    Orbital velocity at target altitude (vis-viva, circular orbit) +
    a typical loss allowance (1300-1900 m/s is the standard textbook
    range for Earth launch; default 1500 m/s is a reasonable mid estimate).

    NOTE: This is a budgeting heuristic, not a substitute for full
    trajectory integration (see guidance/ module for that).
    """
    MU_EARTH = 3.986004418e14   # m^3/s^2
    R_EARTH = 6378137.0          # m

    r = R_EARTH + target_altitude_km * 1000.0
    v_orbit = np.sqrt(MU_EARTH / r)

    return v_orbit + losses_gravity_drag_m_s
