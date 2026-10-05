"""
environment/aero_center.py

Center-of-pressure (Cp) vs center-of-gravity (Cg) shift over propellant
depletion, and the resulting static margin -- the single most important
number in vehicle stability analysis (Cg ahead of Cp = statically
stable weathercocking; Cg behind Cp = unstable, needs active control
authority to fly at all).

Model: a simple two-mass-lump vehicle (propellant mass lumped at its
tank centroid, dry structure mass lumped at its own centroid) gives an
analytic Cg(t) as propellant burns. Cp is modeled with the classic
Barrowman-style approximation: constant along the body for a slender
body of revolution, PLUS a nose-cone contribution, as a function of
Mach number via a simple subsonic-to-supersonic Cp shift (Cp moves aft
as Mach increases past ~1, a well-known and load-bearing effect for
any slender launcher -- ignoring it understates the margin change
through max-Q/transonic).

Zero external dependencies beyond NumPy - CPU-only.
"""

import numpy as np


def center_of_gravity_m(prop_mass_kg: float, dry_mass_kg: float,
                        prop_centroid_m: float, dry_centroid_m: float) -> float:
    """
    Cg measured from the nose tip (m), positive aft. Two-lumped-mass
    model: Cg = (m_prop*x_prop + m_dry*x_dry) / (m_prop + m_dry).
    """
    total = prop_mass_kg + dry_mass_kg
    if total <= 0:
        return dry_centroid_m
    return (prop_mass_kg * prop_centroid_m + dry_mass_kg * dry_centroid_m) / total


def center_of_pressure_m(body_length_m: float, body_diameter_m: float,
                         nose_length_m: float, mach: float) -> float:
    """
    Cp measured from the nose tip (m), Barrowman-style: for a slender
    body of revolution the body itself contributes negligible normal
    force compared to the nose/fins, so Cp sits close to the nose's own
    aerodynamic center (~0.466*nose_length for an ogive nose), then
    shifts AFT as Mach crosses ~1 (classic transonic Cp-aft-shift,
    modeled here as a smooth sigmoid blend to a more-aft supersonic
    value) -- this project has no fins modeled (see docstring
    limitation), so this is nose+body-only and will underestimate Cp's
    forward position for a finned vehicle.
    """
    cp_nose_subsonic = 0.466 * nose_length_m
    # Supersonic Cp for a slender cone-cylinder moves aft, asymptoting
    # toward roughly the body's volumetric centroid for M>>1.
    cp_supersonic = 0.5 * body_length_m
    blend = 1.0 / (1.0 + np.exp(-6.0 * (mach - 1.1)))   # sigmoid centered just past Mach 1
    return float(cp_nose_subsonic * (1.0 - blend) + cp_supersonic * blend)


def static_margin_calibers(cg_m: float, cp_m: float, body_diameter_m: float) -> float:
    """
    Static margin in calibers (body diameters) -- the standard rocketry
    stability metric. Positive = Cp aft of Cg = statically stable.
    Typical amateur/professional target: 1-2 calibers for passive
    aerodynamic stability; a TVC-controlled vehicle can fly with less
    (or even negative) margin since active control substitutes for
    passive weathercock stability, but the ACTIVE margin still needs
    to be known to size control authority.
    """
    return (cp_m - cg_m) / body_diameter_m


def stability_history(prop_mass_kg_series, dry_mass_kg: float, prop_centroid_m: float,
                      dry_centroid_m: float, body_length_m: float, body_diameter_m: float,
                      nose_length_m: float, mach_series):
    """
    Cg, Cp, and static margin over a burn, given time series of
    propellant mass remaining and Mach number (e.g. from
    guidance/gravity_turn.py's simulate_ascent output).
    """
    prop_mass_kg_series = np.asarray(prop_mass_kg_series, dtype=float)
    mach_series = np.asarray(mach_series, dtype=float)
    cg = np.array([center_of_gravity_m(m, dry_mass_kg, prop_centroid_m, dry_centroid_m)
                  for m in prop_mass_kg_series])
    cp = np.array([center_of_pressure_m(body_length_m, body_diameter_m, nose_length_m, mach)
                  for mach in mach_series])
    margin = (cp - cg) / body_diameter_m
    return {"cg_m": cg, "cp_m": cp, "static_margin_calibers": margin}
