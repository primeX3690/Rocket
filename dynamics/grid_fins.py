"""
dynamics/grid_fins.py

Grid-fin aerodynamic control-surface model for the atmospheric-descent
phase, when a returning stage may have little or no active thrust (a
coasting entry/descent phase) and needs aerodynamic control authority
instead — real reusable first stages (Falcon 9) use grid fins for
exactly this: high-drag lattice control surfaces effective from
supersonic through subsonic speed, deployed specifically during the
unpowered portion of descent.

Model: a standard thin-lattice-plate lift/drag model — linear lift-
curve slope at low angle of attack (a real, textbook aerodynamic
approximation for a high-solidity lattice fin, similar in form to a
flat-plate/lattice-wing model), a stall cutoff at high deflection, and
a Mach-dependent effectiveness factor (grid fins lose lift-curve slope
significantly above Mach ~2, and lattice/grid aerodynamic surfaces are
well known in the literature to have reduced supersonic effectiveness
compared to subsonic) -- not a full CFD/wind-tunnel-derived surface,
which is out of scope for this project.

Zero external dependencies beyond NumPy - CPU-only.
"""

import numpy as np


# Effectiveness factor vs Mach: lattice/grid fins are most effective transonic/low-supersonic,
# and lose effectiveness at high supersonic/hypersonic speed (well-documented in the grid-fin literature).
_MACH_TABLE = np.array([0.0, 0.8, 1.2, 2.0, 3.0, 5.0])
_EFFECTIVENESS = np.array([0.6, 1.0, 1.0, 0.75, 0.45, 0.25])


def grid_fin_effectiveness(mach: float) -> float:
    return float(np.interp(mach, _MACH_TABLE, _EFFECTIVENESS))


def grid_fin_lift_coefficient(deflection_rad: float, mach: float,
                              lift_slope_per_rad: float = 2.5, stall_deflection_rad: float = np.radians(20)) -> float:
    """
    Linear lift coefficient up to `stall_deflection_rad`, then a smooth
    falloff (real high-solidity lattice fins stall similarly to a flat
    plate at large deflection — lift stops increasing and drag
    dominates).
    """
    eff = grid_fin_effectiveness(mach)
    d = np.clip(abs(deflection_rad), 0.0, np.pi / 2)
    if d <= stall_deflection_rad:
        cl = lift_slope_per_rad * d
    else:
        cl_stall = lift_slope_per_rad * stall_deflection_rad
        overshoot = d - stall_deflection_rad
        cl = cl_stall * np.exp(-2.0 * overshoot / stall_deflection_rad)   # post-stall falloff
    return float(np.sign(deflection_rad) * cl * eff)


def grid_fin_drag_coefficient(deflection_rad: float, mach: float, cd0: float = 0.8) -> float:
    """
    Grid fins have high intrinsic drag even at zero deflection (the
    lattice structure itself is draggy — a real, well-known
    characteristic and part of why they're only deployed during descent,
    not carried open during ascent), plus induced drag growing with
    deflection angle squared (a standard induced-drag form).
    """
    eff = grid_fin_effectiveness(mach)
    d = np.clip(abs(deflection_rad), 0.0, np.pi / 2)
    return float(cd0 * eff + 1.5 * d ** 2)


def grid_fin_forces_n(deflection_rad: float, mach: float, dynamic_pressure_pa: float,
                      fin_area_m2: float = 0.6, n_fins: int = 4) -> dict:
    """Total lift/drag force (N) from n_fins grid fins at a given deflection, Mach, and dynamic pressure."""
    cl = grid_fin_lift_coefficient(deflection_rad, mach)
    cd = grid_fin_drag_coefficient(deflection_rad, mach)
    lift_n = cl * dynamic_pressure_pa * fin_area_m2 * n_fins
    drag_n = cd * dynamic_pressure_pa * fin_area_m2 * n_fins
    return {"lift_n": lift_n, "drag_n": drag_n, "cl": cl, "cd": cd}


def grid_fin_control_torque_nm(deflection_rad: float, mach: float, dynamic_pressure_pa: float,
                               moment_arm_m: float, fin_area_m2: float = 0.6, n_fins: int = 4) -> float:
    """Pitch/yaw control torque about the vehicle Cg from differential grid-fin lift at the given deflection."""
    forces = grid_fin_forces_n(deflection_rad, mach, dynamic_pressure_pa, fin_area_m2, n_fins)
    return forces["lift_n"] * moment_arm_m
