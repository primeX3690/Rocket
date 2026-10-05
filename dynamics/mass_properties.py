"""
dynamics/mass_properties.py

Time-varying mass-properties model: the full moment-of-inertia tensor
(including off-diagonal cross-products of inertia) as propellant
depletes, for use with dynamics/six_dof.py's InertiaTensor (which was
previously a single constant snapshot the caller had to supply by hand
-- this module derives it from a simple mass-distribution model instead).

Model: the vehicle is treated as a stack of axisymmetric lumped masses
(propellant tank(s), dry structure) along the body's longitudinal (x)
axis, each modeled as a thin disk/cylinder for its own local inertia,
combined via the parallel-axis theorem about the vehicle's current Cg.
A single tank-offset parameter (`tank_lateral_offset_m`) lets the
propellant centroid sit off-axis (e.g. from an asymmetric internal
baffle or a partially-drained off-center sump), which is what produces
non-zero cross terms I_xy/I_xz -- with zero offset, the model correctly
returns a diagonal tensor (an axisymmetric mass distribution truly has
no cross-coupling, so this is not an artificial floor).

Zero external dependencies beyond NumPy - CPU-only.
"""

import numpy as np
from dynamics.six_dof import InertiaTensor


def cylinder_inertia_local(mass_kg: float, radius_m: float, length_m: float):
    """Local (about own centroid) inertia of a uniform solid cylinder, axis along body-x."""
    ixx = 0.5 * mass_kg * radius_m ** 2
    iyy = izz = (1.0 / 12.0) * mass_kg * (3 * radius_m ** 2 + length_m ** 2)
    return np.diag([ixx, iyy, izz])


def parallel_axis_shift(inertia_local: np.ndarray, mass_kg: float, offset_m: np.ndarray) -> np.ndarray:
    """Parallel-axis theorem, full tensor form: I_shifted = I_local + m*(|d|^2*Identity - d⊗d)."""
    d = np.asarray(offset_m, dtype=float)
    return inertia_local + mass_kg * (np.dot(d, d) * np.eye(3) - np.outer(d, d))


def full_inertia_tensor(
    prop_mass_kg: float, prop_radius_m: float, prop_length_m: float, prop_centroid_x_m: float,
    dry_mass_kg: float, dry_radius_m: float, dry_length_m: float, dry_centroid_x_m: float,
    tank_lateral_offset_m: float = 0.0,
):
    """
    Full 3x3 inertia tensor about the vehicle's current Cg, from a
    2-lump (propellant + dry structure) axisymmetric-cylinder model.
    Returns (InertiaTensor for six_dof.py's diagonal-only API, full 3x3
    matrix including cross terms, and the Cg used).
    """
    total_mass = prop_mass_kg + dry_mass_kg
    cg_x = (prop_mass_kg * prop_centroid_x_m + dry_mass_kg * dry_centroid_x_m) / max(total_mass, 1e-9)

    prop_local = cylinder_inertia_local(prop_mass_kg, prop_radius_m, prop_length_m)
    dry_local = cylinder_inertia_local(dry_mass_kg, dry_radius_m, dry_length_m)

    prop_offset = np.array([prop_centroid_x_m - cg_x, tank_lateral_offset_m, 0.0])
    dry_offset = np.array([dry_centroid_x_m - cg_x, 0.0, 0.0])

    I_full = (parallel_axis_shift(prop_local, prop_mass_kg, prop_offset) +
             parallel_axis_shift(dry_local, dry_mass_kg, dry_offset))

    return InertiaTensor(I_full[0, 0], I_full[1, 1], I_full[2, 2]), I_full, cg_x


def inertia_history(prop_mass_kg_series, prop_radius_m: float, prop_length_m: float, prop_centroid_x_m: float,
                    dry_mass_kg: float, dry_radius_m: float, dry_length_m: float, dry_centroid_x_m: float,
                    tank_lateral_offset_m: float = 0.0):
    """Frame-by-frame I_xx/I_yy/I_zz and Cg-x over a burn, from a propellant-mass time series."""
    prop_mass_kg_series = np.asarray(prop_mass_kg_series, dtype=float)
    ixx = np.zeros_like(prop_mass_kg_series)
    iyy = np.zeros_like(prop_mass_kg_series)
    izz = np.zeros_like(prop_mass_kg_series)
    ixy = np.zeros_like(prop_mass_kg_series)
    cg = np.zeros_like(prop_mass_kg_series)
    for i, m_prop in enumerate(prop_mass_kg_series):
        _, I_full, cg_x = full_inertia_tensor(
            m_prop, prop_radius_m, prop_length_m, prop_centroid_x_m,
            dry_mass_kg, dry_radius_m, dry_length_m, dry_centroid_x_m, tank_lateral_offset_m,
        )
        ixx[i], iyy[i], izz[i], ixy[i], cg[i] = I_full[0, 0], I_full[1, 1], I_full[2, 2], I_full[0, 1], cg_x
    return {"ixx": ixx, "iyy": iyy, "izz": izz, "ixy_cross_term": ixy, "cg_x_m": cg}
