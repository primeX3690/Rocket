"""
tests/test_six_dof.py

Verification suite for dynamics/six_dof.py.

Strategy: quaternion math is checked against known identities; rigid-
body ROTATIONAL dynamics is checked against a classical analytic special
case of Euler's equations (torque-free spin about a principal axis is
exactly conserved); and the 6DOF TRANSLATIONAL dynamics is checked for
consistency against the simpler point-mass physics already verified in
propulsion/rocket_equation.py and guidance/gravity_turn.py, in the
degenerate case where attitude is held fixed and thrust points straight
along one inertial axis (the 6DOF model should reduce exactly to 1D
rocket-equation motion in that case).
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dynamics.six_dof import (
    quat_normalize, quat_multiply, quat_to_rotation_matrix, quat_derivative,
    euler_to_quat, SixDOFState, InertiaTensor, six_dof_derivatives,
    integrate_six_dof, simulate_six_dof_free_flight, G0
)

PASS = 0
FAIL = 0


def check(name, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


def test_identity_quaternion_gives_identity_rotation():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    R = quat_to_rotation_matrix(q)
    check("Identity quaternion produces the identity rotation matrix",
          np.allclose(R, np.eye(3)))


def test_quat_multiply_identity_property():
    q_identity = np.array([1.0, 0.0, 0.0, 0.0])
    q = quat_normalize(np.array([0.5, 0.3, 0.1, 0.2]))
    result = quat_multiply(q, q_identity)
    check("Multiplying any quaternion by the identity quaternion returns it unchanged",
          np.allclose(result, q))


def test_90deg_rotation_matches_known_result():
    """A 90-degree rotation about the z-axis should map x-axis to y-axis."""
    half_angle = np.radians(90.0) / 2.0
    q = np.array([np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)])
    R = quat_to_rotation_matrix(q)
    x_axis = np.array([1.0, 0.0, 0.0])
    rotated = R @ x_axis
    check("90-degree z-axis quaternion rotation maps [1,0,0] to [0,1,0] (within tolerance)",
          np.allclose(rotated, [0.0, 1.0, 0.0], atol=1e-9))


def test_euler_to_quat_zero_gives_identity():
    q = euler_to_quat(0.0, 0.0, 0.0)
    check("Zero Euler angles produce the identity quaternion",
          np.allclose(q, [1.0, 0.0, 0.0, 0.0]))


def test_quaternion_norm_preserved_during_free_rotation():
    """
    Repeated integration of the quaternion kinematic equation (with
    normalization each step, as integrate_six_dof does) must keep the
    quaternion at unit norm throughout — a hard numerical requirement
    for attitude representation to remain physically meaningful.
    """
    state = SixDOFState(
        position=[0, 0, 0], velocity=[0, 0, 0],
        quaternion=[1, 0, 0, 0], angular_velocity=[0.5, 0.3, 0.2], mass=1000.0
    )
    inertia = InertiaTensor(100.0, 150.0, 120.0)
    thrust_func = lambda t, s: np.array([0.0, 0.0, 0.0])
    torque_func = lambda t, s: np.array([0.0, 0.0, 0.0])  # torque-free

    result = simulate_six_dof_free_flight(
        state, inertia, thrust_func, torque_func, mdot_kg_s=0.0,
        gravity_inertial=np.array([0, 0, 0]), dt=0.01, t_max=5.0,
    )
    norms = np.linalg.norm(result["quaternion"], axis=1)
    check("Quaternion norm stays at 1.0 (within 1e-6) throughout a 5-second free rotation",
          np.allclose(norms, 1.0, atol=1e-6))


def test_torque_free_spin_about_principal_axis_is_exactly_conserved():
    """
    Classical special case of Euler's rigid-body rotation equations:
    if a body spins PURELY about one of its own principal axes with NO
    external torque, that angular velocity is an exact equilibrium of
    the equations (the gyroscopic cross term omega x I*omega vanishes
    identically for pure single-axis spin), so omega must remain
    constant for the entire simulation — a strong, exact analytic check
    on the rotational dynamics implementation.
    """
    state = SixDOFState(
        position=[0, 0, 0], velocity=[0, 0, 0],
        quaternion=[1, 0, 0, 0],
        angular_velocity=[2.0, 0.0, 0.0],  # pure spin about body x-axis only
        mass=1000.0,
    )
    inertia = InertiaTensor(100.0, 150.0, 120.0)  # asymmetric — makes this a
                                                     # nontrivial check, not a
                                                     # trivially-symmetric case
    thrust_func = lambda t, s: np.array([0.0, 0.0, 0.0])
    torque_func = lambda t, s: np.array([0.0, 0.0, 0.0])

    result = simulate_six_dof_free_flight(
        state, inertia, thrust_func, torque_func, mdot_kg_s=0.0,
        gravity_inertial=np.array([0, 0, 0]), dt=0.005, t_max=3.0,
    )
    omega_x = result["angular_velocity_rad_s"][:, 0]
    omega_y = result["angular_velocity_rad_s"][:, 1]
    omega_z = result["angular_velocity_rad_s"][:, 2]

    check("Pure single-axis spin: omega_x stays constant at 2.0 rad/s (exact analytic result)",
          np.allclose(omega_x, 2.0, atol=1e-4))
    check("Pure single-axis spin: no angular velocity is induced on the other two axes",
          np.allclose(omega_y, 0.0, atol=1e-9) and np.allclose(omega_z, 0.0, atol=1e-9))


def test_gyroscopic_coupling_appears_for_non_principal_spin():
    """
    Contrast case proving the gyroscopic cross-coupling term is actually
    implemented (not accidentally zero): spinning about a combination of
    axes on an ASYMMETRIC body (Ixx != Iyy != Izz) should cause the
    angular velocity components to change over time due to
    omega x I*omega being nonzero — this is exactly the real effect a
    1-axis (pitch-only) model in control/tvc_attitude.py cannot capture.
    """
    state = SixDOFState(
        position=[0, 0, 0], velocity=[0, 0, 0], quaternion=[1, 0, 0, 0],
        angular_velocity=[1.0, 1.0, 0.0],  # spin about combination of x and y
        mass=1000.0,
    )
    inertia = InertiaTensor(100.0, 150.0, 120.0)
    thrust_func = lambda t, s: np.array([0.0, 0.0, 0.0])
    torque_func = lambda t, s: np.array([0.0, 0.0, 0.0])

    result = simulate_six_dof_free_flight(
        state, inertia, thrust_func, torque_func, mdot_kg_s=0.0,
        gravity_inertial=np.array([0, 0, 0]), dt=0.005, t_max=1.0,
    )
    omega_z_final = result["angular_velocity_rad_s"][-1, 2]
    check("Gyroscopic coupling induces nonzero z-axis spin from combined x/y spin on an asymmetric body",
          abs(omega_z_final) > 1e-4)


def test_six_dof_reduces_to_1d_rocket_equation_for_fixed_attitude_axial_thrust():
    """
    Consistency check against the simpler models: with attitude held
    fixed (zero angular velocity, identity quaternion) and thrust
    applied purely along the inertial x-axis with zero gravity, the
    6DOF translational motion must reduce EXACTLY to the same physics
    verified in propulsion/rocket_equation.py's Tsiolkovsky check —
    final speed = Isp*g0*ln(m0/mf).
    """
    isp, thrust_mag, m0 = 300.0, 500000.0, 10000.0
    mdot = thrust_mag / (isp * G0)

    state = SixDOFState(
        position=[0, 0, 0], velocity=[0, 0, 0], quaternion=[1, 0, 0, 0],
        angular_velocity=[0, 0, 0], mass=m0,
    )
    inertia = InertiaTensor(100.0, 150.0, 120.0)
    thrust_func = lambda t, s: np.array([thrust_mag, 0.0, 0.0])  # along body x = inertial x (identity attitude)
    torque_func = lambda t, s: np.array([0.0, 0.0, 0.0])

    dt = 0.01
    result = simulate_six_dof_free_flight(
        state, inertia, thrust_func, torque_func, mdot_kg_s=mdot,
        gravity_inertial=np.array([0.0, 0.0, 0.0]), dt=dt, t_max=5.0,
    )
    final_speed = np.linalg.norm(result["velocity_m_s"][-1])
    final_mass = result["mass_kg"][-1]
    expected_speed = isp * G0 * np.log(m0 / final_mass)

    check("6DOF translational motion matches Tsiolkovsky rocket equation for fixed-attitude axial thrust (<1% error)",
          abs(final_speed - expected_speed) / expected_speed < 0.01)
    check("Motion stays purely along x-axis (y and z velocity remain zero)",
          abs(result["velocity_m_s"][-1, 1]) < 1e-6 and abs(result["velocity_m_s"][-1, 2]) < 1e-6)


def test_off_axis_thrust_produces_3d_motion():
    """
    Sanity check that this is genuinely 3D: thrust applied with a
    component along y (in addition to x) should produce nonzero
    y-velocity — something a 2D (x-y plane only) model could still show,
    but combined with a z-component too, proves the full 3D coupling
    that a 2D gravity-turn model (guidance/gravity_turn.py) cannot
    represent at all.
    """
    state = SixDOFState(
        position=[0, 0, 0], velocity=[0, 0, 0], quaternion=[1, 0, 0, 0],
        angular_velocity=[0, 0, 0], mass=5000.0,
    )
    inertia = InertiaTensor(100.0, 150.0, 120.0)
    thrust_func = lambda t, s: np.array([300000.0, 50000.0, 20000.0])
    torque_func = lambda t, s: np.array([0.0, 0.0, 0.0])

    result = simulate_six_dof_free_flight(
        state, inertia, thrust_func, torque_func, mdot_kg_s=50.0,
        gravity_inertial=np.array([0.0, 0.0, 0.0]), dt=0.01, t_max=2.0,
    )
    vx, vy, vz = result["velocity_m_s"][-1]
    check("Off-axis thrust produces motion in all three inertial axes (genuine 3D/6DOF behavior)",
          abs(vx) > 0 and abs(vy) > 0 and abs(vz) > 0)


if __name__ == "__main__":
    print("Running 6DOF rigid-body dynamics verification suite...\n")
    test_identity_quaternion_gives_identity_rotation()
    test_quat_multiply_identity_property()
    test_90deg_rotation_matches_known_result()
    test_euler_to_quat_zero_gives_identity()
    test_quaternion_norm_preserved_during_free_rotation()
    test_torque_free_spin_about_principal_axis_is_exactly_conserved()
    test_gyroscopic_coupling_appears_for_non_principal_spin()
    test_six_dof_reduces_to_1d_rocket_equation_for_fixed_attitude_axial_thrust()
    test_off_axis_thrust_produces_3d_motion()

    print(f"\n{PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    if FAIL > 0:
        sys.exit(1)
