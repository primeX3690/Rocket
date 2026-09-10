"""
dynamics/six_dof.py

Full 6-Degree-of-Freedom (6DOF) rigid-body launch-vehicle dynamics —
3 translational (x, y, z position/velocity) + 3 rotational (roll,
pitch, yaw attitude/angular-velocity) degrees of freedom, coupled
together. This supersedes the simplified models used elsewhere in the
stack:

    - guidance/gravity_turn.py: 2D point-mass (2DOF translational only,
      no attitude dynamics — "the vehicle points wherever its velocity
      vector points," a standard and valid simplification for ascent
      guidance DESIGN, but not a full vehicle simulation)
    - control/tvc_attitude.py: 1DOF (pitch-axis-only) rotational
      dynamics — cannot represent roll or yaw motion, or the coupling
      between them (e.g. a yaw disturbance inducing roll through
      product-of-inertia terms, or gimbal deflection in one axis
      creating unwanted torque about another).

A real launch vehicle needs all 6: it must track a 3D trajectory
(not just a 2D vertical-plane ascent) and control attitude in 3 axes
simultaneously (pitch AND yaw AND roll), with the equations of motion
properly coupled — this is what "going from 3DOF to 6DOF" means in
flight dynamics, and it's a materially harder problem: attitude can no
longer be represented by a single angle (gimbal lock territory), so
this module uses QUATERNIONS, the standard singularity-free attitude
representation used by every real flight computer.

Reference frames:
    - Inertial frame (I): fixed, Earth-centered for this module's
      purposes (flat, non-rotating — rotating-Earth effects are a
      further extension, not implemented here, and documented as such).
    - Body frame (B): fixed to the vehicle, x-axis along the thrust
      centerline.
    Attitude quaternion q rotates vectors from body frame to inertial
    frame.

Zero external dependencies beyond NumPy — CPU-only.
"""

import numpy as np

G0 = 9.80665


# ---------------------------------------------------------------------------
# Quaternion utilities
# ---------------------------------------------------------------------------

def quat_normalize(q):
    return q / np.linalg.norm(q)


def quat_multiply(q1, q2):
    """Hamilton product, q = [w, x, y, z] scalar-first convention."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_to_rotation_matrix(q):
    """Rotation matrix (body -> inertial) from a unit quaternion."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x ** 2 + y ** 2)],
    ])


def quat_derivative(q, omega_body):
    """
    dq/dt = 0.5 * q (x) [0, omega_body]  (Hamilton product with a
    pure-vector quaternion built from the body-frame angular velocity).
    Standard quaternion kinematic equation.
    """
    omega_quat = np.array([0.0, omega_body[0], omega_body[1], omega_body[2]])
    return 0.5 * quat_multiply(q, omega_quat)


def euler_to_quat(roll, pitch, yaw):
    """Standard aerospace 3-2-1 (yaw-pitch-roll) Euler angle -> quaternion."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


# ---------------------------------------------------------------------------
# 6DOF rigid-body state and dynamics
# ---------------------------------------------------------------------------

class SixDOFState:
    """
    Full rigid-body state:
        position (inertial frame, m):        r = [x, y, z]
        velocity (inertial frame, m/s):       v = [vx, vy, vz]
        attitude quaternion (body->inertial): q = [w, x, y, z]
        body-frame angular velocity (rad/s):  omega = [p, q_rate, r_rate]
        mass (kg):                            m
    """
    def __init__(self, position, velocity, quaternion, angular_velocity, mass):
        self.r = np.array(position, dtype=float)
        self.v = np.array(velocity, dtype=float)
        self.q = quat_normalize(np.array(quaternion, dtype=float))
        self.omega = np.array(angular_velocity, dtype=float)
        self.m = float(mass)


class InertiaTensor:
    """
    Diagonal (principal-axis) inertia tensor — valid for a
    symmetric launch-vehicle body (axisymmetric about the thrust axis),
    which is the standard assumption for this class of vehicle absent
    detailed CAD mass properties.
    """
    def __init__(self, ixx, iyy, izz):
        self.I = np.diag([ixx, iyy, izz])
        self.I_inv = np.diag([1.0 / ixx, 1.0 / iyy, 1.0 / izz])


def six_dof_derivatives(state: SixDOFState, inertia: InertiaTensor,
                         thrust_body_n: np.ndarray, torque_body_nm: np.ndarray,
                         mdot_kg_s: float, gravity_inertial: np.ndarray):
    """
    Compute the full 6DOF state derivative:
        dr/dt     = v
        dv/dt     = (1/m) * R(q) * F_thrust_body + g   (thrust rotated
                    from body frame into inertial frame, plus gravity —
                    the KEY coupling that a pitch-only model cannot
                    represent: thrust direction depends on the FULL 3D
                    attitude, not a single angle)
        dq/dt     = 0.5 * q (x) [0, omega]
        domega/dt = I^-1 * (torque_body - omega x (I*omega))   (Euler's
                    rigid-body rotation equation, including the
                    gyroscopic cross-coupling term omega x I*omega that
                    a 1-axis model sets to zero by construction — this
                    term is exactly what causes an asymmetric rocket
                    to develop roll/yaw coupling from a pure pitch
                    maneuver, a real effect single-axis models miss)
        dm/dt     = -mdot
    """
    R = quat_to_rotation_matrix(state.q)
    thrust_inertial = R @ thrust_body_n

    dr_dt = state.v
    dv_dt = thrust_inertial / state.m + gravity_inertial
    dq_dt = quat_derivative(state.q, state.omega)

    gyroscopic_term = np.cross(state.omega, inertia.I @ state.omega)
    domega_dt = inertia.I_inv @ (torque_body_nm - gyroscopic_term)

    dm_dt = -mdot_kg_s

    return dr_dt, dv_dt, dq_dt, domega_dt, dm_dt


def integrate_six_dof(state: SixDOFState, inertia: InertiaTensor,
                       thrust_body_n, torque_body_nm, mdot_kg_s,
                       gravity_inertial, dt: float):
    """One explicit-Euler integration step, with quaternion re-normalization
    (required every step — integrating the quaternion derivative alone
    slowly drifts off the unit-norm constraint, a well-known numerical
    issue that every real attitude-estimation/propagation system
    corrects for exactly this way)."""
    dr, dv, dq, domega, dm = six_dof_derivatives(
        state, inertia, thrust_body_n, torque_body_nm, mdot_kg_s, gravity_inertial
    )

    state.r = state.r + dr * dt
    state.v = state.v + dv * dt
    state.q = quat_normalize(state.q + dq * dt)
    state.omega = state.omega + domega * dt
    state.m = state.m - mdot_kg_s * dt

    return state


def simulate_six_dof_free_flight(
    initial_state: SixDOFState, inertia: InertiaTensor,
    thrust_body_n_func, torque_body_nm_func, mdot_kg_s: float,
    gravity_inertial=np.array([0.0, 0.0, -9.80665]),
    dt: float = 0.02, t_max: float = 20.0,
):
    """
    Propagate a full 6DOF trajectory. thrust_body_n_func(t, state) and
    torque_body_nm_func(t, state) are callables so guidance/control
    logic can command thrust direction and torques as a function of
    time and current state (closed-loop capable, same pattern as the
    rest of this stack).
    """
    n_steps = int(t_max / dt)
    t_hist = np.zeros(n_steps)
    pos_hist = np.zeros((n_steps, 3))
    vel_hist = np.zeros((n_steps, 3))
    quat_hist = np.zeros((n_steps, 4))
    omega_hist = np.zeros((n_steps, 3))
    mass_hist = np.zeros(n_steps)

    state = initial_state
    t = 0.0
    for i in range(n_steps):
        thrust_body = thrust_body_n_func(t, state)
        torque_body = torque_body_nm_func(t, state)

        state = integrate_six_dof(state, inertia, thrust_body, torque_body,
                                   mdot_kg_s, gravity_inertial, dt)

        t_hist[i] = t
        pos_hist[i] = state.r
        vel_hist[i] = state.v
        quat_hist[i] = state.q
        omega_hist[i] = state.omega
        mass_hist[i] = state.m

        t += dt
        if state.m <= 0:
            break

    return {
        "t": t_hist, "position_m": pos_hist, "velocity_m_s": vel_hist,
        "quaternion": quat_hist, "angular_velocity_rad_s": omega_hist,
        "mass_kg": mass_hist, "final_state": state,
    }
