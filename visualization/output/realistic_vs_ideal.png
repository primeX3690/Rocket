"""
visualization/plots.py

Data visualization for the AscentGNC stack — turns raw simulation
arrays (already produced and verified by every other module) into
plots a human reviewer can actually look at, instead of reading numbers
off a terminal. Uses matplotlib only (no GPU, no heavyweight rendering
engine — consistent with the project's zero-cost CPU-only philosophy).

Every function here takes the SAME result dictionaries already returned
by the simulation functions elsewhere in this project — it does not
recompute or duplicate any physics, it only visualizes what has already
been verified by the test suites.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless rendering, no display needed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)


def plot_ascent_profile(gravity_turn_result: dict, save_path: str):
    """Altitude, speed, and dynamic pressure vs time for a gravity-turn
    ascent (output of guidance/gravity_turn.py's simulate_ascent)."""
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    t = gravity_turn_result["t"]
    axes[0].plot(t, gravity_turn_result["altitude_m"] / 1000.0, color="#1f6feb")
    axes[0].set_ylabel("Altitude (km)")
    axes[0].set_title("Stage-1 Gravity-Turn Ascent Profile")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, gravity_turn_result["speed_m_s"], color="#da3633")
    axes[1].set_ylabel("Speed (m/s)")
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, gravity_turn_result["dynamic_pressure_pa"] / 1000.0, color="#8957e5")
    max_q_t = gravity_turn_result["max_q_time_s"]
    max_q_val = gravity_turn_result["max_q_pa"] / 1000.0
    axes[2].axvline(max_q_t, color="gray", linestyle="--", alpha=0.6)
    axes[2].annotate(f"Max-Q: {max_q_val:.1f} kPa", (max_q_t, max_q_val),
                      textcoords="offset points", xytext=(10, 10))
    axes[2].set_ylabel("Dynamic Pressure (kPa)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_peg_insertion(peg_result: dict, target_radius_m: float, save_path: str):
    """Radius vs time for a PEG-guided insertion burn, with the target
    radius marked (output of guidance/peg.py's simulate_peg_guided_burn)."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    t = peg_result["t"]
    R_EARTH = 6378137.0
    axes[0].plot(t, (peg_result["radius_m"] - R_EARTH) / 1000.0, color="#1f6feb", label="Actual altitude")
    axes[0].axhline((target_radius_m - R_EARTH) / 1000.0, color="#da3633", linestyle="--", label="Target altitude")
    axes[0].set_ylabel("Altitude (km)")
    axes[0].set_title("PEG Closed-Loop Insertion Burn")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, peg_result["velocity_m_s"], color="#da3633")
    axes[1].set_ylabel("Speed (m/s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_monte_carlo_comparison(comparison_result: dict, save_path: str):
    """
    Histogram comparing closed-loop PEG vs open-loop insertion-velocity
    error distributions (output of
    analysis/monte_carlo.py's compare_open_vs_closed_loop_dispersion) —
    the visual version of the project's headline quantified result.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    cl = comparison_result["closed_loop_velocity_error_m_s"]
    ol = comparison_result["open_loop_velocity_error_m_s"]

    ax.hist(ol, bins=30, alpha=0.5, label=f"Open-loop (std={np.std(ol):.1f} m/s)", color="#da3633")
    ax.hist(cl, bins=30, alpha=0.5, label=f"Closed-loop PEG (std={np.std(cl):.1f} m/s)", color="#1f6feb")
    ax.axvline(0, color="black", linestyle=":", alpha=0.5)
    ax.set_xlabel("Insertion Velocity Error (m/s)")
    ax.set_ylabel("Trial Count")
    ax.set_title(f"Monte Carlo Dispersion: Closed-Loop vs Open-Loop "
                 f"({comparison_result['improvement_factor']:.2f}x tighter)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_six_dof_3d_trajectory(six_dof_result: dict, save_path: str):
    """
    3D render of a 6DOF trajectory's center-of-mass path through inertial
    space, color-coded by speed — the genuinely-3D visualization that a
    2D ascent plot cannot provide (output of
    dynamics/six_dof.py's simulate_six_dof_free_flight).
    """
    pos = six_dof_result["position_m"]
    vel = six_dof_result["velocity_m_s"]
    speed = np.linalg.norm(vel, axis=1)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=speed, cmap="plasma", s=4)
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color="gray", alpha=0.3, linewidth=0.5)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m, altitude)")
    ax.set_title("6DOF Trajectory (color = speed)")
    fig.colorbar(sc, ax=ax, label="Speed (m/s)", shrink=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_realistic_vs_ideal_attitude(ideal_result: dict, realistic_result: dict, save_path: str):
    """
    Pitch-error time history comparing idealized vs realistic
    (rate-limited, lagging actuator; gusty wind) attitude-hold
    performance (output of faults/real_world.py's
    simulate_realistic_attitude_hold).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ideal_result["t"], ideal_result["pitch_error_deg"],
            label="Idealized (instant actuator)", color="#1f6feb")
    ax.plot(realistic_result["t"], realistic_result["pitch_error_deg"],
            label="Realistic (rate-limited, lagging actuator + gust)", color="#da3633")
    ax.axhline(0, color="black", linestyle=":", alpha=0.4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch Error (deg)")
    ax.set_title("Real-World Actuator Constraints: Idealized vs Realistic")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
