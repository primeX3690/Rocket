"""
engine_bridge.py

Bridges the platform's HTTP API to the actual AscentGNC simulation
engine (the Rocket-main project). The engine is a SEPARATE codebase on
purpose (library vs service: the simulation core should stay usable
standalone, e.g. from a notebook or a CLI script, without needing this
web platform at all).

Locate the engine via the ASCENTGNC_ENGINE_PATH environment variable
(an absolute path to the Rocket-main project root, the directory that
directly contains guidance/, analysis/, dynamics/, etc.). Falls back to
a sibling directory named "Rocket-main" next to this platform's own
root, for a simple local-dev layout:

    some-folder/
      ascentgnc-platform/   <- this project
      Rocket-main/          <- the engine (unzip Rocket-main-reusable.zip here)

If neither resolves to a valid engine, every function below raises a
clear RuntimeError rather than failing with a confusing ImportError
deep inside numpy/guidance internals.
"""

import os
import sys
import numpy as np

_PLATFORM_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_ENGINE_PATH = os.path.join(os.path.dirname(_PLATFORM_ROOT), "Rocket-main")
_ENGINE_PATH = os.environ.get("ASCENTGNC_ENGINE_PATH", _DEFAULT_ENGINE_PATH)

_engine_ready = False
_engine_error = None

if os.path.isdir(_ENGINE_PATH) and os.path.isdir(os.path.join(_ENGINE_PATH, "guidance")):
    if _ENGINE_PATH not in sys.path:
        sys.path.insert(0, _ENGINE_PATH)
    try:
        from guidance.peg import PEGTarget, simulate_peg_guided_burn           # noqa: E402
        from guidance.anomaly_arbitration import (                            # noqa: E402
            compare_sslv_failure_mode_vs_graceful_degradation, nominal_cutoff_mass_kg,
        )
        from guidance.gravity_turn import simulate_ascent                     # noqa: E402
        from analysis.monte_carlo import run_monte_carlo                      # noqa: E402
        from data.reference_vehicle import REFERENCE_VEHICLE, run_reference_mission  # noqa: E402
        _engine_ready = True
    except ImportError as e:
        _engine_error = str(e)
else:
    _engine_error = (
        f"Engine not found. Looked for a valid Rocket-main project at: {_ENGINE_PATH}\n"
        f"Set the ASCENTGNC_ENGINE_PATH environment variable to the Rocket-main "
        f"project root, or place it at: {_DEFAULT_ENGINE_PATH}"
    )

R_EARTH = 6378137.0
MU_EARTH = 3.986004418e14


def _require_engine():
    if not _engine_ready:
        raise RuntimeError(f"AscentGNC simulation engine is not available: {_engine_error}")


def run_reference_mission_demo() -> dict:
    """The flagship, guaranteed-to-converge demo mission (data/reference_vehicle.py)."""
    _require_engine()
    result = run_reference_mission()
    return {
        "final_altitude_km": float(result["final_altitude_km"]),
        "total_mission_time_s": float(result["total_mission_time_s"]),
        "insertion_error": {k: float(v) for k, v in result["insertion_error"].items()},
        "orbit": {k: (float(v) if not isinstance(v, bool) else v) for k, v in result["orbit"].items()},
        "stage1_max_q_pa": float(result["stage1_max_q_pa"]),
    }


def run_peg_insertion(r0_alt_km: float, v0_m_s: float, gamma0_deg: float, m0_kg: float,
                      thrust_n: float, isp_s: float, target_alt_km: float) -> dict:
    """Runs a single closed-loop PEG orbit-insertion burn with caller-supplied parameters."""
    _require_engine()
    r0 = R_EARTH + r0_alt_km * 1000.0
    target_r = R_EARTH + target_alt_km * 1000.0
    target = PEGTarget(target_r, np.sqrt(MU_EARTH / target_r), 0.0)
    result = simulate_peg_guided_burn(r0, v0_m_s, np.radians(gamma0_deg), m0_kg, thrust_n, isp_s, target)
    return {
        "insertion_error": {k: float(v) for k, v in result["insertion_error"].items()},
        "burn_time_s": float(result["burn_time_s"]),
        "final_mass_kg": float(result["final_mass_kg"]),
        "solver_converged": bool(result["solver_converged"]),
        "orbit": {k: float(v) for k, v in result["orbit"].items() if k != "is_bound"},
    }


def run_ascent_profile(m0_kg: float, m_dry_kg: float, thrust_n: float, isp_s: float,
                       drag_coeff: float, ref_area_m2: float, kick_altitude_m: float,
                       kick_angle_deg: float) -> dict:
    _require_engine()
    result = simulate_ascent(m0_kg, m_dry_kg, thrust_n, isp_s, drag_coeff, ref_area_m2,
                             kick_altitude_m, kick_angle_deg)
    return {
        "max_q_pa": float(result["max_q_pa"]),
        "burnout_altitude_m": float(result["burnout_altitude_m"]),
        "burnout_speed_m_s": float(result["burnout_speed_m_s"]),
        "burn_time_s": float(result["burn_time_s"]),
    }


def run_anomaly_comparison(r0_alt_km: float, v0_m_s: float, gamma0_deg: float, m0_kg: float,
                          thrust_n: float, isp_s: float, target_alt_km: float,
                          anomaly_start_s: float) -> dict:
    """The flagship SSLV-D1-style case study: fallback-vs-graceful-degradation comparison."""
    _require_engine()
    r0 = R_EARTH + r0_alt_km * 1000.0
    target_r = R_EARTH + target_alt_km * 1000.0
    target = PEGTarget(target_r, np.sqrt(MU_EARTH / target_r), 0.0)
    comparison = compare_sslv_failure_mode_vs_graceful_degradation(
        r0, v0_m_s, np.radians(gamma0_deg), m0_kg, thrust_n, isp_s, target,
        anomaly_start_s=anomaly_start_s,
    )
    fb, gd = comparison["sslv_style_fallback"], comparison["graceful_degradation"]
    return {
        "fallback": {"velocity_error_m_s": float(fb["velocity_error_m_s"]),
                    "radius_error_m": float(fb["radius_error_m"]),
                    "permanently_frozen": bool(fb["permanently_frozen"])},
        "graceful_degradation": {"velocity_error_m_s": float(gd["velocity_error_m_s"]),
                                 "radius_error_m": float(gd["radius_error_m"])},
        "velocity_error_improvement_m_s": float(comparison["velocity_error_improvement_m_s"]),
        "radius_error_improvement_m": float(comparison["radius_error_improvement_m"]),
    }


def run_monte_carlo_dispersion(r0_alt_km: float, v0_m_s: float, gamma0_deg: float, m0_kg: float,
                               thrust_n: float, isp_s: float, target_alt_km: float,
                               n_trials: int) -> dict:
    _require_engine()
    n_trials = min(max(int(n_trials), 10), 300)   # server-side cap: protects shared infra from a runaway request
    r0 = R_EARTH + r0_alt_km * 1000.0
    target_r = R_EARTH + target_alt_km * 1000.0
    target = PEGTarget(target_r, np.sqrt(MU_EARTH / target_r), 0.0)
    result = run_monte_carlo(r0, v0_m_s, np.radians(gamma0_deg), m0_kg, thrust_n, isp_s, target,
                            n_trials=n_trials)
    return {
        "n_trials": result["n_trials"], "success_rate": float(result["success_rate"]),
        "velocity_error_mean": float(result["velocity_error_mean"]),
        "velocity_error_std": float(result["velocity_error_std"]),
        "radius_error_mean": float(result["radius_error_mean"]),
        "radius_error_std": float(result["radius_error_std"]),
    }


def engine_status() -> dict:
    return {"ready": _engine_ready, "engine_path": _ENGINE_PATH,
           "error": None if _engine_ready else _engine_error}
