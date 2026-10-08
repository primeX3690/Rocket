AscentGNC — Launch-Vehicle Ascent Guidance, Navigation & Control Stack

What this project is:

AscentGNC is a launch-vehicle ascent guidance, navigation & control (GNC) software stack — the software layer that decides how a rocket climbs straight, corrects its own path in real time, and — if something onboard fails (a sensor, an engine) — keeps the mission alive instead of letting it crash.
This is not a rocket itself. 
It is the brain that would sit inside one: the kind of software that runs on the flight computer of a launch vehicle, the layer that avionics/flight-software teams at companies like Skyroot or Agnikul own and maintain.
Zero-cost, CPU-only, dependency-light (NumPy for the physics; Matplotlib + Streamlit only for the optional plots/dashboard).
479 automated tests passing across 31 test files (run with `python3 run_all_tests.py`), plus a Software-in-the-Loop (SIL) harness that runs the ported control loop on a real ARM Cortex-M3 target under QEMU (see embedded/), and a Python<->C numerical-equivalence test (tests/test_embedded_c_equivalence.py) that verifies the C port actually matches the Python reference, not just that both run.

Why this exists:
India's deep-tech space sector (Skyroot, Agnikul) is growing fast, but their GNC stacks are proprietary and closed — there is no open, verifiable, from-scratch reference implementation that shows this kind of software can be built by one self-taught engineer with no lab, no funding, and no formal CS degree. AscentGNC exists to close that gap: as proof of capability, and as a stepping stone toward joining or building a launch-vehicle avionics team.
Where this is actually useful — real use cases

1. Space launch vehicles (the primary use case).
Every orbital rocket — from a small sounding rocket to a full PSLV-class vehicle — runs on this category of software: when to gimbal the engine (thrust vector control), when to stage, and how to replan toward an achievable lower orbit instead of a hard failure if an engine goes out mid-flight (this repo's engine_out.py).

2. Satellite deployment / orbit insertion.
peg.py implements a linear-tangent-steering guidance law in the Powered-Explicit-Guidance family, closed by a numerical predictor-corrector that re-solves for all three insertion constraints (radius, velocity, AND flight-path angle) every guidance cycle -- not just velocity, which is what an earlier version of this module did (see peg.py's docstring for exactly what is and isn't modeled). Measured result (300-trial Monte Carlo dispersion study, fair same-burn-duration comparison scored on a combined radius+velocity+flight-path-angle metric): closed-loop guidance has an ~11x lower velocity-error std deviation than a fixed-steering open-loop burn (1.5 m/s vs 16.5 m/s), and a ~140x lower combined insertion-error score. See analysis/monte_carlo.py.

3. Defense / dual-use — stated plainly, not hidden.
The underlying physics and guidance math of a launch vehicle and a ballistic missile overlap — this is publicly known and is exactly why every civilian rocket-GNC company, Skyroot included, operates under export-control regimes (ITAR/COCOM). This repo models that reality directly: consumer GPS chips enforce a real limit (GPS lock is dropped once altitude exceeds ~18 km AND speed exceeds ~515 m/s simultaneously — the COCOM rule, specifically to keep them from being usable in missile guidance). Being upfront about this dual-use nature, and building the export-control constraint into the simulation itself, is itself a credibility signal for serious aerospace/defense recruiters and investors.

4. Student / amateur high-power rocketry.
A scaled-down version of this stack is directly usable by college rocketry teams (e.g. Manipal's thrustMIT, IIT Madras's Team Abhyuday) as reference avionics for their own sounding-rocket flight computers.

What makes it different, what makes it advanced:
Zero-cost, CPU-only. No GPU, no paid aerospace tools (STK/GMAT/MATLAB) — just Python + NumPy, runs entirely on a Ryzen 3 / 8 GB RAM laptop.
Verified beyond simulation, on real hardware's instruction set. The control loop was ported to C, cross-compiled for the real ARM Cortex-M3 instruction set, and run under QEMU's ARM core emulation — proving it meets its real-time deadline on an actual ~$2 flight computer class chip (STM32-family) with a 99.8% timing margin, not just that the Python model behaves on a laptop CPU.
Real sensor data, not synthetic guesses. IMU/GPS noise is not an arbitrary constant — it's built from the actual published datasheets of real, buyable parts (MPU-6050, ICM-20602, BMI088 IMUs; u-blox NEO-6M/NEO-M8N GPS).
Models a real, documented failure mode. ISRO's SSLV-D1 mission failure (a transient accelerometer fault triggering a permanent, full open-loop guidance fallback) is reproduced in anomaly_arbitration.py -- a real fault is injected into a real sensor channel, run through an online detector, and the SSLV-D1-style permanent-freeze response is directly compared against isolating only the bad samples while staying closed-loop. Not a hypothetical scenario, and not asserted -- simulated end to end.
479 automated tests, checked against analytic/textbook closed-form values (vis-viva, conservation of angular momentum/energy during coast, quaternion orthonormality) and public order-of-magnitude reference-mission data — not "looks about right."

ascent-gnc/
├── propulsion/
│   └── rocket_equation.py       # Tsiolkovsky, multi-stage delta-V budgeting, nozzle Isp/thrust
├── guidance/
│   ├── atmosphere.py            # US Standard Atmosphere 1976 (density, pressure, dynamic pressure)
│   ├── gravity_turn.py          # 2D ascent integrator + open-loop gravity-turn guidance, max-Q tracking
│   ├── peg.py                   # Linear-tangent-steering guidance, numerical 3-constraint predictor-corrector solve
│   ├── orbit_utils.py           # Osculating-orbit (perigee/apogee/eccentricity) helpers
│   ├── engine_out.py            # Abort-to-Orbit-style engine-out contingency replanning
│   └── anomaly_arbitration.py   # Sensor-fault injection + detection + graceful-degradation vs SSLV-D1-style fallback
├── navigation/
│   ├── state_estimation.py      # Single-axis position/velocity/bias Kalman filter (simple reference case)
│   ├── attitude_ekf.py          # 15-state 3-axis strapdown ESKF: quaternion attitude + gyro/accel bias + GPS fusion
│   └── sensor_datasheets.py     # Real MPU-6050/ICM-20602/BMI088/u-blox datasheet noise models + COCOM GPS limit
├── control/
│   └── tvc_attitude.py          # Thrust Vector Control gimbal loop, analytically-tuned PID
├── dynamics/
│   └── six_dof.py               # Full 6DOF quaternion-based rigid-body dynamics
├── faults/
│   └── real_world.py            # Actuator lag/rate-limits, sensor latency, wind-gust models + compensation
├── embedded/                    # Software-in-the-Loop: control loop ported to C, run on real ARM
│   ├── README.md                #   Cortex-M3 (QEMU) — full method + honest caveats
│   ├── main.c, pid_controller.c/h, startup.c, linker.ld
│   ├── analyze_trace.py         #   parses the QEMU instruction trace into a timing report
│   ├── build_and_run.sh         #   one-command reproduction (needs gcc-arm-none-eabi + qemu-system-arm)
│   └── sample_report.txt        #   a saved run of the above
├── analysis/
│   ├── monte_carlo.py           # Dispersion/reliability analysis, closed-loop vs open-loop comparison
│   └── mission_profile.py       # Full stage1 -> coast -> stage2 -> orbit-insertion mission chain
├── data/
│   ├── reference_missions.py    # Public PSLV-class order-of-magnitude parameters for plausibility checks
│   └── reference_vehicle.py     # A generic 2-stage vehicle design chosen so the full mission converges
├── visualization/
│   ├── plots.py                 # Matplotlib static plot generators (ascent, PEG, Monte Carlo, 6DOF, etc.)
│   └── output/                  # Pre-generated PNG plots (already included)
├── tests/                       # 479 automated tests — one file per module, all runnable standalone
├── run_all_tests.py              # Runs every tests/test_*.py and prints a combined pass/fail summary
├── app.py                       # Interactive Streamlit dashboard (all modules, live sliders)
├── requirements.txt
└── README.md                    # This file



Running the tests:
Every test file is self-contained and runnable directly with python3 — no pytest required. Run from the project root.

python3 run_all_tests.py                   # runs every file below and prints a combined summary

python3 tests/test_propulsion.py           # 12 tests
python3 tests/test_atmosphere.py           # 10 tests
python3 tests/test_gravity_turn.py         # 11 tests
python3 tests/test_navigation.py           # 7 tests
python3 tests/test_control.py              # 9 tests
python3 tests/test_peg.py                  # 8 tests
python3 tests/test_orbit_utils.py          # 10 tests
python3 tests/test_monte_carlo.py          # 6 tests (slower — ~2-4 min, runs real Monte Carlo trials)
python3 tests/test_engine_out.py           # 11 tests
python3 tests/test_mission_profile.py      # 16 tests
python3 tests/test_six_dof.py              # 11 tests
python3 tests/test_real_world.py           # 10 tests
python3 tests/test_visualization.py        # 5 tests
python3 tests/test_anomaly_arbitration.py  # 10 tests
python3 tests/test_sensor_datasheets.py    # 23 tests
python3 tests/test_attitude_ekf.py         # 10 tests
python3 tests/test_reference_vehicle.py    # 7 tests
python3 tests/test_embedded_c_equivalence.py  # 4 tests (needs a host C compiler; skips gracefully without one)

Each file ends with a line like X passed, 0 failed out of X (479 total).

Software-in-the-Loop (real ARM Cortex-M3, not just Python)

embedded/ ports the TVC attitude-hold control loop to C, cross-compiles it for the real ARM Cortex-M3 instruction set, and runs it under QEMU's ARM core emulation — proving the compiled control loop fits its real-time budget on real embedded silicon's ISA, not just that the Python model is correct on a laptop CPU. Measured result: ~1703 ARM Thumb-2 instructions/iteration (worst case, no hardware FPU), ~99.8% timing margin on a 72 MHz Cortex-M3. Full method, numbers, and an honest list of what this does/doesn't prove (it's SIL, not full hardware-in-the-loop) are in embedded/README.md.

Regenerating the static plots
Plots are already included in visualization/output/, but you can regenerate or create new ones:

python3 -c "
from guidance.gravity_turn import simulate_ascent
from visualization.plots import plot_ascent_profile
from data.reference_vehicle import REFERENCE_VEHICLE

v = REFERENCE_VEHICLE
result = simulate_ascent(m0_kg=v['stage1_m0_kg'], m_dry_kg=v['stage1_m_dry_kg'],
                          thrust_n=v['stage1_thrust_n'], isp_s=v['stage1_isp_s'],
                          drag_coeff=v['stage1_drag_coeff'], ref_area_m2=v['stage1_ref_area_m2'],
                          kick_altitude_m=v['stage1_kick_altitude_m'], kick_angle_deg=v['stage1_kick_angle_deg'])
plot_ascent_profile(result, 'visualization/output/ascent_profile.png')
print('Saved!')
"


See visualization/plots.py for the other plot functions (plot_peg_insertion, plot_monte_carlo_comparison, plot_six_dof_3d_trajectory, plot_realistic_vs_ideal_attitude).

Using the modules directly:

# Multi-stage delta-V budget
from propulsion.rocket_equation import Stage, stage_dv_budget
s1 = Stage("Stage-1", m_prop=20000, m_struct=2000, isp_s=280.0)
s2 = Stage("Stage-2", m_prop=3000, m_struct=500, isp_s=340.0)
result = stage_dv_budget([s1, s2], payload_mass=200.0)
print(result["total_dv_m_s"])

# Gravity-turn ascent
from guidance.gravity_turn import simulate_ascent
traj = simulate_ascent(m0_kg=60000, m_dry_kg=12000, thrust_n=850_000, isp_s=285,
                        drag_coeff=0.3, ref_area_m2=2.0, kick_altitude_m=1000, kick_angle_deg=6.0)
print("Max-Q:", traj["max_q_pa"], "Pa")

# Closed-loop PEG orbit insertion
from guidance.peg import PEGTarget, simulate_peg_guided_burn
import numpy as np
R_EARTH, MU_EARTH = 6378137.0, 3.986004418e14
target_r = R_EARTH + 300000
target = PEGTarget(target_r, np.sqrt(MU_EARTH/target_r), 0.0)
peg_result = simulate_peg_guided_burn(r0_m=R_EARTH+180000, v0_m_s=7300, gamma0_rad=np.radians(3.0),
                                       m0_kg=4000, thrust_n=90000, isp_s=320, target=target)
print("Insertion error:", peg_result["insertion_error"])   # radius/velocity/flight-path-angle, all ~0

# Full 2-stage mission, guaranteed to converge (uses the locked reference-vehicle design)
from data.reference_vehicle import run_reference_mission
mission = run_reference_mission()
print("Final altitude:", mission["final_altitude_km"], "km, orbit:", mission["orbit"])

# SSLV-D1-style failure vs graceful degradation: a real accelerometer fault
# is injected and run through an online detector; the two guidance
# architectures' RESPONSE to that same detected fault is what's compared.
from guidance.anomaly_arbitration import compare_sslv_failure_mode_vs_graceful_degradation
comparison = compare_sslv_failure_mode_vs_graceful_degradation(
    r0_m=R_EARTH+300000, v0_m_s=7550, gamma0_rad=np.radians(0.3),
    m0_kg=4000, thrust_n=90000, isp_s=320,
    target=PEGTarget(R_EARTH+356200, np.sqrt(MU_EARTH/(R_EARTH+356200)), 0.0),
    anomaly_start_s=5.0,
)
print("Fallback error:", comparison["sslv_style_fallback"]["velocity_error_m_s"], "m/s")
print("Graceful error:", comparison["graceful_degradation"]["velocity_error_m_s"], "m/s")

Known limitations (stated plainly, not hidden):
- guidance/peg.py's solver is a local Levenberg-Marquardt predictor-corrector, not a globally-convergent method. It converges cleanly for the reference-vehicle design in data/reference_vehicle.py and the states used in the test suite, but is NOT guaranteed to converge from an arbitrary (r0, v0, gamma0, target) combination -- see that module's docstring.
- dynamics/six_dof.py's frame is a local flat tangent-plane approximation (good to a few hundred km / few minutes), not a full rotating oblate-Earth model; navigation/attitude_ekf.py similarly assumes flat, constant gravity.
- guidance/gravity_turn.py, embedded/, and the sensor noise models are all still point-mass/simplified representations of a generic small launcher, not flight-validated hardware or a specific real vehicle.

No package install needed for the physics stack — just run scripts from the project root
directory (imports are relative to the project root). app.py and visualization/plots.py
additionally need `pip install streamlit matplotlib` (see requirements.txt).







---

## Roadmap additions (advanced/extended modules)

Six additional capability areas, each genuinely implemented and tested
(not stubs) and fully additive — nothing in the base stack above was
changed in a breaking way:

1. **Physics/environment** — `environment/wind_models.py` (log-law wind
   shear + jet-stream bump + Dryden turbulence filter), `environment/
   aero_center.py` (Cp/Cg shift & static margin over a burn),
   `dynamics/mass_properties.py` (full inertia tensor incl. cross
   terms from a lumped-mass model), `dynamics/slosh.py` (equivalent-
   pendulum propellant slosh).
2. **Guidance** — `guidance/trajectory_optimization.py` (a real,
   working successive-convexification-STYLE ascent optimizer via
   scipy SLSQP — NOT a true SOCP solver like full SCvx; see that
   module's docstring for the honest scope), `guidance/
   staging_events.py` (stage separation tip-off + fairing jettison
   gated on dynamic pressure).
3. **Navigation** — `navigation/sensor_realism.py` (bias instability,
   angular/velocity random walk, scale-factor error, misalignment),
   `navigation/multi_rate_fusion.py` (async IMU/GNSS/star-tracker
   fusion at each sensor's own rate), `navigation/gnss_integrity.py`
   (autonomous spoofing/jamming detection + inertial fallback).
4. **Embedded/HIL** — `embedded/actuator_realism.py` (deadband +
   backlash), `embedded/rtos_sim.py` (a genuine fixed-priority
   scheduler simulation: RMS priorities, exact response-time analysis,
   tick-by-tick deadline-miss detection — a software MODEL, not a real
   FreeRTOS/Zephyr integration, which needs hardware this project
   doesn't have), `embedded/protocols.py` (real CAN 2.0B CRC-15,
   ARINC 429 words with odd parity, SpaceWire packet framing).
5. **Telemetry security** — `telemetry/secure_log.py` (a genuine
   SHA-256 Merkle tamper-evident log, NOT a zero-knowledge proof — a
   distinct primitive this doesn't claim to be), `telemetry/
   secure_stream.py` (real AES-256-GCM encryption + replay protection
   via the `cryptography` package).
6. **Visualization** — `analysis/dispersion_analysis.py` (CEP,
   confidence dispersion ellipses, parameter-sensitivity correlation —
   wired into app.py's Monte Carlo tab), `visualization/web/
   attitude_viewer.html` (a real, standalone Three.js 3D quaternion
   viewer; open it directly in a browser and load a JSON trajectory
   exported via `visualization/export_trajectory.py`).

All of the above have their own test files (`tests/test_environment.py`,
`test_trajectory_optimization.py`, `test_navigation_advanced.py`,
`test_embedded_advanced.py`, `test_telemetry_security.py`,
`test_visualization_advanced.py`) — 286/286 tests pass in total across
24 files (`python3 run_all_tests.py`).


---

## Reusability (first-stage recovery / propulsive landing)

A separate, genuinely-implemented recovery stack, additive to
everything above (nothing existing was changed in a breaking way):
this project's original scope was ASCENT-only (launch -> orbit); a
returning, reusable first stage is a fundamentally different flight
regime (boostback, entry burn, aerodynamic descent, powered landing,
touchdown) using different guidance algorithms, so it lives in its own
modules rather than being bolted onto the ascent stack.

- **`guidance/powered_descent_guidance.py`** — the real, rigorously-
  derived ZEM/ZEV (Zero-Effort-Miss/Zero-Effort-Velocity) minimum-
  energy closed-loop guidance law (the same guidance-law family behind
  the Apollo Lunar Module's descent guidance), re-derived from first
  principles (calculus of variations on a double integrator) and
  verified against a direct simulation in the test suite — not a
  remembered/copied formula. Includes suicide-burn ignition-altitude
  kinematics and a full closed-loop powered-landing simulator with a
  real deep-throttle engine constraint.
- **`guidance/boostback_and_entry.py`** — boostback-burn delta-v
  targeting and entry-burn deceleration sizing (rocket-equation based),
  including a structural-deceleration-limit check.
- **`dynamics/grid_fins.py`** — a lattice/grid-fin aerodynamic control-
  surface model (lift/drag vs deflection and Mach, with stall and a
  Mach-dependent effectiveness falloff matching published grid-fin
  behavior).
- **`dynamics/landing_legs.py`** — spring-damper touchdown dynamics
  (peak deceleration, bottoming-out detection) and a tip-over stability
  check (Cg offset from tilt + lateral drift vs the landing-leg
  footprint).
- **`analysis/reusability_mission.py`** — ties all of the above into
  ONE genuine round-trip mission: separation -> boostback -> coast to
  entry interface -> entry burn -> drag-limited atmospheric descent ->
  ZEM/ZEV landing burn -> touchdown/tip-over verdict. Uses a real
  PREDICTOR-CORRECTOR boostback-targeting scheme (a quick trial run of
  the coast/entry/descent phases to estimate total time-of-flight,
  THEN sizing the boostback burn from that refined estimate) — a naive
  one-shot kinematic guess was found, during this module's own
  development, to leave a 38+ km landing miss once atmospheric drag
  during descent was modeled correctly; the predictor-corrector fix
  brings that under 1 km for the tested scenario.

**Honestly stated limitations:**
- Boostback/entry/descent targeting in `reusability_mission.py` is
  simplified 2D (vertical-plane) point-mass, matching this project's
  existing ascent-side convention — no 3D cross-range, no active
  grid-fin STEERING loop during descent (the aerodynamic MODEL exists
  in `grid_fins.py`; it is not yet wired into closed-loop lateral
  guidance during the descent phase). This means residual lateral
  velocity at touchdown is corrected only by the final landing burn,
  which is why the tip-over stability check can — correctly and
  informatively — flag some scenarios as unstable due to touchdown
  lateral drift; that is real, useful mission-analysis output, not a
  hidden bug.
- The ZEM/ZEV solver's gains are singular as time-to-go approaches
  zero (a well-known property of the closed-form law); this is handled
  with a time-to-go floor, as documented in that module.

Tested in `tests/test_reusability.py` (43 tests, including a from-
scratch re-derivation check of the ZEM/ZEV law against a direct
double-integrator simulation) — 329/329 tests pass in total across 25
files (`python3 run_all_tests.py`).

## Flight-software layer (C) — `flight_software/`

Board-independent C99 flight software: 3-IMU FDIR with debounce/recovery and voting, a measured-evidence flight-mode state machine with abort logic, a task-liveness watchdog, and a CRC-framed telemetry link, including a **float32 C port of the 15-state ESKF** and a **double-precision C port of the PEG guidance solver** (both numerically checked against the Python originals), GPS latency compensation, a PEG-guided upper-stage burn flown closed-loop in simulation (insertion within 6 m / 0.03 m/s), and an **orbit-scale inertial navigation mode** (Earth-centred pad-aligned frame, J2 gravity, ECEF fixes with Earth rotation and latency handling) that feeds guidance its polar state (filter-derived polar state within 1 m / 0.01 m/s of truth; NEES 4.0 vs ~6 expected); 227 host checks including 28 closed-loop scenario groups, and a full two-stage ascent from the pad to a 199.5 x 211.1 km orbit (ASan/UBSan clean). Host-tested only — **not yet run on hardware**; see `flight_software/README.md` for exact scope and the defects fixed in this pass (notably: the SIL harness's old TVC gains were unstable).
