# flight_software — board-independent C flight-software layer

Until now the repo had the *algorithms* (Python) and one ported control loop
(`embedded/`, SIL under QEMU). It had **no flight-software architecture**: no
mode logic, no sensor redundancy management, no watchdog, no telemetry link.
This directory adds that layer in portable C99 (no heap, no libm, no globals,
no OS), verified on the host.

## What is here

| File | Purpose |
|---|---|
| `fsw_fdir.c/.h` | Per-channel health monitor (range, rate, stuck-at, NaN/Inf) with **debounce + recovery**, permanent exclusion for chronic flapping, and a **3-channel voter** (median-based, catches slow drift the monitor cannot see) |
| `fsw_state.c/.h` | Flight-mode state machine `PAD_SAFE → PAD_ARMED → IGNITION → ASCENT_S1 → STAGE_SEP → ASCENT_S2 → COAST → ORBIT_INSERTED` plus `ABORT`. Transitions need *measured* confirmation (acceleration for N ticks), not just timers; premature cutoff / no-ignition abort; illegal commands rejected and counted |
| `fsw_watchdog.c/.h` | Task-liveness watchdog: hardware dog is kicked only if **every** task checked in |
| `fsw_telemetry.c/.h`, `fsw_crc.c/.h` | Framed telemetry (`EB 90 | type | seq | len | payload | CRC-16/CCITT`), resynchronising parser safe for a UART RX ISR |
| `fsw_eskf.c/.h` | **15-state error-state EKF**, C port of `navigation/attitude_ekf.py` (float32 covariance/attitude, **double nominal position/velocity**, no libm/heap, ~2.9 kB RAM). Gravity models: flat (Python-equivalent), central inverse-square with gradient in the covariance, central + J2; velocity-Verlet integration. Adds: NIS innovation gate with lock-out protection, singular-S rejection, health check, accelerometer leveling, decorrelating `reset_pv` for hand-offs |
| `fsw_frames.c/.h` | Reference frames for orbit-scale navigation: non-rotating, Earth-centred, **pad-aligned inertial frame** (X downrange at the launch azimuth, Z up at the pad), ECEF ↔ inertial with Earth rotation, pad inertial velocity, polar quantities (r, speed, γ, ψ). Uses libm (double) |
| `fsw_peg.c/.h` | **PEG predictor-corrector, C port of `guidance/peg.py`** (double precision, uses libm — the only flight source that does). Bit-identical forward prediction and, in converged cases, identical A/B/T_go and iteration counts vs Python. Adds input validation (status codes instead of exceptions/NaN) and a step cap |
| `fsw_guidance.c/.h` | Runtime around the solver: warm start, cold retry, previous-solution fallback, withdrawal after repeated failures, seqlock publication (heavy solve in a low-priority task, cheap lock-free read in the 50 Hz cycle), θ(t) / time-to-go / cutoff / insertion-window helpers |
| `fsw_math.c/.h` | sqrt/sin/cos/atan2/asin without libm (abs error 3e-6 … 2e-5, measured against libm) |
| `fsw_core.c/.h` | One deterministic cycle: 3×IMU (3-axis accel+gyro) → FDIR on all 6 axes → votes → state machine → pad alignment / ESKF → TVC PID → gimbal → telemetry → watchdog |
| `fsw_hal.h` | The **only** hardware interface (4 callbacks). Body frame: +Z axial/up on the pad, pitch about +Y |
| `port/hal_stm32_template.c` | Shape of a real-board port (**template, not compiled/tested**) |
| `test/fsw_selftest.c` | 210 checks: unit tests + 26 closed-loop scenarios (real ESKF incl. ECI mode, real guidance, 3-D inertial truth) |

Run: `cd flight_software && make test` (needs only gcc/clang; `-lm` is used by the
*test* only). Also wired into `python3 run_all_tests.py` via
`tests/test_flight_software.py` (builds with AddressSanitizer + UBSan too) and
`tests/test_eskf_c_equivalence.py` (35 checks: C ESKF vs the Python reference fed
identical inputs, pitch through ±90°, plus `nm -u` proof that no ESKF/FDIR/state/telemetry object references libm) and
`tests/test_eci_nav.py` (24 checks, numpy/scipy references: frames, gravity + J2, 600 s propagation vs DOP853, gravity-gradient covariance vs finite differences, reset PSD) and
`tests/test_peg_c_equivalence.py` (43 checks: C PEG vs Python — prediction, cold/warm solves, infeasible targets,
closed-loop burns driven by the C solver, input validation, libm confined to `fsw_peg.c` and `fsw_frames.c`).

## Scenarios that are simulated end to end (real `fsw_core` vs. simulated vehicle)

Nominal mission to ORBIT_INSERTED · vehicle tilted 2° on the pad with gyro
bias (pad alignment + bias calibration) · what calibration and GPS each buy ·
IMU stuck-at mid-ascent · one-sample spike
(does not latch — the SSLV-D1 lesson) · 2-sample burst (below debounce) ·
4-sample burst (fault declared, channel **recovers**) · slow drift (invisible to
the monitor, caught by the voter) · two IMUs die in flight (prompt ABORT) · two
IMUs dead on the pad (arm refused) · engine fails to start · premature
cutoff · ground abort in flight and on the pad · slow **gyro** drift (must not
leak into the ESKF) · 400 m GPS outlier · no GPS at all · corrupted filter
covariance in flight (→ abort, gimbal centred).

## Orbit-scale (inertial) navigation: `cfg.nav_eci = 1`

- **Frame.** The filter runs in a non-rotating, Earth-centred frame fixed at liftoff and aligned with the pad
  (`fsw_frames.h`): the pad starts at (0, 0, r_pad) with velocity ω×r (451.8 m/s downrange at 13.7° N). The body
  frame on the pad coincides with it, so "pitch about +Y" is the same quantity as in the flat filter and as
  PEG's plane. Inverse-square gravity (+J2, ~0.013 m/s² at 200 km) replaces constant gravity; its gradient is in
  the covariance. Gyros measure inertial rate directly, so no Earth-rate term enters the attitude kinematics; on
  the pad the Earth's rotation *is* in the gyro average and is subtracted (|Δ| 7e-5 rad/s; it would otherwise be
  15 °/h of attitude drift).
- **Precision.** |p| ≈ 6.4e6 m is beyond float32 (0.5 m resolution): nominal position/velocity are double,
  covariance and attitude float32, the innovation is subtracted in double before rounding.
- **GPS.** `fsw_core_gps_fix_ecef(ecef, age_s)`: rotated into the frame at the *measurement* time, nav history
  interpolated at the fractional age. At orbital speed this matters: a 10 ms error in the reported age costs
  ~65 m (measured 62.6 m vs 11.6 m); GPS time-tagging must be good to ~1 ms.
- **Polar state for PEG is now derived from the filter** every cycle (r, inertial speed, γ, ψ; mass from a
  fuel-gauge model, `fsw_core_set_stage2_mass`).
- **Measured (3-D inertial truth with J2, 3 m ECEF fixes at 1 Hz, simulated IMU):**
  toy mission position error ≤ 4.9 m, velocity ≤ 1.2 m/s after 10 s, **NEES 4.0 over 40 samples (6 dof; a
  consistent filter gives ~6)**; pure inertial 6.2 m / 0.33 m/s over 30 s (J2 ignored: 8.0 m / 0.51 m/s).
  Guided burn on filter-derived polar state: filter vs truth at the end r −1 m, v +0.009 m/s, γ −0.0006°, ψ <1e-5 rad;
  TRUE insertion error r −8 m, v +4.15 m/s, γ −0.004° (the 4 m/s is J2, which PEG's point-mass model does not
  contain; the filter's own J2 setting does not change it).
- **Caveats, all real:**
  - The orbital hand-off in the guided test is a **harness shortcut** (truth jumps to a hand-off state; the filter
    is set to it ± 11 m / 0.1 m/s with a decorrelated covariance). The toy first stage cannot fly to that state, so
    the closed-loop test does not prove the filter through a real S1→S2 trajectory. The first-stage ECI run
    (Scenario 17) is separate and unguided.
  - Spherical Earth (geocentric = geodetic latitude): the ellipsoid is a ~21 km / 0.2° effect to add before
    flying. Gravity is point mass + J2 only; atmospheric drag needs no model (it is in the accelerometers).
  - ψ and PEG assume the motion stays near the launch-azimuth plane; a dog-leg or a launch azimuth other than the
    one the vehicle is pointed along is not handled (the frames support any azimuth; guidance does not).
  - Roll/yaw attitude uncertainty at liftoff is a configured prior (2 mrad roll/pitch from leveling, 10 mrad
    azimuth), not something the code can measure; wrong values make the filter over- or under-confident.
  - No GNSS velocity/Doppler measurement and no velocity-aided time-tag handling.

## Guidance: what runs where, and what the tests show

- **Where it runs.** `fsw_core_guidance_service()` does the solve and belongs in a low-priority
  task/idle loop. The 50 Hz `fsw_core_step()` only *requests* a solve every `cycle_s` (2 s), reads the
  latest published solution through the seqlock, converts θ(t) to a pitch command
  φ = π/2 − θ + ψ (ψ = downrange angle rotating the local horizontal), slew-limits it
  (`pitch_cmd_rate_limit_rps`, 5.7°/s) and sends the engine-cutoff command when T_go expires.
  Host solve time was 0.3 ms (warm) — **meaningless for the target**: doubles are software on a Cortex-M4F;
  measure on the board.
- **Closed-loop result (Scenario 15, point-mass truth integrated along the vehicle's ACTUAL attitude, real
  core + ESKF + PID + slew limit):** 30 kN / 4000 kg upper stage, 180 km → 200 km circular, 183.7 s burn,
  92 solves, 0 failures, cutoff commanded by guidance at T_go → insertion error **r +6 m, v +0.03 m/s,
  γ +0.001°**; attitude loop tracks the guidance command within 0.0065 rad. With the engine delivering
  5 % less thrust than guidance assumes: r −2 m, v −1.0 m/s.
- **Degraded modes, tested:** polar state stops updating (STALE flag, no solves on old data, last solution
  flies and cutoff still commanded — ends 14 m/s off, inside the 15 m/s window, and *not* declared inserted
  because insertion is never judged on stale data); NaN polar state (solver rejects it, solution withdrawn after
  3 consecutive failures, LOST flag, core flies the external pitch program, no guided cutoff, burn ends at
  propellant depletion in COAST).
- **Observations about the Python PEG itself (inherited, not fixed):** (1) the repo's nominal test case
  (180 km, 7300 m/s, γ=3° → 300 km) has a steering law that goes from 88° to **−87° (thrust pointing down)**
  — mathematically fine, physically an unnatural start; (2) a 190 km / 7350 m/s start to 320 km circular is
  infeasible with 2.8 t of propellant and the solver correctly reports not-converged with T pinned at the
  propellant limit (0/72 seeds converge); (3) near the end of a burn with a thrust mismatch the solve is
  propellant-limited and non-converged, and C and Python iterates differ at 1e-7, amplified to ≤0.34 m/s,
  ≤27 m, ≤0.3 s at cutoff.

## Navigation: what it does and what the tests actually show

- **Pad (PAD_SAFE/ARMED):** the voted specific force is low-pass averaged and
  used to level roll/pitch; the voted gyro is averaged (while at rest) into a
  bias estimate. A single accelerometer sample is NOT enough: it gave 0.0038 rad
  tilt error, averaging gives ~0.0003.
- **From liftoff:** the ESKF is initialised from that alignment and bias and
  free-runs on the voted IMUs; fixes passed to `fsw_core_gps_fix()` go through a
  chi-square gate (3 dof, 99 %). 
- **Measured, with the simulated IMU (this is a toy plant, so treat numbers as
  relative, not absolute):**
  - Pure inertial attitude error 0.0015–0.0024 rad; with 3 m GPS fixes the
    ESKF attitude is *worse* (~0.012 rad): position innovations tug attitude.
    The simulated gyro is far quieter than real hardware's in-run drift, so GPS
    only adds noise there. Tune `gps_r_pos` / ESKF process noise on real IMU data.
  - 0.01 rad/s gyro bias: calibrated on the pad → 0.002 rad error; uncalibrated
    and no GPS → 0.26 rad (15°) at t=30 s; uncalibrated **with** GPS → the ESKF
    learns the bias (0.0100 vs true 0.0100) after a ~10 s transient (peak 0.09 rad).
    So: pad calibration is essential; GPS is the backstop, not a substitute.
  - The initial gyro-bias covariance matters: with the Python default (σ =
    0.001 rad/s) the filter treated a 0.01 rad/s bias as a 10σ event and learned
    it far too slowly; when pad calibration is off the core now uses the
    datasheet zero-rate tolerance (`uncal_gyro_bias_sigma_rps`, 1°/s).
- **GPS latency is compensated** (`fsw_core_gps_fix_aged(pos, age_s)`): the core keeps the last 32
  post-predict positions and moves the fix forward with the ESKF's own displacement,
  z' = z + p_now − p_then (error ≈ velocity-error × age), inflating R by age²σ_v². Fixes with a negative/NaN
  age, older than the history, or arriving before liftoff are dropped and counted. Measured with a 0.3 s
  receiver delay: position error 6.95 m compensated vs 41.5 m ignoring it (11 good fixes then get gated out).
  The *age* must come from the receiver's time-of-validity vs a PPS-disciplined clock; this code cannot invent it.
- Cost/size: `sizeof(fsw_eskf_t)` = 2816 B, `sizeof(fsw_core_t)` = 4064 B.
  `fsw_eskf_predict` is ~2×15³ ≈ 6.8 k multiply-adds (analytic estimate, **not
  measured on a target**) — a Cortex-M4F should fit in a 20 ms frame with large
  margin, but verify on the board.

## Defects found and fixed in the ECI pass (all pinned by tests)

- **Gravity integrated first-order** (gravity only at the start of each step): 16.8 m / 0.055 m/s and 452 J/kg
  of energy error after 600 s of free fall. Velocity-Verlet: 0.10 m / 1.2e-4 m/s / 1e-7 J/kg.
- **Setting only the diagonal of P after a hand-off makes it non-positive-definite** (min normalised eigenvalue
  −0.16); in the closed loop roll/yaw covariance collapsed to the floor, the filter became overconfident and
  diverged after cutoff. `fsw_eskf_reset_pv()` decorrelates; the pitfall is documented by a test.
- Attitude covariance at liftoff was one value for all axes (10 mrad); now from alignment quality
  (2 mrad roll/pitch, 10 mrad yaw).

## Defects found and fixed (all pinned by tests)

- **Pitch estimate folded back past ±90°** (`fsw_eskf_pitch` used `asin(2(wy−zx))`): above 1.57 rad the
  estimate moved *opposite* to the true rotation, the PID's sign flipped, and the vehicle spun up and aborted
  within ~3 s. Invisible while pitch stayed below 90°; fatal for an orbital burn that pitches through
  horizontal. Now `atan2(R02, R22)` (full ±π) and the PID error is wrapped.
- **State machine aborted any upper stage with specific force < 11 m/s²** (it reused the >1 g liftoff
  threshold for ASCENT_S2 → false NO_IGNITION abort at 7.5 m/s²). Added `thrust_accel_s2_mps2` (4 m/s²).
- Single-sample pad leveling (3.8 mrad tilt error) → averaged (0.3 mrad); gyro-bias prior too tight when
  uncalibrated; GPS fixes were being treated as fresh. (Earlier passes: see below.)

## Earlier defects found and fixed

1. **`embedded/main.c` TVC gains were unstable.** kp=8, ki=0.5, kd=3 on the
   harness's own plant (T=2e6 N, I=5e4, L=1.2 → loop gain 48 s⁻²) put the
   crossover far above the 50 ms actuator lag; pitch diverged to ~0.9 rad in
   both the Python reference and the C port (the Python↔C equivalence test
   passed because both were *equally* wrong). Gains are now 0.3 / 0.05 / 0.3
   (holds a 0.05 rad step to within 5 %, gimbal use < 1°).
   `tests/test_tvc_stability.py` pins this. `tests/test_embedded_c_equivalence.py`
   was left untouched (it still proves C ≡ Python numerically).
2. State machine did not count commands refused while a critical fault was latched on the pad.

## Honest scope — what this is NOT

- **Not run on hardware.** Host-tested only. `port/hal_stm32_template.c` is a
  template; you still need drivers (SPI IMU, PWM, UART, IWDG) and bench tests.
  SIL/QEMU timing in `embedded/` was measured with the *old* gains; instruction
  mix is unchanged but re-run `embedded/build_and_run.sh` for a fresh report.
- **Two navigation modes.** `nav_eci = 0` (default) is the flat-Earth/constant-gravity filter of the Python
  reference; PEG's polar state then has to come from outside (`fsw_core_set_polar_state()`, used by Scenarios
  15–16 with simulated truth). `nav_eci = 1` is the orbit-scale filter above, which supplies it (Scenario 18).
- **Guidance is only wired into ASCENT_S2.** Only the pitch axis is closed-loop (roll/yaw are estimated, not
  controlled). The S1→S2 hand-off in the closed-loop test is a scripted pitch program plus an arbitrary
  hand-off state, not a consistent first-stage trajectory.
- The PEG solve has no multi-start: from a bad initial guess it can stall (see observations above); the wrapper
  only retries cold once and otherwise falls back.
- **The ESKF uses the same flat-Earth, constant-gravity model as the Python
  reference**: fine for ascent windows of tens of seconds to minutes, not for
  orbit-scale navigation (see `dynamics/six_dof.py` for the inverse-square model).
- **The test plant is a pitch-axis toy** with scripted axial acceleration; it
  validates flight-software *logic* (modes, FDIR, navigation plumbing, watchdog, framing), not
  vehicle performance. Thresholds in `fsw_core_default_cfg()` are reasonable
  starting values for this vehicle class, not flight-qualified numbers.
- No command authentication/CRC on the *uplink*, no flash/NVM fault log, no
  brown-out handling, no pyro/valve driver logic. Those need the real hardware design first.
- Do not build with `-ffast-math` (NaN/Inf detection relies on IEEE compares).
- Stuck-at detection (25 identical samples) is tuned for noisy analogue-ish
  channels; on a very quiet quantised axis it could false-trigger — check on
  real data.
