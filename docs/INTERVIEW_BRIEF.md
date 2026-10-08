# AscentGNC: interview brief (read this first)

## 60-second pitch
A launch-vehicle GNC stack, end to end: **Python reference models** (6-DoF dynamics, ESKF, PEG guidance, atmosphere,
fault models) and a **portable C flight-software layer** that runs the same algorithms the way a flight computer must:
fixed-rate, no heap, no OS, behind a 4-function hardware interface. It is verified against the Python originals,
and flown pad-to-orbit in a closed-loop simulation with atmosphere, drag, staging, J2 gravity, GNSS latency, sensor
faults and engine dispersions. **`run_all_tests.py` reports 479 results across 31 files (the C flight-software layer counts as 5 there but runs 227 checks inside, so roughly 700 individual checks). Everything is host/simulation-verified; nothing has run on hardware.**

## Demo (about 2 minutes)
    sh flight_software/demo.sh          # builds with -Werror, runs the C verification, prints headline numbers
    python3 run_all_tests.py            # whole repository (~8 min): 479 results, 0 failed

## The headline run (Scenario 20, no shortcuts)
Pad alignment, liftoff, pitch table, max-Q 53.7 kPa (Mach 1.79, 10.7 km), staging, PEG from the filter's own state,
guidance-commanded cutoff: **199.5 x 211.1 km orbit** (target 200 circular); filter within 0.6 m / 0.06 m/s of truth;
145/145 guidance solves converged. 4 % weak engines: still inserted. IMU frozen at max-Q: still inserted.
GNSS lost for 250 s: **not** on target (159 x 243 km) and the filter said so (1-sigma 74 km). Quote that one yourself.

## Architecture (one breath)
    3 IMUs -> per-axis FDIR (range/rate/stuck/NaN, debounce+recovery) -> 2-of-3 voter -> flight-mode state machine
           -> ESKF (inertial frame, double position, J2) <- ECEF GNSS (latency + Earth rotation handled)
           -> polar state -> PEG (low-priority task, seqlock hand-over) -> pitch command (slew-limited) -> TVC PID
    + task-liveness watchdog + CRC-framed telemetry + single HAL

## Bugs the tests found (good stories: say "found by testing", it is true)
1. **Pitch from asin() folds back past 90 degrees**: the estimate moved opposite to the real rotation, the PID sign flipped,
   the vehicle spun up. Hidden because older flights never passed horizontal. Fixed (atan2) and pinned.
2. The SIL harness's TVC gains were **unstable on their own plant** (pitch diverged to 52 deg); Python and C agreed because both were wrong.
3. State machine aborted any upper stage under 1.1 g (reused the liftoff threshold) -> false "no ignition".
4. Gravity integrated first-order: 17 m / 0.055 m/s error after 600 s -> velocity-Verlet (0.1 m).
5. Editing only the diagonal of a covariance after a hand-off makes it non-positive-definite -> filter overconfident, diverged.
6. A fuel-gauge mass model turned a 4 % weak engine into a 45 m/s cutoff shortfall -> effective mass from measured acceleration.
7. My own C-wrapper tests silently SKIPPED on a build error. Fixed and proven by breaking the build on purpose.

## What NOT to claim
Not flight-proven. Not hardware-tested. Vehicle numbers are illustrative (not Skyroot/Agnikul/ISRO data). Aerodynamics
are zero-lift (no wind, no loads, no bending/slosh). Spherical Earth. Pitch axis only is closed-loop; the attitude
plant is a toy. GNSS time-tag must be good to ~1 ms (10 ms = 65 m at orbital speed). PEG cold-start can stall on bad
guesses (no multi-start). Guidance assumes motion near the launch plane (no dog-leg). Timing on a real MCU is unmeasured
(doubles are software on a Cortex-M4F; budget the PEG solve on the target).

## Questions to expect
- *Why an error-state EKF?* Small-angle error states keep the covariance well-conditioned and the quaternion on the manifold; nominal state propagates nonlinearly.
- *Why double for position?* |p| ~ 6.4e6 m; float32 resolution is 0.5 m. Covariance stays float32 (error states are small).
- *How do you know the filter is consistent?* NEES over 40 samples = 4.0 (6 dof, expected ~6); plus a numpy finite-difference check of the gravity-gradient covariance block.
- *What about a stuck or drifting sensor?* Debounced monitor + median voter; a single spike never latches (the SSLV-D1 lesson); slow drift is caught by the voter, not the monitor (both tested).
- *Why is PEG not in the control interrupt?* Doubles in software + iterative solve: low-priority task, seqlock publication, the 50 Hz cycle only reads.
- *What would you do next on hardware?* Drivers + PPS-disciplined GNSS time-tag, measure PEG solve time, processor-in-the-loop with the same test vectors, then bench IMU injection.
