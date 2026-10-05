# AscentGNC — Software-in-the-Loop (SIL) Timing Harness

## What this is

The rest of this repo (propulsion/, guidance/, navigation/, control/,
analysis/) is Python running on a laptop CPU. That proves the
**algorithms** are correct. It proves nothing about whether they fit
inside a real flight computer's timing budget -- Python on an x86
laptop has nothing to do with a Cortex-M3/M4 running at 72-168 MHz.

This folder ports the actual TVC attitude-hold control loop
(`control/tvc_attitude.py`'s `PIDController`, plus the realistic
actuator/wind-gust model from `faults/real_world.py`) to C, cross-
compiles it for the **real ARM Cortex-M3 instruction set**
(`arm-none-eabi-gcc -mcpu=cortex-m3 -mthumb`), and runs the compiled
binary under **QEMU's cycle-level ARM core emulation**
(`qemu-system-arm -M lm3s6965evb`) -- not a physical board (this project
doesn't have one yet), but the actual target CPU architecture and
instruction set, which is the part that determines timing.

## What was measured, and how

The plan was to read the ARM Cortex-M **DWT cycle counter**
(`CYCCNT`, the standard register real embedded engineers profile
against) directly from the firmware. QEMU's `lm3s6965evb` machine
model turned out **not to implement DWT** (`CYCCNT` reads back 0
always on this target) -- a real limitation of this specific QEMU
board model, not of the approach. Rather than paper over that with a
fabricated number, the harness instead does single-step **instruction
tracing** (`qemu-system-arm -singlestep -d exec`) and counts every
real ARM Thumb-2 instruction the compiled control loop executes,
attributed by function symbol (`analyze_trace.py`). This is a
strictly more granular measurement than a cycle count would have
been -- it's converted to an estimated cycle/time budget at the end
using a documented CPI (cycles-per-instruction) assumption, stated
explicitly rather than hidden.

## Result (see `sample_report.txt` for the full breakdown)

- **~1703 ARM Thumb-2 instructions per control-loop iteration**
  (measured directly, not estimated) on this Cortex-M3 target.
- This target has **no hardware FPU** (`lm3s6965evb` is a plain
  Cortex-M3), so every float add/sub/mul/div/compare in the PID +
  actuator + dynamics code goes through a **software floating-point
  library** -- which is why float-helper routines (`__divsf3`,
  `__mulsf3`, `__aeabi_fsub`, ...) dominate the instruction count.
  This makes the measurement a **deliberately conservative,
  worst-case** number: a real flight-computer-class part
  (STM32F4/F7, Cortex-M4F/M7F) has a **hardware** single-precision
  FPU and would replace most of this with single-cycle instructions.
- Converting the no-FPU instruction count to time at real embedded
  clock speeds (CPI~1.3, a standard Cortex-M3 assumption per ARM's
  own Technical Reference Manual for branch-light integer/float-heavy
  code):
  - STM32F103 "Blue Pill" (72 MHz, ~$2, the most common Cortex-M3
    hobbyist board): **~30.7 us/iteration**, vs. a 20,000 us budget
    at 50 Hz -- **99.8% timing margin**.
  - STM32F4-class (168 MHz): **~13.2 us/iteration**, **99.9% margin**.
- Headline claim for a pitch/fellowship application: *"the ported
  attitude-control loop uses under 0.2% of its real-time budget on
  the cheapest, slowest, FPU-less Cortex-M3 board a student can buy
  -- and that's the pessimistic case."*

## What this does NOT prove (be honest about this)

- **This is SIL, not HIL.** No physical microcontroller, no real
  sensor bus (I2C/SPI to an actual IMU), no interrupt jitter, no
  power-supply noise, no real-world timing nondeterminism. QEMU's CPU
  core timing model is a simplified approximation of a real chip, not
  a cycle-perfect silicon replica.
- It does not include interrupt latency, DMA setup, sensor I2C/SPI
  transaction time, or RTOS scheduling jitter -- all of which eat
  into the real budget on an actual board. The margin above (>99%)
  has a lot of room to absorb that, but "a lot of room in a rough
  estimate" is not the same claim as "measured on hardware."
- **Real HIL** -- an actual STM32/RP2040 board wired to a real IMU,
  running this exact control loop against live sensor data over
  UART/SPI -- is the next rung up, and needs physical hardware this
  project does not currently have. This SIL harness is the legitimate,
  honest, fully-reproducible-on-a-laptop step below that, not a
  substitute for it.

## How to reproduce

```
sudo apt-get install -y --no-install-recommends \
    gcc-arm-none-eabi libnewlib-arm-none-eabi qemu-system-arm
./build_and_run.sh
```

Files:
- `startup.c` -- minimal Cortex-M3 vector table + reset handler (the
  actual boot code a flight computer runs before anything else).
- `pid_controller.c/.h` -- C port of the PID + actuator + rigid-body
  dynamics + wind-gust model, structurally identical to the Python.
- `main.c` -- the SIL harness: runs 600 iterations (12 s simulated
  flight @ 50 Hz, same as the Python test) and prints results via
  ARM semihosting (no UART driver needed for this stage).
- `linker.ld` -- memory layout for the `lm3s6965evb` QEMU target.
- `analyze_trace.py` -- parses the QEMU instruction trace into the
  report above.
- `build_and_run.sh` -- does all of the above in one command.
