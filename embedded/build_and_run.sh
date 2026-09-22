#!/usr/bin/env bash
# embedded/build_and_run.sh
#
# Reproduces the AscentGNC Software-in-the-Loop (SIL) timing result end
# to end: cross-compiles the ported TVC control loop for a real ARM
# Cortex-M3 target, boots it under QEMU's cycle-level ARM core
# emulation, traces every instruction it actually executes, and
# reports the resulting per-iteration instruction count / estimated
# timing margin against a real flight-computer's control-loop deadline.
#
# Requires (Ubuntu/Debian): sudo apt-get install -y --no-install-recommends \
#     gcc-arm-none-eabi libnewlib-arm-none-eabi qemu-system-arm
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1/4: cross-compiling for ARM Cortex-M3 (ARMv7-M Thumb-2) =="
arm-none-eabi-gcc -mcpu=cortex-m3 -mthumb -O2 -Wall \
  -ffreestanding -nostartfiles -T linker.ld \
  startup.c pid_controller.c main.c \
  --specs=rdimon.specs \
  -Wl,--start-group -lc -lrdimon -lgcc -Wl,--end-group \
  -o firmware.elf
arm-none-eabi-size firmware.elf

echo
echo "== 2/4: sanity-run under QEMU (semihosting console output) =="
timeout 15 qemu-system-arm -M lm3s6965evb -nographic -semihosting -kernel firmware.elf

echo
echo "== 3/4: single-step instruction trace (this is the slow step) =="
rm -f trace.log
timeout 180 qemu-system-arm -M lm3s6965evb -nographic -semihosting \
  -singlestep -d exec -D trace.log -kernel firmware.elf > /dev/null

echo
echo "== 4/4: analyzing trace =="
python3 analyze_trace.py trace.log

rm -f trace.log   # ~85 MB, regenerate by re-running this script
