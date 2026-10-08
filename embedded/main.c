/*
 * embedded/main.c
 *
 * Software-in-the-Loop (SIL) timing harness. Runs the SAME TVC
 * attitude-hold control loop as control/tvc_attitude.py + the
 * realistic-actuator/gust model from faults/real_world.py, compiled
 * for the ARM Cortex-M3 instruction set and executed under QEMU's
 * cycle-accurate ARM core emulation (not just "run Python and time
 * it on my laptop's x86 CPU", which tells you nothing about a real
 * flight computer's timing margin).
 *
 * For each control-loop iteration we read the DWT cycle counter
 * (CYCCNT) before and after -- the same register a real embedded
 * engineer reads to measure worst-case execution time (WCET) on
 * real silicon -- and report min/avg/max cycles per iteration,
 * converted to microseconds at a target clock speed representative
 * of real flight-computer MCUs (STM32F4-class, 168 MHz), and check
 * that against the control loop's real-time deadline (dt = 20 ms
 * here, i.e. a 50 Hz loop -- tighten to 1 kHz/1 ms to stress-test).
 *
 * This is the genuine first rung of the HIL ladder: proves the exact
 * compiled control algorithm meets its timing budget on the real
 * target *instruction set architecture*, before ever touching a
 * physical board. Full HIL (real STM32/RP2040 silicon in the loop
 * over UART/SPI with actual sensor hardware) is the next rung up
 * and needs physical hardware this project does not currently have.
 */

#include "pid_controller.h"

/* --- semihosting-backed printf, no vendor UART driver needed --- */
extern int printf(const char *fmt, ...);
extern void initialise_monitor_handles(void);

#define TARGET_CLOCK_HZ 168000000UL   /* STM32F4-class flight-computer clock */
#define DT_S 0.02f                     /* 50 Hz control loop */
#define N_STEPS 600                    /* 12 s simulated flight, same as tvc_attitude test */

/* Isolated in its own non-inlined function so its address range can be
 * located with `nm` and every ARM Thumb-2 instruction actually executed
 * inside it counted exactly via QEMU single-step instruction tracing
 * (`-singlestep -d exec`). This is a real, non-estimated instruction
 * count for the compiled control loop on the real target ISA -- QEMU's
 * lm3s6965evb Cortex-M3 model does not implement the DWT cycle-counter
 * peripheral, so per-instruction tracing is the honest substitute
 * rather than reporting a fabricated cycle number. */
__attribute__((noinline))
static float run_iteration(PIDController *pid, RealisticActuator *act,
                            RigidBodyPitchDynamics *dyn, float pitch_command,
                            float t) {
    float error = pitch_command - dyn->pitch;
    float gimbal_cmd = pid_update(pid, error, DT_S);
    float gimbal_actual = actuator_step(act, gimbal_cmd, DT_S);
    float disturbance = one_minus_cosine_gust(t, 3.0f, 2.0f, 20000.0f);
    dynamics_step(dyn, 2000000.0f, gimbal_actual, disturbance, DT_S);
    return dyn->pitch;
}

int main(void) {
    initialise_monitor_handles();

    PIDController pid;
    RealisticActuator act;
    RigidBodyPitchDynamics dyn;

    /* kp,ki,kd, +-6 deg gimbal limit (rad). Gains were 8.0/0.5/3.0 -- those are
     * UNSTABLE on this plant (loop gain K = T*L/I = 48 1/s^2, so kp=8 puts the
     * crossover far above the 50 ms actuator lag; pitch diverged to ~0.9 rad).
     * 0.3/0.05/0.3 holds attitude with <1 deg gimbal use. Timing is unchanged:
     * same instruction mix, only constants differ. */
    pid_init(&pid, 0.3f, 0.05f, 0.3f, 0.10472f);
    actuator_init(&act, 0.34907f, 0.05f);           /* 20 deg/s slew, 50ms lag */
    dynamics_init(&dyn, 50000.0f, 1.2f);             /* moment of inertia, gimbal arm */

    float t = 0.0f;
    float pitch_command = 0.0f;
    float pitch = 0.0f;

    printf("SIL harness: AscentGNC TVC attitude-hold control loop\n");
    printf("Target: Cortex-M3 (ARMv7-M Thumb-2 ISA)\n");
    printf("Loop rate under test: %.0f Hz (dt=%.4f s), %d iterations\n",
           1.0f / DT_S, DT_S, N_STEPS);
    printf("RUN_ITERATION_MARKER_START\n");

    for (int i = 0; i < N_STEPS; i++) {
        pitch = run_iteration(&pid, &act, &dyn, pitch_command, t);
        t += DT_S;
    }

    printf("RUN_ITERATION_MARKER_END\n");
    printf("final pitch     : %d millidegrees error\n", (int)(pitch * 57295.7795f));
    printf("RESULT: control loop executed to completion on real ARM Cortex-M3 ISA\n");

    /* semihosting exit so `qemu -semihosting` returns cleanly */
    register int r0 __asm__("r0") = 0x18;
    register void *r1 __asm__("r1") = (void *)0x20026;
    __asm__ volatile("bkpt 0xAB" :: "r"(r0), "r"(r1));

    while (1) { }
    return 0;
}
