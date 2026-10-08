/* port/hal_stm32_template.c -- TEMPLATE, NOT COMPILED OR TESTED ON HARDWARE.
 *
 * To run flight_software on a real board you implement fsw_hal_t (4 callbacks)
 * and call fsw_core_step() at a fixed rate. Nothing else in flight_software/
 * touches hardware. This file shows the shape for an STM32F4 + 3x SPI IMUs +
 * UART telemetry + IWDG. Every `TODO` is a vendor-HAL call you must supply.
 *
 * Wiring plan this template assumes:
 *   IMU0/1/2 : three separate SPI IMUs (e.g. BMI088 / ICM-20602 / MPU-6050
 *              class parts, per navigation/sensor_datasheets.py), own CS pins
 *   Gimbal   : one PWM channel, 1000-2000 us = -limit..+limit
 *   TLM      : UART TX (DMA), 115200 8N1 or faster
 *   Watchdog : IWDG, window sized to ~5x the control period
 *
 * Timing: drive fsw_core_step() from a hardware timer tick (50 Hz). Do the
 * SPI reads inside the tick or from a previous DMA transfer -- never block
 * on a bus without a timeout (return nonzero from read_imu instead; the
 * core treats that as a bad sample and FDIR handles it).
 */
#include "../fsw_core.h"

/* ---- board-specific, supply these -------------------------------------- */
/* Fill accel[3] (specific force, m/s^2) and gyro[3] (rad/s) in the BODY frame:
 * +Z along the thrust axis (up on the pad), pitch = rotation about +Y.
 * Apply the board's IMU-mounting rotation HERE so the core only sees body axes. */
extern int  board_spi_read_imu(int idx, float accel[3], float gyro[3], uint32_t timeout_us);
/* Returns 1 when a NEW GNSS fix is available: ECEF position [m] and its age [s] = (local time now) - (receiver
 * time-of-validity), from a PPS-disciplined clock. The age must be good to ~1 ms: at orbital speed 10 ms is ~65 m. */
extern int  board_gps_poll(double ecef_m[3], float *age_s);
extern void board_pwm_set_gimbal_us(uint16_t pulse_us);
extern void board_iwdg_kick(void);
extern size_t board_uart_write(const uint8_t *buf, size_t len);
extern void board_timer_start_hz(unsigned hz, void (*isr_cb)(void));
extern void board_engine_cutoff(void);               /* close the main valve / command the engine controller; must be idempotent */

static fsw_core_t g_core;
static volatile uint8_t g_tick;           /* set by timer ISR, consumed by main loop */

static int hal_read_imu(void *ctx, int idx, fsw_imu_sample_t *out)
{
    (void)ctx;
    return board_spi_read_imu(idx, out->accel, out->gyro, 300u);
}

static void hal_write_gimbal(void *ctx, float rad)
{
    float lim = ((fsw_core_t *)ctx)->cfg.gimbal_limit_rad;
    float x = rad / lim;                              /* -1..+1 */
    if (x > 1.0f) x = 1.0f;
    if (x < -1.0f) x = -1.0f;
    board_pwm_set_gimbal_us((uint16_t)(1500.0f + 500.0f * x));
}

static void hal_kick(void *ctx)               { (void)ctx; board_iwdg_kick(); }
static size_t hal_tlm(void *ctx, const uint8_t *b, size_t n) { (void)ctx; return board_uart_write(b, n); }

static void hal_engine_cutoff(void *ctx) { (void)ctx; board_engine_cutoff(); }

static void on_tick(void) { g_tick = 1; }

int main(void)
{
    fsw_hal_t hal;
    fsw_core_cfg_t cfg;

    /* TODO: clock, GPIO, SPI, UART, PWM, IWDG init (vendor HAL / CubeMX) */

    hal.ctx = &g_core;
    hal.read_imu = hal_read_imu;
    hal.write_gimbal = hal_write_gimbal;
    hal.kick_watchdog = hal_kick;
    hal.tlm_write = hal_tlm;
    hal.engine_cutoff = hal_engine_cutoff;      /* optional: only used by the guided stage */

    fsw_core_default_cfg(&cfg);
    /* cfg.nav_eci = 1; cfg.site_lat_rad / site_lon_rad / site_radius_m / launch_azimuth_rad = YOUR pad (the defaults are an
     * example site). After fsw_core_init(): fsw_core_set_stage2_mass(&g_core, <kg>) and fsw_core_attach_guidance(). */
    fsw_core_init(&g_core, &hal, &cfg);

    board_timer_start_hz(50u, on_tick);

    for (;;) {
        if (g_tick) {
            g_tick = 0;
            {
                double ecef[3];
                float age_s;
                if (board_gps_poll(ecef, &age_s)) {
                    fsw_core_gps_fix_ecef(&g_core, ecef, age_s);   /* cfg.nav_eci = 1; flat mode uses fsw_core_gps_fix_aged() */
                }
            }
            fsw_core_step(&g_core);
            /* TODO: poll UART RX -> fsw_core_cmd_arm()/launch()/abort() */
        }
        /* Guidance: the heavy PEG solve (double precision, libm) runs HERE, outside the 50 Hz tick, and only
         * when the control cycle has requested one (every ~2 s in the guided stage). Cost on a Cortex-M4F is
         * unmeasured (doubles are software); budget it on the target before trusting a 2 s cadence. The
         * polar state (r, v, gamma, mass, downrange) must come from an orbit-scale navigation filter via
         * fsw_core_set_polar_state(); the flat-Earth ESKF cannot provide it. Attach with
         * fsw_core_attach_guidance(&g_core, &g_guid) after fsw_guidance_init(). */
        fsw_core_guidance_service(&g_core);
        /* TODO: __WFI() */
    }
}
