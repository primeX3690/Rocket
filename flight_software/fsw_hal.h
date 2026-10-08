/* fsw_hal.h -- the ONLY interface between flight logic and hardware.
 * Everything else in flight_software/ is board-independent and is tested on
 * the host. Porting to a real board = implementing these four callbacks
 * (see port/hal_stm32_template.c).
 *
 * Body-frame convention (matches navigation/attitude_ekf.py): +Z is the
 * vehicle's AXIAL (thrust) axis, pointing "up" on the pad; pitch is a rotation
 * about +Y. IMU samples are SPECIFIC FORCE (what an accelerometer reads: +g
 * along +Z at rest on the pad, thrust/mass during burn), in m/s^2, and body
 * rates in rad/s. */
#ifndef FSW_HAL_H
#define FSW_HAL_H
#include <stddef.h>
#include <stdint.h>

typedef struct {
    float accel[3];   /* body x,y,z specific force [m/s^2] */
    float gyro[3];    /* body x,y,z angular rate   [rad/s] */
} fsw_imu_sample_t;

#define FSW_AXIS_PITCH_GYRO 1   /* gyro[1] = rotation about +Y */
#define FSW_AXIS_AXIAL      2   /* accel[2] = along the thrust axis */

typedef struct {
    void  *ctx;
    /* idx 0..2. Return 0 on success, nonzero on bus error/timeout. */
    int    (*read_imu)(void *ctx, int idx, fsw_imu_sample_t *out);
    void   (*write_gimbal)(void *ctx, float pitch_rad);
    void   (*kick_watchdog)(void *ctx);
    size_t (*tlm_write)(void *ctx, const uint8_t *buf, size_t len);
    /* OPTIONAL (may be NULL): commanded main-engine cutoff, called once when guidance
     * time-to-go expires in the guided stage. Must be idempotent and fast. */
    void   (*engine_cutoff)(void *ctx);
} fsw_hal_t;

#endif
