/* fsw_eskf.h -- 15-state strapdown error-state EKF, C port of
 * navigation/attitude_ekf.py (StrapdownAttitudeEKF).
 *
 * Nominal state : q (body->nav, w,x,y,z), v, p (nav frame, Z up), bg, ba
 * Error state   : [dtheta(3) dv(3) dp(3) dbg(3) dba(3)]   (same order as Python)
 * Propagation   : gyro-driven quaternion, accelerometer-driven v/p, flat-Earth
 *                 constant gravity (identical simplification to the Python ref).
 * Measurement   : 3-axis position fix, Joseph-form covariance update.
 *
 * Differences from the Python reference (all deliberate, all tested):
 *   - float32 instead of float64.  P is re-symmetrised after every predict/update
 *     and its diagonal is floored at 1e-14 (float32 can otherwise lose positive-
 *     definiteness over thousands of steps).
 *   - Optional NIS (chi-square) innovation gate on position fixes, OFF by default
 *     so the numbers match Python exactly; fsw_core turns it on.  After
 *     `max_consecutive_rejects` rejected fixes in a row the next fix is accepted
 *     ungated (otherwise a filter that has genuinely drifted would lock itself out).
 *   - A singular innovation covariance rejects the update instead of inverting garbage.
 *   - fsw_eskf_healthy(): finite state, |q|~1, finite non-negative P diagonal.
 *   - fsw_eskf_align_accel(): static roll/pitch leveling from the accelerometer.
 *
 * No heap, no libm, no globals; all scratch lives inside the struct (~3.4 kB).
 * Single-threaded: do not call from two contexts on the same instance. */
#ifndef FSW_ESKF_H
#define FSW_ESKF_H
#include <stdint.h>

#define FSW_ESKF_N 15

/* Gravity models.
 *  FLAT    : constant gravity_nav vector, flat Earth -- identical to the Python reference.
 *  CENTRAL : inverse-square  g = -mu p / |p|^3  with its gradient in the covariance (inertial frame,
 *            origin at the Earth's centre; p, v are ECI-like and MUST be double: |p| ~ 6.4e6 m is
 *            beyond float32's 0.5 m resolution).
 *  CENTRAL_J2 : CENTRAL plus the J2 oblateness acceleration about the axis `polar_axis`
 *            (~0.014 m/s^2 at 200 km: ~3 m/s of velocity error over a 200 s burn if ignored).
 *            The J2 gradient is neglected in the covariance (0.3 % of the point-mass gradient). */
#define FSW_ESKF_GRAV_FLAT        0
#define FSW_ESKF_GRAV_CENTRAL     1
#define FSW_ESKF_GRAV_CENTRAL_J2  2

typedef struct {
    float gyro_noise_psd, gyro_bias_psd, accel_noise_psd, accel_bias_psd;
    float gravity_nav[3];              /* FLAT only                              */
    float nis_gate;                    /* <= 0 disables gating                   */
    uint8_t max_consecutive_rejects;   /* gating lock-out protection             */
    uint8_t gravity_model;             /* FSW_ESKF_GRAV_*                        */
    double  mu, j2, r_eq;              /* CENTRAL / CENTRAL_J2                   */
    double  polar_axis[3];             /* unit vector of Earth's spin axis in the filter frame (J2) */
} fsw_eskf_cfg_t;

typedef struct {
    fsw_eskf_cfg_t cfg;
    float q[4], bg[3], ba[3];
    double v[3], p[3];                 /* nominal velocity / position: double (see gravity models) */
    float P[FSW_ESKF_N][FSW_ESKF_N];
    float F[FSW_ESKF_N][FSW_ESKF_N];   /* scratch */
    float T[FSW_ESKF_N][FSW_ESKF_N];   /* scratch */
    uint32_t n_predict, n_update, n_rejected;
    uint8_t consecutive_rejects;
    uint8_t last_update_accepted;
} fsw_eskf_t;

/* Same defaults as the Python class (gravity 9.80665 down, gate disabled). */
void fsw_eskf_default_cfg(fsw_eskf_cfg_t *cfg);

/* Any of q0/v0/p0/bg0/ba0 may be NULL (-> identity / zeros), like the Python ctor. */
void fsw_eskf_init(fsw_eskf_t *e, const fsw_eskf_cfg_t *cfg,
                   const float *q0, const double *v0, const double *p0,
                   const float *bg0, const float *ba0);

void fsw_eskf_predict(fsw_eskf_t *e, const float gyro[3], const float accel[3], float dt);

/* Returns 1 if the fix was applied, 0 if rejected (gated / singular / bad input). */
int  fsw_eskf_update_position(fsw_eskf_t *e, const double pos[3], float r_pos);

/* Re-initialise position and velocity from an external source (hand-off, re-acquisition): sets the
 * nominal state, DECORRELATES rows/columns 3..8 from the rest of P (setting only the diagonal of a
 * covariance that still carries old cross terms makes it non-positive-definite) and sets the diagonal. */
void fsw_eskf_reset_pv(fsw_eskf_t *e, const double p[3], const double v[3], float sigma_p, float sigma_v);
/* Set the attitude covariance (body-frame small-angle errors about x, y, z), decorrelated from the rest. */
void fsw_eskf_set_att_sigma(fsw_eskf_t *e, float sx, float sy, float sz);

int  fsw_eskf_healthy(const fsw_eskf_t *e);

/* Level roll/pitch (yaw = 0) from a static specific-force measurement, using
 * the Z-up convention: at rest f_body = R^T * (0,0,+g). Writes e->q only.
 * Returns 0 and leaves q unchanged if |f| is not within 20 % of g. */
int  fsw_eskf_align_accel(fsw_eskf_t *e, const float accel[3]);

/* Euler angles (rad), ZYX convention, same formulas as Python's euler_deg. NOTE: pitch here is
 * asin()-based and singular at +-90 deg; use fsw_eskf_pitch() for control/guidance. */
void  fsw_eskf_euler(const fsw_eskf_t *e, float *roll, float *pitch, float *yaw);
/* Thrust-axis pitch from vertical about +Y (rad, full +-pi range, no gimbal lock). */
float fsw_eskf_pitch(const fsw_eskf_t *e);

/* Flat getters (used by the Python equivalence test via ctypes). */
unsigned fsw_eskf_sizeof(void);
void fsw_eskf_get(const fsw_eskf_t *e, float *q4, float *v3, float *p3, float *bg3, float *ba3);   /* v,p rounded to float */
void fsw_eskf_get_pv(const fsw_eskf_t *e, double *v3, double *p3);
/* Gravity acceleration the filter uses at position p (test hook). */
void fsw_eskf_gravity(const fsw_eskf_t *e, const double p[3], double g[3]);
void fsw_eskf_get_P(const fsw_eskf_t *e, float *out225);
void fsw_eskf_set_P_diag(fsw_eskf_t *e, int i, float value);   /* test hook */

#endif
