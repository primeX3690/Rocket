/* fsw_core.h -- one deterministic flight cycle, called at a fixed rate
 * (e.g. from a 50 Hz timer tick or a scheduler task).
 *
 * Cycle: read 3 IMUs (3-axis accel + gyro each) -> per-channel FDIR on all 6
 * axes -> 2-of-3 votes -> flight-mode state machine -> navigation -> TVC PID
 * (thrusting modes only) -> gimbal write -> telemetry -> watchdog service.
 *
 * Navigation:
 *   PAD_SAFE / PAD_ARMED : accelerometer leveling (roll/pitch) + gyro-bias
 *                          averaging while the vehicle is at rest.
 *   IGNITION             : alignment frozen.
 *   from ASCENT_S1 on    : the 15-state ESKF (fsw_eskf.c, port of
 *                          navigation/attitude_ekf.py) is initialised from the
 *                          pad alignment and free-runs on the voted IMU data,
 *                          fusing position fixes given to fsw_core_gps_fix().
 *
 * HONEST SCOPE: PEG guidance (guidance/peg.py) is NOT ported to C yet --
 * guidance enters via fsw_core_set_pitch_cmd(). The navigation frame is the
 * same locally-flat, constant-gravity frame as the Python ESKF; that is valid
 * for the ascent-phase windows it was written for, not for orbit-scale
 * navigation. */
#ifndef FSW_CORE_H
#define FSW_CORE_H
#include "fsw_hal.h"
#include "fsw_fdir.h"
#include "fsw_state.h"
#include "fsw_watchdog.h"
#include "fsw_eskf.h"
#include "fsw_guidance.h"
#include "fsw_frames.h"
#include "fsw_pitch_program.h"
#include "../embedded/pid_controller.h"

#define FSW_TASK_SENSORS  (1u << 0)
#define FSW_TASK_FDIR     (1u << 1)
#define FSW_TASK_STATE    (1u << 2)
#define FSW_TASK_CONTROL  (1u << 3)
#define FSW_TASK_TLM      (1u << 4)
#define FSW_TASK_ALL      0x1Fu

#define FSW_GPS_HIST      32   /* navigation positions kept for latency compensation (0.64 s at 50 Hz) */

#define FSW_TLM_TYPE_HK   0x01u
#define FSW_TLM_HK_LEN    40u

/* guid_flags bits (telemetry byte 32) */
#define FSW_GUID_ATTACHED     (1u << 0)
#define FSW_GUID_VALID        (1u << 1)   /* a solution is being flown                         */
#define FSW_GUID_DEGRADED     (1u << 2)   /* solution did not converge (target unreachable?)   */
#define FSW_GUID_LOST         (1u << 3)   /* guided stage but no valid solution: flying the external pitch program */
#define FSW_GUID_STALE_STATE  (1u << 4)   /* polar state too old to start a new solve           */
#define FSW_GUID_CUTOFF_SENT  (1u << 5)
#define FSW_GUID_INSERTED     (1u << 6)   /* polar state inside the insertion window            */

/* fault_flags bits */
#define FSW_FLAG_IMU0_FAULT   (1u << 0)   /* any axis of IMU i not OK        */
#define FSW_FLAG_IMU1_FAULT   (1u << 1)
#define FSW_FLAG_IMU2_FAULT   (1u << 2)
#define FSW_FLAG_DEGRADED     (1u << 3)   /* some axis voted with < 3 inputs */
#define FSW_FLAG_TRUST_LOST   (1u << 4)   /* some axis < min_trusted_imus    */
#define FSW_FLAG_WDG_TRIPPED  (1u << 5)
#define FSW_FLAG_NAV_FAULT    (1u << 6)   /* ESKF failed its health check    */
#define FSW_FLAG_NAV_STALE    (1u << 7)   /* ESKF could not be propagated    */

typedef struct {
    float                dt_s;
    fsw_monitor_cfg_t    accel_mon, gyro_mon;     /* applied to all 3 axes of each IMU */
    float                accel_vote_tol, gyro_vote_tol;
    uint8_t              min_trusted_imus;        /* default policy: 2        */
    uint16_t             crit_persist_ticks;      /* trust-lost / nav-fault debounce */
    fsw_state_cfg_t      state;
    float                kp, ki, kd, gimbal_limit_rad;
    uint16_t             tlm_every_n;
    uint16_t             wdg_miss_limit;
    fsw_eskf_cfg_t       eskf;
    float                gps_r_pos;               /* position-fix variance [m^2] */
    float                gps_default_latency_s;   /* age assumed by fsw_core_gps_fix(); 0 = fix is fresh */
    float                pad_gyro_bias_tau_s;     /* gyro bias averaging time constant on the pad */
    float                pad_rest_rate_rps;       /* only average bias while |w| below this */
    uint8_t              mass_from_accel;         /* ECI + guidance: guided-stage mass from the measured specific force (default 1), else the fuel-gauge model */
    float                att_sigma_rollpitch_rad; /* 1-sigma attitude uncertainty at liftoff: leveling accuracy (averaged accelerometer) */
    float                att_sigma_yaw_rad;       /* ... azimuth: not observable on the pad; comes from the pad survey */
    /* ---- orbit-scale (inertial) navigation: nav_eci = 1 ----
     * Filter frame = non-rotating, Earth-centred, pad-aligned (see fsw_frames.h). Gravity is inverse-square
     * (+J2). Position fixes come in ECEF via fsw_core_gps_fix_ecef(). The polar state PEG needs is derived
     * from the filter every cycle. nav_eci = 0 keeps the flat-Earth, pad-frame behaviour. Defaults describe
     * an EXAMPLE site (13.72 N, 80.23 E, due-east launch): set your own. */
    uint8_t              nav_eci;
    uint8_t              nav_j2;
    double               site_lat_rad, site_lon_rad, site_radius_m, launch_azimuth_rad, omega_earth;
    float                pitch_cmd_rate_limit_rps;/* slew limit on the commanded pitch (any source) */
    float                guid_state_max_age_s;    /* polar state older than this is not used to start a solve */
    float                uncal_gyro_bias_sigma_rps; /* 1-sigma prior for gyro bias when NOT pad-calibrated (datasheet zero-rate tolerance; BMI088: 1 dps) */
    uint8_t              pad_bias_cal_enabled;    /* 0: ESKF starts with zero gyro bias and must estimate it (needs GPS) */
} fsw_core_cfg_t;

typedef struct {
    fsw_hal_t         hal;
    fsw_core_cfg_t    cfg;
    fsw_monitor_t     acc_mon[3][3], gyr_mon[3][3];   /* [imu][axis] */
    fsw_state_t       st;
    fsw_wdg_t         wdg;
    PIDController     pid;
    fsw_eskf_t        nav;
    fsw_inputs_t      in;                  /* pending commands, cleared per cycle */
    fsw_vote_t        acc_vote[3], gyr_vote[3];       /* [axis] */
    float             pitch_est, pitch_cmd, pitch_cmd_ext, gimbal_cmd;   /* pitch_cmd = slew-limited value fed to the PID */
    /* --- guidance (optional) --- */
    fsw_guidance_t   *guid;
    fsw_guid_solution_t guid_cache;                 /* last consistent copy read from the seqlock */
    volatile uint32_t polar_seq;                    /* odd while the nav bridge is writing the polar state */
    fsw_peg_state_t   polar;                        /* r, v, gamma, mass: from the orbit-scale navigation */
    double            polar_psi_rad;                /* downrange angle: local horizontal vs pad horizontal */
    double            polar_stamp_s;
    uint8_t           polar_valid;
    double            last_guid_req_s;
    fsw_pitch_program_t pprog;                      /* first-stage open-loop pitch table (optional) */
    uint32_t          liftoff_cycle;
    uint8_t           guid_request, guid_flags, cutoff_sent;
    uint32_t          guid_serviced;
    float             pad_q[4], pad_bg[3], pad_f[3];
    uint8_t           pad_aligned, pad_bg_seeded, pad_f_seeded;
    uint8_t           nav_active, nav_ok, nav_stale;
    uint16_t          crit_count, nav_bad_count;
    double            gps_pos[3];
    float             gps_age_s;
    uint8_t           gps_is_ecef;
    fsw_site_t        site;
    double            stage2_m0_kg, mass_est_kg;   /* mass model for the polar state (see fsw_core_set_stage2_mass) */
    double            s2_accel_lp;                 /* low-passed guided-stage axial specific force */
    uint8_t           s2_accel_seeded;
    uint8_t           gps_pending;
    double            nav_pos_hist[FSW_GPS_HIST][3];   /* post-predict position of nav cycle n at [n % FSW_GPS_HIST] */
    uint32_t          nav_cycles;                       /* index of the newest stored entry */
    uint32_t          gps_dropped;                      /* fixes discarded: too old / bad age / nav not ready */
    uint32_t          gps_compensated;
    uint32_t          cycle;
    uint8_t           tlm_seq;
    uint8_t           fault_flags;
    uint32_t          tlm_frames_sent;
} fsw_core_t;

void fsw_core_default_cfg(fsw_core_cfg_t *cfg);
void fsw_core_init(fsw_core_t *c, const fsw_hal_t *hal, const fsw_core_cfg_t *cfg);

void fsw_core_cmd_arm(fsw_core_t *c);
void fsw_core_cmd_disarm(fsw_core_t *c);
void fsw_core_cmd_launch(fsw_core_t *c);
void fsw_core_cmd_abort(fsw_core_t *c);
void fsw_core_set_insertion_ok(fsw_core_t *c, int ok);
void fsw_core_set_pitch_cmd(fsw_core_t *c, float pitch_rad);   /* external pitch program (from vertical, rad); slew-limited */

/* ---- guidance ----
 * fsw_core_attach_guidance: the guided stage is ASCENT_S2. While a valid solution exists the
 * pitch command (from vertical, in the pad-nav frame) is  phi = pi/2 - theta(t) + psi  where
 * theta is PEG's thrust pitch from the LOCAL horizontal and psi the downrange angle that
 * rotates the local horizontal away from the pad horizontal. With no valid solution the
 * external pitch program is flown and FSW_GUID_LOST is raised.
 *
 * fsw_core_set_polar_state: call from the navigation task with the state in the polar frame
 * (r from Earth centre [m], speed [m/s], flight-path angle [rad], mass [kg], downrange psi [rad]).
 * THE ESKF IS FLAT-EARTH: it cannot supply this at orbit scale; an inertial-frame/orbit-scale
 * navigation filter is a prerequisite for flying this on a vehicle.
 *
 * fsw_core_guidance_service: HEAVY (one PEG solve). Call from a low-priority task / idle loop,
 * NEVER from the control interrupt. It only does work when the control cycle has requested a solve. */
/* First-stage steering: a pitch-vs-time table, flown from liftoff through ASCENT_S1 and STAGE_SEP, and held as
 * the fallback in ASCENT_S2 if guidance has no valid solution. Returns 0 on success (see fsw_pitch_program_set;
 * the slew limit of the attitude loop is used as the maximum slope). With no table the external pitch command
 * (fsw_core_set_pitch_cmd) is used as before. */
int  fsw_core_set_pitch_program(fsw_core_t *c, const float *t_s, const float *phi_rad, int n);
void fsw_core_attach_guidance(fsw_core_t *c, fsw_guidance_t *g);
void fsw_core_set_polar_state(fsw_core_t *c, double r_m, double v_mps, double gamma_rad, double mass_kg, double psi_rad);
void fsw_core_guidance_service(fsw_core_t *c);
/* Position fix in the launch-pad navigation frame (x,y horizontal, z up), metres.
 * Latest fix wins; consumed on the next cycle where navigation is active.
 *
 * LATENCY: a receiver reports where the vehicle WAS when it measured, typically
 * 100+ ms ago (~30 m at 300 m/s). `age_s` is the time between that measurement
 * instant and THIS call (derive it from the receiver's time-of-validity vs the
 * PPS-disciplined local clock). The core moves the fix forward to "now" using
 * the ESKF's own displacement over those `age_s` seconds (z' = z + p_now - p_then,
 * accurate to velocity_error * age) and inflates the measurement variance by
 * age^2 * sigma_v^2. Fixes older than (FSW_GPS_HIST-1) cycles, with a negative or
 * non-finite age, or arriving before navigation started, are dropped (gps_dropped).
 * fsw_core_gps_fix() uses cfg.gps_default_latency_s. */
void fsw_core_gps_fix_aged(fsw_core_t *c, const float pos_m[3], float age_s);
void fsw_core_gps_fix(fsw_core_t *c, const float pos_m[3]);
/* ECI mode (cfg.nav_eci = 1): receiver position in ECEF [m], measured `age_s` before this call. The core
 * rotates it into the filter frame at the MEASUREMENT time (Earth rotation: 465 m/s at the equator, i.e. 9 m
 * per 20 ms), interpolates the stored navigation history at the fractional age (at orbital speed one 20 ms
 * cycle is 154 m, so rounding the age to a whole cycle is not good enough), and then proceeds as
 * fsw_core_gps_fix_aged(). In flat mode this is dropped, and the flat variants are dropped in ECI mode. */
void fsw_core_gps_fix_ecef(fsw_core_t *c, const double ecef_m[3], float age_s);
/* ECI mode: initial mass of the guided stage. The polar state's mass is, by default, the EFFECTIVE mass
 * m_eff = F_nominal / a_measured (low-passed axial specific force): PEG's dynamics only depend on F/m, so this
 * absorbs a thrust dispersion that a fuel-gauge model (m0 - mdot*t) turns into a ~4 % mass error and an early
 * cutoff (measured: 4 % weak engines -> 45 m/s short with the gauge model). cfg.mass_from_accel = 0 selects
 * the gauge model. Without guidance attached no polar state is derived. */
void fsw_core_set_stage2_mass(fsw_core_t *c, double m0_kg);

void fsw_core_step(fsw_core_t *c);

#endif
