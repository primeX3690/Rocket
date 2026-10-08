/* fsw_fdir.h -- Fault Detection, Isolation & Recovery for redundant sensors.
 *
 * Two layers:
 *  1. fsw_monitor_t : per-channel health (range, rate-of-change, stuck-at,
 *     NaN/Inf), with DEBOUNCE (a fault needs N consecutive bad samples) and
 *     RECOVERY (M consecutive good samples clear it). A single transient
 *     therefore never latches -- this is the lesson of the SSLV-D1 anomaly
 *     modelled in guidance/anomaly_arbitration.py. A channel that keeps
 *     faulting (> max_fault_events) is finally EXCLUDED permanently.
 *  2. fsw_vote3 : 3-channel voter. Outliers vs. the median are voted out
 *     even if their own monitor still says OK (catches slow drift/bias).
 *
 * Do NOT build with -ffast-math: NaN/Inf detection relies on IEEE compares. */
#ifndef FSW_FDIR_H
#define FSW_FDIR_H
#include <stdint.h>

typedef struct {
    float    min, max;            /* plausible range                          */
    float    max_rate;            /* max |dx/dt| in units/s vs last good      */
    uint16_t stuck_samples;       /* identical consecutive samples => stuck   */
    uint16_t fail_persist;        /* consecutive bad samples => FAULT         */
    uint16_t recover_persist;     /* consecutive good samples => back to OK   */
    uint8_t  max_fault_events;    /* more FAULT declarations => EXCLUDED      */
} fsw_monitor_cfg_t;

typedef enum {
    FSW_SENSOR_OK = 0,
    FSW_SENSOR_FAULT,
    FSW_SENSOR_EXCLUDED
} fsw_sensor_status_t;

typedef struct {
    fsw_monitor_cfg_t   cfg;
    float               prev_raw;
    float               last_good;
    uint8_t             has_prev;
    uint16_t            same_count;
    uint16_t            bad_count;
    uint16_t            good_count;
    uint8_t             fault_events;
    fsw_sensor_status_t status;
} fsw_monitor_t;

void                fsw_monitor_init(fsw_monitor_t *m, const fsw_monitor_cfg_t *cfg);
fsw_sensor_status_t fsw_monitor_update(fsw_monitor_t *m, float x, float dt);

typedef struct {
    float   value;      /* mean of the channels that were used              */
    uint8_t n_used;     /* 0 = nothing trustworthy, 1 = no cross-check, 2-3 */
    uint8_t used_mask;  /* bit i set => channel i contributed               */
} fsw_vote_t;

/* healthy[i] != 0 => channel i is eligible (its monitor is OK).
 * tol = max allowed distance from the consensus value. */
fsw_vote_t fsw_vote3(const float v[3], const uint8_t healthy[3], float tol);

#endif
