/* fsw_state.h -- flight-mode state machine.
 *
 *  PAD_SAFE -> PAD_ARMED -> IGNITION -> ASCENT_S1 -> STAGE_SEP -> ASCENT_S2
 *           -> COAST -> ORBIT_INSERTED          (nominal, terminal)
 *  any pad/flight state --abort/critical-fault--> PAD_SAFE (pad) or ABORT
 *
 * Transitions are driven by MEASURED evidence (voted specific force), not
 * only by timers: ignition and burnout must be confirmed by acceleration
 * for `confirm_ticks` consecutive cycles. Burnout earlier than
 * `min_burn_s` after confirmed thrust is treated as an engine failure
 * (premature cutoff) and aborts. Commands that are illegal in the current
 * mode are rejected and counted, never silently honoured.
 * Pure logic: no I/O, no heap, no globals. */
#ifndef FSW_STATE_H
#define FSW_STATE_H
#include <stdint.h>

typedef enum {
    FSW_PAD_SAFE = 0,
    FSW_PAD_ARMED,
    FSW_IGNITION,
    FSW_ASCENT_S1,
    FSW_STAGE_SEP,
    FSW_ASCENT_S2,
    FSW_COAST,
    FSW_ORBIT_INSERTED,
    FSW_ABORT
} fsw_mode_t;

typedef enum {
    FSW_REASON_NONE = 0,
    FSW_REASON_ARM,
    FSW_REASON_DISARM,
    FSW_REASON_LAUNCH,
    FSW_REASON_LIFTOFF,
    FSW_REASON_NO_IGNITION,
    FSW_REASON_MECO,
    FSW_REASON_SEP_DONE,
    FSW_REASON_SECO,
    FSW_REASON_INSERTION,
    FSW_REASON_ABORT_CMD,
    FSW_REASON_FDIR_CRITICAL,
    FSW_REASON_PREMATURE_CUTOFF
} fsw_reason_t;

typedef struct {
    float    thrust_accel_mps2;   /* liftoff/S1: must exceed 1 g on the pad (specific force)            */
    float    thrust_accel_s2_mps2;/* upper stage: specific force is thrust/m and may be << 1 g (TWR<1
                                   * stages are normal in orbit); must still exceed cutoff_accel_mps2 */
    float    cutoff_accel_mps2;   /* at/below this while thrusting => cutoff     */
    float    ignition_timeout_s;
    float    min_burn_s;
    float    sep_delay_s;
    uint16_t confirm_ticks;
} fsw_state_cfg_t;

typedef struct {
    float   accel_mps2;      /* voted axial specific force                  */
    uint8_t cmd_arm, cmd_disarm, cmd_launch, cmd_abort;
    uint8_t imu_trusted;     /* enough agreeing IMU channels right now      */
    uint8_t fdir_critical;   /* trust lost persistently                     */
    uint8_t insertion_ok;    /* guidance says target orbit achieved         */
} fsw_inputs_t;

typedef struct {
    fsw_state_cfg_t cfg;
    fsw_mode_t      mode;
    fsw_reason_t    last_reason;
    float           t_in_state_s;
    float           t_burn_s;
    uint16_t        thrust_count, cutoff_count;
    uint8_t         thrust_seen;
    uint16_t        transitions;
    uint16_t        rejected_cmds;
} fsw_state_t;

void       fsw_state_init(fsw_state_t *s, const fsw_state_cfg_t *cfg);
fsw_mode_t fsw_state_step(fsw_state_t *s, const fsw_inputs_t *in, float dt);
int        fsw_mode_is_thrusting(fsw_mode_t m);   /* S1 or S2: TVC active   */
int        fsw_mode_is_terminal(fsw_mode_t m);

#endif
