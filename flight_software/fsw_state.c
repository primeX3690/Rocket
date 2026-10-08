#include "fsw_state.h"

static void enter(fsw_state_t *s, fsw_mode_t m, fsw_reason_t r)
{
    s->mode = m;
    s->last_reason = r;
    s->t_in_state_s = 0.0f;
    s->t_burn_s = 0.0f;
    s->thrust_count = 0;
    s->cutoff_count = 0;
    s->thrust_seen = (m == FSW_ASCENT_S1) ? 1u : 0u;  /* IGNITION already confirmed thrust */
    s->transitions++;
}

void fsw_state_init(fsw_state_t *s, const fsw_state_cfg_t *cfg)
{
    s->cfg = *cfg;
    s->mode = FSW_PAD_SAFE;
    s->last_reason = FSW_REASON_NONE;
    s->t_in_state_s = 0.0f;
    s->t_burn_s = 0.0f;
    s->thrust_count = 0;
    s->cutoff_count = 0;
    s->thrust_seen = 0;
    s->transitions = 0;
    s->rejected_cmds = 0;
}

int fsw_mode_is_thrusting(fsw_mode_t m)
{
    return (m == FSW_ASCENT_S1) || (m == FSW_ASCENT_S2);
}

int fsw_mode_is_terminal(fsw_mode_t m)
{
    return (m == FSW_ORBIT_INSERTED) || (m == FSW_ABORT);
}

static void thrust_phase(fsw_state_t *s, const fsw_inputs_t *in, float dt,
                         fsw_mode_t next, fsw_reason_t next_reason)
{
    if (!s->thrust_seen) {
        float thr = (s->mode == FSW_ASCENT_S2) ? s->cfg.thrust_accel_s2_mps2 : s->cfg.thrust_accel_mps2;
        if (in->accel_mps2 >= thr) {
            s->thrust_count++;
        } else {
            s->thrust_count = 0;
        }
        if (s->thrust_count >= s->cfg.confirm_ticks) {
            s->thrust_seen = 1;
            s->t_burn_s = 0.0f;
            s->cutoff_count = 0;
        } else if (s->t_in_state_s > s->cfg.ignition_timeout_s) {
            enter(s, FSW_ABORT, FSW_REASON_NO_IGNITION);
        }
        return;
    }

    s->t_burn_s += dt;
    if (in->accel_mps2 <= s->cfg.cutoff_accel_mps2) {
        s->cutoff_count++;
    } else {
        s->cutoff_count = 0;
    }
    if (s->cutoff_count >= s->cfg.confirm_ticks) {
        if (s->t_burn_s < s->cfg.min_burn_s) {
            enter(s, FSW_ABORT, FSW_REASON_PREMATURE_CUTOFF);
        } else {
            enter(s, next, next_reason);
        }
    }
}

fsw_mode_t fsw_state_step(fsw_state_t *s, const fsw_inputs_t *in, float dt)
{
    int any_cmd = (in->cmd_arm || in->cmd_disarm || in->cmd_launch);

    s->t_in_state_s += dt;

    if (fsw_mode_is_terminal(s->mode)) {
        if (any_cmd) {
            s->rejected_cmds++;
        }
        return s->mode;
    }

    if (in->cmd_abort || in->fdir_critical) {
        fsw_reason_t r = in->cmd_abort ? FSW_REASON_ABORT_CMD : FSW_REASON_FDIR_CRITICAL;
        if (s->mode == FSW_PAD_SAFE) {
            if (any_cmd) {
                s->rejected_cmds++;               /* refused, and counted */
            }
            return s->mode;                       /* already safe */
        }
        if (s->mode == FSW_PAD_ARMED) {
            enter(s, FSW_PAD_SAFE, r);            /* scrub, nothing lit */
        } else {
            enter(s, FSW_ABORT, r);
        }
        return s->mode;
    }

    switch (s->mode) {
    case FSW_PAD_SAFE:
        if (in->cmd_arm) {
            if (in->imu_trusted) { enter(s, FSW_PAD_ARMED, FSW_REASON_ARM); }
            else                 { s->rejected_cmds++; }
        } else if (in->cmd_launch || in->cmd_disarm) {
            s->rejected_cmds++;
        }
        break;
    case FSW_PAD_ARMED:
        if (in->cmd_disarm) {
            enter(s, FSW_PAD_SAFE, FSW_REASON_DISARM);
        } else if (in->cmd_launch) {
            if (in->imu_trusted) { enter(s, FSW_IGNITION, FSW_REASON_LAUNCH); }
            else                 { s->rejected_cmds++; }
        } else if (in->cmd_arm) {
            s->rejected_cmds++;
        }
        break;
    case FSW_IGNITION:
        if (any_cmd) { s->rejected_cmds++; }
        if (in->accel_mps2 >= s->cfg.thrust_accel_mps2) { s->thrust_count++; }
        else                                            { s->thrust_count = 0; }
        if (s->thrust_count >= s->cfg.confirm_ticks) {
            enter(s, FSW_ASCENT_S1, FSW_REASON_LIFTOFF);
        } else if (s->t_in_state_s > s->cfg.ignition_timeout_s) {
            enter(s, FSW_ABORT, FSW_REASON_NO_IGNITION);
        }
        break;
    case FSW_ASCENT_S1:
        if (any_cmd) { s->rejected_cmds++; }
        thrust_phase(s, in, dt, FSW_STAGE_SEP, FSW_REASON_MECO);
        break;
    case FSW_STAGE_SEP:
        if (any_cmd) { s->rejected_cmds++; }
        if (s->t_in_state_s >= s->cfg.sep_delay_s) {
            enter(s, FSW_ASCENT_S2, FSW_REASON_SEP_DONE);
        }
        break;
    case FSW_ASCENT_S2:
        if (any_cmd) { s->rejected_cmds++; }
        thrust_phase(s, in, dt, FSW_COAST, FSW_REASON_SECO);
        break;
    case FSW_COAST:
        if (any_cmd) { s->rejected_cmds++; }
        if (in->insertion_ok) {
            enter(s, FSW_ORBIT_INSERTED, FSW_REASON_INSERTION);
        }
        break;
    default:
        break;
    }
    return s->mode;
}
