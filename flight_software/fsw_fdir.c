#include "fsw_fdir.h"

static int is_finite_f(float x)
{
    /* NaN fails x == x; +/-Inf fails the range compare. */
    return (x == x) && (x <= 3.4e38f) && (x >= -3.4e38f);
}

static float absf(float x) { return (x < 0.0f) ? -x : x; }

void fsw_monitor_init(fsw_monitor_t *m, const fsw_monitor_cfg_t *cfg)
{
    m->cfg = *cfg;
    m->prev_raw = 0.0f;
    m->last_good = 0.0f;
    m->has_prev = 0;
    m->same_count = 0;
    m->bad_count = 0;
    m->good_count = 0;
    m->fault_events = 0;
    m->status = FSW_SENSOR_OK;
}

fsw_sensor_status_t fsw_monitor_update(fsw_monitor_t *m, float x, float dt)
{
    int bad = 0;

    if (m->status == FSW_SENSOR_EXCLUDED) {
        return m->status;
    }

    if (!is_finite_f(x)) {
        bad = 1;
        m->same_count = 0;
    } else {
        if (x < m->cfg.min || x > m->cfg.max) {
            bad = 1;
        }
        if (m->has_prev && x == m->prev_raw) {
            if (m->same_count < 65535u) {
                m->same_count++;
            }
        } else {
            m->same_count = 0;
        }
        if (m->cfg.stuck_samples > 0u && m->same_count >= m->cfg.stuck_samples) {
            bad = 1;
        }
        /* Rate check only while healthy, against the last GOOD sample, so one
         * spike costs exactly one bad sample (not two) and a real step change
         * after a fault cannot lock the channel out. */
        if (m->status == FSW_SENSOR_OK && m->has_prev && dt > 0.0f &&
            absf(x - m->last_good) / dt > m->cfg.max_rate) {
            bad = 1;
        }
        m->prev_raw = x;
        m->has_prev = 1;
    }

    if (m->status == FSW_SENSOR_OK) {
        if (bad) {
            m->bad_count++;
            m->good_count = 0;
            if (m->bad_count >= m->cfg.fail_persist) {
                m->status = FSW_SENSOR_FAULT;
                m->fault_events++;
                if (m->fault_events > m->cfg.max_fault_events) {
                    m->status = FSW_SENSOR_EXCLUDED;
                }
                m->good_count = 0;
            }
        } else {
            m->bad_count = 0;
            m->last_good = x;
        }
    } else { /* FAULT */
        if (bad) {
            m->good_count = 0;
        } else {
            m->good_count++;
            m->last_good = x;
            if (m->good_count >= m->cfg.recover_persist) {
                m->status = FSW_SENSOR_OK;
                m->bad_count = 0;
                m->good_count = 0;
            }
        }
    }
    return m->status;
}

fsw_vote_t fsw_vote3(const float v[3], const uint8_t healthy[3], float tol)
{
    fsw_vote_t r;
    float h[3];
    uint8_t idx[3];
    int n = 0, i;

    r.value = 0.0f;
    r.n_used = 0;
    r.used_mask = 0;

    for (i = 0; i < 3; i++) {
        if (healthy[i] && is_finite_f(v[i])) {
            h[n] = v[i];
            idx[n] = (uint8_t)i;
            n++;
        }
    }

    if (n == 0) {
        return r;
    }
    if (n == 1) {
        r.value = h[0];
        r.n_used = 1;
        r.used_mask = (uint8_t)(1u << idx[0]);
        return r;
    }
    if (n == 2) {
        if (absf(h[0] - h[1]) <= tol) {
            r.value = 0.5f * (h[0] + h[1]);
            r.n_used = 2;
            r.used_mask = (uint8_t)((1u << idx[0]) | (1u << idx[1]));
        } else {
            /* Two healthy channels that disagree: cannot tell which is right. */
            r.value = 0.5f * (h[0] + h[1]);
            r.n_used = 0;
        }
        return r;
    }

    /* n == 3: median, then keep everything within tol of it. */
    {
        float med;
        float sum = 0.0f;
        float a = h[0], b = h[1], c = h[2];
        if ((a >= b && a <= c) || (a <= b && a >= c)) {
            med = a;
        } else if ((b >= a && b <= c) || (b <= a && b >= c)) {
            med = b;
        } else {
            med = c;
        }
        for (i = 0; i < 3; i++) {
            if (absf(h[i] - med) <= tol) {
                sum += h[i];
                r.n_used++;
                r.used_mask = (uint8_t)(r.used_mask | (1u << idx[i]));
            }
        }
        r.value = sum / (float)r.n_used;
    }
    return r;
}
