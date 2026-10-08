#include "fsw_pitch_program.h"
#include "fsw_math.h"

int fsw_pitch_program_set(fsw_pitch_program_t *p, const float *t, const float *phi, int n, float max_rate_rps)
{
    int i;
    p->n = 0;
    if (n < 1 || n > FSW_PITCH_PROGRAM_MAX) { return 1; }
    for (i = 0; i < n; i++) {
        if (!fsw_isfinitef(t[i]) || !fsw_isfinitef(phi[i]) || fsw_absf(phi[i]) > FSW_PI_F) { return 2; }
        if (i > 0) {
            float dt = t[i] - t[i - 1];
            if (!(dt > 0.0f)) { return 3; }
            if (fsw_absf(phi[i] - phi[i - 1]) > max_rate_rps * dt) { return 4; }
        }
    }
    for (i = 0; i < n; i++) { p->t[i] = t[i]; p->phi[i] = phi[i]; }
    p->n = n;
    return 0;
}

float fsw_pitch_program_eval(const fsw_pitch_program_t *p, float ts)
{
    int i;
    if (p->n < 1) { return 0.0f; }
    if (!(ts > p->t[0])) { return p->phi[0]; }
    for (i = 1; i < p->n; i++) {
        if (ts <= p->t[i]) {
            float f = (ts - p->t[i - 1]) / (p->t[i] - p->t[i - 1]);
            return p->phi[i - 1] + f * (p->phi[i] - p->phi[i - 1]);
        }
    }
    return p->phi[p->n - 1];
}
