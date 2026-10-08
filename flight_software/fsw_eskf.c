#include "fsw_eskf.h"
#include "fsw_math.h"
#include <string.h>

#define N FSW_ESKF_N
#define P_DIAG_FLOOR 1e-14f
#define P_DIAG_CEIL  1e12f

/* ---------------- small helpers (3-vectors / quaternions) ---------------- */

static void quat_normalize(float q[4])
{
    float n2 = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
    float n = fsw_sqrtf(n2);
    if (n > 0.0f && fsw_isfinitef(n)) {
        float inv = 1.0f / n;
        q[0] *= inv; q[1] *= inv; q[2] *= inv; q[3] *= inv;
    } else {
        q[0] = 1.0f; q[1] = 0.0f; q[2] = 0.0f; q[3] = 0.0f;   /* unrecoverable; health check flags NaN separately */
    }
}

static void quat_mult(const float a[4], const float b[4], float out[4])
{
    out[0] = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3];
    out[1] = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2];
    out[2] = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1];
    out[3] = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0];
}

/* Exact quaternion for a rotation vector. Series for small angles (accurate
 * to float precision below 0.5 rad, and exact-limit safe at 0), libm-free
 * sin/cos beyond that. */
static void quat_from_rotvec(const float rv[3], float out[4])
{
    float a2 = rv[0]*rv[0] + rv[1]*rv[1] + rv[2]*rv[2];
    float angle = fsw_sqrtf(a2);
    float c, s_over_angle;
    if (angle < 0.5f) {
        float h2 = 0.25f * a2;     /* (angle/2)^2 */
        c = 1.0f - h2 * (0.5f - h2 * (1.0f/24.0f - h2 * (1.0f/720.0f - h2 * (1.0f/40320.0f))));
        s_over_angle = 0.5f * (1.0f - h2 * (1.0f/6.0f - h2 * (1.0f/120.0f - h2 * (1.0f/5040.0f - h2 * (1.0f/362880.0f)))));
    } else {
        float h = 0.5f * angle;
        c = fsw_cosf(h);
        s_over_angle = fsw_sinf(h) / angle;
    }
    out[0] = c;
    out[1] = rv[0] * s_over_angle;
    out[2] = rv[1] * s_over_angle;
    out[3] = rv[2] * s_over_angle;
}

static void quat_to_R(const float q[4], float R[3][3])
{
    float w = q[0], x = q[1], y = q[2], z = q[3];
    R[0][0] = 1.0f - 2.0f*(y*y + z*z); R[0][1] = 2.0f*(x*y - w*z);        R[0][2] = 2.0f*(x*z + w*y);
    R[1][0] = 2.0f*(x*y + w*z);        R[1][1] = 1.0f - 2.0f*(x*x + z*z); R[1][2] = 2.0f*(y*z - w*x);
    R[2][0] = 2.0f*(x*z - w*y);        R[2][1] = 2.0f*(y*z + w*x);        R[2][2] = 1.0f - 2.0f*(x*x + y*y);
}

static void skew(const float v[3], float S[3][3])
{
    S[0][0] = 0.0f;  S[0][1] = -v[2]; S[0][2] = v[1];
    S[1][0] = v[2];  S[1][1] = 0.0f;  S[1][2] = -v[0];
    S[2][0] = -v[1]; S[2][1] = v[0];  S[2][2] = 0.0f;
}

static void symmetrize_and_floor(float P[N][N])
{
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = i + 1; j < N; j++) {
            float m = 0.5f * (P[i][j] + P[j][i]);
            P[i][j] = m;
            P[j][i] = m;
        }
        if (P[i][i] < P_DIAG_FLOOR) {          /* NaN compares false -> left as NaN so health check sees it */
            P[i][i] = P_DIAG_FLOOR;
        }
    }
}

/* ---------------- public API ---------------- */

void fsw_eskf_default_cfg(fsw_eskf_cfg_t *c)
{
    c->gyro_noise_psd = 1e-6f;
    c->gyro_bias_psd = 1e-9f;
    c->accel_noise_psd = 1e-4f;
    c->accel_bias_psd = 1e-8f;
    c->gravity_nav[0] = 0.0f;
    c->gravity_nav[1] = 0.0f;
    c->gravity_nav[2] = -9.80665f;
    c->nis_gate = 0.0f;
    c->max_consecutive_rejects = 5;
    c->gravity_model = FSW_ESKF_GRAV_FLAT;
    c->mu = 3.986004418e14;
    c->j2 = 1.08262668e-3;
    c->r_eq = 6378137.0;
    c->polar_axis[0] = 0.0; c->polar_axis[1] = 0.0; c->polar_axis[2] = 1.0;
}

void fsw_eskf_gravity(const fsw_eskf_t *e, const double p[3], double g[3])
{
    if (e->cfg.gravity_model == FSW_ESKF_GRAV_FLAT) {
        g[0] = (double)e->cfg.gravity_nav[0]; g[1] = (double)e->cfg.gravity_nav[1]; g[2] = (double)e->cfg.gravity_nav[2];
        return;
    }
    {
        double r2 = p[0]*p[0] + p[1]*p[1] + p[2]*p[2];
        double r = fsw_sqrt_d(r2);
        double k;
        if (!(r > 1.0e3)) {                      /* inside the Earth / not initialised: no gravity rather than Inf */
            g[0] = g[1] = g[2] = 0.0;
            return;
        }
        k = -e->cfg.mu / (r2 * r);
        g[0] = k * p[0]; g[1] = k * p[1]; g[2] = k * p[2];
        if (e->cfg.gravity_model == FSW_ESKF_GRAV_CENTRAL_J2) {
            const double *ax = e->cfg.polar_axis;
            double z = p[0]*ax[0] + p[1]*ax[1] + p[2]*ax[2];
            double z2r2 = z * z / r2;
            double kj = -1.5 * e->cfg.j2 * e->cfg.mu * e->cfg.r_eq * e->cfg.r_eq / (r2 * r2 * r);
            g[0] += kj * ((1.0 - 5.0 * z2r2) * p[0] + 2.0 * z * ax[0]);
            g[1] += kj * ((1.0 - 5.0 * z2r2) * p[1] + 2.0 * z * ax[1]);
            g[2] += kj * ((1.0 - 5.0 * z2r2) * p[2] + 2.0 * z * ax[2]);
        }
    }
}

void fsw_eskf_init(fsw_eskf_t *e, const fsw_eskf_cfg_t *cfg,
                   const float *q0, const double *v0, const double *p0,
                   const float *bg0, const float *ba0)
{
    int i, j;
    memset(e, 0, sizeof(*e));
    e->cfg = *cfg;
    if (q0 != 0) { memcpy(e->q, q0, sizeof(e->q)); quat_normalize(e->q); }
    else         { e->q[0] = 1.0f; }
    if (v0 != 0)  { memcpy(e->v, v0, sizeof(e->v)); }
    if (p0 != 0)  { memcpy(e->p, p0, sizeof(e->p)); }
    if (bg0 != 0) { memcpy(e->bg, bg0, sizeof(e->bg)); }
    if (ba0 != 0) { memcpy(e->ba, ba0, sizeof(e->ba)); }

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) { e->P[i][j] = 0.0f; }
    }
    for (i = 0; i < 3; i++)  { e->P[i][i] = 1e-4f; }          /* attitude  */
    for (i = 3; i < 6; i++)  { e->P[i][i] = 1.0f; }           /* velocity  */
    for (i = 6; i < 9; i++)  { e->P[i][i] = 10.0f; }          /* position  */
    for (i = 9; i < 12; i++) { e->P[i][i] = 1e-6f; }          /* gyro bias */
    for (i = 12; i < 15; i++){ e->P[i][i] = 1e-4f; }          /* accel bias*/
    e->last_update_accepted = 1;
}

void fsw_eskf_predict(fsw_eskf_t *e, const float gyro[3], const float accel[3], float dt)
{
    float omega[3], a_b[3], rv[3], dq[4], qn[4];
    float R[3][3], S[3][3];
    double a_n[3], g[3];
    int i, j, k;
    double dtd = (double)dt, dt2h = 0.5 * (double)dt * (double)dt;

    for (i = 0; i < 3; i++) {
        omega[i] = gyro[i] - e->bg[i];
        a_b[i] = accel[i] - e->ba[i];
        rv[i] = omega[i] * dt;
    }

    quat_from_rotvec(rv, dq);
    quat_mult(e->q, dq, qn);
    quat_normalize(qn);
    memcpy(e->q, qn, sizeof(qn));
    quat_to_R(e->q, R);

    fsw_eskf_gravity(e, e->p, g);                    /* evaluated at the start of the step */
    for (i = 0; i < 3; i++) {
        a_n[i] = (double)R[i][0]*(double)a_b[0] + (double)R[i][1]*(double)a_b[1] + (double)R[i][2]*(double)a_b[2] + g[i];
    }
    /* gravity gradient (point-mass part) for the covariance: G = -mu/r^3 (I - 3 rhat rhat^T) */
    {
        float G[3][3];
        int central = (e->cfg.gravity_model != FSW_ESKF_GRAV_FLAT);
        if (central) {
            double r2 = e->p[0]*e->p[0] + e->p[1]*e->p[1] + e->p[2]*e->p[2];
            double r = fsw_sqrt_d(r2);
            if (r > 1.0e3) {
                double kk = -e->cfg.mu / (r2 * r);
                for (i = 0; i < 3; i++) {
                    for (j = 0; j < 3; j++) {
                        double rr = e->p[i] * e->p[j] / r2;
                        G[i][j] = (float)(kk * (((i == j) ? 1.0 : 0.0) - 3.0 * rr));
                    }
                }
            } else {
                central = 0;
            }
        }
        {
            /* Velocity-Verlet: position uses the start-of-step acceleration, velocity uses the AVERAGE gravity of
             * the step (specific force is a single sample, held). Gravity at the step start alone is only
             * first-order in v: ~1.9e-6 m/s per 20 ms step at orbital speed, same sign every step = 17 m / 0.055 m/s
             * after 600 s. With constant (flat) gravity this reduces to the old update exactly. */
            double gn[3], pn[3];
            for (i = 0; i < 3; i++) { pn[i] = e->p[i] + e->v[i]*dtd + a_n[i]*dt2h; }
            fsw_eskf_gravity(e, pn, gn);
            for (i = 0; i < 3; i++) {
                e->v[i] = e->v[i] + (a_n[i] - g[i] + 0.5 * (g[i] + gn[i])) * dtd;
                e->p[i] = pn[i];
            }
        }
        /* F is built below; stash G in T's top-left block via a second pass after F is initialised */
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) { e->F[i][j] = (i == j) ? 1.0f : 0.0f; }
        }
        if (central) {
            for (i = 0; i < 3; i++) {
                for (j = 0; j < 3; j++) { e->F[3 + i][6 + j] = G[i][j] * dt; }
            }
        }
    }

    /* F = I + sparse blocks (identical structure to the Python code; gravity-gradient block added above). */
    skew(omega, S);
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) { e->F[i][j] -= S[i][j] * dt; }       /* I - skew(w) dt          */
        e->F[i][9 + i] = -dt;                                          /* d(theta)/d(bg)          */
        e->F[6 + i][3 + i] = dt;                                       /* d(p)/d(v)               */
    }
    skew(a_b, S);                                                      /* -R skew(a_b) dt         */
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            float acc = 0.0f;
            for (k = 0; k < 3; k++) { acc += R[i][k] * S[k][j]; }
            e->F[3 + i][j] = -acc * dt;
            e->F[3 + i][12 + j] = -R[i][j] * dt;                       /* d(v)/d(ba)              */
        }
    }

    /* P = F P F^T + Q */
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            float acc = 0.0f;
            for (k = 0; k < N; k++) { acc += e->F[i][k] * e->P[k][j]; }
            e->T[i][j] = acc;
        }
    }
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            float acc = 0.0f;
            for (k = 0; k < N; k++) { acc += e->T[i][k] * e->F[j][k]; }
            e->P[i][j] = acc;
        }
    }
    for (i = 0; i < 3; i++) {
        e->P[i][i]           += e->cfg.gyro_noise_psd * dt;
        e->P[3 + i][3 + i]   += e->cfg.accel_noise_psd * dt;
        e->P[9 + i][9 + i]   += e->cfg.gyro_bias_psd * dt;
        e->P[12 + i][12 + i] += e->cfg.accel_bias_psd * dt;
    }
    symmetrize_and_floor(e->P);
    e->n_predict++;
}

/* 3x3 inverse by adjugate. Returns 0 if (near-)singular. */
static int inv3(float A[3][3], float out[3][3])
{
    float c00 = A[1][1]*A[2][2] - A[1][2]*A[2][1];
    float c01 = A[1][2]*A[2][0] - A[1][0]*A[2][2];
    float c02 = A[1][0]*A[2][1] - A[1][1]*A[2][0];
    float det = A[0][0]*c00 + A[0][1]*c01 + A[0][2]*c02;
    float scale = fsw_absf(A[0][0]) + fsw_absf(A[1][1]) + fsw_absf(A[2][2]);
    float inv_det;
    if (!fsw_isfinitef(det) || fsw_absf(det) <= 1e-12f * scale * scale * scale) {
        return 0;
    }
    inv_det = 1.0f / det;
    out[0][0] = c00 * inv_det;
    out[1][0] = c01 * inv_det;
    out[2][0] = c02 * inv_det;
    out[0][1] = (A[0][2]*A[2][1] - A[0][1]*A[2][2]) * inv_det;
    out[1][1] = (A[0][0]*A[2][2] - A[0][2]*A[2][0]) * inv_det;
    out[2][1] = (A[0][1]*A[2][0] - A[0][0]*A[2][1]) * inv_det;
    out[0][2] = (A[0][1]*A[1][2] - A[0][2]*A[1][1]) * inv_det;
    out[1][2] = (A[0][2]*A[1][0] - A[0][0]*A[1][2]) * inv_det;
    out[2][2] = (A[0][0]*A[1][1] - A[0][1]*A[1][0]) * inv_det;
    return 1;
}

int fsw_eskf_update_position(fsw_eskf_t *e, const double pos[3], float r_pos)
{
    float y[3], S[3][3], Si[3][3], K[N][3], dx[N], dq[4], qn[4], rv[3];
    int i, j, k;
    int force_accept;

    if (!(r_pos > 0.0f) || !fsw_isfinitef(r_pos)) { e->last_update_accepted = 0; return 0; }
    for (i = 0; i < 3; i++) {
        double yd = pos[i] - e->p[i];                   /* subtract in double, THEN round: float(p) alone loses ~0.5 m */
        if (!(yd == yd) || yd > 3.0e38 || yd < -3.0e38) { e->last_update_accepted = 0; return 0; }
        y[i] = (float)yd;
    }
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) { S[i][j] = e->P[6 + i][6 + j] + ((i == j) ? r_pos : 0.0f); }
    }
    if (!inv3(S, Si)) { e->last_update_accepted = 0; return 0; }

    force_accept = (e->cfg.max_consecutive_rejects > 0) &&
                   (e->consecutive_rejects >= e->cfg.max_consecutive_rejects);
    if (e->cfg.nis_gate > 0.0f && !force_accept) {
        float nis = 0.0f;
        for (i = 0; i < 3; i++) {
            for (j = 0; j < 3; j++) { nis += y[i] * Si[i][j] * y[j]; }
        }
        if (!(nis <= e->cfg.nis_gate)) {              /* also rejects NaN */
            e->consecutive_rejects++;
            e->n_rejected++;
            e->last_update_accepted = 0;
            return 0;
        }
    }
    e->consecutive_rejects = 0;

    /* K = P H^T S^-1, with H selecting the position block -> P H^T = P[:, 6:9] */
    for (i = 0; i < N; i++) {
        for (j = 0; j < 3; j++) {
            float acc = 0.0f;
            for (k = 0; k < 3; k++) { acc += e->P[i][6 + k] * Si[k][j]; }
            K[i][j] = acc;
        }
    }
    for (i = 0; i < N; i++) {
        dx[i] = K[i][0]*y[0] + K[i][1]*y[1] + K[i][2]*y[2];
    }

    /* inject (same order as the Python _inject) */
    for (i = 0; i < 3; i++) { rv[i] = dx[i]; }
    quat_from_rotvec(rv, dq);
    quat_mult(e->q, dq, qn);
    quat_normalize(qn);
    memcpy(e->q, qn, sizeof(qn));
    for (i = 0; i < 3; i++) {
        e->v[i]  += (double)dx[3 + i];
        e->p[i]  += (double)dx[6 + i];
        e->bg[i] += dx[9 + i];
        e->ba[i] += dx[12 + i];
    }

    /* Joseph form: P = (I-KH) P (I-KH)^T + K R K^T ; (I-KH) differs from I only in columns 6..8 */
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) { e->F[i][j] = (i == j) ? 1.0f : 0.0f; }
        for (j = 0; j < 3; j++) { e->F[i][6 + j] -= K[i][j]; }
    }
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            float acc = 0.0f;
            for (k = 0; k < N; k++) { acc += e->F[i][k] * e->P[k][j]; }
            e->T[i][j] = acc;
        }
    }
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            float acc = 0.0f;
            for (k = 0; k < N; k++) { acc += e->T[i][k] * e->F[j][k]; }
            for (k = 0; k < 3; k++) { acc += K[i][k] * r_pos * K[j][k]; }
            e->P[i][j] = acc;
        }
    }
    symmetrize_and_floor(e->P);
    e->n_update++;
    e->last_update_accepted = 1;
    return 1;
}

static void decorrelate(fsw_eskf_t *e, int lo, int hi)
{
    int i, j;
    for (i = lo; i < hi; i++) {
        for (j = 0; j < N; j++) {
            if (j < lo || j >= hi) { e->P[i][j] = 0.0f; e->P[j][i] = 0.0f; }
        }
    }
}

void fsw_eskf_reset_pv(fsw_eskf_t *e, const double p[3], const double v[3], float sigma_p, float sigma_v)
{
    int i, j;
    for (i = 0; i < 3; i++) { e->p[i] = p[i]; e->v[i] = v[i]; }
    decorrelate(e, 3, 9);
    for (i = 3; i < 9; i++) { for (j = 3; j < 9; j++) { e->P[i][j] = 0.0f; } }
    for (i = 3; i < 6; i++) { e->P[i][i] = sigma_v * sigma_v; }
    for (i = 6; i < 9; i++) { e->P[i][i] = sigma_p * sigma_p; }
}

void fsw_eskf_set_att_sigma(fsw_eskf_t *e, float sx, float sy, float sz)
{
    int i, j;
    decorrelate(e, 0, 3);
    for (i = 0; i < 3; i++) { for (j = 0; j < 3; j++) { e->P[i][j] = 0.0f; } }
    e->P[0][0] = sx * sx; e->P[1][1] = sy * sy; e->P[2][2] = sz * sz;
}

int fsw_eskf_healthy(const fsw_eskf_t *e)
{
    int i;
    float n2;
    for (i = 0; i < 4; i++) { if (!fsw_isfinitef(e->q[i])) { return 0; } }
    for (i = 0; i < 3; i++) {
        double vv = e->v[i], pp = e->p[i];
        if (!(vv == vv) || vv > 1.0e300 || vv < -1.0e300 || !(pp == pp) || pp > 1.0e300 || pp < -1.0e300 ||
            !fsw_isfinitef(e->bg[i]) || !fsw_isfinitef(e->ba[i])) { return 0; }
    }
    n2 = e->q[0]*e->q[0] + e->q[1]*e->q[1] + e->q[2]*e->q[2] + e->q[3]*e->q[3];
    if (!(n2 > 0.998f && n2 < 1.002f)) { return 0; }
    for (i = 0; i < N; i++) {
        float d = e->P[i][i];
        if (!fsw_isfinitef(d) || d < 0.0f || d > P_DIAG_CEIL) { return 0; }
    }
    return 1;
}

int fsw_eskf_align_accel(fsw_eskf_t *e, const float f[3])
{
    /* f_body/g = (-sin th, cos th sin ph, cos th cos ph) for R = Rz(0) Ry(th) Rx(ph). */
    float g = 9.80665f;
    float n = fsw_sqrtf(f[0]*f[0] + f[1]*f[1] + f[2]*f[2]);
    float roll, pitch, qy[4], qx[4], qa[4];
    if (!fsw_isfinitef(n) || n < 0.8f * g || n > 1.2f * g) {
        return 0;
    }
    pitch = fsw_atan2f(-f[0], fsw_sqrtf(f[1]*f[1] + f[2]*f[2]));
    roll  = fsw_atan2f(f[1], f[2]);
    qy[0] = fsw_cosf(0.5f * pitch); qy[1] = 0.0f; qy[2] = fsw_sinf(0.5f * pitch); qy[3] = 0.0f;
    qx[0] = fsw_cosf(0.5f * roll);  qx[1] = fsw_sinf(0.5f * roll); qx[2] = 0.0f;  qx[3] = 0.0f;
    quat_mult(qy, qx, qa);
    quat_normalize(qa);
    memcpy(e->q, qa, sizeof(qa));
    return 1;
}

void fsw_eskf_euler(const fsw_eskf_t *e, float *roll, float *pitch, float *yaw)
{
    float w = e->q[0], x = e->q[1], y = e->q[2], z = e->q[3];
    *roll  = fsw_atan2f(2.0f*(w*x + y*z), 1.0f - 2.0f*(x*x + y*y));
    *pitch = fsw_asinf(2.0f*(w*y - z*x));
    *yaw   = fsw_atan2f(2.0f*(w*z + x*y), 1.0f - 2.0f*(y*y + z*z));
}

float fsw_eskf_pitch(const fsw_eskf_t *e)
{
    /* Angle of the body +Z (thrust) axis from nav +Z, measured in the nav X-Z plane: atan2(R02, R22).
     * Valid over the whole circle. The Euler form asin(2(wy - zx)) is singular at +-90 deg and folds
     * back past it (the estimate then moves OPPOSITE to the true rotation) -- fatal for a vehicle
     * that pitches over through the horizontal during an orbital burn. */
    float w = e->q[0], x = e->q[1], y = e->q[2], z = e->q[3];
    return fsw_atan2f(2.0f*(x*z + w*y), 1.0f - 2.0f*(x*x + y*y));
}

unsigned fsw_eskf_sizeof(void) { return (unsigned)sizeof(fsw_eskf_t); }

void fsw_eskf_get(const fsw_eskf_t *e, float *q4, float *v3, float *p3, float *bg3, float *ba3)
{
    memcpy(q4, e->q, sizeof(e->q));
    int i;
    for (i = 0; i < 3; i++) { v3[i] = (float)e->v[i]; p3[i] = (float)e->p[i]; }
    memcpy(bg3, e->bg, sizeof(e->bg));
    memcpy(ba3, e->ba, sizeof(e->ba));
}

void fsw_eskf_get_pv(const fsw_eskf_t *e, double *v3, double *p3)
{
    memcpy(v3, e->v, sizeof(e->v));
    memcpy(p3, e->p, sizeof(e->p));
}

void fsw_eskf_get_P(const fsw_eskf_t *e, float *out225)
{
    memcpy(out225, e->P, sizeof(e->P));
}

void fsw_eskf_set_P_diag(fsw_eskf_t *e, int i, float value)
{
    if (i >= 0 && i < N) { e->P[i][i] = value; }
}
