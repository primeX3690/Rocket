#include "fsw_peg.h"
#include <math.h>

static const double RES_SCALE[3] = { 100.0, 1.0, 1.0e-3 };   /* m, m/s, rad -> comparable units */

static int finite(double x) { return (x == x) && (x <= 1.7e308) && (x >= -1.7e308); }

double fsw_peg_theta(double A, double B, double tau) { return atan(A + B * tau); }

void fsw_peg_derivs(double r, double v, double gamma, double m, double thrust_n, double mdot,
                    double theta, double mu, double out[4])
{
    double g = mu / (r * r);
    double chi = theta - gamma;
    out[0] = v * sin(gamma);
    out[1] = thrust_n * cos(chi) / m - g * sin(gamma);
    out[2] = thrust_n * sin(chi) / (m * v) + (v / r - g / v) * cos(gamma);
    out[3] = -mdot;
}

fsw_peg_state_t fsw_peg_rk4_step(const fsw_peg_state_t *s, double thrust_n, double mdot,
                                 double theta, double dt, double mu)
{
    double k1[4], k2[4], k3[4], k4[4];
    fsw_peg_state_t out;
    fsw_peg_derivs(s->r, s->v, s->gamma, s->m, thrust_n, mdot, theta, mu, k1);
    fsw_peg_derivs(s->r + 0.5*dt*k1[0], s->v + 0.5*dt*k1[1], s->gamma + 0.5*dt*k1[2], s->m + 0.5*dt*k1[3],
                   thrust_n, mdot, theta, mu, k2);
    fsw_peg_derivs(s->r + 0.5*dt*k2[0], s->v + 0.5*dt*k2[1], s->gamma + 0.5*dt*k2[2], s->m + 0.5*dt*k2[3],
                   thrust_n, mdot, theta, mu, k3);
    fsw_peg_derivs(s->r + dt*k3[0], s->v + dt*k3[1], s->gamma + dt*k3[2], s->m + dt*k3[3],
                   thrust_n, mdot, theta, mu, k4);
    out.r     = s->r     + dt / 6.0 * (k1[0] + 2.0*k2[0] + 2.0*k3[0] + k4[0]);
    out.v     = s->v     + dt / 6.0 * (k1[1] + 2.0*k2[1] + 2.0*k3[1] + k4[1]);
    out.gamma = s->gamma + dt / 6.0 * (k1[2] + 2.0*k2[2] + 2.0*k3[2] + k4[2]);
    out.m     = s->m     + dt / 6.0 * (k1[3] + 2.0*k2[3] + 2.0*k3[3] + k4[3]);
    return out;
}

fsw_peg_state_t fsw_peg_predict_cutoff(const fsw_peg_state_t *s0, double A, double B, double T,
                                       double thrust_n, double mdot, double mu, int n_steps)
{
    int n = n_steps, k;
    double h;
    fsw_peg_state_t s = *s0;
    if (n <= 0) {
        n = (int)ceil(T / 1.5);
        if (n < 12) { n = 12; }
    }
    if (n > FSW_PEG_MAX_STEPS) { n = FSW_PEG_MAX_STEPS; }
    h = T / (double)n;
    for (k = 0; k < n; k++) {
        double theta = atan(A + B * ((double)k + 0.5) * h);
        s = fsw_peg_rk4_step(&s, thrust_n, mdot, theta, h, mu);
    }
    return s;
}

static void residual(const fsw_peg_state_t *c, const fsw_peg_target_t *t, double res[3])
{
    res[0] = c->r - t->r_t;
    res[1] = c->v - t->v_t;
    res[2] = c->gamma - t->gamma_t;
}

static double cost_of(const double res[3])
{
    double a = res[0] / RES_SCALE[0], b = res[1] / RES_SCALE[1], c = res[2] / RES_SCALE[2];
    return a*a + b*b + c*c;
}

static double clipd(double x, double lo, double hi) { return (x < lo) ? lo : ((x > hi) ? hi : x); }

/* Solve H x = b (3x3) by Gaussian elimination with partial pivoting. 0 on singular. */
static int solve3(double Hin[3][3], const double bin[3], double x[3])
{
    double M[3][4];
    int i, j, k;
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) { M[i][j] = Hin[i][j]; }
        M[i][3] = bin[i];
    }
    for (k = 0; k < 3; k++) {
        int piv = k;
        double best = fabs(M[k][k]);
        for (i = k + 1; i < 3; i++) {
            if (fabs(M[i][k]) > best) { best = fabs(M[i][k]); piv = i; }
        }
        if (!(best > 0.0) || !finite(best)) { return 0; }
        if (piv != k) {
            for (j = 0; j < 4; j++) { double tmp = M[k][j]; M[k][j] = M[piv][j]; M[piv][j] = tmp; }
        }
        for (i = k + 1; i < 3; i++) {
            double f = M[i][k] / M[k][k];
            for (j = k; j < 4; j++) { M[i][j] -= f * M[k][j]; }
        }
    }
    for (i = 2; i >= 0; i--) {
        double acc = M[i][3];
        for (j = i + 1; j < 3; j++) { acc -= M[i][j] * x[j]; }
        x[i] = acc / M[i][i];
        if (!finite(x[i])) { return 0; }
    }
    return 1;
}

static int within_tol(const double res[3], const double tol[3])
{
    return (fabs(res[0]) < tol[0]) && (fabs(res[1]) < tol[1]) && (fabs(res[2]) < tol[2]);
}

void fsw_peg_solve(const fsw_peg_state_t *s, double thrust_n, double mdot,
                   const fsw_peg_target_t *target, double m_dry_kg, double mu,
                   const double guess[3], int max_iter, const double tol_in[3],
                   fsw_peg_solution_t *out)
{
    static const double default_tol[3] = { FSW_PEG_TOL_R, FSW_PEG_TOL_V, FSW_PEG_TOL_GAMMA };
    static const double steps[3] = { 1.0e-3, 1.0e-4, 0.05 };
    const double *tol = (tol_in != 0) ? tol_in : default_tol;
    double x[3], res[3], cost, lam = 1.0e-2, T_max, ve;
    int it = 0, converged = 0, i, j, loop_broke;

    out->A = out->B = out->T = 0.0;
    out->res_r = out->res_v = out->res_gamma = 0.0;
    out->iterations = 0;

    if (!finite(s->r) || !finite(s->v) || !finite(s->gamma) || !finite(s->m) ||
        !finite(thrust_n) || !finite(mdot) || !finite(m_dry_kg) || !finite(mu) ||
        !finite(target->r_t) || !finite(target->v_t) || !finite(target->gamma_t) ||
        s->r <= 1.0e6 || s->v < 1.0 || s->m <= m_dry_kg || m_dry_kg < 0.0 ||
        thrust_n <= 0.0 || mdot <= 0.0 || mu <= 0.0 || target->r_t <= 1.0e6 || target->v_t < 1.0 ||
        max_iter < 1 || max_iter > 100) {
        out->status = FSW_PEG_BAD_INPUT;
        return;
    }

    T_max = (s->m - m_dry_kg) / mdot;
    if (T_max < 1.0) { T_max = 1.0; }
    ve = thrust_n / mdot;

    if (guess == 0) {
        double E0 = 0.5 * s->v * s->v - mu / s->r;
        double Et = 0.5 * target->v_t * target->v_t - mu / target->r_t;
        double den = 0.5 * (s->v + target->v_t);
        double dv_est, T0;
        if (den < 1.0) { den = 1.0; }
        dv_est = (Et - E0) / den;
        if (dv_est < 5.0) { dv_est = 5.0; }
        T0 = (s->m / mdot) * (1.0 - exp(-dv_est / ve));
        x[0] = tan(s->gamma + 0.05);
        x[1] = 0.0;
        x[2] = clipd(T0, 1.0, T_max);
    } else {
        x[0] = guess[0]; x[1] = guess[1]; x[2] = guess[2];
        if (!finite(x[0]) || !finite(x[1]) || !finite(x[2])) {
            out->status = FSW_PEG_BAD_INPUT;
            return;
        }
        x[2] = clipd(x[2], 0.2, T_max);
    }

    {
        fsw_peg_state_t c = fsw_peg_predict_cutoff(s, x[0], x[1], x[2], thrust_n, mdot, mu, 0);
        residual(&c, target, res);
    }
    if (!finite(res[0]) || !finite(res[1]) || !finite(res[2])) {
        out->status = FSW_PEG_NUMERIC_FAIL;
        out->A = x[0]; out->B = x[1]; out->T = x[2];
        return;
    }
    cost = cost_of(res);

    loop_broke = 0;
    for (it = 1; it <= max_iter; it++) {
        double J[3][3], Js[3][3], rs[3], JtJ[3][3], g[3];
        int improved = 0, attempt;

        if (within_tol(res, tol)) {
            converged = 1;
            it -= 1;
            loop_broke = 1;
            break;
        }
        for (j = 0; j < 3; j++) {
            double xp[3], rp[3];
            fsw_peg_state_t c;
            xp[0] = x[0]; xp[1] = x[1]; xp[2] = x[2];
            xp[j] += steps[j];
            if (j == 2 && xp[2] > T_max) { xp[2] = T_max; }
            if (xp[j] == x[j]) { xp[j] = x[j] - steps[j]; }
            c = fsw_peg_predict_cutoff(s, xp[0], xp[1], xp[2], thrust_n, mdot, mu, 0);
            residual(&c, target, rp);
            for (i = 0; i < 3; i++) { J[i][j] = (rp[i] - res[i]) / (xp[j] - x[j]); }
        }
        for (i = 0; i < 3; i++) {
            rs[i] = res[i] / RES_SCALE[i];
            for (j = 0; j < 3; j++) { Js[i][j] = J[i][j] / RES_SCALE[i]; }
        }
        for (i = 0; i < 3; i++) {
            for (j = 0; j < 3; j++) {
                double acc = 0.0;
                int k;
                for (k = 0; k < 3; k++) { acc += Js[k][i] * Js[k][j]; }
                JtJ[i][j] = acc;
            }
            g[i] = 0.0;
            for (j = 0; j < 3; j++) { g[i] += Js[j][i] * rs[j]; }
        }
        for (attempt = 0; attempt < 8; attempt++) {
            double H[3][3], rhs[3], dx[3], xn[3], resn[3], costn;
            fsw_peg_state_t c;
            for (i = 0; i < 3; i++) {
                for (j = 0; j < 3; j++) { H[i][j] = JtJ[i][j]; }
                {
                    double d = JtJ[i][i];
                    if (d < 1.0e-12) { d = 1.0e-12; }
                    H[i][i] += lam * d;
                }
                rhs[i] = -g[i];
            }
            if (!solve3(H, rhs, dx)) {
                lam *= 10.0;
                continue;
            }
            xn[0] = x[0] + dx[0]; xn[1] = x[1] + dx[1]; xn[2] = clipd(x[2] + dx[2], 0.2, T_max);
            c = fsw_peg_predict_cutoff(s, xn[0], xn[1], xn[2], thrust_n, mdot, mu, 0);
            residual(&c, target, resn);
            costn = cost_of(resn);
            if (costn < cost) {                       /* NaN compares false -> step rejected */
                x[0] = xn[0]; x[1] = xn[1]; x[2] = xn[2];
                res[0] = resn[0]; res[1] = resn[1]; res[2] = resn[2];
                cost = costn;
                lam = lam / 5.0;
                if (lam < 1.0e-9) { lam = 1.0e-9; }
                improved = 1;
                break;
            }
            lam *= 5.0;
        }
        if (!improved) {
            loop_broke = 1;
            break;
        }
    }
    if (!loop_broke) {
        it = max_iter;                                /* Python: loop ran to completion */
    }
    converged = within_tol(res, tol);

    out->A = x[0]; out->B = x[1]; out->T = x[2];
    out->res_r = res[0]; out->res_v = res[1]; out->res_gamma = res[2];
    out->iterations = it;
    out->status = converged ? FSW_PEG_CONVERGED : FSW_PEG_NOT_CONVERGED;
}
