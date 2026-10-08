#include "fsw_guidance.h"
#include <math.h>
#include <string.h>

static int finite(double x) { return (x == x) && (x <= 1.7e308) && (x >= -1.7e308); }

static double cost3(double rr, double rv, double rg)
{
    double a = rr / 100.0, b = rv / 1.0, c = rg / 1.0e-3;
    return a*a + b*b + c*c;
}

void fsw_guidance_default_cfg(fsw_guidance_cfg_t *c)
{
    memset(c, 0, sizeof(*c));
    c->mu = FSW_PEG_MU_EARTH;
    c->cycle_s = 2.0;
    c->max_iter = 25;
    c->max_consecutive_failures = 3;
    c->t_go_cap_s = 900.0;
    c->ins_tol_r_m = 5000.0;          /* same acceptance as tests/test_peg.py: 15 m/s, 0.5 deg */
    c->ins_tol_v_mps = 15.0;
    c->ins_tol_gamma_rad = 0.0087;
}

int fsw_guidance_init(fsw_guidance_t *g, const fsw_guidance_cfg_t *cfg)
{
    memset(g, 0, sizeof(*g));
    g->cfg = *cfg;
    if (!finite(cfg->thrust_n) || !finite(cfg->isp_s) || !finite(cfg->m_dry_kg) || !finite(cfg->mu) ||
        cfg->thrust_n <= 0.0 || cfg->isp_s <= 0.0 || cfg->m_dry_kg < 0.0 || cfg->mu <= 0.0 ||
        !finite(cfg->target.r_t) || !finite(cfg->target.v_t) || !finite(cfg->target.gamma_t) ||
        cfg->target.r_t <= 1.0e6 || cfg->target.v_t < 1.0 ||
        cfg->cycle_s <= 0.0 || cfg->max_iter < 1 || cfg->max_iter > 100 || cfg->t_go_cap_s < 1.0) {
        return 1;
    }
    g->mdot = cfg->thrust_n / (cfg->isp_s * FSW_PEG_G0);
    return 0;
}

static void publish(fsw_guidance_t *g, double A, double B, double T, double t_s, int valid, int degraded)
{
    g->seq++;                          /* odd: writer active */
    FSW_BARRIER();
    g->sol.A = A; g->sol.B = B; g->sol.T = T; g->sol.t_solved_s = t_s;
    g->sol.valid = (uint8_t)valid; g->sol.degraded = (uint8_t)degraded;
    FSW_BARRIER();
    g->sol.seq = g->seq + 1u;
    g->seq++;                          /* even: stable */
    FSW_BARRIER();
}

fsw_guid_result_t fsw_guidance_update(fsw_guidance_t *g, double t_state_s, const fsw_peg_state_t *est)
{
    const fsw_guidance_cfg_t *c = &g->cfg;
    fsw_peg_solution_t ps, pc;
    double guess[3], elapsed = 0.0;
    int have_prior = g->sol.valid, use_guess = 0, usable;
    fsw_guid_result_t result;

    g->solves++;
    if (have_prior) {
        elapsed = t_state_s - g->sol.t_solved_s;
        if (finite(elapsed) && elapsed >= 0.0 && elapsed < g->sol.T - 0.2) {
            guess[0] = g->sol.A + g->sol.B * elapsed;      /* re-reference tan(theta) = A + B*tau to now */
            guess[1] = g->sol.B;
            guess[2] = g->sol.T - elapsed;
            use_guess = 1;
        }
    }

    fsw_peg_solve(est, c->thrust_n, g->mdot, &c->target, c->m_dry_kg, c->mu,
                  use_guess ? guess : 0, c->max_iter, 0, &ps);
    if (use_guess && ps.status != FSW_PEG_CONVERGED && ps.status != FSW_PEG_BAD_INPUT) {
        g->cold_retries++;
        fsw_peg_solve(est, c->thrust_n, g->mdot, &c->target, c->m_dry_kg, c->mu, 0, c->max_iter, 0, &pc);
        if (pc.status == FSW_PEG_CONVERGED ||
            (pc.status == FSW_PEG_NOT_CONVERGED && ps.status == FSW_PEG_NUMERIC_FAIL) ||
            (pc.status == FSW_PEG_NOT_CONVERGED && ps.status == FSW_PEG_NOT_CONVERGED &&
             cost3(pc.res_r, pc.res_v, pc.res_gamma) < cost3(ps.res_r, ps.res_v, ps.res_gamma))) {
            ps = pc;
        }
    }

    usable = (ps.status == FSW_PEG_CONVERGED || ps.status == FSW_PEG_NOT_CONVERGED) &&
             finite(ps.A) && finite(ps.B) && finite(ps.T) && ps.T >= 0.2 && ps.T <= c->t_go_cap_s;

    if (usable && ps.status == FSW_PEG_CONVERGED) {
        publish(g, ps.A, ps.B, ps.T, t_state_s, 1, 0);
        g->consecutive_failures = 0;
        g->converged++;
        g->last_res[0] = ps.res_r; g->last_res[1] = ps.res_v; g->last_res[2] = ps.res_gamma;
        g->last_iterations = ps.iterations;
        return FSW_GUID_PUBLISHED;
    }

    g->consecutive_failures++;
    result = FSW_GUID_FAILED;

    if (usable) {                                          /* not converged: best effort vs previous */
        int prefer_new = 1;
        if (have_prior && use_guess) {
            fsw_peg_state_t cut = fsw_peg_predict_cutoff(est, guess[0], guess[1], guess[2],
                                                         c->thrust_n, g->mdot, c->mu, 0);
            double old_cost = cost3(cut.r - c->target.r_t, cut.v - c->target.v_t, cut.gamma - c->target.gamma_t);
            double new_cost = cost3(ps.res_r, ps.res_v, ps.res_gamma);
            if (finite(old_cost) && old_cost <= new_cost) { prefer_new = 0; }
        }
        if (prefer_new) {
            publish(g, ps.A, ps.B, ps.T, t_state_s, 1, 1);
            g->degraded++;
            g->last_res[0] = ps.res_r; g->last_res[1] = ps.res_v; g->last_res[2] = ps.res_gamma;
            g->last_iterations = ps.iterations;
            result = FSW_GUID_PUBLISHED_DEGRADED;
        } else {
            g->kept_previous++;
            result = FSW_GUID_KEPT_PREVIOUS;
        }
    } else if (have_prior) {
        g->kept_previous++;
        result = FSW_GUID_KEPT_PREVIOUS;
    }

    if (g->consecutive_failures > g->cfg.max_consecutive_failures && result != FSW_GUID_PUBLISHED_DEGRADED) {
        publish(g, g->sol.A, g->sol.B, g->sol.T, g->sol.t_solved_s, 0, 1);   /* withdrawn */
        result = FSW_GUID_FAILED;
    }
    if (result == FSW_GUID_FAILED) { g->failed++; }
    return result;
}

int fsw_guidance_read(const fsw_guidance_t *g, fsw_guid_solution_t *out)
{
    uint32_t s1, s2;
    fsw_guid_solution_t tmp;
    s1 = g->seq;
    FSW_BARRIER();
    if (s1 & 1u) { return 0; }
    tmp = g->sol;
    FSW_BARRIER();
    s2 = g->seq;
    if (s1 != s2) { return 0; }
    *out = tmp;
    return 1;
}

int fsw_guid_theta(const fsw_guid_solution_t *s, double t_now_s, double *theta)
{
    double tau;
    if (!s->valid) { return 0; }
    tau = t_now_s - s->t_solved_s;
    if (!(tau >= 0.0)) { tau = 0.0; }
    if (tau > s->T) { tau = s->T; }                       /* hold the final attitude after T_go */
    *theta = fsw_peg_theta(s->A, s->B, tau);
    return 1;
}

double fsw_guid_time_to_go(const fsw_guid_solution_t *s, double t_now_s)
{
    double ttg;
    if (!s->valid) { return -1.0; }
    ttg = s->T - (t_now_s - s->t_solved_s);
    return (ttg > 0.0) ? ttg : 0.0;
}

int fsw_guid_cutoff_due(const fsw_guid_solution_t *s, double t_now_s)
{
    return s->valid && ((t_now_s - s->t_solved_s) >= s->T);
}

int fsw_guidance_insertion_ok(const fsw_guidance_cfg_t *c, const fsw_peg_state_t *s)
{
    if (!finite(s->r) || !finite(s->v) || !finite(s->gamma)) { return 0; }
    return (fabs(s->r - c->target.r_t) <= c->ins_tol_r_m) &&
           (fabs(s->v - c->target.v_t) <= c->ins_tol_v_mps) &&
           (fabs(s->gamma - c->target.gamma_t) <= c->ins_tol_gamma_rad);
}
