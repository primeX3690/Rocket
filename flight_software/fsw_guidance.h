/* fsw_guidance.h -- runtime wrapper around the PEG solver: everything a flight
 * computer needs beyond the math.
 *
 *  - fsw_guidance_update() : the HEAVY part (one predictor-corrector solve, double
 *    precision, libm). Run it from a LOW-PRIORITY task / the idle loop, every
 *    cfg.cycle_s (2 s), NOT from the 50 Hz control interrupt.
 *  - fsw_guidance_read()   : the CHEAP part, safe from the control cycle. Publication
 *    is a sequence lock (writer: seq odd while writing; reader: copy, re-check seq).
 *    A reader that preempts a writer sees an odd/changed seq and keeps its previous
 *    copy -- it never waits and never sees a torn solution. On a real target compile
 *    with FSW_BARRIER() mapped to your DMB/compiler fence (defaults to
 *    __sync_synchronize on GCC/Clang).
 *  - Acceptance policy: converged -> publish. Not converged -> keep whichever is
 *    closer to the target from the CURRENT state: the previous solution (re-referenced
 *    to now) or the new best-effort one; mark degraded. Bad input / numeric failure ->
 *    keep the previous solution. More than max_consecutive_failures failures in a row
 *    -> solution withdrawn (valid=0): the caller must fall back to something else.
 *  - A warm-start solve that fails is retried cold once.
 *
 * No heap, no globals. Time is a double of seconds on any monotonic clock. */
#ifndef FSW_GUIDANCE_H
#define FSW_GUIDANCE_H
#include <stdint.h>
#include "fsw_peg.h"

#if defined(__GNUC__)
#define FSW_BARRIER() __sync_synchronize()
#else
#define FSW_BARRIER() ((void)0)
#endif

typedef struct {
    fsw_peg_target_t target;
    double   thrust_n, isp_s, m_dry_kg, mu;
    double   cycle_s;                    /* solve cadence                                  */
    int      max_iter;
    uint16_t max_consecutive_failures;
    double   t_go_cap_s;                 /* solutions with a longer burn are rejected      */
    double   ins_tol_r_m, ins_tol_v_mps, ins_tol_gamma_rad;   /* "orbit inserted" window   */
} fsw_guidance_cfg_t;

typedef struct {
    double   A, B, T;                    /* tan(theta) = A + B*tau, burn time-to-go at t_solved */
    double   t_solved_s;                 /* time the input state was valid                 */
    uint8_t  valid, degraded;
    uint32_t seq;                        /* copy of the publication counter                */
} fsw_guid_solution_t;

typedef enum {
    FSW_GUID_PUBLISHED = 0,              /* converged solution published                   */
    FSW_GUID_PUBLISHED_DEGRADED,         /* best-effort published (target not reachable / not converged) */
    FSW_GUID_KEPT_PREVIOUS,              /* new solve unusable or worse; previous kept     */
    FSW_GUID_FAILED                      /* nothing valid to fly                           */
} fsw_guid_result_t;

typedef struct {
    fsw_guidance_cfg_t cfg;
    double   mdot;
    volatile uint32_t seq;
    fsw_guid_solution_t sol;             /* the published copy (written under the seqlock) */
    uint16_t consecutive_failures;
    uint32_t solves, converged, degraded, kept_previous, failed, cold_retries;
    double   last_res[3];                /* residual of the last published solution        */
    int      last_iterations;
} fsw_guidance_t;

/* Fills mu, cadence, iteration limit, failure limit, caps and insertion window; the
 * caller sets target, thrust_n, isp_s and m_dry_kg. */
void fsw_guidance_default_cfg(fsw_guidance_cfg_t *cfg);
/* Returns 0 on success, nonzero if the vehicle/target numbers are not physical. */
int  fsw_guidance_init(fsw_guidance_t *g, const fsw_guidance_cfg_t *cfg);

/* HEAVY. est is the vehicle state valid at t_state_s. */
fsw_guid_result_t fsw_guidance_update(fsw_guidance_t *g, double t_state_s, const fsw_peg_state_t *est);

/* CHEAP. Returns 1 and fills *out if a consistent copy was obtained, else 0 (keep the old one). */
int  fsw_guidance_read(const fsw_guidance_t *g, fsw_guid_solution_t *out);

/* Operations on a copied solution (pure). */
int    fsw_guid_theta(const fsw_guid_solution_t *s, double t_now_s, double *theta_rad);  /* 0 if !valid */
double fsw_guid_time_to_go(const fsw_guid_solution_t *s, double t_now_s);               /* <0 if !valid */
int    fsw_guid_cutoff_due(const fsw_guid_solution_t *s, double t_now_s);

int    fsw_guidance_insertion_ok(const fsw_guidance_cfg_t *cfg, const fsw_peg_state_t *s);

#endif
