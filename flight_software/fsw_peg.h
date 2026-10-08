/* fsw_peg.h -- C port of the predictor-corrector PEG solver in guidance/peg.py
 * (point_mass_derivs, rk4_step, predict_cutoff, solve_peg_predictor_corrector).
 *
 * Linear-tangent steering  tan(theta) = A + B*tau  (theta from LOCAL HORIZONTAL),
 * three unknowns (A, B, T_go) solved by Levenberg-Marquardt so that forward
 * integration of the point-mass equations lands on (radius, speed, flight-path
 * angle). Same algorithm, same constants, same tolerances as the Python.
 *
 * DOUBLE PRECISION ON PURPOSE: r ~ 6.5e6 m has a float32 resolution of 0.5 m,
 * the convergence tolerance is 2 m and the Jacobian is finite-differenced, so
 * float32 would make the solve noise-limited. This module therefore uses
 * libm in double (sin/cos/atan/tan/exp/ceil). On a Cortex-M4F (single-precision
 * FPU) doubles run in software: the solve belongs in a LOW-PRIORITY task, not in
 * the 50 Hz control cycle (see fsw_guidance.h). Cost is bounded: <= max_iter
 * iterations x 4 forward predictions x <= FSW_PEG_MAX_STEPS RK4 steps.
 *
 * Differences from the Python reference (all for safety, none change results
 * inside the normal envelope): inputs are validated and a status code returned
 * instead of raising/NaN-propagating; the predictor step count is capped at
 * FSW_PEG_MAX_STEPS (only matters for burns > 600 s). No heap, no globals. */
#ifndef FSW_PEG_H
#define FSW_PEG_H

#define FSW_PEG_MU_EARTH   3.986004418e14
#define FSW_PEG_G0         9.80665
#define FSW_PEG_MAX_STEPS  400

typedef struct { double r, v, gamma, m; } fsw_peg_state_t;          /* m, m/s, rad, kg */
typedef struct { double r_t, v_t, gamma_t; } fsw_peg_target_t;

typedef enum {
    FSW_PEG_CONVERGED = 0,       /* all three residuals inside tolerance                       */
    FSW_PEG_NOT_CONVERGED,       /* best effort returned; residuals say how far off            */
    FSW_PEG_BAD_INPUT,           /* non-finite / non-physical input; sol untouched except status */
    FSW_PEG_NUMERIC_FAIL         /* initial prediction was not finite                          */
} fsw_peg_status_t;

typedef struct {
    double A, B, T;              /* steering coefficients and time-to-go                       */
    double res_r, res_v, res_gamma;   /* predicted cutoff minus target                         */
    int    iterations;
    fsw_peg_status_t status;
} fsw_peg_solution_t;

#define FSW_PEG_TOL_R      2.0
#define FSW_PEG_TOL_V      0.02
#define FSW_PEG_TOL_GAMMA  2.0e-5

void fsw_peg_derivs(double r, double v, double gamma, double m, double thrust_n, double mdot,
                    double theta, double mu, double out[4]);
fsw_peg_state_t fsw_peg_rk4_step(const fsw_peg_state_t *s, double thrust_n, double mdot,
                                 double theta, double dt, double mu);
/* n_steps <= 0 -> max(12, ceil(T/1.5)), capped at FSW_PEG_MAX_STEPS. */
fsw_peg_state_t fsw_peg_predict_cutoff(const fsw_peg_state_t *s, double A, double B, double T,
                                       double thrust_n, double mdot, double mu, int n_steps);

/* guess may be NULL (cold start) or {A, B, T}. tol may be NULL (defaults above). */
void fsw_peg_solve(const fsw_peg_state_t *s, double thrust_n, double mdot,
                   const fsw_peg_target_t *target, double m_dry_kg, double mu,
                   const double guess[3], int max_iter, const double tol[3],
                   fsw_peg_solution_t *out);

/* Steering angle (rad from local horizontal) tau seconds after the solve. */
double fsw_peg_theta(double A, double B, double tau);

#endif
