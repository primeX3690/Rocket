#include "pid_controller.h"

#define PI 3.14159265358979323846f

/* --- crude single-precision cosf(), no libm on this bare target ---
 * Degree-10 EVEN minimax polynomial in x (fitted numerically over
 * [-pi, pi], max error ~2.4e-6) -- NOT a truncated Taylor series. A
 * truncated 6th-order Taylor series (1 - x^2/2 + x^4/24 - x^6/720)
 * looks similar in structure but is only accurate near x=0: at x=pi
 * it is off by ~21% (returns -1.211 instead of -1.0), because Taylor
 * series are a LOCAL approximation around one point and get worse the
 * further you evaluate from that point -- this had exactly that bug in
 * an earlier version of this file. A real flight computer would use a
 * vendor CMSIS-DSP arm_cos_f32() (hardware/table-optimized); this
 * fitted minimax polynomial is a deliberately simple, verified-accurate
 * stand-in so the demo has zero external deps. */
static float cos_approx(float x) {
    while (x > PI) x -= 2.0f * PI;
    while (x < -PI) x += 2.0f * PI;
    float x2 = x * x;
    return 9.99999442e-01f
         + x2 * (-4.99995572e-01f
         + x2 * (4.16610230e-02f
         + x2 * (-1.38627136e-03f
         + x2 * (2.42527302e-05f
         + x2 * (-2.21917677e-07f)))));
}

/* sinf() via cos(x - pi/2), same accuracy as cos_approx (was MISSING
 * entirely from an earlier version of this file, forcing the gimbal
 * geometry below to use a small-angle sin(x)~=x approximation that
 * the Python reference (control/tvc_attitude.py) does not make). */
static float sin_approx(float x) {
    return cos_approx(x - PI / 2.0f);
}

/* Matches control/tvc_attitude.py's PIDController: anti-windup via a
 * clamped integral term (default limit = 10x the output limit), not
 * just output saturation -- an earlier version of this file clamped
 * only the output, letting the integral term wind up unboundedly
 * during a saturated period, unlike the Python reference. */
void pid_init(PIDController *pid, float kp, float ki, float kd, float output_limit) {
    pid->kp = kp; pid->ki = ki; pid->kd = kd;
    pid->integral = 0.0f;
    pid->prev_error = 0.0f;
    pid->output_limit = output_limit;
    pid->integral_limit = output_limit * 10.0f;
    pid->has_prev = 0;
}

float pid_update(PIDController *pid, float error, float dt) {
    pid->integral += error * dt;
    if (pid->integral > pid->integral_limit) pid->integral = pid->integral_limit;
    if (pid->integral < -pid->integral_limit) pid->integral = -pid->integral_limit;

    float derivative = pid->has_prev ? (error - pid->prev_error) / dt : 0.0f;
    pid->has_prev = 1;
    pid->prev_error = error;

    float output = pid->kp * error + pid->ki * pid->integral + pid->kd * derivative;
    if (output > pid->output_limit) output = pid->output_limit;
    if (output < -pid->output_limit) output = -pid->output_limit;
    return output;
}

void actuator_init(RealisticActuator *act, float max_slew_rate_rad_s, float tau_s) {
    act->max_slew_rate = max_slew_rate_rad_s;
    act->tau = tau_s;
    act->actual_angle = 0.0f;
}

float actuator_step(RealisticActuator *act, float commanded_angle, float dt) {
    float error = commanded_angle - act->actual_angle;
    float rate = error / act->tau;
    if (rate > act->max_slew_rate) rate = act->max_slew_rate;
    if (rate < -act->max_slew_rate) rate = -act->max_slew_rate;
    float step_move = rate * dt;
    if (step_move > error && step_move > 0) step_move = error;
    if (step_move < error && step_move < 0) step_move = error;
    act->actual_angle += step_move;
    return act->actual_angle;
}

void dynamics_init(RigidBodyPitchDynamics *dyn, float moment_of_inertia, float gimbal_arm) {
    dyn->moment_of_inertia = moment_of_inertia;
    dyn->gimbal_arm = gimbal_arm;
    dyn->pitch = 0.0f;
    dyn->pitch_rate = 0.0f;
}

/* Matches control/tvc_attitude.py's RigidBodyPitchDynamics.step():
 * torque = thrust * sin(gimbal_deflection) * L. An earlier version of
 * this file used `gimbal_angle` directly (a small-angle sin(x)~=x
 * approximation the Python reference does NOT make), which diverges
 * from the Python model as gimbal deflection approaches its +-6deg
 * limit (sin(6deg)=0.1045 vs the angle itself 0.1047 rad -- a small
 * but real, avoidable mismatch given sin_approx already exists here). */
void dynamics_step(RigidBodyPitchDynamics *dyn, float thrust_n, float gimbal_angle,
                    float disturbance_nm, float dt) {
    float torque = thrust_n * sin_approx(gimbal_angle) * dyn->gimbal_arm + disturbance_nm;
    float angular_accel = torque / dyn->moment_of_inertia;
    dyn->pitch_rate += angular_accel * dt;
    dyn->pitch += dyn->pitch_rate * dt;
}

float one_minus_cosine_gust(float t, float gust_start_s, float gust_length_s,
                             float gust_peak_torque_nm) {
    if (t < gust_start_s || t > gust_start_s + gust_length_s) return 0.0f;
    float phase = 2.0f * PI * (t - gust_start_s) / gust_length_s;
    return (gust_peak_torque_nm / 2.0f) * (1.0f - cos_approx(phase));
}
