#include "pid_controller.h"

#define PI 3.14159265358979323846f

/* --- crude single-precision cosf(), no libm on this bare target ---
 * 6th-order Taylor/minimax-ish approximation, range-reduced to
 * [-pi, pi], which is all the gust model ever needs. A real flight
 * computer either uses a vendor CMSIS-DSP arm_cos_f32() (hardware/
 * table-optimized) or a proper minimax polynomial -- this is a
 * deliberately simple stand-in so the demo has zero external deps. */
static float cos_approx(float x) {
    while (x > PI) x -= 2.0f * PI;
    while (x < -PI) x += 2.0f * PI;
    float x2 = x * x;
    return 1.0f - x2 / 2.0f + (x2 * x2) / 24.0f - (x2 * x2 * x2) / 720.0f;
}

void pid_init(PIDController *pid, float kp, float ki, float kd, float output_limit) {
    pid->kp = kp; pid->ki = ki; pid->kd = kd;
    pid->integral = 0.0f;
    pid->prev_error = 0.0f;
    pid->output_limit = output_limit;
    pid->has_prev = 0;
}

float pid_update(PIDController *pid, float error, float dt) {
    pid->integral += error * dt;
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

void dynamics_step(RigidBodyPitchDynamics *dyn, float thrust_n, float gimbal_angle,
                    float disturbance_nm, float dt) {
    float torque = thrust_n * dyn->gimbal_arm * gimbal_angle + disturbance_nm;
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
