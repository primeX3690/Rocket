/*
 * embedded/pid_controller.h
 *
 * C port of control/tvc_attitude.py's PIDController + the actuator/
 * rigid-body model from faults/real_world.py, for on-target (Cortex-M)
 * timing verification. This is not a rewrite-for-fun -- it is the
 * literal same control law, ported so it can be compiled for the real
 * instruction set a flight computer uses and profiled for worst-case
 * execution time (WCET) instead of Python wall-clock time, which tells
 * you nothing about a real MCU's timing budget.
 */

#ifndef PID_CONTROLLER_H
#define PID_CONTROLLER_H

typedef struct {
    float kp, ki, kd;
    float integral;
    float prev_error;
    float output_limit;
    int has_prev;
} PIDController;

void pid_init(PIDController *pid, float kp, float ki, float kd, float output_limit);
float pid_update(PIDController *pid, float error, float dt);

typedef struct {
    float max_slew_rate;   /* rad/s */
    float tau;              /* s */
    float actual_angle;     /* rad */
} RealisticActuator;

void actuator_init(RealisticActuator *act, float max_slew_rate_rad_s, float tau_s);
float actuator_step(RealisticActuator *act, float commanded_angle, float dt);

typedef struct {
    float moment_of_inertia;
    float gimbal_arm;
    float pitch;
    float pitch_rate;
} RigidBodyPitchDynamics;

void dynamics_init(RigidBodyPitchDynamics *dyn, float moment_of_inertia, float gimbal_arm);
void dynamics_step(RigidBodyPitchDynamics *dyn, float thrust_n, float gimbal_angle,
                    float disturbance_nm, float dt);

float one_minus_cosine_gust(float t, float gust_start_s, float gust_length_s,
                             float gust_peak_torque_nm);

#endif
