#include "fsw_core.h"
#include "fsw_telemetry.h"
#include "fsw_math.h"
#include "fsw_peg.h"
#include <string.h>

static float fsw_nan(void)
{
    union { uint32_t u; float f; } x;
    x.u = 0x7FC00000u;
    return x.f;
}

static void put_u32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v & 0xFFu);
    p[1] = (uint8_t)((v >> 8) & 0xFFu);
    p[2] = (uint8_t)((v >> 16) & 0xFFu);
    p[3] = (uint8_t)((v >> 24) & 0xFFu);
}

static void put_f32(uint8_t *p, float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    put_u32(p, u);
}

void fsw_core_default_cfg(fsw_core_cfg_t *cfg)
{
    cfg->dt_s = 0.02f;                         /* 50 Hz, as in embedded/main.c */

    cfg->accel_mon.min = -50.0f;               /* ~ -5 g .. +20 g              */
    cfg->accel_mon.max = 200.0f;
    cfg->accel_mon.max_rate = 5000.0f;         /* staging drops ~1.5e3 m/s^3   */
    cfg->accel_mon.stuck_samples = 25;         /* 0.5 s of bit-identical data  */
    cfg->accel_mon.fail_persist = 3;
    cfg->accel_mon.recover_persist = 50;
    cfg->accel_mon.max_fault_events = 3;

    cfg->gyro_mon.min = -1.0f;                 /* +/- 57 deg/s                 */
    cfg->gyro_mon.max = 1.0f;
    cfg->gyro_mon.max_rate = 50.0f;
    cfg->gyro_mon.stuck_samples = 25;
    cfg->gyro_mon.fail_persist = 3;
    cfg->gyro_mon.recover_persist = 50;
    cfg->gyro_mon.max_fault_events = 3;

    cfg->accel_vote_tol = 1.0f;
    cfg->gyro_vote_tol = 0.02f;
    cfg->min_trusted_imus = 2;
    cfg->crit_persist_ticks = 5;

    cfg->state.thrust_accel_mps2 = 11.0f;      /* > 1 g (9.81) at rest on pad  */
    cfg->state.thrust_accel_s2_mps2 = 4.0f;    /* upper stages can be well below 1 g */
    cfg->state.cutoff_accel_mps2 = 2.0f;
    cfg->state.ignition_timeout_s = 3.0f;
    cfg->state.min_burn_s = 4.0f;
    cfg->state.sep_delay_s = 1.5f;
    cfg->state.confirm_ticks = 5;

    cfg->kp = 0.3f; cfg->ki = 0.05f; cfg->kd = 0.3f;  /* stable for plant K=T*L/I=48; see README */
    cfg->gimbal_limit_rad = 0.10472f;                  /* +/- 6 deg                    */
    cfg->tlm_every_n = 5;                              /* 10 Hz housekeeping           */
    cfg->wdg_miss_limit = 3;

    fsw_eskf_default_cfg(&cfg->eskf);
    cfg->eskf.nis_gate = 11.34f;               /* chi-square, 3 dof, 99 %      */
    cfg->eskf.max_consecutive_rejects = 5;
    cfg->gps_r_pos = 9.0f;                     /* (3 m)^2                      */
    cfg->gps_default_latency_s = 0.0f;
    cfg->pad_gyro_bias_tau_s = 0.5f;
    cfg->pitch_cmd_rate_limit_rps = 0.1f;      /* 5.7 deg/s */
    cfg->guid_state_max_age_s = 1.0f;
    cfg->pad_rest_rate_rps = 0.05f;
    cfg->att_sigma_rollpitch_rad = 2.0e-3f;
    cfg->att_sigma_yaw_rad = 1.0e-2f;
    cfg->nav_eci = 0;
    cfg->nav_j2 = 1;
    cfg->site_lat_rad = 13.72 * 3.14159265358979323846 / 180.0;
    cfg->site_lon_rad = 80.23 * 3.14159265358979323846 / 180.0;
    cfg->site_radius_m = 6378137.0 + 10.0;
    cfg->launch_azimuth_rad = 0.5 * 3.14159265358979323846;
    cfg->omega_earth = FSW_OMEGA_EARTH;
    cfg->pad_bias_cal_enabled = 1;
    cfg->uncal_gyro_bias_sigma_rps = 0.0175f;   /* 1 deg/s */
}

void fsw_core_init(fsw_core_t *c, const fsw_hal_t *hal, const fsw_core_cfg_t *cfg)
{
    int i, a;
    memset(c, 0, sizeof(*c));
    c->hal = *hal;
    c->cfg = *cfg;
    for (i = 0; i < 3; i++) {
        for (a = 0; a < 3; a++) {
            fsw_monitor_init(&c->acc_mon[i][a], &cfg->accel_mon);
            fsw_monitor_init(&c->gyr_mon[i][a], &cfg->gyro_mon);
        }
    }
    fsw_state_init(&c->st, &cfg->state);
    fsw_wdg_init(&c->wdg, FSW_TASK_ALL, cfg->wdg_miss_limit);
    pid_init(&c->pid, cfg->kp, cfg->ki, cfg->kd, cfg->gimbal_limit_rad);
    c->pad_q[0] = 1.0f;
    c->nav_ok = 1;
    if (cfg->nav_eci) {
        int i;
        fsw_site_init(&c->site, cfg->site_lat_rad, cfg->site_lon_rad, cfg->site_radius_m, cfg->launch_azimuth_rad, cfg->omega_earth);
        c->cfg.eskf.gravity_model = cfg->nav_j2 ? FSW_ESKF_GRAV_CENTRAL_J2 : FSW_ESKF_GRAV_CENTRAL;
        for (i = 0; i < 3; i++) { c->cfg.eskf.polar_axis[i] = c->site.axis[i]; }
    }
}

void fsw_core_cmd_arm(fsw_core_t *c)     { c->in.cmd_arm = 1; }
void fsw_core_cmd_disarm(fsw_core_t *c)  { c->in.cmd_disarm = 1; }
void fsw_core_cmd_launch(fsw_core_t *c)  { c->in.cmd_launch = 1; }
void fsw_core_cmd_abort(fsw_core_t *c)   { c->in.cmd_abort = 1; }
void fsw_core_set_insertion_ok(fsw_core_t *c, int ok) { c->in.insertion_ok = (uint8_t)(ok != 0); }
void fsw_core_set_pitch_cmd(fsw_core_t *c, float r)   { c->pitch_cmd_ext = r; }

void fsw_core_attach_guidance(fsw_core_t *c, fsw_guidance_t *g)
{
    c->guid = g;
    c->guid_flags = (uint8_t)(g != 0 ? FSW_GUID_ATTACHED : 0u);
}

static void publish_polar(fsw_core_t *c, double r_m, double v_mps, double gamma_rad, double mass_kg, double psi_rad)
{
    c->polar_seq++;                         /* odd: writing */
    FSW_BARRIER();
    c->polar.r = r_m; c->polar.v = v_mps; c->polar.gamma = gamma_rad; c->polar.m = mass_kg;
    c->polar_psi_rad = psi_rad;
    c->polar_stamp_s = (double)c->cycle * (double)c->cfg.dt_s;
    c->polar_valid = 1;
    FSW_BARRIER();
    c->polar_seq++;                         /* even: stable */
}

void fsw_core_set_polar_state(fsw_core_t *c, double r_m, double v_mps, double gamma_rad, double mass_kg, double psi_rad)
{
    if (c->cfg.nav_eci) { return; }         /* ECI mode derives it from the filter; an external value is ignored */
    publish_polar(c, r_m, v_mps, gamma_rad, mass_kg, psi_rad);
}

void fsw_core_set_stage2_mass(fsw_core_t *c, double m0_kg)
{
    c->stage2_m0_kg = m0_kg;
    c->mass_est_kg = m0_kg;
}

void fsw_core_guidance_service(fsw_core_t *c)
{
    uint32_t s1, s2;
    fsw_peg_state_t est;
    double stamp;
    if (c->guid == 0 || !c->guid_request) {
        return;
    }
    s1 = c->polar_seq;
    FSW_BARRIER();
    if (s1 & 1u) { return; }                /* nav bridge is mid-write: try again next call */
    est = c->polar;
    stamp = c->polar_stamp_s;
    FSW_BARRIER();
    s2 = c->polar_seq;
    if (s1 != s2) { return; }
    c->guid_request = 0;
    (void)fsw_guidance_update(c->guid, stamp, &est);
    c->guid_serviced++;
}

void fsw_core_gps_fix_aged(fsw_core_t *c, const float pos_m[3], float age_s)
{
    if (c->cfg.nav_eci) { c->gps_dropped++; return; }     /* flat-frame fix in ECI mode: wrong frame */
    if (!fsw_isfinitef(age_s) || age_s < 0.0f ||
        !fsw_isfinitef(pos_m[0]) || !fsw_isfinitef(pos_m[1]) || !fsw_isfinitef(pos_m[2])) {
        c->gps_dropped++;
        return;
    }
    c->gps_pos[0] = (double)pos_m[0]; c->gps_pos[1] = (double)pos_m[1]; c->gps_pos[2] = (double)pos_m[2];
    c->gps_age_s = age_s;
    c->gps_is_ecef = 0;
    c->gps_pending = 1;
}

void fsw_core_gps_fix_ecef(fsw_core_t *c, const double ecef_m[3], float age_s)
{
    int i;
    if (!c->cfg.nav_eci || !fsw_isfinitef(age_s) || age_s < 0.0f) { c->gps_dropped++; return; }
    for (i = 0; i < 3; i++) {
        double x = ecef_m[i];
        if (!(x == x) || x > 1.0e12 || x < -1.0e12) { c->gps_dropped++; return; }
    }
    c->gps_pos[0] = ecef_m[0]; c->gps_pos[1] = ecef_m[1]; c->gps_pos[2] = ecef_m[2];
    c->gps_age_s = age_s;
    c->gps_is_ecef = 1;
    c->gps_pending = 1;
}

void fsw_core_gps_fix(fsw_core_t *c, const float pos_m[3])
{
    fsw_core_gps_fix_aged(c, pos_m, c->cfg.gps_default_latency_s);
}

/* Apply the pending fix, moved forward from its measurement instant to the newest nav state. */
static void apply_gps_fix(fsw_core_t *c)
{
    /* The fix was taken `age` before the call; the call happens at the START of this cycle while the newest
     * stored state is the END of this cycle, so the measurement lies (age/dt + 1) entries back -- a FRACTIONAL
     * number: we interpolate (curvature error a*dt^2/8 ~ 5e-4 m) instead of rounding (up to dt/2 * v = 77 m at
     * orbital speed). */
    double back_f = (double)c->gps_age_s / (double)c->cfg.dt_s + 1.0;
    uint32_t b0;
    double frac, z[3], then[3];
    float r_eff, vvar, age_tot;
    int a;
    c->gps_pending = 0;
    if (back_f > (double)(FSW_GPS_HIST - 2)) {
        c->gps_dropped++;
        return;
    }
    b0 = (uint32_t)back_f;
    frac = back_f - (double)b0;
    if (c->nav_cycles < b0 + (frac > 0.0 ? 1u : 0u)) {
        c->gps_dropped++;                      /* measurement predates the stored history / liftoff */
        return;
    }
    {
        const double *now = c->nav_pos_hist[c->nav_cycles % FSW_GPS_HIST];
        const double *e0 = c->nav_pos_hist[(c->nav_cycles - b0) % FSW_GPS_HIST];
        const double *e1 = (frac > 0.0) ? c->nav_pos_hist[(c->nav_cycles - b0 - 1u) % FSW_GPS_HIST] : e0;
        for (a = 0; a < 3; a++) { then[a] = (1.0 - frac) * e0[a] + frac * e1[a]; }
        if (c->gps_is_ecef) {
            double t_meas = ((double)c->nav_cycles - back_f) * (double)c->cfg.dt_s;     /* seconds since t_L */
            fsw_ecef_to_inertial(&c->site, c->gps_pos, t_meas, z);
        } else {
            for (a = 0; a < 3; a++) { z[a] = c->gps_pos[a]; }
        }
        for (a = 0; a < 3; a++) { z[a] += now[a] - then[a]; }
    }
    age_tot = (float)(back_f * (double)c->cfg.dt_s);
    vvar = (c->nav.P[3][3] + c->nav.P[4][4] + c->nav.P[5][5]) * (1.0f / 3.0f);
    r_eff = c->cfg.gps_r_pos + age_tot * age_tot * vvar;
    if (back_f > 1.5) { c->gps_compensated++; }
    fsw_eskf_update_position(&c->nav, z, r_eff);
}

static void send_housekeeping(fsw_core_t *c)
{
    uint8_t payload[FSW_TLM_HK_LEN];
    uint8_t frame[FSW_TLM_MAX_FRAME];
    size_t n;

    payload[0] = (uint8_t)c->st.mode;
    payload[1] = (uint8_t)c->st.last_reason;
    payload[2] = c->fault_flags;
    payload[3] = (uint8_t)((c->acc_vote[FSW_AXIS_AXIAL].n_used << 4) |
                           (c->gyr_vote[FSW_AXIS_PITCH_GYRO].n_used & 0x0Fu));
    put_u32(&payload[4], c->cycle);
    put_f32(&payload[8], c->acc_vote[FSW_AXIS_AXIAL].value);
    put_f32(&payload[12], c->gyr_vote[FSW_AXIS_PITCH_GYRO].value);
    put_f32(&payload[16], c->pitch_est);
    put_f32(&payload[20], c->gimbal_cmd);
    if (c->nav_active && c->cfg.nav_eci) {
        double rr = fsw_sqrt_d(c->nav.p[0]*c->nav.p[0] + c->nav.p[1]*c->nav.p[1] + c->nav.p[2]*c->nav.p[2]);
        double vr = (rr > 0.0) ? (c->nav.p[0]*c->nav.v[0] + c->nav.p[1]*c->nav.v[1] + c->nav.p[2]*c->nav.v[2]) / rr : 0.0;
        put_f32(&payload[24], (float)(rr - c->cfg.site_radius_m));   /* altitude above the pad radius */
        put_f32(&payload[28], (float)vr);                            /* radial speed                  */
    } else {
        put_f32(&payload[24], c->nav_active ? (float)c->nav.p[2] : 0.0f);
        put_f32(&payload[28], c->nav_active ? (float)c->nav.v[2] : 0.0f);
    }
    payload[32] = c->guid_flags;
    payload[33] = 0; payload[34] = 0; payload[35] = 0;
    {
        float ttg = -1.0f;
        if (c->guid != 0 && c->guid_cache.valid) {
            ttg = (float)fsw_guid_time_to_go(&c->guid_cache, (double)c->cycle * (double)c->cfg.dt_s);
        }
        put_f32(&payload[36], ttg);                                 /* PEG time-to-go, -1 = none */
    }

    n = fsw_tlm_encode(frame, sizeof(frame), FSW_TLM_TYPE_HK, c->tlm_seq++,
                       payload, (uint8_t)FSW_TLM_HK_LEN);
    if (n > 0u && c->hal.tlm_write != 0) {
        c->hal.tlm_write(c->hal.ctx, frame, n);
        c->tlm_frames_sent++;
    }
}

/* v_body = R(q)^T v_inertial */
static void rotate_to_body(const float q[4], const double v[3], double out[3])
{
    double w = q[0], x = q[1], y = q[2], z = q[3];
    double R00 = 1.0 - 2.0*(y*y + z*z), R01 = 2.0*(x*y - w*z),       R02 = 2.0*(x*z + w*y);
    double R10 = 2.0*(x*y + w*z),       R11 = 1.0 - 2.0*(x*x + z*z), R12 = 2.0*(y*z - w*x);
    double R20 = 2.0*(x*z - w*y),       R21 = 2.0*(y*z + w*x),       R22 = 1.0 - 2.0*(x*x + y*y);
    out[0] = R00*v[0] + R10*v[1] + R20*v[2];
    out[1] = R01*v[0] + R11*v[1] + R21*v[2];
    out[2] = R02*v[0] + R12*v[1] + R22*v[2];
}

static float pitch_of_quat(const float q[4])
{
    return fsw_atan2f(2.0f * (q[1] * q[3] + q[0] * q[2]), 1.0f - 2.0f * (q[1] * q[1] + q[2] * q[2]));
}

static float wrap_pi(float a)
{
    while (a > FSW_PI_F)  { a -= 2.0f * FSW_PI_F; }
    while (a < -FSW_PI_F) { a += 2.0f * FSW_PI_F; }
    return a;
}

/* All six voted axes carry at least one usable channel. */
static int all_axes_have_data(const fsw_core_t *c)
{
    int a;
    for (a = 0; a < 3; a++) {
        if (c->acc_vote[a].n_used < 1u || c->gyr_vote[a].n_used < 1u) {
            return 0;
        }
    }
    return 1;
}

static void pad_alignment(fsw_core_t *c)
{
    float f[3], w[3];
    int a;
    if (!all_axes_have_data(c)) {
        return;
    }
    for (a = 0; a < 3; a++) {
        f[a] = c->acc_vote[a].value;
        w[a] = c->gyr_vote[a].value;
    }
    /* Gyro bias: exponential average while the vehicle is at rest. */
    {
        float rest = c->cfg.pad_rest_rate_rps;
        float alpha = c->cfg.dt_s / c->cfg.pad_gyro_bias_tau_s;
        int at_rest = (fsw_absf(w[0] - c->pad_bg[0]) < rest) &&
                      (fsw_absf(w[1] - c->pad_bg[1]) < rest) &&
                      (fsw_absf(w[2] - c->pad_bg[2]) < rest);
        if (alpha > 1.0f) { alpha = 1.0f; }
        if (!c->cfg.pad_bias_cal_enabled) {
            /* pad_bg stays at zero */
        } else if (!c->pad_bg_seeded) {
            for (a = 0; a < 3; a++) { c->pad_bg[a] = w[a]; }
            c->pad_bg_seeded = 1;
        } else if (at_rest) {
            for (a = 0; a < 3; a++) { c->pad_bg[a] += alpha * (w[a] - c->pad_bg[a]); }
        }
    }
    /* Leveling from the AVERAGED specific-force direction (valid only if |f| ~ g).
     * A single sample is far too noisy: sigma_tilt ~ sigma_accel / g (~3 mrad here). */
    {
        float alpha = c->cfg.dt_s / c->cfg.pad_gyro_bias_tau_s;
        float n2, n;
        if (alpha > 1.0f) { alpha = 1.0f; }
        if (!c->pad_f_seeded) {
            for (a = 0; a < 3; a++) { c->pad_f[a] = f[a]; }
            c->pad_f_seeded = 1;
        } else {
            for (a = 0; a < 3; a++) { c->pad_f[a] += alpha * (f[a] - c->pad_f[a]); }
        }
        for (a = 0; a < 3; a++) { f[a] = c->pad_f[a]; }
        n2 = f[0]*f[0] + f[1]*f[1] + f[2]*f[2];
        n = fsw_sqrtf(n2);
        if (n > 0.8f * 9.80665f && n < 1.2f * 9.80665f) {
            float pitch = fsw_atan2f(-f[0], fsw_sqrtf(f[1]*f[1] + f[2]*f[2]));
            float roll = fsw_atan2f(f[1], f[2]);
            float cy = fsw_cosf(0.5f * pitch), sy = fsw_sinf(0.5f * pitch);
            float cx = fsw_cosf(0.5f * roll),  sx = fsw_sinf(0.5f * roll);
            /* q = qy(pitch) (x) qx(roll) */
            float q0 = cy * cx, q1 = cy * sx, q2 = sy * cx, q3 = -sy * sx;
            float nn = fsw_sqrtf(q0*q0 + q1*q1 + q2*q2 + q3*q3);
            c->pad_q[0] = q0 / nn; c->pad_q[1] = q1 / nn; c->pad_q[2] = q2 / nn; c->pad_q[3] = q3 / nn;
            c->pad_aligned = 1;
        }
    }
}

void fsw_core_step(fsw_core_t *c)
{
    fsw_imu_sample_t s[3];
    uint8_t min_n;
    int i, a, thrusting;
    fsw_mode_t prev_mode, mode;

    /* --- 1. sensors ------------------------------------------------------ */
    for (i = 0; i < 3; i++) {
        if (c->hal.read_imu(c->hal.ctx, i, &s[i]) != 0) {
            for (a = 0; a < 3; a++) {
                s[i].accel[a] = fsw_nan();            /* bus error => bad samples */
                s[i].gyro[a] = fsw_nan();
            }
        }
    }
    fsw_wdg_checkin(&c->wdg, FSW_TASK_SENSORS);

    /* --- 2. FDIR + voting (every axis of every IMU) ---------------------- */
    c->fault_flags = 0;
    for (a = 0; a < 3; a++) {
        float av[3], gv[3];
        uint8_t aok[3], gok[3];
        for (i = 0; i < 3; i++) {
            fsw_sensor_status_t sa = fsw_monitor_update(&c->acc_mon[i][a], s[i].accel[a], c->cfg.dt_s);
            fsw_sensor_status_t sg = fsw_monitor_update(&c->gyr_mon[i][a], s[i].gyro[a], c->cfg.dt_s);
            av[i] = s[i].accel[a];
            gv[i] = s[i].gyro[a];
            aok[i] = (uint8_t)(sa == FSW_SENSOR_OK);
            gok[i] = (uint8_t)(sg == FSW_SENSOR_OK);
            if (sa != FSW_SENSOR_OK || sg != FSW_SENSOR_OK) {
                c->fault_flags = (uint8_t)(c->fault_flags | (1u << i));
            }
        }
        c->acc_vote[a] = fsw_vote3(av, aok, c->cfg.accel_vote_tol);
        c->gyr_vote[a] = fsw_vote3(gv, gok, c->cfg.gyro_vote_tol);
    }

    min_n = 3;
    for (a = 0; a < 3; a++) {
        if (c->acc_vote[a].n_used < min_n) { min_n = c->acc_vote[a].n_used; }
        if (c->gyr_vote[a].n_used < min_n) { min_n = c->gyr_vote[a].n_used; }
    }
    c->in.imu_trusted = (uint8_t)(min_n >= c->cfg.min_trusted_imus);
    if (min_n < 3u) {
        c->fault_flags |= FSW_FLAG_DEGRADED;
    }
    if (!c->in.imu_trusted) {
        c->fault_flags |= FSW_FLAG_TRUST_LOST;
        if (c->crit_count < 65535u) { c->crit_count++; }
    } else {
        c->crit_count = 0;
    }
    c->in.accel_mps2 = c->acc_vote[FSW_AXIS_AXIAL].value;
    fsw_wdg_checkin(&c->wdg, FSW_TASK_FDIR);

    /* --- 3. navigation --------------------------------------------------- */
    prev_mode = c->st.mode;
    c->nav_stale = 0;
    if (prev_mode == FSW_PAD_SAFE || prev_mode == FSW_PAD_ARMED) {
        pad_alignment(c);
        c->pitch_est = pitch_of_quat(c->pad_q);
    } else if (prev_mode == FSW_IGNITION) {
        /* alignment frozen */
    } else if (c->nav_active) {
        if (all_axes_have_data(c)) {
            float g[3], f[3];
            for (a = 0; a < 3; a++) { f[a] = c->acc_vote[a].value; g[a] = c->gyr_vote[a].value; }
            fsw_eskf_predict(&c->nav, g, f, c->cfg.dt_s);
            c->nav_cycles++;
            memcpy(c->nav_pos_hist[c->nav_cycles % FSW_GPS_HIST], c->nav.p, sizeof(c->nav.p));
            if (c->gps_pending) {
                apply_gps_fix(c);
            }
            if (c->cfg.nav_eci && c->guid != 0) {
                /* guided-stage mass: m0 - mdot * (time since thrust was confirmed), model not measurement */
                if (prev_mode == FSW_ASCENT_S2 && c->stage2_m0_kg > 0.0) {
                    double burnt = c->st.thrust_seen ? ((double)c->st.t_burn_s + (double)c->cfg.state.confirm_ticks * (double)c->cfg.dt_s) : 0.0;
                    c->mass_est_kg = c->stage2_m0_kg - c->guid->mdot * burnt;
                }
                {
                    double pr, pv, pg, ppsi;
                    fsw_polar_from_inertial(c->nav.p, c->nav.v, &pr, &pv, &pg, &ppsi);
                    publish_polar(c, pr, pv, pg, c->mass_est_kg, ppsi);
                }
            }
        } else {
            c->nav_stale = 1;
            if (c->gps_pending) { c->gps_age_s += c->cfg.dt_s; }   /* the fix keeps ageing while we cannot use it */
        }
    }
    if (c->nav_active) {
        c->nav_ok = (uint8_t)fsw_eskf_healthy(&c->nav);
        if (c->nav_ok) {
            c->pitch_est = fsw_eskf_pitch(&c->nav);
            c->nav_bad_count = 0;
        } else if (c->nav_bad_count < 65535u) {
            c->nav_bad_count++;
        }
        if (!c->nav_ok) { c->fault_flags |= FSW_FLAG_NAV_FAULT; }
        if (c->nav_stale) { c->fault_flags |= FSW_FLAG_NAV_STALE; }
    }
    c->in.fdir_critical = (uint8_t)((c->crit_count >= c->cfg.crit_persist_ticks) ||
                                    (c->nav_bad_count >= c->cfg.crit_persist_ticks));

    /* --- 4. state machine ------------------------------------------------ */
    mode = fsw_state_step(&c->st, &c->in, c->cfg.dt_s);
    c->in.cmd_arm = c->in.cmd_disarm = c->in.cmd_launch = c->in.cmd_abort = 0;
    fsw_wdg_checkin(&c->wdg, FSW_TASK_STATE);

    if (prev_mode == FSW_IGNITION && mode == FSW_ASCENT_S1) {
        /* Liftoff: start free-running navigation from the pad alignment. */
        if (c->cfg.nav_eci) {
            double p0[3], v0[3], w[3], wb[3];
            float bg0[3];
            fsw_site_pad_state(&c->site, p0, v0);
            fsw_site_earth_rate(&c->site, w);
            rotate_to_body(c->pad_q, w, wb);
            /* On the pad the gyros read bias + the Earth's rotation seen in the body frame; remove the latter. */
            for (a = 0; a < 3; a++) { bg0[a] = c->cfg.pad_bias_cal_enabled ? (float)((double)c->pad_bg[a] - wb[a]) : 0.0f; }
            fsw_eskf_init(&c->nav, &c->cfg.eskf, c->pad_q, v0, p0, bg0, 0);
        } else {
            fsw_eskf_init(&c->nav, &c->cfg.eskf, c->pad_q, 0, 0, c->pad_bg, 0);
        }
        fsw_eskf_set_att_sigma(&c->nav, c->cfg.att_sigma_rollpitch_rad, c->cfg.att_sigma_rollpitch_rad, c->cfg.att_sigma_yaw_rad);
        if (!c->cfg.pad_bias_cal_enabled) {
            /* Zero-rate offset is unknown up to the datasheet tolerance: say so, or the
             * filter will (wrongly) trust a 0.001 rad/s prior and be slow to learn it. */
            float v = c->cfg.uncal_gyro_bias_sigma_rps * c->cfg.uncal_gyro_bias_sigma_rps;
            fsw_eskf_set_P_diag(&c->nav, 9, v);
            fsw_eskf_set_P_diag(&c->nav, 10, v);
            fsw_eskf_set_P_diag(&c->nav, 11, v);
        }
        c->nav_active = 1;
        c->nav_ok = 1;
        c->nav_bad_count = 0;
        c->gps_pending = 0;
        c->nav_cycles = 0;
        memcpy(c->nav_pos_hist[0], c->nav.p, sizeof(c->nav.p));
        c->pitch_est = fsw_eskf_pitch(&c->nav);
    }

    /* --- 4b. guidance (guided stage = ASCENT_S2) ------------------------ */
    {
        float target = c->pitch_cmd_ext;
        double t_now = (double)c->cycle * (double)c->cfg.dt_s;
        if (c->guid != 0) {
            fsw_guid_solution_t tmp;
            uint8_t gf = (uint8_t)(c->guid_flags & (FSW_GUID_ATTACHED | FSW_GUID_CUTOFF_SENT | FSW_GUID_INSERTED));   /* these latch */
            if (fsw_guidance_read(c->guid, &tmp)) { c->guid_cache = tmp; }
            if (mode == FSW_ASCENT_S2) {
                double theta;
                int fresh = c->polar_valid && ((t_now - c->polar_stamp_s) <= (double)c->cfg.guid_state_max_age_s);
                if (fsw_guid_theta(&c->guid_cache, t_now, &theta)) {
                    gf |= FSW_GUID_VALID;
                    if (c->guid_cache.degraded) { gf |= FSW_GUID_DEGRADED; }
                    target = (float)(1.5707963267948966 - theta + (c->polar_valid ? c->polar_psi_rad : 0.0));
                } else {
                    gf |= FSW_GUID_LOST;                 /* flying the external pitch program */
                }
                if (!fresh) { gf |= FSW_GUID_STALE_STATE; }
                if (!c->cutoff_sent) {
                    if (fresh && !c->guid_request && (t_now - c->last_guid_req_s) >= c->guid->cfg.cycle_s) {
                        c->guid_request = 1;
                        c->last_guid_req_s = t_now;
                    }
                    if (fsw_guid_cutoff_due(&c->guid_cache, t_now)) {
                        c->cutoff_sent = 1;
                        if (c->hal.engine_cutoff != 0) { c->hal.engine_cutoff(c->hal.ctx); }
                    }
                }
            }
            if (c->cutoff_sent) { gf |= FSW_GUID_CUTOFF_SENT; }
            if (mode == FSW_COAST && c->polar_valid &&
                (t_now - c->polar_stamp_s) <= (double)c->cfg.guid_state_max_age_s) {   /* never judge insertion on stale data */
                int ok = fsw_guidance_insertion_ok(&c->guid->cfg, &c->polar);
                c->in.insertion_ok = (uint8_t)ok;
                if (ok) { gf |= FSW_GUID_INSERTED; }
            }
            c->guid_flags = gf;
        }
        if (!c->nav_active) {
            c->pitch_cmd = c->pitch_est;                 /* no slewing from a stale value before liftoff */
        } else {
            float step = c->cfg.pitch_cmd_rate_limit_rps * c->cfg.dt_s;
            float d = wrap_pi(target - c->pitch_cmd);
            if (d > step) { d = step; }
            if (d < -step) { d = -step; }
            c->pitch_cmd += d;
        }
    }

    /* --- 5. control ------------------------------------------------------ */
    thrusting = fsw_mode_is_thrusting(mode);
    if (thrusting && c->nav_active && c->nav_ok && !c->nav_stale) {
        c->gimbal_cmd = pid_update(&c->pid, wrap_pi(c->pitch_cmd - c->pitch_est), c->cfg.dt_s);
    } else {
        c->gimbal_cmd = 0.0f;                        /* nozzle centred when not thrusting / blind */
        pid_init(&c->pid, c->cfg.kp, c->cfg.ki, c->cfg.kd, c->cfg.gimbal_limit_rad);
    }
    c->hal.write_gimbal(c->hal.ctx, c->gimbal_cmd);
    fsw_wdg_checkin(&c->wdg, FSW_TASK_CONTROL);

    /* --- 6. telemetry ---------------------------------------------------- */
    if (c->cfg.tlm_every_n > 0u && (c->cycle % c->cfg.tlm_every_n) == 0u) {
        send_housekeeping(c);
    }
    fsw_wdg_checkin(&c->wdg, FSW_TASK_TLM);

    /* --- 7. watchdog ----------------------------------------------------- */
    if (fsw_wdg_service(&c->wdg) && c->hal.kick_watchdog != 0) {
        c->hal.kick_watchdog(c->hal.ctx);
    }
    if (c->wdg.tripped) {
        c->fault_flags |= FSW_FLAG_WDG_TRIPPED;
    }
    c->cycle++;
}
