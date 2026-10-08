/* test/fsw_selftest.c -- host-side verification of the flight-software layer.
 *
 * Part A: unit tests (math layer vs libm, CRC, telemetry framing/resync, FDIR
 *         monitor, voter, state machine, watchdog). The ESKF itself is verified
 *         against the Python reference in tests/test_eskf_c_equivalence.py.
 * Part B: closed-loop scenarios. The REAL fsw_core runs against a simulated
 *         vehicle: scripted engines that light/cut according to what the
 *         flight software commands, a rigid-body pitch plant + rate-limited
 *         gimbal (re-using embedded/pid_controller.c), three noisy IMUs with
 *         injectable faults. No heap, no libm, plain C99.
 *
 * The plant is a pitch-axis toy: axial acceleration is scripted, not
 * integrated from mass/thrust; truth position/velocity are integrated from it
 * with the same flat-Earth kinematics the ESKF assumes. It validates
 * flight-software LOGIC (modes, FDIR, navigation plumbing, watchdog, framing),
 * not vehicle performance. */
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "../fsw_core.h"
#include "../fsw_crc.h"
#include "../fsw_telemetry.h"
#include "../fsw_math.h"
#include "../fsw_peg.h"
#include "../fsw_guidance.h"
#include <time.h>
#include <math.h>

static int g_pass = 0, g_fail = 0;
#define CHECK(name, cond) do { if (cond) { g_pass++; printf("  [PASS] %s\n", name); } \
                               else { g_fail++; printf("  [FAIL] %s\n", name); } } while (0)

/* ======================= Part A: unit tests ======================= */


static void test_math(void)
{
    double ms = 0, mc = 0, ma = 0, m2 = 0, mq = 0, msq = 0;
    int i, j;
    printf("Math layer vs libm\n");
    for (i = -20000; i <= 20000; i += 3) {
        double x = i * 0.001, e;
        e = fabs((double)fsw_sinf((float)x) - sin(x)); if (e > ms) ms = e;
        e = fabs((double)fsw_cosf((float)x) - cos(x)); if (e > mc) mc = e;
    }
    for (i = -100000; i <= 100000; i += 7) {
        double x = i * 0.001, e = fabs((double)fsw_atanf((float)x) - atan(x)); if (e > ma) ma = e;
    }
    for (i = -300; i <= 300; i += 3) {
        for (j = -300; j <= 300; j += 7) {
            double e;
            if (i == 0 && j == 0) continue;
            e = fabs((double)fsw_atan2f((float)i, (float)j) - atan2((double)i, (double)j)); if (e > m2) m2 = e;
        }
    }
    for (i = -1000; i <= 1000; i++) {
        double x = i * 0.001, e = fabs((double)fsw_asinf((float)x) - asin(x)); if (e > mq) mq = e;
    }
    for (i = 1; i < 2000000; i += 997) {
        double x = i * 1e-3, e = fabs((double)fsw_sqrtf((float)x) - sqrt(x)) / sqrt(x); if (e > msq) msq = e;
    }
    CHECK("sin abs error < 5e-6", ms < 5e-6);
    CHECK("cos abs error < 5e-6", mc < 5e-6);
    CHECK("atan abs error < 2e-5", ma < 2e-5);
    CHECK("atan2 abs error < 2e-5 over all quadrants", m2 < 2e-5);
    CHECK("asin abs error < 3e-5 on [-1,1]", mq < 3e-5);
    CHECK("sqrt relative error < 3e-7", msq < 3e-7);
    CHECK("sqrt(0)=0, sqrt(-1)=0", fsw_sqrtf(0.0f) == 0.0f && fsw_sqrtf(-1.0f) == 0.0f);
    CHECK("sqrt(NaN) is NaN (not silently 0)", fsw_sqrtf(0.0f / 0.0f * 0.0f + (float)NAN) != fsw_sqrtf(0.0f / 0.0f * 0.0f + (float)NAN));
    CHECK("atan2(0,0)=0 and asin clamps beyond +-1", fsw_atan2f(0.0f, 0.0f) == 0.0f && fsw_asinf(1.5f) > 1.57f && fsw_asinf(-1.5f) < -1.57f);
}

static void test_crc(void)
{
    const uint8_t v[] = { '1','2','3','4','5','6','7','8','9' };
    printf("CRC-16/CCITT-FALSE\n");
    CHECK("check value 0x29B1 for \"123456789\"", fsw_crc16_ccitt(v, 9, 0xFFFF) == 0x29B1);
    CHECK("empty input returns init", fsw_crc16_ccitt(v, 0, 0xFFFF) == 0xFFFF);
}

static void test_telemetry(void)
{
    uint8_t buf[256], pay[64];
    fsw_tlm_frame_t f;
    fsw_tlm_parser_t p;
    size_t n, i;
    int got;

    printf("Telemetry framing\n");
    for (i = 0; i < 64; i++) pay[i] = (uint8_t)(i * 7 + 3);

    n = fsw_tlm_encode(buf, sizeof(buf), 0x42, 9, pay, 10);
    CHECK("encode length = 5 + len + 2", n == 17);
    CHECK("encode rejects payload > 64", fsw_tlm_encode(buf, sizeof(buf), 1, 0, pay, 65) == 0);
    CHECK("encode rejects too-small buffer", fsw_tlm_encode(buf, 16, 0x42, 9, pay, 10) == 0);

    fsw_tlm_parser_init(&p);
    got = 0;
    for (i = 0; i < n; i++) got = fsw_tlm_parser_feed(&p, buf[i], &f);
    CHECK("round-trip frame accepted", got == 1 && f.type == 0x42 && f.seq == 9 && f.len == 10 &&
                                       memcmp(f.payload, pay, 10) == 0);

    n = fsw_tlm_encode(buf, sizeof(buf), 0x01, 0, pay, 0);
    fsw_tlm_parser_init(&p);
    got = 0;
    for (i = 0; i < n; i++) got = fsw_tlm_parser_feed(&p, buf[i], &f);
    CHECK("zero-length payload frame accepted", got == 1 && f.len == 0);

    n = fsw_tlm_encode(buf, sizeof(buf), 0x01, 0, pay, 64);
    fsw_tlm_parser_init(&p);
    got = 0;
    for (i = 0; i < n; i++) got = fsw_tlm_parser_feed(&p, buf[i], &f);
    CHECK("max-length (64) payload frame accepted", got == 1 && f.len == 64 && memcmp(f.payload, pay, 64) == 0);

    /* single flipped bit must be rejected */
    n = fsw_tlm_encode(buf, sizeof(buf), 0x42, 9, pay, 10);
    buf[8] ^= 0x04;
    fsw_tlm_parser_init(&p);
    got = 0;
    for (i = 0; i < n; i++) got |= fsw_tlm_parser_feed(&p, buf[i], &f);
    CHECK("1-bit corruption rejected, crc_errors counted", got == 0 && p.crc_errors >= 1 && p.frames_ok == 0);

    /* garbage before a good frame */
    {
        uint8_t stream[64]; size_t m = 0, k;
        stream[m++] = 0x00; stream[m++] = 0xEB; stream[m++] = 0x13; stream[m++] = 0xEB; stream[m++] = 0xEB;
        k = fsw_tlm_encode(&stream[m], sizeof(stream) - m, 0x07, 1, pay, 6);
        m += k;
        fsw_tlm_parser_init(&p);
        got = 0;
        for (i = 0; i < m; i++) got |= fsw_tlm_parser_feed(&p, stream[i], &f);
        CHECK("resync past garbage/false sync bytes", got == 1 && p.frames_ok == 1 && f.type == 0x07);
    }

    /* a valid frame that begins INSIDE a corrupted/interrupted one is not lost */
    {
        uint8_t stream[200]; size_t m = 0, k; int frames = 0;
        stream[m++] = 0xEB; stream[m++] = 0x90; stream[m++] = 0x05; stream[m++] = 0x00; stream[m++] = 20;
        stream[m++] = 0xAA; stream[m++] = 0xBB; stream[m++] = 0xCC;      /* header says 20 bytes, only 3 arrive */
        k = fsw_tlm_encode(&stream[m], sizeof(stream) - m, 0x21, 4, pay, 12);
        m += k;
        k = fsw_tlm_encode(&stream[m], sizeof(stream) - m, 0x22, 5, pay, 12);
        m += k;
        fsw_tlm_parser_init(&p);
        for (i = 0; i < m; i++) {
            if (fsw_tlm_parser_feed(&p, stream[i], &f)) {
                frames++;
                while (fsw_tlm_parser_poll(&p, &f)) frames++;
            }
        }
        CHECK("frame embedded in a truncated frame is recovered (both survive)", frames == 2 && p.frames_ok == 2);
    }
}

static fsw_monitor_cfg_t mon_cfg(void)
{
    fsw_monitor_cfg_t c;
    c.min = -10.0f; c.max = 10.0f; c.max_rate = 100.0f;
    c.stuck_samples = 25; c.fail_persist = 3; c.recover_persist = 50; c.max_fault_events = 3;
    return c;
}

static void test_monitor(void)
{
    fsw_monitor_cfg_t c = mon_cfg();
    fsw_monitor_t m;
    int i;
    float v;
    float nan_v;
    { union { uint32_t u; float f; } x; x.u = 0x7FC00000u; nan_v = x.f; }

    printf("FDIR monitor\n");
    fsw_monitor_init(&m, &c);
    for (i = 0; i < 20; i++) fsw_monitor_update(&m, 1.0f + 0.001f * (float)i, 0.02f);
    CHECK("healthy channel stays OK", m.status == FSW_SENSOR_OK);

    fsw_monitor_update(&m, 99.0f, 0.02f);
    fsw_monitor_update(&m, 99.0f, 0.02f);
    CHECK("2 consecutive bad samples do NOT latch (debounce)", m.status == FSW_SENSOR_OK);
    fsw_monitor_update(&m, 1.0f, 0.02f);
    CHECK("good sample resets bad counter", m.bad_count == 0);

    for (i = 0; i < 3; i++) fsw_monitor_update(&m, 99.0f, 0.02f);
    CHECK("3 consecutive bad samples => FAULT", m.status == FSW_SENSOR_FAULT && m.fault_events == 1);
    for (i = 0; i < 49; i++) fsw_monitor_update(&m, 1.0f + 0.001f * (float)i, 0.02f);
    CHECK("49 good samples: still FAULT", m.status == FSW_SENSOR_FAULT);
    fsw_monitor_update(&m, 1.5f, 0.02f);
    CHECK("50th good sample: recovers to OK (no permanent freeze)", m.status == FSW_SENSOR_OK);

    fsw_monitor_init(&m, &c);
    fsw_monitor_update(&m, 0.0f, 0.02f);
    fsw_monitor_update(&m, 5.0f, 0.02f);   /* 250 /s > 100 /s */
    CHECK("rate-of-change violation counts as bad sample", m.bad_count == 1);

    fsw_monitor_init(&m, &c);
    for (i = 0; i < 3; i++) fsw_monitor_update(&m, nan_v, 0.02f);
    CHECK("NaN samples => FAULT", m.status == FSW_SENSOR_FAULT);

    fsw_monitor_init(&m, &c);
    v = 2.0f;
    for (i = 0; i < 24; i++) fsw_monitor_update(&m, v, 0.02f);
    CHECK("24 identical samples: not yet stuck", m.status == FSW_SENSOR_OK);
    for (i = 0; i < 5; i++) fsw_monitor_update(&m, v, 0.02f);
    CHECK("sustained identical samples => stuck-at FAULT", m.status == FSW_SENSOR_FAULT);

    fsw_monitor_init(&m, &c);
    for (i = 0; i < 4; i++) {
        int k;
        for (k = 0; k < 3; k++) fsw_monitor_update(&m, 99.0f, 0.02f);
        for (k = 0; k < 60; k++) fsw_monitor_update(&m, 1.0f + 0.001f * (float)k, 0.02f);
    }
    CHECK("repeatedly flapping channel => EXCLUDED permanently", m.status == FSW_SENSOR_EXCLUDED);
    for (i = 0; i < 100; i++) fsw_monitor_update(&m, 1.0f + 0.001f * (float)i, 0.02f);
    CHECK("EXCLUDED never recovers", m.status == FSW_SENSOR_EXCLUDED);
}

static void test_vote(void)
{
    uint8_t all[3] = {1, 1, 1};
    uint8_t two[3] = {1, 0, 1};
    uint8_t one[3] = {0, 0, 1};
    uint8_t none[3] = {0, 0, 0};
    float a[3] = {10.0f, 10.1f, 9.9f};
    float b[3] = {10.0f, 55.0f, 10.2f};
    float c[3] = {10.0f, 20.0f, 30.0f};
    float d[3] = {10.0f, 0.0f, 14.0f};
    fsw_vote_t r;

    printf("3-channel voter\n");
    r = fsw_vote3(a, all, 0.5f);
    CHECK("3 agreeing channels: n_used=3, mean", r.n_used == 3 && r.used_mask == 7 && r.value > 9.99f && r.value < 10.01f);
    r = fsw_vote3(b, all, 0.5f);
    CHECK("outlier voted out even though 'healthy'", r.n_used == 2 && r.used_mask == 5 && r.value > 10.0f && r.value < 10.2f);
    r = fsw_vote3(c, all, 0.5f);
    CHECK("three-way disagreement: only the median survives", r.n_used == 1 && r.used_mask == 2);
    r = fsw_vote3(d, two, 0.5f);
    CHECK("two healthy but disagreeing => untrusted (n_used=0)", r.n_used == 0);
    r = fsw_vote3(a, two, 0.5f);
    CHECK("two healthy and agreeing => n_used=2", r.n_used == 2 && r.used_mask == 5);
    r = fsw_vote3(a, one, 0.5f);
    CHECK("single healthy channel => n_used=1 (no cross-check)", r.n_used == 1 && r.used_mask == 4);
    r = fsw_vote3(a, none, 0.5f);
    CHECK("no healthy channel => n_used=0", r.n_used == 0);
}

static void test_state_unit(void)
{
    fsw_core_cfg_t cc;
    fsw_state_t s;
    fsw_inputs_t in;

    printf("Flight-mode state machine (unit)\n");
    fsw_core_default_cfg(&cc);
    fsw_state_init(&s, &cc.state);
    memset(&in, 0, sizeof(in));
    in.accel_mps2 = 9.81f; in.imu_trusted = 1;

    in.cmd_launch = 1;
    fsw_state_step(&s, &in, 0.02f);
    CHECK("launch command rejected in PAD_SAFE", s.mode == FSW_PAD_SAFE && s.rejected_cmds == 1);

    in.cmd_launch = 0; in.cmd_arm = 1; in.imu_trusted = 0;
    fsw_state_step(&s, &in, 0.02f);
    CHECK("arm rejected while IMUs untrusted", s.mode == FSW_PAD_SAFE && s.rejected_cmds == 2);

    in.imu_trusted = 1;
    fsw_state_step(&s, &in, 0.02f);
    CHECK("arm accepted when IMUs trusted", s.mode == FSW_PAD_ARMED);

    in.cmd_arm = 0; in.fdir_critical = 1;
    fsw_state_step(&s, &in, 0.02f);
    CHECK("critical fault while ARMED scrubs back to PAD_SAFE", s.mode == FSW_PAD_SAFE && s.last_reason == FSW_REASON_FDIR_CRITICAL);

    in.fdir_critical = 0; in.cmd_arm = 1;
    fsw_state_step(&s, &in, 0.02f);
    in.cmd_arm = 0; in.cmd_launch = 1;
    fsw_state_step(&s, &in, 0.02f);
    in.cmd_launch = 0;
    CHECK("launch accepted from PAD_ARMED", s.mode == FSW_IGNITION);

    in.accel_mps2 = 9.81f;   /* on-pad specific force is 1 g: must NOT count as liftoff */
    { int i; for (i = 0; i < 20; i++) fsw_state_step(&s, &in, 0.02f); }
    CHECK("1 g on the pad does not confirm ignition", s.mode == FSW_IGNITION);

    in.accel_mps2 = 14.0f;
    { int i; for (i = 0; i < 4; i++) fsw_state_step(&s, &in, 0.02f); }
    CHECK("4 of 5 confirm ticks: still IGNITION", s.mode == FSW_IGNITION);
    fsw_state_step(&s, &in, 0.02f);
    CHECK("5th confirm tick: ASCENT_S1 (liftoff)", s.mode == FSW_ASCENT_S1 && s.last_reason == FSW_REASON_LIFTOFF);

    in.cmd_abort = 1;
    fsw_state_step(&s, &in, 0.02f);
    CHECK("abort command in flight => ABORT", s.mode == FSW_ABORT && s.last_reason == FSW_REASON_ABORT_CMD);
    in.cmd_abort = 0; in.cmd_arm = 1;
    fsw_state_step(&s, &in, 0.02f);
    CHECK("ABORT is terminal; later commands rejected", s.mode == FSW_ABORT && s.rejected_cmds >= 3);
}

static void test_watchdog(void)
{
    fsw_wdg_t w;
    printf("Watchdog\n");
    fsw_wdg_init(&w, FSW_TASK_ALL, 3);
    fsw_wdg_checkin(&w, FSW_TASK_ALL);
    CHECK("all tasks checked in => kick", fsw_wdg_service(&w) == 1 && w.kicks == 1);
    fsw_wdg_checkin(&w, FSW_TASK_SENSORS | FSW_TASK_FDIR | FSW_TASK_STATE | FSW_TASK_TLM);
    CHECK("one task missing (CONTROL) => NO kick", fsw_wdg_service(&w) == 0 && w.kicks == 1);
    fsw_wdg_checkin(&w, FSW_TASK_SENSORS);
    fsw_wdg_service(&w);
    fsw_wdg_checkin(&w, FSW_TASK_SENSORS);
    fsw_wdg_service(&w);
    CHECK("persistent stall trips the software flag", w.tripped == 1);
    fsw_wdg_checkin(&w, FSW_TASK_ALL);
    CHECK("healthy cycle kicks again and clears the miss counter", fsw_wdg_service(&w) == 1 && w.miss_count == 0);
}


static void test_state_s2_low_twr(void)
{
    fsw_core_cfg_t cc;
    fsw_state_t s;
    fsw_inputs_t in;
    int i;
    printf("State machine: upper stage with specific force well below 1 g\n");
    fsw_core_default_cfg(&cc);
    fsw_state_init(&s, &cc.state);
    memset(&in, 0, sizeof(in));
    in.imu_trusted = 1; in.accel_mps2 = 9.81f;
    in.cmd_arm = 1; fsw_state_step(&s, &in, 0.02f); in.cmd_arm = 0;
    in.cmd_launch = 1; fsw_state_step(&s, &in, 0.02f); in.cmd_launch = 0;
    in.accel_mps2 = 14.0f; for (i = 0; i < 6; i++) fsw_state_step(&s, &in, 0.02f);
    for (i = 0; i < 500; i++) fsw_state_step(&s, &in, 0.02f);
    in.accel_mps2 = 0.1f; for (i = 0; i < 10; i++) fsw_state_step(&s, &in, 0.02f);
    for (i = 0; i < 80; i++) fsw_state_step(&s, &in, 0.02f);
    CHECK("reached ASCENT_S2", s.mode == FSW_ASCENT_S2);
    in.accel_mps2 = 7.5f;                       /* 30 kN / 4000 kg */
    for (i = 0; i < 300; i++) fsw_state_step(&s, &in, 0.02f);   /* 6 s: longer than min_burn_s */
    CHECK("S2 thrust at 7.5 m/s^2 (0.76 g) is recognised: no NO_IGNITION abort", s.mode == FSW_ASCENT_S2);
    in.accel_mps2 = 0.1f; for (i = 0; i < 10; i++) fsw_state_step(&s, &in, 0.02f);
    CHECK("cutoff afterwards still detected (SECO -> COAST)", s.mode == FSW_COAST && s.last_reason == FSW_REASON_SECO);

    fsw_state_init(&s, &cc.state);
    in.accel_mps2 = 9.81f; in.cmd_arm = 1; fsw_state_step(&s, &in, 0.02f); in.cmd_arm = 0;
    in.cmd_launch = 1; fsw_state_step(&s, &in, 0.02f); in.cmd_launch = 0;
    in.accel_mps2 = 7.5f; for (i = 0; i < 200; i++) fsw_state_step(&s, &in, 0.02f);
    CHECK("liftoff still needs > 1 g: 7.5 m/s^2 on the pad is NOT a liftoff (NO_IGNITION)", s.mode == FSW_ABORT && s.last_reason == FSW_REASON_NO_IGNITION);
}

/* ---- guidance unit tests: the Python test_peg.py vehicle (4000 kg, 90 kN, 320 s, 180 km -> 300 km) ---- */
static void nominal_guid_cfg(fsw_guidance_cfg_t *c)
{
    double r = 6378137.0 + 300000.0;
    fsw_guidance_default_cfg(c);
    c->target.r_t = r; c->target.v_t = sqrt(FSW_PEG_MU_EARTH / r); c->target.gamma_t = 0.0;
    c->thrust_n = 90000.0; c->isp_s = 320.0; c->m_dry_kg = 1000.0;
}

static void test_guidance_unit(void)
{
    static fsw_guidance_t g;
    fsw_guidance_cfg_t cfg;
    fsw_peg_state_t st = { 6378137.0 + 180000.0, 7300.0, 3.0 * 3.14159265358979 / 180.0, 4000.0 };
    fsw_guid_solution_t sol, keep;
    double th, ttg;
    int i, ok;
    fsw_guid_result_t r;

    printf("Guidance runtime (PEG wrapper)\n");
    nominal_guid_cfg(&cfg);
    CHECK("init accepts a physical configuration", fsw_guidance_init(&g, &cfg) == 0);
    { fsw_guidance_cfg_t bad = cfg; bad.thrust_n = -1.0; fsw_guidance_t gb; CHECK("init rejects negative thrust", fsw_guidance_init(&gb, &bad) != 0); }
    { fsw_guidance_cfg_t bad = cfg; bad.isp_s = (double)NAN; fsw_guidance_t gb; CHECK("init rejects NaN Isp", fsw_guidance_init(&gb, &bad) != 0); }
    { fsw_guidance_cfg_t bad = cfg; bad.target.r_t = 100.0; fsw_guidance_t gb; CHECK("init rejects a target inside the Earth", fsw_guidance_init(&gb, &bad) != 0); }

    CHECK("no solution before the first solve", fsw_guidance_read(&g, &sol) == 1 && !sol.valid);
    CHECK("pure helpers refuse an invalid solution", fsw_guid_theta(&sol, 0.0, &th) == 0 && fsw_guid_time_to_go(&sol, 0.0) < 0.0 && !fsw_guid_cutoff_due(&sol, 1e9));

    r = fsw_guidance_update(&g, 0.0, &st);
    ok = fsw_guidance_read(&g, &sol);
    CHECK("cold solve converges and is published", r == FSW_GUID_PUBLISHED && ok && sol.valid && !sol.degraded);
    CHECK("time-to-go matches the Python solve (100.817 s)", fabs(sol.T - 100.81707) < 1e-3);
    CHECK("residual at cutoff inside the PEG tolerance", fabs(g.last_res[0]) < 2.0 && fabs(g.last_res[1]) < 0.02 && fabs(g.last_res[2]) < 2e-5);
    CHECK("theta(tau=0) = atan(A)", fsw_guid_theta(&sol, 0.0, &th) && fabs(th - atan(sol.A)) < 1e-12);
    CHECK("time-to-go counts down", fabs(fsw_guid_time_to_go(&sol, 10.0) - (sol.T - 10.0)) < 1e-9);
    CHECK("theta held at the final value after T_go", fsw_guid_theta(&sol, 1e6, &th) && fabs(th - atan(sol.A + sol.B * sol.T)) < 1e-12);
    CHECK("cutoff not due before T_go, due at T_go", !fsw_guid_cutoff_due(&sol, sol.T - 0.01) && fsw_guid_cutoff_due(&sol, sol.T));
    CHECK("seq is even (stable) after publication", (sol.seq & 1u) == 0u && (g.seq & 1u) == 0u);

    /* fly 2 s of the nominal solution, re-solve: warm start must be cheap */
    {
        fsw_peg_state_t s2 = st;
        double mdot = g.mdot;
        for (i = 0; i < 40; i++) {
            double theta = atan(sol.A + sol.B * ((double)i + 0.5) * 0.05);
            s2 = fsw_peg_rk4_step(&s2, 90000.0, mdot, theta, 0.05, FSW_PEG_MU_EARTH);
        }
        r = fsw_guidance_update(&g, 2.0, &s2);
        CHECK("warm re-solve 2 s later converges in <= 3 iterations", r == FSW_GUID_PUBLISHED && g.last_iterations <= 3);
        fsw_guidance_read(&g, &sol);
        CHECK("T_go decreased by about the elapsed 2 s", fabs(sol.T - (100.817 - 2.0)) < 0.2);
    }

    /* failure handling */
    fsw_guidance_read(&g, &keep);
    {
        fsw_peg_state_t bad = st; bad.v = (double)NAN;
        r = fsw_guidance_update(&g, 4.0, &bad);
        fsw_guidance_read(&g, &sol);
        CHECK("NaN state: previous solution kept, still valid", r == FSW_GUID_KEPT_PREVIOUS && sol.valid && sol.A == keep.A && sol.T == keep.T);
        r = fsw_guidance_update(&g, 6.0, &bad);
        r = fsw_guidance_update(&g, 8.0, &bad);
        fsw_guidance_read(&g, &sol);
        CHECK("3 consecutive failures: still flying the previous solution", sol.valid);
        r = fsw_guidance_update(&g, 10.0, &bad);
        fsw_guidance_read(&g, &sol);
        CHECK("4th consecutive failure withdraws the solution (valid=0)", r == FSW_GUID_FAILED && !sol.valid);
        r = fsw_guidance_update(&g, 12.0, &st);
        fsw_guidance_read(&g, &sol);
        CHECK("a good state afterwards recovers guidance", r == FSW_GUID_PUBLISHED && sol.valid && g.consecutive_failures == 0);
    }

    /* infeasible target: best effort is published, flagged degraded */
    {
        static fsw_guidance_t g2;
        fsw_guidance_cfg_t c2 = cfg;
        double r6 = 6378137.0 + 600000.0;
        c2.target.r_t = r6; c2.target.v_t = sqrt(FSW_PEG_MU_EARTH / r6);
        fsw_guidance_init(&g2, &c2);
        r = fsw_guidance_update(&g2, 0.0, &st);
        fsw_guidance_read(&g2, &sol);
        CHECK("unreachable target: best effort published and flagged degraded", r == FSW_GUID_PUBLISHED_DEGRADED && sol.valid && sol.degraded);
        CHECK("... burning to the propellant limit (T = (m-m_dry)/mdot)", fabs(sol.T - (4000.0 - 1000.0) / g2.mdot) < 0.5);
    }

    /* seqlock behaviour */
    {
        fsw_guid_solution_t untouched;
        memset(&untouched, 0xAB, sizeof(untouched));
        g.seq |= 1u;                                           /* pretend a writer is mid-update */
        CHECK("reader refuses to copy while a write is in progress (odd seq)", fsw_guidance_read(&g, &untouched) == 0 && ((unsigned char *)&untouched)[0] == 0xAB);
        g.seq += 1u;                                           /* writer finished */
        CHECK("reader works again once the sequence is even", fsw_guidance_read(&g, &untouched) == 1);
    }

    /* insertion window */
    {
        fsw_peg_state_t s = { cfg.target.r_t + 1000.0, cfg.target.v_t + 5.0, 0.002, 1000.0 };
        CHECK("insertion window accepts 1 km / 5 m/s / 0.11 deg", fsw_guidance_insertion_ok(&cfg, &s));
        s.v = cfg.target.v_t + 40.0;
        CHECK("insertion window rejects 40 m/s", !fsw_guidance_insertion_ok(&cfg, &s));
        s.v = cfg.target.v_t; s.gamma = 0.05;
        CHECK("insertion window rejects 2.9 deg flight-path angle", !fsw_guidance_insertion_ok(&cfg, &s));
        s.gamma = 0.0; s.r = (double)NAN;
        CHECK("insertion window rejects NaN", !fsw_guidance_insertion_ok(&cfg, &s));
    }
    (void)ttg;
}

/* ======================= Part B: closed-loop scenarios ======================= */

typedef enum { IMU_OK = 0, IMU_STUCK, IMU_SPIKE, IMU_DEAD, IMU_BURST, IMU_DRIFT, IMU_GYRO_DRIFT } imu_mode_t;

typedef struct {
    imu_mode_t mode;
    double t_on, t_off;
    int held, count;
    float ha[3], hg[3];
} imu_fault_t;

#define TX_CAP 262144
#define MAX_VISITED 16
#define G0 9.80665

typedef struct {
    /* scenario knobs */
    imu_fault_t imu[3];
    double burn1_s, burn2_s;
    int engine1_no_start;
    double engine1_cut_after_s;     /* <0: none */
    double abort_at_s;              /* <0: none */
    double launch_at_s;             /* <0: never */
    double gyro_bias[3];            /* true gyro bias, same on all 3 IMUs */
    double pad_tilt_rad;            /* initial pitch of the vehicle on the pad */
    int gps_enabled;
    double gps_sigma_m;
    double gps_delay_s;             /* receiver latency: fix is delivered this long after it was measured */
    int gps_report_age;             /* 1: tell the core the true age (compensated); 0: pretend it is fresh */
    struct { uint32_t at; float pos[3]; double ecef[3]; double ecef_t; int used; } gps_q[8];
    double gps_outlier_at_s;        /* <0: none; one 400 m outlier at this time */
    /* plant */
    double t;
    RigidBodyPitchDynamics dyn;
    RealisticActuator act;
    double e1_start, e1_end, e2_start, e2_end;
    float last_gimbal;
    uint32_t rng;
    double coast_t;
    int armed_sent, launch_sent, abort_sent;
    /* truth kinematics (nav frame, Z up) */
    double tp[3], tv[3];
    /* HAL capture */
    uint32_t kicks;
    uint8_t tx[TX_CAP];
    size_t txn;
    /* observations */
    fsw_core_t core;
    fsw_mode_t visited[MAX_VISITED];
    int n_visited;
    float max_abs_gimbal, max_abs_gimbal_nonthrust, max_acc_vote_err;
    uint8_t ever_fault_flags;
    double t_abort;
    float pitch_err_8s;
    int pitch_err_8s_taken;
    float max_pitch_est_err, max_pos_err_after_5s, final_pos_err;
    float pad_pitch_est_at_launch;
    int pad_pitch_taken;
    /* ---- guided upper-stage scenarios: point-mass truth in the polar frame ---- */
    int guided;
    double pm_F_true, pm_isp, pm_m_dry;
    fsw_peg_state_t pm;             /* truth: r, v, gamma, m                                   */
    double psi;                     /* downrange angle                                          */
    int pm_started, pm_engine, pm_cut_by_cmd, pm_depleted;
    double pm_ign_t, pm_cut_t, phi_program;
    double polar_bias[3];           /* navigation error given to guidance (r, v, gamma)         */
    double polar_dropout_at_s, polar_nan_at_s;
    fsw_guidance_t guid;
    double max_solve_ms;
    float max_track_err;            /* |applied pitch command - true pitch| during S2            */
    uint8_t ever_guid_flags;
    /* ---- ECI navigation truth ---- */
    int eci, eci_started, handoff_done;
    double tpI[3], tvI[3];          /* truth in the inertial navigation frame                    */
    double t_L;                     /* time the navigation frame was fixed (nav init)            */
    double max_pos3, max_vel3, max_pos3_after10, max_vel3_after10;
    double handoff_pos_err, handoff_vel_err;   /* residual estimate error at the hand-off (m, m/s) */
    int j2_in_truth;
    uint32_t ecef_fixes;
    double nees_sum; int nees_n;
    double bg_y_at_init;
    double gps_age_error_s;         /* wrongly reported part of the fix age (receiver timing error) */
    double tp_hist[64][3];          /* truth position at the start of each cycle (for fixes measured between cycles) */
    uint32_t cycle_L;
} sim_t;

static void (*g_cfg_hook)(fsw_core_cfg_t *) = 0;
static fsw_site_t g_site;                 /* the sim's own copy of the launch site, for truth <-> ECEF */

static float noise(sim_t *s)
{
    s->rng = s->rng * 1664525u + 1013904223u;
    return (float)((s->rng >> 8) & 0xFFFFu) / 32768.0f - 1.0f;
}

static int flight_started(const sim_t *s) { return s->e1_start >= 0 && s->t >= s->e1_start; }

static float truth_axial_accel(const sim_t *s)
{
    double t = s->t;
    if (s->guided && s->pm_started) return s->pm_engine ? (float)(s->pm_F_true / s->pm.m) : 0.1f;
    if (s->e2_start >= 0 && t >= s->e2_start && t < s->e2_end) return (float)(12.0 + 1.0 * (t - s->e2_start));
    if (s->e2_start >= 0 && t >= s->e2_end)                     return 0.1f;
    if (s->e1_start >= 0 && t >= s->e1_start && t < s->e1_end) return (float)(14.0 + 1.5 * (t - s->e1_start));
    if (s->e1_start >= 0 && t >= s->e1_end)                     return 0.1f;
    return (float)G0;
}

static int engine_on(const sim_t *s)
{
    if (s->guided && s->pm_started) return s->pm_engine;
    return (s->e1_start >= 0 && s->t >= s->e1_start && s->t < s->e1_end) ||
           (s->e2_start >= 0 && s->t >= s->e2_start && s->t < s->e2_end);
}

/* Specific force in the body frame. Standing on the pad it is the reaction to gravity,
 * R^T (0,0,g), which for a pitch-tilted vehicle has an x component; in flight it is
 * the scripted axial thrust acceleration. */
static void truth_specific_force(const sim_t *s, float f[3])
{
    if (!flight_started(s)) {
        double th = s->dyn.pitch;
        f[0] = (float)(-G0 * sin(th));
        f[1] = 0.0f;
        f[2] = (float)(G0 * cos(th));
    } else {
        f[0] = 0.0f; f[1] = 0.0f; f[2] = truth_axial_accel(s);
    }
}

static int hal_read_imu(void *ctx, int idx, fsw_imu_sample_t *out)
{
    sim_t *s = (sim_t *)ctx;
    imu_fault_t *f = &s->imu[idx];
    float a[3], g[3];
    int k;

    truth_specific_force(s, a);
    g[0] = (float)s->gyro_bias[0]; g[1] = (float)(s->dyn.pitch_rate + s->gyro_bias[1]); g[2] = (float)s->gyro_bias[2];
    if (s->eci && !flight_started(s)) {          /* standing on the rotating Earth: body frame = pad frame, due-east launch */
        g[1] += (float)(FSW_OMEGA_EARTH * cos(g_site.lat));
        g[2] += (float)(FSW_OMEGA_EARTH * sin(g_site.lat));
    }
    for (k = 0; k < 3; k++) {
        a[k] += noise(s) * 0.05f;
        g[k] += noise(s) * 0.0005f;
    }

    if (f->mode != IMU_OK && s->t >= f->t_on - 1e-9) {
        switch (f->mode) {
        case IMU_STUCK:
            if (!f->held) { f->held = 1; for (k = 0; k < 3; k++) { f->ha[k] = a[k]; f->hg[k] = g[k]; } }
            for (k = 0; k < 3; k++) { a[k] = f->ha[k]; g[k] = f->hg[k]; }
            break;
        case IMU_SPIKE:
            if (f->count == 0) a[2] += 120.0f;
            f->count++;
            break;
        case IMU_BURST:
            if (s->t < f->t_off - 1e-9) { for (k = 0; k < 3; k++) { a[k] = 500.0f; g[k] = 5.0f; } }
            break;
        case IMU_DEAD:
            return 1;
        case IMU_DRIFT:
            a[2] += 0.5f * (float)(s->t - f->t_on);
            break;
        case IMU_GYRO_DRIFT:
            g[1] += 0.01f * (float)(s->t - f->t_on);     /* 0.01 rad/s per second: slow, in-range, never trips the monitor */
            break;
        default:
            break;
        }
    }
    for (k = 0; k < 3; k++) { out->accel[k] = a[k]; out->gyro[k] = g[k]; }
    return 0;
}

static void hal_write_gimbal(void *ctx, float rad) { ((sim_t *)ctx)->last_gimbal = rad; }
static void hal_kick(void *ctx) { ((sim_t *)ctx)->kicks++; }
static void hal_engine_cutoff(void *ctx)
{
    sim_t *s = (sim_t *)ctx;
    if (s->pm_engine) { s->pm_engine = 0; s->pm_cut_by_cmd = 1; s->pm_cut_t = s->t; }
}
static size_t hal_tlm(void *ctx, const uint8_t *b, size_t n)
{
    sim_t *s = (sim_t *)ctx;
    if (s->txn + n <= TX_CAP) { memcpy(&s->tx[s->txn], b, n); s->txn += n; }
    return n;
}

static void sim_init(sim_t *s)
{
    fsw_hal_t hal;
    fsw_core_cfg_t cfg;
    int i;
    memset(s, 0, sizeof(*s));
    s->burn1_s = 8.0; s->burn2_s = 6.0;
    s->engine1_cut_after_s = -1.0; s->abort_at_s = -1.0; s->launch_at_s = 0.6;
    s->e1_start = s->e1_end = s->e2_start = s->e2_end = -1.0;
    s->gps_enabled = 1; s->gps_sigma_m = 3.0; s->gps_outlier_at_s = -1.0; s->gps_report_age = 1;
    s->rng = 12345u;
    for (i = 0; i < 3; i++) s->imu[i].mode = IMU_OK;
    dynamics_init(&s->dyn, 50000.0f, 1.2f);
    actuator_init(&s->act, 0.34907f, 0.05f);
    hal.ctx = s; hal.read_imu = hal_read_imu; hal.write_gimbal = hal_write_gimbal;
    hal.kick_watchdog = hal_kick; hal.tlm_write = hal_tlm; hal.engine_cutoff = hal_engine_cutoff;
    fsw_core_default_cfg(&cfg);
    if (g_cfg_hook) g_cfg_hook(&cfg);
    fsw_core_init(&s->core, &hal, &cfg);
    s->visited[s->n_visited++] = FSW_PAD_SAFE;
    s->t_abort = -1.0;
}

static void sim_start_tilted(sim_t *s, double tilt)
{
    s->pad_tilt_rad = tilt;
    s->dyn.pitch = (float)tilt;
}

static void sim_run(sim_t *s, double t_end)
{
    const float dt = 0.02f;
    while (s->t < t_end - 1e-9) {
        fsw_mode_t before = s->core.st.mode, after;
        float truth, thrust, dist, gimbal_actual;

        /* engines light only when the flight software commands it */
        if (before == FSW_IGNITION && s->e1_start < 0 && !s->engine1_no_start) {
            s->e1_start = s->t + 0.4;
            s->e1_end = s->e1_start + (s->engine1_cut_after_s > 0 ? s->engine1_cut_after_s : s->burn1_s);
        }
        if (before == FSW_ASCENT_S2 && s->e2_start < 0) {
            s->e2_start = s->t + 0.3;
            s->e2_end = s->e2_start + s->burn2_s;
            if (s->guided) {              /* hand-off state for the upper-stage burn (arbitrary, not the toy S1's) */
                s->pm.r = 6378137.0 + 180000.0; s->pm.v = 6000.0; s->pm.gamma = 3.0 * 3.14159265358979 / 180.0; s->pm.m = 4000.0;
                s->psi = 0.0; s->pm_started = 1; s->pm_ign_t = s->e2_start;
                if (s->eci) {
                    /* TEST-HARNESS SHORTCUT: fast-forward past the first stage. Truth jumps to the hand-off orbit state;
                     * the filter's nominal state jumps to the same plus a realistic residual error, its history is
                     * rebased, and its covariance is set to match. A real vehicle navigates continuously from liftoff. */
                    int k;
                    double er = s->pm.r;
                    s->tpI[0] = 0.0; s->tpI[1] = 0.0; s->tpI[2] = er;
                    s->tvI[0] = s->pm.v * cos(s->pm.gamma); s->tvI[1] = 0.0; s->tvI[2] = s->pm.v * sin(s->pm.gamma);
                    for (k = 0; k < 3; k++) {
                        double ep = noise(s) * 5.0 * 1.732, ev = noise(s) * 0.05 * 1.732;       /* sigma 5 m, 0.05 m/s */
                        s->core.nav.p[k] = s->tpI[k] + ep;
                        s->core.nav.v[k] = s->tvI[k] + ev;
                        s->handoff_pos_err += ep * ep; s->handoff_vel_err += ev * ev;
                    }
                    s->handoff_pos_err = sqrt(s->handoff_pos_err); s->handoff_vel_err = sqrt(s->handoff_vel_err);
                    for (k = 0; k < FSW_GPS_HIST; k++) memcpy(s->core.nav_pos_hist[k], s->core.nav.p, sizeof(s->core.nav.p));
                    fsw_eskf_reset_pv(&s->core.nav, s->core.nav.p, s->core.nav.v, 5.0f, 0.05f);   /* state already set above; decorrelate + set P */
                    s->handoff_done = 1;
                }
            }
        }
        if (s->guided && s->pm_started && !s->pm_engine && !s->pm_cut_by_cmd && !s->pm_depleted && s->t >= s->pm_ign_t) s->pm_engine = 1;

        if (!s->armed_sent && s->t >= 0.2) { fsw_core_cmd_arm(&s->core); s->armed_sent = 1; }
        if (!s->launch_sent && s->launch_at_s >= 0 && s->t >= s->launch_at_s) { fsw_core_cmd_launch(&s->core); s->launch_sent = 1; }
        if (!s->abort_sent && s->abort_at_s >= 0 && s->t >= s->abort_at_s) { fsw_core_cmd_abort(&s->core); s->abort_sent = 1; }
        if (before == FSW_COAST && !s->guided) { s->coast_t += dt; if (s->coast_t > 3.0) fsw_core_set_insertion_ok(&s->core, 1); }
        if (s->guided) fsw_core_set_pitch_cmd(&s->core, (s->t >= 3.0) ? (float)s->phi_program : 0.0f);   /* external pitch program */
        else           fsw_core_set_pitch_cmd(&s->core, (s->t >= 5.0) ? 0.05f : 0.0f);
        if (s->guided && s->pm_started && !s->eci) {     /* navigation -> guidance bridge (truth + injected nav error) */
            int dropped = (s->polar_dropout_at_s >= 0 && s->t >= s->polar_dropout_at_s);
            if (!dropped) {
                double rr = s->pm.r + s->polar_bias[0];
                if (s->polar_nan_at_s >= 0 && s->t >= s->polar_nan_at_s) rr = (double)NAN;
                fsw_core_set_polar_state(&s->core, rr, s->pm.v + s->polar_bias[1], s->pm.gamma + s->polar_bias[2], s->pm.m, s->psi);
            }
        }

        /* 1 Hz position fix once the vehicle is flying */
        if (s->eci && s->eci_started) { memcpy(s->tp_hist[s->core.cycle % 64u], s->tpI, sizeof(s->tpI)); }
        if (s->eci && s->gps_enabled && s->eci_started && s->core.nav_active && ((s->core.cycle % 50u) == 25u)) {
            /* The receiver measured gps_delay_s ago, i.e. BETWEEN cycle boundaries: interpolate the truth history there. */
            double back = s->gps_delay_s / dt, fr, pos[3], ecef[3];
            uint32_t ib = (uint32_t)back;
            int k;
            fr = back - (double)ib;
            if (s->core.cycle >= s->cycle_L + ib + 2u && ib + 2u < 64u) {
                const double *a = s->tp_hist[(s->core.cycle - ib) % 64u], *b = s->tp_hist[(s->core.cycle - ib - 1u) % 64u];
                for (k = 0; k < 3; k++) pos[k] = (1.0 - fr) * a[k] + fr * b[k];
                fsw_inertial_to_ecef(&g_site, pos, (s->t - s->gps_delay_s) - s->t_L, ecef);
                for (k = 0; k < 3; k++) ecef[k] += noise(s) * s->gps_sigma_m * 1.732;
                fsw_core_gps_fix_ecef(&s->core, ecef, s->gps_report_age ? (float)(s->gps_delay_s + s->gps_age_error_s) : 0.0f);
                s->ecef_fixes++;
            }
        }
        if (!s->eci && s->gps_enabled && flight_started(s) && ((s->core.cycle % 50u) == 25u)) {
            float fix[3];
            int k;
            for (k = 0; k < 3; k++) fix[k] = (float)(s->tp[k] + noise(s) * s->gps_sigma_m * 1.732);   /* uniform, sigma = gps_sigma_m */
            if (s->gps_outlier_at_s >= 0 && s->t >= s->gps_outlier_at_s) { fix[0] += 400.0f; s->gps_outlier_at_s = -1.0; }
            {   /* the receiver reports it gps_delay_s late */
                int q;
                for (q = 0; q < 8; q++) if (!s->gps_q[q].used) break;
                if (q < 8) {
                    s->gps_q[q].used = 1;
                    s->gps_q[q].at = s->core.cycle + (uint32_t)(s->gps_delay_s / dt + 0.5);
                    memcpy(s->gps_q[q].pos, fix, sizeof(fix));
                }
            }
        }
        {
            int q;
            for (q = 0; q < 8; q++) {
                if (s->gps_q[q].used && s->core.cycle >= s->gps_q[q].at) {
                    if (s->eci) { fsw_core_gps_fix_ecef(&s->core, s->gps_q[q].ecef, s->gps_report_age ? (float)(s->gps_delay_s + s->gps_age_error_s) : 0.0f); s->ecef_fixes++; }
                    else fsw_core_gps_fix_aged(&s->core, s->gps_q[q].pos, s->gps_report_age ? (float)s->gps_delay_s : 0.0f);
                    s->gps_q[q].used = 0;
                }
            }
        }

        truth = truth_axial_accel(s);
        fsw_core_step(&s->core);
        if (s->guided) {                                  /* the heavy solve runs OUTSIDE the control step */
            clock_t c0 = clock();
            uint32_t before_n = s->core.guid_serviced;
            fsw_core_guidance_service(&s->core);
            if (s->core.guid_serviced != before_n) {
                double ms = 1000.0 * (double)(clock() - c0) / (double)CLOCKS_PER_SEC;
                if (ms > s->max_solve_ms) s->max_solve_ms = ms;
            }
        }
        after = s->core.st.mode;

        if (s->eci && before == FSW_IGNITION && after == FSW_ASCENT_S1 && !s->eci_started) {
            double p0[3], v0[3];
            fsw_site_pad_state(&g_site, p0, v0);
            memcpy(s->tpI, p0, sizeof(p0)); memcpy(s->tvI, v0, sizeof(v0));
            s->eci_started = 1;
            s->cycle_L = s->core.cycle + 1u;
            s->t_L = s->t + dt;
            s->bg_y_at_init = s->core.nav.bg[1];
        }
        if (before == FSW_IGNITION && after == FSW_ASCENT_S1 && !s->pad_pitch_taken) {
            s->pad_pitch_est_at_launch = s->core.pitch_est; s->pad_pitch_taken = 1;
        }
        if (after != s->visited[s->n_visited - 1] && s->n_visited < MAX_VISITED) s->visited[s->n_visited++] = after;
        if (after == FSW_ABORT && s->t_abort < 0) s->t_abort = s->t;
        s->ever_fault_flags |= (uint8_t)(s->core.fault_flags & 0x07u);
        s->ever_guid_flags |= s->core.guid_flags;
        {
            float e = s->core.acc_vote[FSW_AXIS_AXIAL].value - truth; if (e < 0) e = -e;
            if (flight_started(s) && s->core.acc_vote[FSW_AXIS_AXIAL].n_used > 0 && e > s->max_acc_vote_err) s->max_acc_vote_err = e;
        }
        {
            float g = s->last_gimbal < 0 ? -s->last_gimbal : s->last_gimbal;
            if (g > s->max_abs_gimbal) s->max_abs_gimbal = g;
            if (!fsw_mode_is_thrusting(after) && g > s->max_abs_gimbal_nonthrust) s->max_abs_gimbal_nonthrust = g;
        }

        if (s->guided && s->pm_started) {                 /* point-mass truth, thrust along the ACTUAL attitude */
            double mdot = s->pm_F_true / (s->pm_isp * FSW_PEG_G0);
            double theta_act = 1.5707963267948966 - (double)s->dyn.pitch + s->psi;
            double F = s->pm_engine ? s->pm_F_true : 0.0;
            if (s->pm_engine && s->pm.m - mdot * dt <= s->pm_m_dry) { s->pm_engine = 0; s->pm_depleted = 1; s->pm_cut_t = s->t; F = 0.0; }
            s->pm = fsw_peg_rk4_step(&s->pm, F, s->pm_engine ? mdot : 0.0, theta_act, dt, FSW_PEG_MU_EARTH);
            s->psi += s->pm.v * cos(s->pm.gamma) / s->pm.r * dt;
            if (after == FSW_ASCENT_S2) {
                float e = s->core.pitch_cmd - s->dyn.pitch; if (e < 0) e = -e;
                if (e > s->max_track_err) s->max_track_err = e;
            }
        }
        gimbal_actual = actuator_step(&s->act, s->last_gimbal, dt);
        thrust = engine_on(s) ? 2000000.0f : 0.0f;
        dist = engine_on(s) ? one_minus_cosine_gust((float)s->t, 3.0f, 2.0f, 20000.0f) : 0.0f;
        dynamics_step(&s->dyn, thrust, gimbal_actual, dist, dt);

        /* truth kinematics: same flat-Earth model the ESKF assumes */
        if (flight_started(s)) {
            double th = s->dyn.pitch, c = cos(th), sn = sin(th);
            double fa = truth;
            double an[3];
            int k;
            an[0] = sn * fa;            /* R = Ry(th): body z axis -> (sin th, 0, cos th) */
            an[1] = 0.0;
            an[2] = c * fa - G0;
            for (k = 0; k < 3; k++) {
                s->tp[k] += s->tv[k] * dt + 0.5 * an[k] * dt * dt;
                s->tv[k] += an[k] * dt;
            }
        }
        if (s->eci && s->eci_started && s->t >= s->t_L - 1e-9) {
            /* RK4 in the inertial frame; thrust acceleration along the body +Z axis, held over the step */
            double fa = truth, th = s->dyn.pitch, tb[3] = { sin(th) * fa, 0.0, cos(th) * fa };
            double mu = FSW_PEG_MU_EARTH, j2 = 1.08262668e-3, req = 6378137.0, ax[3];
            double k1p[3], k1v[3], k2p[3], k2v[3], k3p[3], k3v[3], k4p[3], k4v[3], pp[3], vv[3];
            int k, st;
            ax[0] = g_site.axis[0]; ax[1] = g_site.axis[1]; ax[2] = g_site.axis[2];
            for (st = 0; st < 4; st++) {
                double *pk = (st == 0) ? s->tpI : pp, *vk = (st == 0) ? s->tvI : vv, a[3], r2, r, kk, z, z2r2, kj;
                double *op = (st == 0) ? k1p : (st == 1) ? k2p : (st == 2) ? k3p : k4p;
                double *ov = (st == 0) ? k1v : (st == 1) ? k2v : (st == 2) ? k3v : k4v;
                if (st == 1) for (k = 0; k < 3; k++) { pp[k] = s->tpI[k] + 0.5 * dt * k1p[k]; vv[k] = s->tvI[k] + 0.5 * dt * k1v[k]; }
                if (st == 2) for (k = 0; k < 3; k++) { pp[k] = s->tpI[k] + 0.5 * dt * k2p[k]; vv[k] = s->tvI[k] + 0.5 * dt * k2v[k]; }
                if (st == 3) for (k = 0; k < 3; k++) { pp[k] = s->tpI[k] + dt * k3p[k];       vv[k] = s->tvI[k] + dt * k3v[k]; }
                r2 = pk[0]*pk[0] + pk[1]*pk[1] + pk[2]*pk[2]; r = sqrt(r2); kk = -mu / (r2 * r);
                for (k = 0; k < 3; k++) a[k] = kk * pk[k] + tb[k];
                if (s->j2_in_truth) {
                    z = pk[0]*ax[0] + pk[1]*ax[1] + pk[2]*ax[2]; z2r2 = z * z / r2;
                    kj = -1.5 * j2 * mu * req * req / (r2 * r2 * r);
                    for (k = 0; k < 3; k++) a[k] += kj * ((1.0 - 5.0 * z2r2) * pk[k] + 2.0 * z * ax[k]);
                }
                for (k = 0; k < 3; k++) { op[k] = vk[k]; ov[k] = a[k]; }
            }
            for (k = 0; k < 3; k++) {
                s->tpI[k] += dt / 6.0 * (k1p[k] + 2.0 * k2p[k] + 2.0 * k3p[k] + k4p[k]);
                s->tvI[k] += dt / 6.0 * (k1v[k] + 2.0 * k2v[k] + 2.0 * k3v[k] + k4v[k]);
            }
        }
        s->t += dt;

        if (s->eci && s->eci_started && s->core.nav_active) {
            double ep = 0.0, ev = 0.0; int k;
            for (k = 0; k < 3; k++) {
                double dp = s->core.nav.p[k] - s->tpI[k], dv = s->core.nav.v[k] - s->tvI[k];
                ep += dp * dp; ev += dv * dv;
            }
            ep = sqrt(ep); ev = sqrt(ev);
            if (ep > s->max_pos3) s->max_pos3 = ep;
            if (ev > s->max_vel3) s->max_vel3 = ev;
            if (s->t - s->t_L > 10.0) {
                if (ep > s->max_pos3_after10) s->max_pos3_after10 = ep;
                if (ev > s->max_vel3_after10) s->max_vel3_after10 = ev;
            }
        }
        if (!s->eci && s->core.nav_active) {
            float e = s->core.pitch_est - s->dyn.pitch; if (e < 0) e = -e;
            if (e > s->max_pitch_est_err) s->max_pitch_est_err = e;
            {
                float dx = (float)(s->core.nav.p[0] - s->tp[0]), dy = (float)(s->core.nav.p[1] - s->tp[1]), dz = (float)(s->core.nav.p[2] - s->tp[2]);
                float pe = (float)sqrt((double)(dx*dx + dy*dy + dz*dz));
                s->final_pos_err = pe;
                if (s->t > 5.0 && pe > s->max_pos_err_after_5s) s->max_pos_err_after_5s = pe;
            }
        }
        if (!s->pitch_err_8s_taken && s->t >= 8.0) {
            float e = s->dyn.pitch - 0.05f; if (e < 0) e = -e;
            s->pitch_err_8s = e; s->pitch_err_8s_taken = 1;
        }
    }
}

static int visited_is(const sim_t *s, const fsw_mode_t *exp, int n)
{
    int i;
    if (s->n_visited != n) return 0;
    for (i = 0; i < n; i++) if (s->visited[i] != exp[i]) return 0;
    return 1;
}

static sim_t S;   /* large; keep off the stack */
static float g_baseline_pitch_err = 0.0f;   /* nominal-run ESKF pitch error, set by scenario 1 */

static void test_nominal(void)
{
    static const fsw_mode_t seq[] = { FSW_PAD_SAFE, FSW_PAD_ARMED, FSW_IGNITION, FSW_ASCENT_S1,
                                      FSW_STAGE_SEP, FSW_ASCENT_S2, FSW_COAST, FSW_ORBIT_INSERTED };
    fsw_tlm_parser_t p; fsw_tlm_frame_t f; size_t i; int frames = 0, seq_ok = 1, have_prev = 0; uint8_t prev = 0;
    uint8_t last_mode = 255;

    printf("Scenario 1: nominal two-stage mission (real ESKF in the loop)\n");
    sim_init(&S);
    sim_run(&S, 30.0);
    printf("     ESKF: max pitch err %.5f rad, max pos err after 5 s %.2f m, gps applied %u rejected %u\n",
           S.max_pitch_est_err, S.max_pos_err_after_5s, S.core.nav.n_update, S.core.nav.n_rejected);
    CHECK("full mode sequence PAD_SAFE..ORBIT_INSERTED, in order", visited_is(&S, seq, 8));
    CHECK("final mode ORBIT_INSERTED", S.core.st.mode == FSW_ORBIT_INSERTED);
    CHECK("no rejected commands", S.core.st.rejected_cmds == 0);
    CHECK("gimbal never exceeds +/-6 deg limit", S.max_abs_gimbal <= 0.10472f + 1e-5f);
    CHECK("gimbal centred whenever not thrusting", S.max_abs_gimbal_nonthrust == 0.0f);
    CHECK("navigation went live at liftoff", S.core.nav_active && S.core.nav_ok);
    g_baseline_pitch_err = S.max_pitch_est_err;
    /* With 3 m (1-sigma) fixes the ESKF lets position innovations tug attitude, so the
     * GPS-fused attitude is good to ~0.7 deg here, WORSE than the pure gyro run in this
     * (very quiet) simulated IMU -- see scenario 12. Bound set from that physics, not tuned to pass. */
    CHECK("ESKF pitch within 0.02 rad (1.1 deg) of truth for the whole flight", S.max_pitch_est_err < 0.02f);
    CHECK("closed loop on ESKF attitude: true pitch within 0.015 rad of the 0.05 rad command at t=8 s", S.pitch_err_8s < 0.015f);
    CHECK("ESKF position error stays < 12 m after 5 s (3 m 1-sigma fixes at 1 Hz)", S.max_pos_err_after_5s < 12.0f);
    CHECK("position fixes were fused (>= 15 accepted), none gated in a clean run", S.core.nav.n_update >= 15 && S.core.nav.n_rejected == 0);
    CHECK("voted axial accel within 0.2 m/s^2 of truth all flight", S.max_acc_vote_err < 0.2f);
    CHECK("no sensor fault flags in a clean run", S.ever_fault_flags == 0 && (S.core.fault_flags & 0xC7u) == 0);
    CHECK("watchdog kicked every cycle, never tripped", S.kicks == S.core.cycle && S.core.wdg.tripped == 0);

    fsw_tlm_parser_init(&p);
    for (i = 0; i < S.txn; i++) {
        if (fsw_tlm_parser_feed(&p, S.tx[i], &f)) {
            do {
                frames++;
                if (have_prev && (uint8_t)(prev + 1u) != f.seq) seq_ok = 0;
                prev = f.seq; have_prev = 1; last_mode = f.payload[0];
            } while (fsw_tlm_parser_poll(&p, &f));
        }
    }
    CHECK("telemetry stream: every frame parses, zero CRC errors", p.crc_errors == 0 && (uint32_t)frames == S.core.tlm_frames_sent && frames > 100);
    CHECK("telemetry sequence numbers are contiguous", seq_ok);
    CHECK("last telemetry frame reports ORBIT_INSERTED", last_mode == (uint8_t)FSW_ORBIT_INSERTED);
    CHECK("housekeeping frame is 40 bytes of payload", FSW_TLM_HK_LEN == 40u && f.len == 40u);
}

static void test_pad_alignment_and_bias(void)
{
    printf("Scenario 1b: vehicle tilted 2 deg on the pad, 0.002 rad/s gyro bias, 3 s of pad calibration\n");
    sim_init(&S);
    sim_start_tilted(&S, 0.0349066);              /* 2 deg */
    S.gyro_bias[0] = 0.001; S.gyro_bias[1] = 0.002; S.gyro_bias[2] = -0.0015;
    S.launch_at_s = 3.0;
    sim_run(&S, 30.0);
    printf("     pad pitch est %.5f (true 0.03491), final bg_y est %.5f (true 0.002), max pitch err %.5f rad\n",
           S.pad_pitch_est_at_launch, S.core.nav.bg[1], S.max_pitch_est_err);
    CHECK("pad alignment (averaged) recovers the 2 deg tilt to within 0.001 rad", fabs(S.pad_pitch_est_at_launch - 0.0349066) < 0.001);
    CHECK("pad gyro-bias calibration captures the bias (< 0.0005 rad/s error at liftoff)", fabs((double)S.core.pad_bg[1] - 0.002) < 0.0005);
    CHECK("pitch estimate stays within 0.02 rad of truth through the whole flight despite bias + tilt", S.max_pitch_est_err < 0.02f);
    CHECK("mission completes", S.core.st.mode == FSW_ORBIT_INSERTED);
}

static void no_pad_cal(fsw_core_cfg_t *c) { c->pad_bias_cal_enabled = 0; }

static void test_bias_handling(void)
{
    float err_cal_nogps, final_nocal_nogps, final_nocal_gps, peak_nocal_gps;
    double bg_est;
    printf("Scenario 1c: 0.01 rad/s gyro bias -- what do pad calibration and GPS each buy? (30 s flight)\n");

    sim_init(&S); S.gyro_bias[1] = 0.01; S.launch_at_s = 3.0; S.gps_enabled = 0;
    sim_run(&S, 30.0);
    err_cal_nogps = S.max_pitch_est_err;

    g_cfg_hook = no_pad_cal;
    sim_init(&S); S.gyro_bias[1] = 0.01; S.launch_at_s = 3.0; S.gps_enabled = 0;
    sim_run(&S, 30.0);
    final_nocal_nogps = (float)fabs((double)S.core.pitch_est - (double)S.dyn.pitch);

    sim_init(&S); S.gyro_bias[1] = 0.01; S.launch_at_s = 3.0; S.gps_enabled = 1;
    sim_run(&S, 30.0);
    final_nocal_gps = (float)fabs((double)S.core.pitch_est - (double)S.dyn.pitch);
    peak_nocal_gps = S.max_pitch_est_err;
    bg_est = S.core.nav.bg[1];
    g_cfg_hook = 0;

    printf("     calibrated/no GPS: max err %.4f rad | uncalibrated/no GPS: final err %.4f rad | uncalibrated + GPS: final err %.4f, transient peak %.4f rad, bg_y est %.4f (true 0.0100)\n",
           err_cal_nogps, final_nocal_nogps, final_nocal_gps, peak_nocal_gps, bg_est);
    CHECK("pad calibration alone removes the bias (error < 0.01 rad, no GPS needed)", err_cal_nogps < 0.01f);
    CHECK("no calibration, no GPS: attitude drifts ~bias*t (> 0.2 rad = 11 deg by t=30 s)", final_nocal_nogps > 0.2f);
    CHECK("no calibration but GPS: ESKF learns the bias online (est within 0.0015 rad/s of truth)", fabs(bg_est - 0.01) < 0.0015);
    CHECK("... and ends with attitude error < 0.01 rad", final_nocal_gps < 0.01f);
    CHECK("... after a convergence transient that stays < 0.12 rad (documented: ~10 s to converge)", peak_nocal_gps < 0.12f);
}

static void test_imu_stuck(void)
{
    printf("Scenario 2: IMU 1 freezes (stuck-at, all axes) mid-ascent\n");
    sim_init(&S);
    S.imu[1].mode = IMU_STUCK; S.imu[1].t_on = 5.0;
    sim_run(&S, 30.0);
    CHECK("stuck IMU detected and flagged", (S.ever_fault_flags & FSW_FLAG_IMU1_FAULT) != 0);
    CHECK("healthy IMUs not blamed", (S.ever_fault_flags & (FSW_FLAG_IMU0_FAULT | FSW_FLAG_IMU2_FAULT)) == 0);
    CHECK("mission continues on 2-of-3 and reaches ORBIT_INSERTED", S.core.st.mode == FSW_ORBIT_INSERTED);
    CHECK("final vote uses 2 channels, excludes IMU 1",
          S.core.acc_vote[2].n_used == 2 && (S.core.acc_vote[2].used_mask & 2u) == 0 &&
          S.core.gyr_vote[1].n_used == 2 && (S.core.gyr_vote[1].used_mask & 2u) == 0);
    CHECK("vote error stays small during the failure", S.max_acc_vote_err < 0.5f);
    CHECK("attitude estimate unaffected by the frozen IMU (within 0.005 rad of the nominal run)", S.max_pitch_est_err < g_baseline_pitch_err + 0.005f);
}

static void test_transients(void)
{
    printf("Scenario 3: one-sample spike on IMU 2 (SSLV-D1 lesson: do not over-react)\n");
    sim_init(&S);
    S.imu[2].mode = IMU_SPIKE; S.imu[2].t_on = 6.0;
    sim_run(&S, 30.0);
    CHECK("single spike never latches a fault", S.ever_fault_flags == 0);
    CHECK("spike voted out: vote error < 0.5 m/s^2", S.max_acc_vote_err < 0.5f);
    CHECK("mission unaffected, attitude estimate within 0.005 rad of the nominal run", S.core.st.mode == FSW_ORBIT_INSERTED && S.max_pitch_est_err < g_baseline_pitch_err + 0.005f);

    printf("Scenario 4a: 2-sample burst of garbage on IMU 2 (below debounce)\n");
    sim_init(&S);
    S.imu[2].mode = IMU_BURST; S.imu[2].t_on = 6.0; S.imu[2].t_off = 6.04;
    sim_run(&S, 30.0);
    CHECK("2-sample burst does not latch", S.ever_fault_flags == 0 && S.core.st.mode == FSW_ORBIT_INSERTED);
    CHECK("garbage never reached the filter (within 0.005 rad of the nominal run)", S.max_pitch_est_err < g_baseline_pitch_err + 0.005f);

    printf("Scenario 4b: 4-sample burst -> fault declared, then channel RECOVERS\n");
    sim_init(&S);
    S.imu[2].mode = IMU_BURST; S.imu[2].t_on = 6.0; S.imu[2].t_off = 6.08;
    sim_run(&S, 30.0);
    CHECK("burst >= debounce declares a fault", (S.ever_fault_flags & FSW_FLAG_IMU2_FAULT) != 0 && S.core.acc_mon[2][2].fault_events == 1);
    CHECK("channel recovers after clean samples (not frozen out forever)", (S.core.fault_flags & 0x07u) == 0 && S.core.acc_vote[2].n_used == 3);
    CHECK("mission completes closed-loop throughout, attitude estimate intact", S.core.st.mode == FSW_ORBIT_INSERTED && S.max_pitch_est_err < g_baseline_pitch_err + 0.005f);
}

static void test_drift(void)
{
    printf("Scenario 5: slow accelerometer drift on IMU 0 (monitor-invisible)\n");
    sim_init(&S);
    S.imu[0].mode = IMU_DRIFT; S.imu[0].t_on = 3.0;
    sim_run(&S, 30.0);
    CHECK("drift never trips the per-channel monitor", (S.ever_fault_flags & FSW_FLAG_IMU0_FAULT) == 0);
    CHECK("voter excludes the drifting channel at the end", S.core.acc_vote[2].n_used == 2 && (S.core.acc_vote[2].used_mask & 1u) == 0);
    CHECK("voted accel error bounded (< 0.6 m/s^2)", S.max_acc_vote_err < 0.6f);
    CHECK("mission completes", S.core.st.mode == FSW_ORBIT_INSERTED);

    printf("Scenario 5b: slow GYRO drift on IMU 1 (0.01 rad/s per s): the voter must keep it out of the ESKF\n");
    sim_init(&S);
    S.imu[1].mode = IMU_GYRO_DRIFT; S.imu[1].t_on = 4.0;
    sim_run(&S, 30.0);
    CHECK("gyro drift never trips the per-channel monitor", (S.ever_fault_flags & FSW_FLAG_IMU1_FAULT) == 0);
    CHECK("voter excludes the drifting gyro at the end", S.core.gyr_vote[1].n_used == 2 && (S.core.gyr_vote[1].used_mask & 2u) == 0);
    CHECK("attitude within 0.005 rad of the nominal run (a leaked drift would reach ~0.26 rad/s by t=30 s)", S.max_pitch_est_err < g_baseline_pitch_err + 0.005f);
    CHECK("mission completes", S.core.st.mode == FSW_ORBIT_INSERTED);
}

static void test_two_dead(void)
{
    printf("Scenario 6: two IMUs die in flight (redundancy lost)\n");
    sim_init(&S);
    S.imu[0].mode = IMU_DEAD; S.imu[0].t_on = 5.0;
    S.imu[1].mode = IMU_DEAD; S.imu[1].t_on = 6.0;
    sim_run(&S, 30.0);
    CHECK("ABORT with reason FDIR_CRITICAL", S.core.st.mode == FSW_ABORT && S.core.st.last_reason == FSW_REASON_FDIR_CRITICAL);
    CHECK("abort happens promptly after trust is lost (< 0.6 s)", S.t_abort > 6.0 && S.t_abort < 6.6);
    CHECK("gimbal centred after abort", S.last_gimbal == 0.0f);
    CHECK("TRUST_LOST flag raised", (S.core.fault_flags & FSW_FLAG_TRUST_LOST) != 0);

    printf("Scenario 7: two IMUs already dead on the pad -> arm refused\n");
    sim_init(&S);
    S.imu[0].mode = IMU_DEAD; S.imu[0].t_on = 0.0;
    S.imu[1].mode = IMU_DEAD; S.imu[1].t_on = 0.0;
    sim_run(&S, 5.0);
    CHECK("vehicle stays PAD_SAFE, arm and launch rejected", S.core.st.mode == FSW_PAD_SAFE && S.core.st.rejected_cmds >= 1 && S.e1_start < 0);
}

static void test_engine_faults(void)
{
    printf("Scenario 8: engine fails to start\n");
    sim_init(&S);
    S.engine1_no_start = 1;
    sim_run(&S, 10.0);
    CHECK("ABORT with reason NO_IGNITION, never reaches ASCENT_S1",
          S.core.st.mode == FSW_ABORT && S.core.st.last_reason == FSW_REASON_NO_IGNITION && S.n_visited == 4);
    CHECK("navigation never went live without liftoff", !S.core.nav_active);

    printf("Scenario 9: engine cuts off prematurely at 2 s of burn\n");
    sim_init(&S);
    S.engine1_cut_after_s = 2.0;
    sim_run(&S, 15.0);
    CHECK("ABORT with reason PREMATURE_CUTOFF while in ASCENT_S1",
          S.core.st.mode == FSW_ABORT && S.core.st.last_reason == FSW_REASON_PREMATURE_CUTOFF);
}

static void test_abort_cmd(void)
{
    printf("Scenario 10: ground/range abort commands\n");
    sim_init(&S);
    S.abort_at_s = 5.0;
    sim_run(&S, 10.0);
    CHECK("abort command during ascent => ABORT (ABORT_CMD)", S.core.st.mode == FSW_ABORT && S.core.st.last_reason == FSW_REASON_ABORT_CMD);

    sim_init(&S);
    S.abort_at_s = 0.4;   /* armed at 0.2 s, launch not yet sent */
    sim_run(&S, 3.0);
    CHECK("abort while ARMED on the pad => PAD_SAFE, engine never lit", S.core.st.mode == FSW_PAD_SAFE && S.e1_start < 0);
}

static void test_navigation_faults(void)
{
    printf("Scenario 11: 400 m position-fix outlier at t=12 s (bad GPS multipath / spoof-like jump)\n");
    sim_init(&S);
    S.gps_outlier_at_s = 12.0;
    sim_run(&S, 30.0);
    CHECK("outlier rejected by the innovation gate", S.core.nav.n_rejected >= 1);
    CHECK("estimate not corrupted: position error stays < 12 m", S.max_pos_err_after_5s < 12.0f);
    CHECK("mission completes", S.core.st.mode == FSW_ORBIT_INSERTED);

    printf("Scenario 12: GPS unavailable for the whole flight (pure inertial)\n");
    sim_init(&S);
    S.gps_enabled = 0;
    sim_run(&S, 30.0);
    printf("     inertial-only: max pitch err %.5f rad, final pos err %.2f m\n", S.max_pitch_est_err, S.final_pos_err);
    CHECK("no fixes were applied", S.core.nav.n_update == 0);
    CHECK("pure-inertial attitude is accurate (< 0.005 rad, better than the GPS-fused run here)", S.max_pitch_est_err < 0.005f);
    CHECK("mission completes without GPS", S.core.st.mode == FSW_ORBIT_INSERTED);

    printf("Scenario 13: navigation filter corrupted in flight (covariance goes NaN)\n");
    sim_init(&S);
    sim_run(&S, 6.0);
    fsw_eskf_set_P_diag(&S.core.nav, 4, (float)NAN);
    sim_run(&S, 8.0);
    CHECK("NAV_FAULT flagged", (S.core.fault_flags & FSW_FLAG_NAV_FAULT) != 0 || S.core.st.mode == FSW_ABORT);
    CHECK("persistent nav fault => ABORT (FDIR_CRITICAL)", S.core.st.mode == FSW_ABORT && S.core.st.last_reason == FSW_REASON_FDIR_CRITICAL);
    CHECK("gimbal centred (control never runs on a corrupt estimate)", S.last_gimbal == 0.0f);
}


static void test_gps_latency(void)
{
    float pos_err_comp, pos_err_raw, pitch_comp, pitch_raw;
    uint32_t rej_comp, rej_raw;
    float fix[3] = {1.0f, 2.0f, 3.0f};
    uint32_t d0;
    printf("Scenario 14: GPS receiver reports 0.3 s late (~30 m at flight speeds)\n");
    sim_init(&S); S.gps_delay_s = 0.3; S.gps_report_age = 1;
    sim_run(&S, 30.0);
    pos_err_comp = S.max_pos_err_after_5s; pitch_comp = S.max_pitch_est_err; rej_comp = S.core.nav.n_rejected;
    CHECK("compensated run completes the mission", S.core.st.mode == FSW_ORBIT_INSERTED);
    sim_init(&S); S.gps_delay_s = 0.3; S.gps_report_age = 0;
    sim_run(&S, 30.0);
    pos_err_raw = S.max_pos_err_after_5s; pitch_raw = S.max_pitch_est_err; rej_raw = S.core.nav.n_rejected;
    printf("     age-compensated: pos err %.2f m, pitch err %.4f rad, gated %u | latency ignored: pos err %.2f m, pitch err %.4f rad, gated %u\n",
           pos_err_comp, pitch_comp, rej_comp, pos_err_raw, pitch_raw, rej_raw);
    CHECK("compensated: position error stays < 12 m despite 0.3 s latency", pos_err_comp < 12.0f);
    CHECK("compensated: no valid fix is wrongly gated out", rej_comp == 0);
    CHECK("ignoring the latency is measurably worse (position error at least 1.5x larger)", pos_err_raw > 1.5f * pos_err_comp);
    CHECK("ignoring the latency drives fixes into the innovation gate (>= 1 rejection)", rej_raw >= 1);

    printf("Scenario 14b: bad fix ages are refused\n");
    sim_init(&S);
    sim_run(&S, 6.0);                       /* nav is live */
    d0 = S.core.gps_dropped;
    fsw_core_gps_fix_aged(&S.core, fix, -0.1f);
    fsw_core_gps_fix_aged(&S.core, fix, (float)NAN);
    CHECK("negative and NaN ages dropped at the API", S.core.gps_dropped == d0 + 2 && !S.core.gps_pending);
    fsw_core_gps_fix_aged(&S.core, fix, 5.0f);     /* accepted by the API, but older than the history */
    fsw_core_step(&S.core);
    CHECK("fix older than the stored history dropped at apply time", S.core.gps_dropped == d0 + 3 && !S.core.gps_pending);
    fix[0] = (float)NAN;
    fsw_core_gps_fix_aged(&S.core, fix, 0.1f);
    CHECK("NaN position dropped at the API", S.core.gps_dropped == d0 + 4);
    sim_init(&S);                           /* still on the pad: no navigation yet */
    fsw_core_gps_fix_aged(&S.core, (float[3]){0, 0, 0}, 0.1f);
    sim_run(&S, 0.5);
    CHECK("fix arriving before navigation is live never reaches the filter", S.core.nav.n_update == 0);
}

static void guided_setup(sim_t *s)
{
    fsw_guidance_cfg_t gc;
    double r = 6378137.0 + 200000.0;
    sim_init(s);
    s->guided = 1;
    s->burn1_s = 24.0;
    s->phi_program = 1.5708 - 0.0925;                 /* first PEG solve asks for theta0 = 5.3 deg from horizontal */
    s->pm_F_true = 30000.0; s->pm_isp = 320.0; s->pm_m_dry = 1000.0;
    s->polar_dropout_at_s = -1.0; s->polar_nan_at_s = -1.0;
    fsw_guidance_default_cfg(&gc);
    gc.target.r_t = r; gc.target.v_t = sqrt(FSW_PEG_MU_EARTH / r); gc.target.gamma_t = 0.0;
    gc.thrust_n = 30000.0; gc.isp_s = 320.0; gc.m_dry_kg = 1000.0;
    fsw_guidance_init(&s->guid, &gc);
    fsw_core_attach_guidance(&s->core, &s->guid);
}

static void test_guided_burn(void)
{
    static const fsw_mode_t seq[] = { FSW_PAD_SAFE, FSW_PAD_ARMED, FSW_IGNITION, FSW_ASCENT_S1,
                                      FSW_STAGE_SEP, FSW_ASCENT_S2, FSW_COAST, FSW_ORBIT_INSERTED };
    fsw_tlm_parser_t p; fsw_tlm_frame_t f; size_t i; int ok_frames = 0, guid_valid_frames = 0; float last_ttg = 1e9f, max_ttg_rise = 0.0f, first_ttg = -1.0f; int ttg_mono = 1;
    double dv, dr, dg, tgt_v;

    printf("Scenario 15: PEG-guided upper-stage burn, closed loop (point-mass truth, real core, real ESKF)\n");
    guided_setup(&S);
    sim_run(&S, 250.0);
    tgt_v = S.guid.cfg.target.v_t;
    dv = S.pm.v - tgt_v; dr = S.pm.r - S.guid.cfg.target.r_t; dg = S.pm.gamma;
    printf("     burn %.1f s | cutoff by command at t=%.1f s | final err: r %+.0f m, v %+.2f m/s, gamma %+.3f deg | solves %u (cold retries %u, degraded %u, failed %u) | max solve %.1f ms (HOST)\n",
           S.pm_cut_t - S.pm_ign_t, S.pm_cut_t, dr, dv, dg * 57.29578, S.guid.solves, S.guid.cold_retries, S.guid.degraded, S.guid.failed, S.max_solve_ms);
    printf("     max |pitch command - true pitch| in S2 = %.4f rad; ESKF max pitch err %.4f rad\n", S.max_track_err, S.max_pitch_est_err);
    CHECK("full mode sequence, ending in ORBIT_INSERTED", visited_is(&S, seq, 8));
    CHECK("main engine was cut by the GUIDANCE command (not by propellant depletion)", S.pm_cut_by_cmd && !S.pm_depleted && S.core.cutoff_sent);
    CHECK("burn time within 1 s of the PEG prediction made at the first solve (~184 s)", fabs((S.pm_cut_t - S.pm_ign_t) - 184.0) < 6.0);
    CHECK("true insertion error: |v| < 15 m/s", fabs(dv) < 15.0);
    CHECK("true insertion error: |gamma| < 0.5 deg", fabs(dg) < 0.0087);
    CHECK("true insertion error: |r| < 5 km", fabs(dr) < 5000.0);
    CHECK("insertion declared by the guidance window check on the polar state", (S.core.guid_flags & FSW_GUID_INSERTED) != 0);
    CHECK("guidance solved every ~2 s through the burn (>= 85 solves), none failed", S.guid.solves >= 85 && S.guid.failed == 0);
    CHECK("every solve ran through fsw_core_guidance_service, none inside the control step", S.core.guid_serviced == S.guid.solves);
    CHECK("attitude loop tracks the slew-limited guidance command within 0.12 rad throughout S2", S.max_track_err < 0.12f);
    CHECK("gimbal limit respected", S.max_abs_gimbal <= 0.10472f + 1e-5f);
    CHECK("no ESKF fault, no watchdog trip", (S.core.fault_flags & (FSW_FLAG_NAV_FAULT | FSW_FLAG_WDG_TRIPPED)) == 0);

    fsw_tlm_parser_init(&p);
    for (i = 0; i < S.txn; i++) {
        if (fsw_tlm_parser_feed(&p, S.tx[i], &f)) {
            do {
                ok_frames++;
                if (f.payload[32] & FSW_GUID_VALID) {
                    uint32_t u; float ttg;
                    guid_valid_frames++;
                    u = (uint32_t)f.payload[36] | ((uint32_t)f.payload[37] << 8) | ((uint32_t)f.payload[38] << 16) | ((uint32_t)f.payload[39] << 24);
                    memcpy(&ttg, &u, 4);
                    if (first_ttg < 0.0f) first_ttg = ttg;
                    if (last_ttg < 1e8f && ttg - last_ttg > max_ttg_rise) max_ttg_rise = ttg - last_ttg;
                    last_ttg = ttg;
                }
            } while (fsw_tlm_parser_poll(&p, &f));
        }
    }
    CHECK("telemetry carries the guidance flags: VALID throughout the guided burn", p.crc_errors == 0 && guid_valid_frames > 80);
    printf("     telemetry T_go: first %.1f s, last %.2f s, largest re-solve increase %.3f s\n", first_ttg, last_ttg, max_ttg_rise);
    ttg_mono = (max_ttg_rise < 0.5f);
    CHECK("telemetry time-to-go runs ~184 s -> ~0 and never rises by more than 0.5 s (re-solve adjustments only)", ttg_mono && first_ttg > 170.0f && last_ttg < 1.0f);
}

static void test_guided_faults(void)
{
    printf("Scenario 16a: engine delivers 5 %% less thrust than guidance assumes (closed loop must absorb it)\n");
    guided_setup(&S);
    S.pm_F_true = 0.95 * 30000.0;
    sim_run(&S, 280.0);
    printf("     final err: r %+.0f m, v %+.2f m/s, gamma %+.3f deg, burn %.1f s\n", S.pm.r - S.guid.cfg.target.r_t, S.pm.v - S.guid.cfg.target.v_t, S.pm.gamma * 57.29578, S.pm_cut_t - S.pm_ign_t);
    CHECK("still inserted: |v| < 15 m/s, |gamma| < 0.5 deg, |r| < 5 km",
          fabs(S.pm.v - S.guid.cfg.target.v_t) < 15.0 && fabs(S.pm.gamma) < 0.0087 && fabs(S.pm.r - S.guid.cfg.target.r_t) < 5000.0);
    CHECK("mission ends ORBIT_INSERTED", S.core.st.mode == FSW_ORBIT_INSERTED);

    printf("Scenario 16b: navigation bridge dies 60 s into the burn (polar state stops updating)\n");
    guided_setup(&S);
    S.polar_dropout_at_s = 26.0 + 60.0;
    sim_run(&S, 250.0);
    printf("     STALE flag %d, solves %u, final err: v %+.2f m/s, mode %d\n", (S.ever_guid_flags & FSW_GUID_STALE_STATE) != 0, S.guid.solves, S.pm.v - S.guid.cfg.target.v_t, S.core.st.mode);
    CHECK("STALE_STATE raised, no new solves started on old data", (S.ever_guid_flags & FSW_GUID_STALE_STATE) != 0 && S.guid.solves < 50);
    CHECK("last solution keeps flying and the guided cutoff is still sent at its T_go", S.core.cutoff_sent && S.pm_cut_by_cmd);
    CHECK("no abort: vehicle ends in COAST (insertion cannot be confirmed without fresh state)", S.core.st.mode == FSW_COAST);

    printf("Scenario 16c: polar state turns to NaN 80 s into the burn (bad navigation data)\n");
    guided_setup(&S);
    S.polar_nan_at_s = 26.0 + 80.0;
    sim_run(&S, 380.0);                         /* the fallback burn runs to propellant depletion (~340 s) */
    printf("     guidance failures %u, withdrawn=%d, LOST flag %d, cutoff sent %d, mode %d\n", S.guid.failed, !S.core.guid_cache.valid, (S.ever_guid_flags & FSW_GUID_LOST) != 0, S.core.cutoff_sent, S.core.st.mode);
    CHECK("NaN state rejected by the solver; solution withdrawn after repeated failures", S.guid.failed >= 1 && !S.core.guid_cache.valid);
    CHECK("LOST flagged and the core falls back to the external pitch program (no crash, no NaN command)", (S.ever_guid_flags & FSW_GUID_LOST) != 0 && S.core.pitch_cmd == S.core.pitch_cmd);
    CHECK("no guided cutoff was commanded from withdrawn guidance", !S.core.cutoff_sent || S.pm_cut_by_cmd == 0 || S.pm_cut_t < 26.0 + 80.0 + 8.0);
    CHECK("burn ends by propellant depletion and the state machine reaches COAST (not INSERTED)", S.pm_depleted && S.core.st.mode == FSW_COAST);
}

/* ======================= ECI navigation scenarios ======================= */

static void eci_cfg(fsw_core_cfg_t *c)
{
    c->nav_eci = 1;
    fsw_site_init(&g_site, c->site_lat_rad, c->site_lon_rad, c->site_radius_m, c->launch_azimuth_rad, c->omega_earth);
}
static void eci_cfg_nojs(fsw_core_cfg_t *c) { eci_cfg(c); c->nav_j2 = 0; }

/* NEES of (velocity, position) error against the filter's own covariance: consistent filter -> mean ~ 6 */
static double nees6(const sim_t *s)
{
    double e[6], P[6][6], M[6][7], acc = 0.0;
    int i, j, k;
    for (i = 0; i < 3; i++) { e[i] = s->core.nav.v[i] - s->tvI[i]; e[3 + i] = s->core.nav.p[i] - s->tpI[i]; }
    for (i = 0; i < 6; i++) for (j = 0; j < 6; j++) P[i][j] = (double)s->core.nav.P[3 + i][3 + j];
    for (i = 0; i < 6; i++) { for (j = 0; j < 6; j++) M[i][j] = P[i][j]; M[i][6] = e[i]; }
    for (k = 0; k < 6; k++) {
        int piv = k;
        for (i = k + 1; i < 6; i++) if (fabs(M[i][k]) > fabs(M[piv][k])) piv = i;
        if (fabs(M[piv][k]) < 1e-30) return -1.0;
        if (piv != k) for (j = 0; j < 7; j++) { double tmp = M[k][j]; M[k][j] = M[piv][j]; M[piv][j] = tmp; }
        for (i = k + 1; i < 6; i++) { double f = M[i][k] / M[k][k]; for (j = k; j < 7; j++) M[i][j] -= f * M[k][j]; }
    }
    { double x[6]; for (i = 5; i >= 0; i--) { double a = M[i][6]; for (j = i + 1; j < 6; j++) a -= M[i][j] * x[j]; x[i] = a / M[i][i]; } for (i = 0; i < 6; i++) acc += e[i] * x[i]; }
    return acc;
}

static void eci_setup_plain(sim_t *s, void (*hook)(fsw_core_cfg_t *))
{
    g_cfg_hook = hook;
    sim_init(s);
    g_cfg_hook = 0;
    s->eci = 1; s->j2_in_truth = 1;
}

static void test_eci_nominal(void)
{
    static const fsw_mode_t seq[] = { FSW_PAD_SAFE, FSW_PAD_ARMED, FSW_IGNITION, FSW_ASCENT_S1,
                                      FSW_STAGE_SEP, FSW_ASCENT_S2, FSW_COAST, FSW_ORBIT_INSERTED };
    double p_j2, v_j2, p_nj2, v_nj2, p_ing, v_ing, nees_acc = 0.0;
    int nn = 0, run, tk;
    double bg_err, padbg_err;

    printf("Scenario 17: inertial (ECI) navigation through the toy mission: gravity, J2, Earth rotation, ECEF fixes\n");
    eci_setup_plain(&S, eci_cfg);
    S.gyro_bias[1] = 0.002; S.launch_at_s = 3.0;
    sim_run(&S, 30.0);
    p_j2 = S.max_pos3_after10; v_j2 = S.max_vel3_after10;
    padbg_err = fabs((double)S.core.pad_bg[1] - 0.002);
    printf("     nav vs truth after 10 s: max |dp| %.2f m, max |dv| %.3f m/s | pad gyro mean leaks Earth rate: |pad_bg - bias| = %.1e rad/s (Omega*cos(lat) = %.1e)\n",
           p_j2, v_j2, padbg_err, FSW_OMEGA_EARTH * cos(g_site.lat));
    CHECK("mission sequence completes with the inertial navigation in the loop", visited_is(&S, seq, 8) && S.core.st.mode == FSW_ORBIT_INSERTED);
    CHECK("ECI position error stays < 20 m (3 m 1-sigma ECEF fixes, 1 Hz) after 10 s", p_j2 < 20.0);
    /* GPS-driven velocity noise: 3 m fixes at 1 Hz give ~1 m/s excursions in this short mission; NEES below is the consistency criterion. */
    CHECK("ECI velocity error stays < 2 m/s after 10 s", v_j2 < 2.0);
    CHECK("pitch estimate (inertial frame) tracks truth within 0.02 rad", S.max_pitch_est_err < 0.02f);
    bg_err = fabs(S.bg_y_at_init - 0.002);
    CHECK("the pad gyro average contains the Earth's rotation (the effect exists)", padbg_err > 0.5 * FSW_OMEGA_EARTH * cos(g_site.lat));
    CHECK("... and the filter's bias state at liftoff does not (|bg - true bias| < 3e-5 rad/s: Earth rate removed)", bg_err < 3e-5);
    CHECK("ECEF fixes were fused (>= 15) and none gated in a clean run", S.core.nav.n_update >= 15 && S.core.nav.n_rejected == 0);

    eci_setup_plain(&S, eci_cfg);
    S.gps_enabled = 0; S.launch_at_s = 3.0;
    sim_run(&S, 30.0);
    p_ing = S.max_pos3_after10; v_ing = S.max_vel3_after10;
    eci_setup_plain(&S, eci_cfg_nojs);
    S.gps_enabled = 0; S.launch_at_s = 3.0;
    sim_run(&S, 30.0);
    p_nj2 = S.max_pos3_after10; v_nj2 = S.max_vel3_after10;
    printf("     pure inertial (no GPS), 30 s: J2 modelled %.1f m, %.3f m/s | J2 ignored %.1f m, %.3f m/s | GPS-aided: %.1f m, %.3f m/s\n", p_ing, v_ing, p_nj2, v_nj2, p_j2, v_j2);
    CHECK("pure inertial: ignoring J2 (truth has it, ~0.016 m/s^2) is worse than modelling it (position)", p_nj2 > p_ing && v_nj2 > v_ing);
    CHECK("pure inertial over the 30 s mission stays within 50 m / 1 m/s with J2 modelled (IMU noise only, no bias in this sim)", p_ing < 50.0 && v_ing < 1.0);

    for (run = 0; run < 8; run++) {
        eci_setup_plain(&S, eci_cfg);
        S.rng = 777u + 13u * (uint32_t)run; S.launch_at_s = 3.0;
        for (tk = 0; tk < 5; tk++) {
            double n;
            sim_run(&S, 12.0 + 4.0 * tk);
            n = nees6(&S);
            if (n >= 0.0) { nees_acc += n; nn++; }
        }
    }
    printf("     NEES (6 dof: velocity + position) averaged over %d samples = %.2f (consistent filter: ~6)\n", nn, nees_acc / (double)nn);
    CHECK("filter is statistically consistent: mean NEES within [1.5, 20] (6 dof)", nn >= 30 && nees_acc / (double)nn > 1.5 && nees_acc / (double)nn < 20.0);
}

static void guided_setup_eci(sim_t *s, void (*hook)(fsw_core_cfg_t *))
{
    g_cfg_hook = hook;
    guided_setup(s);
    g_cfg_hook = 0;
    s->eci = 1; s->j2_in_truth = 1;
    fsw_core_set_stage2_mass(&s->core, 4000.0);
}

static void truth_polar(const sim_t *s, double *r, double *v, double *g, double *psi) { fsw_polar_from_inertial(s->tpI, s->tvI, r, v, g, psi); }

static void test_eci_guided(void)
{
    static const fsw_mode_t seq[] = { FSW_PAD_SAFE, FSW_PAD_ARMED, FSW_IGNITION, FSW_ASCENT_S1,
                                      FSW_STAGE_SEP, FSW_ASCENT_S2, FSW_COAST, FSW_ORBIT_INSERTED };
    double r, v, g, psi, dv, dr, dg, nr, nv, ng, npsi;
    double dv_nj2;

    printf("Scenario 18: PEG-guided burn with the polar state DERIVED FROM THE ECI FILTER (truth: 3-D inertial point mass + J2)\n");
    guided_setup_eci(&S, eci_cfg);
    sim_run(&S, 250.0);
    truth_polar(&S, &r, &v, &g, &psi);
    fsw_polar_from_inertial(S.core.nav.p, S.core.nav.v, &nr, &nv, &ng, &npsi);
    dv = v - S.guid.cfg.target.v_t; dr = r - S.guid.cfg.target.r_t; dg = g;
    printf("     hand-off residual injected: %.1f m, %.3f m/s | cutoff by command at t=%.1f s, burn %.1f s | TRUE final err: r %+.0f m, v %+.2f m/s, gamma %+.3f deg\n",
           S.handoff_pos_err, S.handoff_vel_err, S.pm_cut_t, S.pm_cut_t - S.pm_ign_t, dr, dv, dg * 57.29578);
    printf("     filter polar state vs truth at the end: r %+.1f m, v %+.3f m/s, gamma %+.4f deg, psi %+.5f rad | solves %u failed %u\n",
           nr - r, nv - v, (ng - g) * 57.29578, npsi - psi, S.guid.solves, S.guid.failed);
    CHECK("full mode sequence, ending in ORBIT_INSERTED (insertion judged on the filter's own polar state)", visited_is(&S, seq, 8));
    CHECK("cutoff commanded by guidance from filter-derived time-to-go", S.pm_cut_by_cmd && !S.pm_depleted && S.core.cutoff_sent);
    CHECK("TRUE insertion error: |v| < 15 m/s, |gamma| < 0.5 deg, |r| < 5 km", fabs(dv) < 15.0 && fabs(dg) < 0.0087 && fabs(dr) < 5000.0);
    CHECK("filter's polar state at the end within 30 m / 0.5 m/s / 0.02 deg of truth", fabs(nr - r) < 30.0 && fabs(nv - v) < 0.5 && fabs(ng - g) < 3.5e-4);
    CHECK("downrange angle psi agrees with truth to 1e-4 rad (~0.6 km of arc)", fabs(npsi - psi) < 1e-4);
    CHECK("guided solves ran from the filter state: >= 85 solves, none failed", S.guid.solves >= 85 && S.guid.failed == 0);
    CHECK("attitude loop tracks the guidance command within 0.12 rad", S.max_track_err < 0.12f);
    CHECK("no filter fault flagged through the burn", (S.core.fault_flags & FSW_FLAG_NAV_FAULT) == 0);

    guided_setup_eci(&S, eci_cfg_nojs);
    sim_run(&S, 250.0);
    truth_polar(&S, &r, &v, &g, &psi);
    dv_nj2 = v - S.guid.cfg.target.v_t;
    printf("     filter WITHOUT J2 (truth has it): TRUE final v err %+.2f m/s (vs %+.2f with J2), mode %d\n", dv_nj2, dv, S.core.st.mode);
    CHECK("still inserts without J2 in the filter (GPS absorbs it): |v| < 15 m/s, |gamma| < 0.5 deg", fabs(dv_nj2) < 15.0 && fabs(g) < 0.0087);
}

static void test_eci_gps(void)
{
    double e_exact, e_off, e_frac;
    uint32_t d0;
    double ecef[3] = { 5.0e6, 1.0e6, 3.0e6 };
    float flatfix[3] = { 1.0f, 2.0f, 3.0f };

    printf("Scenario 19: ECEF position fixes -- API rules, fractional age, timing sensitivity at orbital speed\n");
    eci_setup_plain(&S, eci_cfg);
    sim_run(&S, 6.0);
    d0 = S.core.gps_dropped;
    fsw_core_gps_fix_aged(&S.core, flatfix, 0.0f);
    CHECK("ECI mode: a flat pad-frame fix is refused (wrong frame)", S.core.gps_dropped == d0 + 1 && !S.core.gps_pending);
    fsw_core_gps_fix_ecef(&S.core, ecef, -0.1f);
    fsw_core_gps_fix_ecef(&S.core, ecef, (float)NAN);
    ecef[1] = (double)NAN; fsw_core_gps_fix_ecef(&S.core, ecef, 0.1f); ecef[1] = 1.0e6;
    CHECK("negative age, NaN age and NaN position are refused at the API", S.core.gps_dropped == d0 + 4 && !S.core.gps_pending);
    fsw_core_set_polar_state(&S.core, 6.6e6, 7000.0, 0.0, 1000.0, 0.0);
    CHECK("an external polar state is ignored in ECI mode (the core derives its own)", !S.core.polar_valid);
    sim_init(&S);                                           /* flat core */
    sim_run(&S, 6.0);
    d0 = S.core.gps_dropped;
    fsw_core_gps_fix_ecef(&S.core, ecef, 0.0f);
    CHECK("flat mode: an ECEF fix is refused", S.core.gps_dropped == d0 + 1 && !S.core.gps_pending);

    /* orbital regime: 0.31 s receiver delay (15.5 cycles: not a whole number), first 110 s of the guided burn */
    guided_setup_eci(&S, eci_cfg); S.gps_delay_s = 0.31; S.gps_age_error_s = 0.0;
    sim_run(&S, 110.0);
    e_exact = S.max_pos3_after10;
    guided_setup_eci(&S, eci_cfg); S.gps_delay_s = 0.30; S.gps_age_error_s = 0.0;
    sim_run(&S, 110.0);
    e_frac = S.max_pos3_after10;
    guided_setup_eci(&S, eci_cfg); S.gps_delay_s = 0.31; S.gps_age_error_s = -0.010;     /* receiver says 0.30 s, it was 0.31 s */
    sim_run(&S, 110.0);
    e_off = S.max_pos3_after10;
    printf("     max position error: age exact (0.31 s, 15.5 cycles) %.1f m | whole-cycle age (0.30 s) %.1f m | age wrong by 10 ms %.1f m (speed ~6-7 km/s: 10 ms = 65 m)\n", e_exact, e_frac, e_off);
    CHECK("a fractional age (0.31 s = 15.5 cycles) is handled by interpolation: position error < 25 m", e_exact < 25.0);
    CHECK("whole-cycle delay also fine (< 25 m)", e_frac < 25.0);
    CHECK("at orbital speed a 10 ms error in the reported age is clearly visible (error >= 3x larger): GPS time-tagging needs ms accuracy", e_off > 3.0 * e_exact);
}

int main(void)
{
    printf("== AscentGNC flight_software self-test ==\n");
    test_math();
    test_crc();
    test_telemetry();
    test_monitor();
    test_vote();
    test_state_unit();
    test_state_s2_low_twr();
    test_guidance_unit();
    test_watchdog();
    test_nominal();
    test_pad_alignment_and_bias();
    test_bias_handling();
    test_imu_stuck();
    test_transients();
    test_drift();
    test_two_dead();
    test_engine_faults();
    test_abort_cmd();
    test_navigation_faults();
    test_gps_latency();
    test_guided_burn();
    test_guided_faults();
    test_eci_nominal();
    test_eci_guided();
    test_eci_gps();
    printf("%d passed, %d failed out of %d\n", g_pass, g_fail, g_pass + g_fail);
    return g_fail == 0 ? 0 : 1;
}
