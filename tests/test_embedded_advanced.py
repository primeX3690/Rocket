"""
tests/test_embedded_advanced.py

Verification suite for embedded/actuator_realism.py, embedded/rtos_sim.py,
and embedded/protocols.py.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from embedded.actuator_realism import DeadbandBacklashActuator
from embedded.rtos_sim import (
    PeriodicTask, utilization_bound_schedulable, exact_response_time_analysis,
    simulate_scheduler_timeline,
)
from embedded.protocols import (
    encode_can_frame, decode_and_verify_can_frame, can_crc15,
    encode_arinc429_word, decode_arinc429_word,
    spacewire_encode_packet, spacewire_decode_packets, spacewire_time_code,
)

PASS = 0
FAIL = 0


def check(name, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


def test_deadband_blocks_small_commands():
    act = DeadbandBacklashActuator(deadband_deg=0.2, backlash_deg=0.0)
    outputs = [act.step(np.radians(0.05), 0.01) for _ in range(50)]
    check("A command smaller than the deadband produces zero actuator motion",
          max(abs(o) for o in outputs) < 1e-9)


def test_large_command_moves_actuator():
    act = DeadbandBacklashActuator(deadband_deg=0.1, backlash_deg=0.0, max_slew_rate_deg_s=20.0)
    for _ in range(300):
        act.step(np.radians(5.0), 0.01)
    check("A large, sustained command drives the actuator close to the commanded angle",
          abs(np.degrees(act.actual_angle_rad) - 5.0) < 0.5)


def test_backlash_delays_direction_reversal():
    act = DeadbandBacklashActuator(deadband_deg=0.0, backlash_deg=1.0, max_slew_rate_deg_s=20.0)
    for _ in range(300):
        act.step(np.radians(5.0), 0.01)
    angle_before = act.actual_angle_rad
    for _ in range(3):
        act.step(np.radians(-5.0), 0.01)
    movement = angle_before - act.actual_angle_rad
    unconstrained_movement = np.radians(20.0) * 0.01 * 3
    check("Backlash causes LESS movement than an unconstrained rate-limited reversal would give",
          0 <= movement < unconstrained_movement)


def test_can_crc_detects_corruption():
    frame = encode_can_frame(0x123, b"\x01\x02\x03\x04")
    check("A freshly-encoded CAN frame verifies as valid", decode_and_verify_can_frame(frame))
    corrupted = dict(frame)
    corrupted["crc15"] = frame["crc15"] ^ 0x1
    check("A CAN frame with a flipped CRC bit is detected as invalid",
          not decode_and_verify_can_frame(corrupted))


def test_can_crc_is_deterministic_and_sensitive_to_payload():
    check("Identical payloads produce identical CRC-15",
          can_crc15("0" * 11 + "0100" + "00000001") == can_crc15("0" * 11 + "0100" + "00000001"))
    check("Different payloads produce different CRC-15 (no trivial collision for this pair)",
          can_crc15("0" * 11 + "0100" + "00000001") != can_crc15("0" * 11 + "0100" + "00000010"))


def test_arinc429_round_trip():
    word = encode_arinc429_word(label_octal=0o310, sdi=1, data_value=12345, ssm=0b11)
    decoded = decode_arinc429_word(word)
    check("ARINC 429 label round-trips correctly", decoded["label_octal"] == 0o310)
    check("ARINC 429 SDI round-trips correctly", decoded["sdi"] == 1)
    check("ARINC 429 data value round-trips correctly", decoded["data_value"] == 12345)
    check("ARINC 429 SSM round-trips correctly", decoded["ssm"] == 0b11)
    check("ARINC 429 parity checks out on an unmodified word", decoded["parity_ok"])


def test_arinc429_parity_detects_bit_flip():
    word = encode_arinc429_word(label_octal=0o201, sdi=0, data_value=500, ssm=0)
    flipped = word ^ (1 << 5)   # flip a data bit, not the parity bit itself
    check("Flipping a single data bit is caught by the odd-parity check",
          not decode_arinc429_word(flipped)["parity_ok"])


def test_spacewire_packet_round_trip():
    chars = spacewire_encode_packet(b"hello") + spacewire_encode_packet(b"world", use_eep=True)
    packets = spacewire_decode_packets(chars)
    check("SpaceWire decoding recovers exactly 2 packets from the concatenated stream",
          len(packets) == 2)
    check("First SpaceWire packet's data round-trips exactly", packets[0]["data"] == b"hello")
    check("First SpaceWire packet terminated normally (EOP) and is marked valid",
          packets[0]["terminator"] == "EOP" and packets[0]["valid"])
    check("Second SpaceWire packet terminated with an error marker (EEP) and is marked invalid",
          packets[1]["terminator"] == "EEP" and not packets[1]["valid"])


def test_spacewire_time_code_is_6_bit():
    check("SpaceWire time-code value stays within its 6-bit range",
          0 <= spacewire_time_code(45) <= 0x3F)
    check("SpaceWire time-code wraps correctly at the 6-bit boundary",
          spacewire_time_code(64) == 0)


def test_rms_schedulable_task_set_meets_all_deadlines():
    tasks = [
        PeriodicTask("guidance", period_s=0.020, execution_time_s=0.004),
        PeriodicTask("control", period_s=0.005, execution_time_s=0.0015),
        PeriodicTask("telemetry", period_s=0.100, execution_time_s=0.010),
    ]
    bound_check = utilization_bound_schedulable(tasks)
    check("A realistic, lightly-loaded flight-computer task set passes the RMS utilization bound",
          bound_check["guaranteed_schedulable"])
    rta = exact_response_time_analysis(tasks)
    check("Exact response-time analysis confirms every task meets its deadline",
          all(r["meets_deadline"] for r in rta.values()))
    sim = simulate_scheduler_timeline(tasks, sim_duration_s=0.5, tick_s=0.0002)
    check("A literal tick-by-tick scheduler simulation confirms zero missed deadlines",
          sim["schedulable"])


def test_overloaded_task_set_misses_a_deadline():
    tasks = [
        PeriodicTask("a", period_s=0.010, execution_time_s=0.006),
        PeriodicTask("b", period_s=0.010, execution_time_s=0.006),
    ]
    rta = exact_response_time_analysis(tasks)
    check("An overloaded (>100% CPU) task set's lower-priority task misses its deadline",
          not rta["b"]["meets_deadline"])
    check("The overloaded task set's higher-priority task still meets its deadline",
          rta["a"]["meets_deadline"])


if __name__ == "__main__":
    print("Running embedded advanced (actuator realism / RTOS scheduling / protocols) suite...\n")
    test_deadband_blocks_small_commands()
    test_large_command_moves_actuator()
    test_backlash_delays_direction_reversal()
    test_can_crc_detects_corruption()
    test_can_crc_is_deterministic_and_sensitive_to_payload()
    test_arinc429_round_trip()
    test_arinc429_parity_detects_bit_flip()
    test_spacewire_packet_round_trip()
    test_spacewire_time_code_is_6_bit()
    test_rms_schedulable_task_set_meets_all_deadlines()
    test_overloaded_task_set_misses_a_deadline()

    print(f"\n{PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    if FAIL > 0:
        sys.exit(1)
