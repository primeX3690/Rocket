"""
tests/test_telemetry_security.py

Verification suite for telemetry/secure_log.py (Merkle tamper-evident
logging) and telemetry/secure_stream.py (AES-256-GCM encrypted,
authenticated, replay-resistant telemetry).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from telemetry.secure_log import MerkleTelemetryLog
from telemetry.secure_stream import SecureTelemetryChannel, ReplayGuard, _CRYPTOGRAPHY_AVAILABLE

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


def test_merkle_log_root_is_deterministic():
    log1 = MerkleTelemetryLog()
    log2 = MerkleTelemetryLog()
    for i in range(15):
        record = {"t": i * 0.1, "altitude_m": i * 50.0}
        log1.append(record)
        log2.append(record)
    check("Two logs with identical records produce identical root hashes",
          log1.root_hash() == log2.root_hash())


def test_merkle_log_root_changes_on_any_record_difference():
    log1 = MerkleTelemetryLog()
    log2 = MerkleTelemetryLog()
    for i in range(15):
        log1.append({"t": i * 0.1, "altitude_m": i * 50.0})
        log2.append({"t": i * 0.1, "altitude_m": i * 50.0 + (0.001 if i == 7 else 0.0)})
    check("A tiny difference in a single record changes the entire log's root hash",
          log1.root_hash() != log2.root_hash())


def test_merkle_inclusion_proof_verifies_correctly():
    log = MerkleTelemetryLog()
    for i in range(23):   # odd count exercises the "duplicate last node" path
        log.append({"t": i * 0.1, "value": i})
    root = log.root_hash()
    for idx in (0, 7, 22):
        proof = log.inclusion_proof(idx)
        check(f"Inclusion proof for record {idx} verifies against the true root",
              log.verify_inclusion(log.records[idx], idx, proof, root))


def test_merkle_inclusion_proof_rejects_wrong_record():
    log = MerkleTelemetryLog()
    for i in range(10):
        log.append({"t": i * 0.1, "value": i})
    root = log.root_hash()
    proof = log.inclusion_proof(3)
    check("An inclusion proof for record 3 does NOT verify a substituted (wrong) record",
          not log.verify_inclusion({"t": 999, "value": -1}, 3, proof, root))


def test_merkle_tamper_detection():
    log = MerkleTelemetryLog()
    for i in range(12):
        log.append({"t": i * 0.1, "altitude_m": i * 50.0})
    check("Substituting a tampered record is detected via a changed root hash",
          log.detect_tamper(5, {"t": 999.0, "altitude_m": -9999.0}))


def test_secure_channel_round_trip():
    if not _CRYPTOGRAPHY_AVAILABLE:
        print("  [SKIP] 'cryptography' package not installed.")
        return
    channel = SecureTelemetryChannel()
    packet = channel.transmit(b"altitude=1000m")
    result = channel.receive(packet)
    check("A legitimately transmitted packet is accepted and decrypts to the original plaintext",
          result["accept"] and result["plaintext"] == b"altitude=1000m")


def test_secure_channel_rejects_replay():
    if not _CRYPTOGRAPHY_AVAILABLE:
        print("  [SKIP] 'cryptography' package not installed.")
        return
    channel = SecureTelemetryChannel()
    packet = channel.transmit(b"altitude=1000m")
    channel.receive(packet)
    replay_result = channel.receive(packet)
    check("Replaying an already-received packet is rejected",
          not replay_result["accept"] and replay_result["reason"] == "replay_detected")


def test_secure_channel_rejects_tampered_ciphertext():
    if not _CRYPTOGRAPHY_AVAILABLE:
        print("  [SKIP] 'cryptography' package not installed.")
        return
    channel = SecureTelemetryChannel()
    packet = channel.transmit(b"altitude=1000m")
    tampered = dict(packet)
    tampered_bytes = bytearray(tampered["ciphertext"])
    tampered_bytes[0] ^= 0xFF
    tampered["ciphertext"] = bytes(tampered_bytes)
    result = channel.receive(tampered)
    check("A packet with tampered ciphertext fails AEAD authentication and is rejected",
          not result["accept"] and result["reason"] == "authentication_failed")


def test_secure_channel_different_keys_cannot_decrypt():
    if not _CRYPTOGRAPHY_AVAILABLE:
        print("  [SKIP] 'cryptography' package not installed.")
        return
    channel_a = SecureTelemetryChannel()
    channel_b = SecureTelemetryChannel()   # independently generated key
    packet = channel_a.transmit(b"secret telemetry")
    result = channel_b.receive(packet)
    check("A receiver with a DIFFERENT key cannot decrypt/authenticate the packet",
          not result["accept"])


def test_replay_guard_accepts_in_order_and_rejects_old():
    guard = ReplayGuard(max_reorder_window=5)
    for i in range(10):
        result = guard.check_and_record(i)
        check(f"ReplayGuard accepts fresh in-order sequence number {i}", result["accept"])
    check("ReplayGuard rejects a sequence number far outside the reorder window",
          not guard.check_and_record(0)["accept"])


if __name__ == "__main__":
    print("Running telemetry security (Merkle log / AES-GCM stream) verification suite...\n")
    test_merkle_log_root_is_deterministic()
    test_merkle_log_root_changes_on_any_record_difference()
    test_merkle_inclusion_proof_verifies_correctly()
    test_merkle_inclusion_proof_rejects_wrong_record()
    test_merkle_tamper_detection()
    test_secure_channel_round_trip()
    test_secure_channel_rejects_replay()
    test_secure_channel_rejects_tampered_ciphertext()
    test_secure_channel_different_keys_cannot_decrypt()
    test_replay_guard_accepts_in_order_and_rejects_old()

    print(f"\n{PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    if FAIL > 0:
        sys.exit(1)
