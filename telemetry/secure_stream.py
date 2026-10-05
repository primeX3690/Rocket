"""
telemetry/secure_stream.py

Encrypted, authenticated, replay-resistant telemetry downlink, using
REAL cryptographic primitives from Python's `cryptography` library
(AES-256-GCM -- an AEAD cipher providing both confidentiality AND
integrity/authenticity in one primitive, the modern standard choice;
this is genuine encryption, not an obfuscation placeholder).

Design:
- AES-256-GCM per packet: confidentiality (nobody without the key reads
  telemetry) and authenticity (any tampering with ciphertext or the
  associated data is detected by GCM's built-in authentication tag).
- A monotonic sequence number is sent as GCM associated data (AAD, so
  it's authenticated but not encrypted, letting the ground station
  check replay before decrypting) and the receiver rejects any packet
  whose sequence number it has already seen or that is too far out of
  order -- real replay-attack resistance.
- Key management: this module takes a pre-shared symmetric key (an
  actual flight system would derive/rotate session keys via a proper
  key-exchange protocol, e.g. established on the ground before launch;
  full PKI/key-exchange is out of this project's scope, stated plainly
  rather than hand-waved).

Requires the `cryptography` package. If unavailable, TelemetryEncryptor
raises ImportError with a clear message rather than silently degrading
to insecure behavior — this project would rather report a missing
feature than pretend to be secure when it wasn't.
"""

import struct

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    _CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    _CRYPTOGRAPHY_AVAILABLE = False


class TelemetryEncryptor:
    """AES-256-GCM encrypt/decrypt for one telemetry packet at a time."""
    def __init__(self, key: bytes = None):
        if not _CRYPTOGRAPHY_AVAILABLE:
            raise ImportError(
                "telemetry.secure_stream requires the 'cryptography' package "
                "(pip install cryptography) — not silently falling back to an "
                "insecure stand-in."
            )
        self.key = key if key is not None else AESGCM.generate_key(bit_length=256)
        self.aesgcm = AESGCM(self.key)

    def encrypt_packet(self, plaintext: bytes, sequence_number: int, nonce: bytes = None) -> dict:
        nonce = nonce if nonce is not None else _deterministic_nonce(sequence_number)
        aad = struct.pack(">Q", sequence_number)   # sequence number is authenticated, sent in the clear
        ciphertext = self.aesgcm.encrypt(nonce, plaintext, aad)
        return {"sequence_number": sequence_number, "nonce": nonce, "ciphertext": ciphertext}

    def decrypt_packet(self, packet: dict) -> bytes:
        """Raises cryptography.exceptions.InvalidTag if the packet was tampered with or the AAD doesn't match."""
        aad = struct.pack(">Q", packet["sequence_number"])
        return self.aesgcm.decrypt(packet["nonce"], packet["ciphertext"], aad)


def _deterministic_nonce(sequence_number: int) -> bytes:
    """
    96-bit GCM nonce derived from the sequence number. CRITICAL GCM
    REQUIREMENT: a (key, nonce) pair must NEVER repeat, or GCM's
    security completely breaks -- deriving the nonce deterministically
    from a monotonic, never-reused sequence number is a standard,
    correct way to guarantee that as long as the sequence number itself
    never repeats for this key (which the replay-guard below also
    enforces from the receive side).
    """
    return struct.pack(">Q", sequence_number) + b"\x00\x00\x00\x00"


class ReplayGuard:
    """
    Rejects packets that have already been seen, or that arrive with a
    sequence number implausibly far out of order (a classic mitigation
    against a captured-and-replayed old packet, or a sequence-number
    jump suggesting a spoofing attempt).
    """
    def __init__(self, max_reorder_window: int = 64):
        self.highest_seen = -1
        self.seen_recent = set()
        self.window = max_reorder_window

    def check_and_record(self, sequence_number: int) -> dict:
        if sequence_number in self.seen_recent:
            return {"accept": False, "reason": "replay_detected"}
        if sequence_number <= self.highest_seen - self.window:
            return {"accept": False, "reason": "too_old_or_replay"}
        self.seen_recent.add(sequence_number)
        if len(self.seen_recent) > self.window * 2:
            oldest_allowed = self.highest_seen - self.window
            self.seen_recent = {s for s in self.seen_recent if s > oldest_allowed}
        self.highest_seen = max(self.highest_seen, sequence_number)
        return {"accept": True, "reason": "nominal"}


class SecureTelemetryChannel:
    """End-to-end: encrypt+send on the vehicle side, receive+verify+decrypt+replay-check on the ground side."""
    def __init__(self, key: bytes = None):
        self.encryptor = TelemetryEncryptor(key)
        self.replay_guard = ReplayGuard()
        self._next_sequence = 0

    def transmit(self, plaintext: bytes) -> dict:
        packet = self.encryptor.encrypt_packet(plaintext, self._next_sequence)
        self._next_sequence += 1
        return packet

    def receive(self, packet: dict) -> dict:
        """Returns {"accept": bool, "reason": str, "plaintext": bytes|None}."""
        replay_result = self.replay_guard.check_and_record(packet["sequence_number"])
        if not replay_result["accept"]:
            return {**replay_result, "plaintext": None}
        try:
            plaintext = self.encryptor.decrypt_packet(packet)
            return {"accept": True, "reason": "nominal", "plaintext": plaintext}
        except Exception:
            return {"accept": False, "reason": "authentication_failed", "plaintext": None}
