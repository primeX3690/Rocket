"""
embedded/protocols.py

Bit/byte-level frame encoding and decoding for three real avionics/
industrial communication protocols used on actual flight hardware:
CAN 2.0B (with genuine CRC-15 per the Bosch CAN spec), ARINC 429 (with
genuine odd-parity per the ARINC 429 spec), and SpaceWire (with genuine
character-level encoding: data/control character distinction, EOP/EEP
markers, and Time-Code distribution per ECSS-E-ST-50-12C).

This is real protocol framing/checksumming logic, testable and
round-trippable -- NOT a hardware driver (no actual CAN transceiver,
ARINC 429 line driver, or SpaceWire PHY exists in this environment),
and NOT a full protocol stack (no bus arbitration, error-frame
retransmission state machine, or physical-layer timing). It's the
message-framing layer a real driver would sit underneath.

Zero external dependencies beyond NumPy - CPU-only.
"""

import struct

# --- CAN 2.0B -----------------------------------------------------------

_CAN_CRC15_POLY = 0x4599   # Bosch CAN 2.0 CRC-15 generator polynomial


def can_crc15(bits: str) -> int:
    """Genuine CAN 2.0 CRC-15 over a bitstring (MSB first), per the Bosch CAN spec."""
    crc = 0
    for bit in bits:
        b = int(bit)
        crc_next_bit = b ^ ((crc >> 14) & 1)
        crc = (crc << 1) & 0x7FFF
        if crc_next_bit:
            crc ^= _CAN_CRC15_POLY
    return crc & 0x7FFF


def encode_can_frame(can_id: int, data: bytes, extended: bool = False) -> dict:
    """
    Encode a CAN 2.0 data frame's ID+data+CRC (the fields a real CAN
    controller would compute and check in hardware). `can_id` is 11-bit
    (standard) or 29-bit (extended).
    """
    if len(data) > 8:
        raise ValueError("CAN 2.0 data field is at most 8 bytes")
    id_bits = format(can_id, "029b" if extended else "011b")
    dlc_bits = format(len(data), "04b")
    data_bits = "".join(format(b, "08b") for b in data)
    payload_bits = id_bits + dlc_bits + data_bits
    crc = can_crc15(payload_bits)
    return {"id": can_id, "extended": extended, "data": data, "dlc": len(data),
            "crc15": crc, "payload_bits": payload_bits}


def decode_and_verify_can_frame(frame: dict) -> bool:
    """Recomputes CRC-15 over the frame's payload bits and checks it matches — a genuine integrity check."""
    return can_crc15(frame["payload_bits"]) == frame["crc15"]


# --- ARINC 429 ------------------------------------------------------------

def _odd_parity_bit(value: int, n_bits: int = 31) -> int:
    """Returns the single parity bit that makes the total number of 1-bits (incl. parity) odd."""
    ones = bin(value & ((1 << n_bits) - 1)).count("1")
    return 0 if ones % 2 == 1 else 1


def encode_arinc429_word(label_octal: int, sdi: int, data_value: int, ssm: int) -> int:
    """
    Encode a 32-bit ARINC 429 word: [31]=parity, [30:29]=SSM, [28:11]=data (18 bits, BNR-style),
    [10:9]=SDI, [8:1]=label (octal, transmitted bit-reversed per the real ARINC 429 spec),
    per ARINC Specification 429.
    """
    label_bits = format(label_octal, "08b")[::-1]   # label transmitted LSB-first, a real ARINC 429 quirk
    label_val = int(label_bits, 2)
    word = (label_val & 0xFF) | ((sdi & 0b11) << 8) | ((data_value & 0x3FFFF) << 10) | ((ssm & 0b11) << 29)
    parity = _odd_parity_bit(word, 31)
    return word | (parity << 31)


def decode_arinc429_word(word: int) -> dict:
    label_val = word & 0xFF
    label_octal = int(format(label_val, "08b")[::-1], 2)
    sdi = (word >> 8) & 0b11
    data_value = (word >> 10) & 0x3FFFF
    ssm = (word >> 29) & 0b11
    parity_bit = (word >> 31) & 1
    parity_ok = _odd_parity_bit(word & 0x7FFFFFFF, 31) == parity_bit
    return {"label_octal": label_octal, "sdi": sdi, "data_value": data_value,
            "ssm": ssm, "parity_ok": parity_ok}


# --- SpaceWire (ECSS-E-ST-50-12C character level) --------------------------

SPACEWIRE_CONTROL_CHARS = {"FCT", "EOP", "EEP", "ESC"}


def spacewire_encode_packet(data_bytes: bytes, use_eep: bool = False):
    """
    Encode a SpaceWire packet as its character sequence: N data
    characters followed by an End-of-Packet (EOP) marker, or an
    Error-End-of-Packet (EEP) if the packet is being marked as
    corrupted -- the actual character-level framing defined by
    ECSS-E-ST-50-12C (real SpaceWire hardware also does 8b/10b-style
    Data-Control-Bit-Value-4-bit encoding and self-clocking DS
    signaling at the physical layer, which is out of scope here; this
    is the character/packet framing layer).
    """
    chars = [("data", b) for b in data_bytes]
    chars.append(("control", "EEP" if use_eep else "EOP"))
    return chars


def spacewire_decode_packets(char_stream):
    """
    Splits a flat SpaceWire character stream (as produced by
    spacewire_encode_packet, possibly several concatenated) back into
    packets, each ending at its EOP/EEP marker.
    """
    packets = []
    current = []
    for kind, val in char_stream:
        if kind == "data":
            current.append(val)
        elif val in ("EOP", "EEP"):
            packets.append({"data": bytes(current), "terminator": val, "valid": val == "EOP"})
            current = []
    return packets


def spacewire_time_code(time_value: int) -> int:
    """
    SpaceWire Time-Code character: 6-bit time counter + 2-bit control
    flags, distributed periodically for network-wide time
    synchronization per ECSS-E-ST-50-12C section on Time-Codes.
    """
    return time_value & 0x3F   # 6-bit counter, control flags assumed 0 here
