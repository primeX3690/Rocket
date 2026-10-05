"""
telemetry/secure_log.py

Tamper-evident flight telemetry logging via a genuine Merkle hash tree
(SHA-256, Python's real `hashlib`, not a toy hash) -- the same
construction used by Certificate Transparency logs and blockchains to
make tampering with any past entry detectable without needing to
re-check every entry.

Each telemetry record is hashed; hashes are paired and re-hashed up a
binary tree to a single ROOT HASH. Publishing (or downlinking) the root
hash periodically lets a ground station verify, after the fact, that
NONE of the logged records were altered -- changing even one bit of one
historical record changes its leaf hash, which cascades up and changes
the root hash, which will no longer match any previously-published
root. This is real, standard cryptographic tamper-evidence, not a
"zero-knowledge proof" (a distinct, much more specialized cryptographic
primitive for proving a statement without revealing the underlying
data, which is not what tamper-evident LOGGING needs and is not
implemented here — that claim in an earlier product-roadmap version of
this idea overstated what a Merkle log actually provides).

Zero external dependencies beyond Python's stdlib `hashlib` - CPU-only.
"""

import hashlib
import json


def _hash(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _leaf_hash(record: dict) -> bytes:
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    return _hash(b"\x00" + canonical)   # 0x00 prefix: standard leaf/node domain separation


def _node_hash(left: bytes, right: bytes) -> bytes:
    return _hash(b"\x01" + left + right)   # 0x01 prefix: distinguishes internal nodes from leaves


class MerkleTelemetryLog:
    """
    Append-only tamper-evident telemetry log. Records are appended in
    real time during flight; `root_hash()` gives the current Merkle
    root, which a ground station can archive/publish at intervals (a
    "checkpoint") to make everything logged up to that point tamper-
    evident against later modification.
    """
    def __init__(self):
        self.records = []
        self.leaf_hashes = []

    def append(self, record: dict):
        self.records.append(record)
        self.leaf_hashes.append(_leaf_hash(record))
        return len(self.records) - 1   # index of the appended record

    def root_hash(self) -> bytes:
        if not self.leaf_hashes:
            return _hash(b"")
        layer = list(self.leaf_hashes)
        while len(layer) > 1:
            if len(layer) % 2 == 1:
                layer.append(layer[-1])   # duplicate last node for an odd layer (standard Merkle convention)
            layer = [_node_hash(layer[i], layer[i + 1]) for i in range(0, len(layer), 2)]
        return layer[0]

    def inclusion_proof(self, index: int) -> list:
        """
        Merkle inclusion (audit) proof for the record at `index`: the
        sibling hashes needed to recompute the root from that one leaf,
        WITHOUT needing every other record -- this is what makes Merkle
        proofs efficient (O(log n) instead of O(n)).
        """
        layer = list(self.leaf_hashes)
        proof = []
        idx = index
        while len(layer) > 1:
            if len(layer) % 2 == 1:
                layer.append(layer[-1])
            sibling_idx = idx + 1 if idx % 2 == 0 else idx - 1
            proof.append((layer[sibling_idx], "right" if idx % 2 == 0 else "left"))
            layer = [_node_hash(layer[i], layer[i + 1]) for i in range(0, len(layer), 2)]
            idx //= 2
        return proof

    def verify_inclusion(self, record: dict, index: int, proof: list, expected_root: bytes) -> bool:
        """Recomputes the root from ONE record + its proof and checks it matches the published/expected root."""
        current = _leaf_hash(record)
        for sibling_hash, side in proof:
            current = _node_hash(current, sibling_hash) if side == "right" else _node_hash(sibling_hash, current)
        return current == expected_root

    def detect_tamper(self, index: int, tampered_record: dict) -> bool:
        """Convenience: True if substituting `tampered_record` at `index` changes the log's root hash."""
        original_root = self.root_hash()
        original = self.records[index]
        self.records[index] = tampered_record
        self.leaf_hashes[index] = _leaf_hash(tampered_record)
        tampered_root = self.root_hash()
        self.records[index] = original   # restore
        self.leaf_hashes[index] = _leaf_hash(original)
        return tampered_root != original_root
