"""
auth.py

Password hashing and JWT session tokens.

Password hashing uses PBKDF2-HMAC-SHA256 (hashlib.pbkdf2_hmac, Python
stdlib, NIST-approved) with a per-user random salt and 310,000
iterations (OWASP's 2023 minimum recommendation for PBKDF2-SHA256).
This is a real, secure choice -- NOT a toy hash -- chosen specifically
because it needs zero external dependencies (bcrypt/argon2 need a
compiled package); if/when this moves to FastAPI+passlib in
production, passlib's bcrypt or argon2 backend is a fine (arguably
preferable) swap, but PBKDF2 at this iteration count is not a
downgrade in real security terms.

JWT session tokens use PyJWT with HS256, a signing secret, and a short
expiry -- standard practice for a stateless session token.

SECURITY NOTE for production deployment: JWT_SECRET below MUST be set
via a real environment variable (a long, random value) and never
committed to source control; the fallback value here is ONLY for local
development/testing and is deliberately labeled as such.
"""

import hashlib
import hmac
import os
import secrets
import time
import jwt

PBKDF2_ITERATIONS = 310_000
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_SECONDS = 24 * 3600

JWT_SECRET = os.environ.get("ASCENTGNC_JWT_SECRET", "dev-only-insecure-secret-CHANGE-IN-PRODUCTION")


def hash_password(password: str, salt: str = None) -> tuple:
    """Returns (hash_hex, salt_hex). Generates a new random salt if not provided."""
    if salt is None:
        salt = secrets.token_hex(16)
    salt_bytes = bytes.fromhex(salt)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, PBKDF2_ITERATIONS)
    return derived.hex(), salt


def verify_password(password: str, stored_hash: str, stored_salt: str) -> bool:
    """Constant-time comparison (hmac.compare_digest) to avoid timing side-channel attacks."""
    computed_hash, _ = hash_password(password, stored_salt)
    return hmac.compare_digest(computed_hash, stored_hash)


def issue_jwt(user_id: int, org_id: int, role: str) -> str:
    payload = {
        "sub": str(user_id),
        "org_id": org_id,
        "role": role,
        "iat": int(time.time()),
        "exp": int(time.time()) + JWT_EXPIRY_SECONDS,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_jwt(token: str) -> dict:
    """Raises jwt.InvalidTokenError (or a subclass, e.g. ExpiredSignatureError) if invalid/expired."""
    payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    return payload


def generate_api_key() -> tuple:
    """
    Returns (plaintext_key, key_hash). The plaintext key is shown to the
    user EXACTLY ONCE at creation time (standard practice, like GitHub
    personal access tokens) -- only its SHA-256 hash is stored, so a
    database leak does not expose usable API keys.
    """
    plaintext = "agnc_" + secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(plaintext.encode("utf-8")).hexdigest()
    return plaintext, key_hash


def hash_api_key(plaintext_key: str) -> str:
    return hashlib.sha256(plaintext_key.encode("utf-8")).hexdigest()
