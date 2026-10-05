"""
tests/test_platform.py

Full verification suite for the AscentGNC platform MVP: auth.py's
cryptographic primitives, db.py's data layer, and a real end-to-end
HTTP integration test (starts server.py as an actual subprocess,
listening on a real port, and drives it with stdlib urllib — no mocks,
no test client shortcuts).

Run: python3 tests/test_platform.py
Requires: PyJWT (pip install pyjwt), and the AscentGNC simulation
engine reachable via ASCENTGNC_ENGINE_PATH (defaults to a sibling
"Rocket-main" directory — see engine_bridge.py).
"""

import sys
import os
import time
import json
import subprocess
import urllib.request
import urllib.error
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend", "app"))
import auth

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


# --- auth.py unit tests ---

def test_password_hash_verify_round_trip():
    h, s = auth.hash_password("correct horse battery staple")
    check("Correct password verifies successfully", auth.verify_password("correct horse battery staple", h, s))
    check("Wrong password is rejected", not auth.verify_password("wrong password", h, s))


def test_password_hash_uses_unique_salt_per_call():
    h1, s1 = auth.hash_password("samepassword")
    h2, s2 = auth.hash_password("samepassword")
    check("Two hashes of the same password use different salts", s1 != s2)
    check("Two hashes of the same password produce different hash output (due to salt)", h1 != h2)


def test_jwt_round_trip_and_tamper_detection():
    token = auth.issue_jwt(user_id=42, org_id=7, role="admin")
    payload = auth.verify_jwt(token)
    check("JWT payload round-trips user_id correctly", payload["sub"] == "42")
    check("JWT payload round-trips org_id correctly", payload["org_id"] == 7)
    check("JWT payload round-trips role correctly", payload["role"] == "admin")
    try:
        auth.verify_jwt(token[:-4] + "xxxx")
        check("Tampered JWT signature is rejected", False)
    except Exception:
        check("Tampered JWT signature is rejected", True)


def test_api_key_generation_and_hashing():
    plaintext, key_hash = auth.generate_api_key()
    check("Generated API key has the expected 'agnc_' prefix", plaintext.startswith("agnc_"))
    check("Hashing the same plaintext key reproduces the same hash", auth.hash_api_key(plaintext) == key_hash)
    plaintext2, _ = auth.generate_api_key()
    check("Two generated API keys are different", plaintext != plaintext2)


# --- db.py integration tests (real SQLite, temp file) ---

def test_db_user_and_org_lifecycle():
    import db
    db.DB_PATH = tempfile.mktemp(suffix=".db")
    db.init_db()

    org_id = db.create_organization("Acme Rockets", plan="free")
    h, s = auth.hash_password("testpass123")
    user_id = db.create_user(org_id, "acme@example.com", h, s, role="admin")

    fetched = db.get_user_by_email("acme@example.com")
    check("Created user can be fetched by email", fetched is not None and fetched["id"] == user_id)
    check("Fetched user belongs to the correct organization", fetched["org_id"] == org_id)

    try:
        db.create_user(org_id, "acme@example.com", h, s)
        check("Duplicate email is rejected by a UNIQUE constraint", False)
    except Exception:
        check("Duplicate email is rejected by a UNIQUE constraint", True)

    # This specific check guards against a real bug found during
    # development: an earlier version left the SQLite connection open
    # after the IntegrityError above, causing every SUBSEQUENT call to
    # fail with "database is locked".
    try:
        db.create_organization("Second Org", plan="free")
        check("Database is NOT locked after a previous call raised an exception", True)
    except Exception as e:
        check(f"Database is NOT locked after a previous call raised an exception ({e})", False)


def test_db_api_key_lifecycle():
    import db
    db.DB_PATH = tempfile.mktemp(suffix=".db")
    db.init_db()
    org_id = db.create_organization("Key Test Org")

    plaintext, key_hash = auth.generate_api_key()
    key_id = db.create_api_key(org_id, key_hash, label="test key")

    looked_up = db.get_api_key_by_hash(key_hash)
    check("A freshly-created API key is found by its hash", looked_up is not None)
    check("Looked-up key belongs to the correct organization", looked_up["org_id"] == org_id)

    revoked = db.revoke_api_key(key_id, org_id)
    check("Revoking an existing key succeeds", revoked)
    check("A revoked key is no longer found by get_api_key_by_hash", db.get_api_key_by_hash(key_hash) is None)


def test_db_simulation_run_logging_and_quota_counting():
    import db
    db.DB_PATH = tempfile.mktemp(suffix=".db")
    db.init_db()
    org_id = db.create_organization("Usage Test Org")
    user_id = db.create_user(org_id, "usage@test.com", *auth.hash_password("pw12345678"))

    for i in range(5):
        db.log_simulation_run(org_id, user_id, "peg_insertion", {"trial": i}, {"ok": True})

    check("count_simulation_runs returns the exact number of logged runs", db.count_simulation_runs(org_id) == 5)
    check("list_recent_runs returns the logged runs", len(db.list_recent_runs(org_id, limit=10)) == 5)


# --- Full HTTP integration test (real subprocess, real port, real requests) ---

def _http(method, url, token=None, api_key=None, body=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    if api_key:
        headers["Authorization"] = "ApiKey " + api_key
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_full_http_integration():
    engine_path = os.environ.get(
        "ASCENTGNC_ENGINE_PATH",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "Rocket-main"),
    )
    engine_path = os.path.abspath(engine_path)
    if not os.path.isdir(os.path.join(engine_path, "guidance")):
        print(f"  [SKIP] Simulation engine not found at {engine_path} — skipping HTTP integration test. "
              f"Set ASCENTGNC_ENGINE_PATH to run this test.")
        return

    port = 8321
    db_path = tempfile.mktemp(suffix=".db")
    env = dict(os.environ)
    env["PORT"] = str(port)
    env["ASCENTGNC_ENGINE_PATH"] = engine_path
    env["ASCENTGNC_DB_PATH"] = db_path   # isolates this test run from any existing data/platform.db
    server_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend", "app", "server.py")
    proc = subprocess.Popen([sys.executable, server_script], env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        for _ in range(30):
            try:
                status, _ = _http("GET", f"http://localhost:{port}/api/status")
                if status == 200:
                    break
            except Exception:
                time.sleep(0.2)
        else:
            check("Server started and responded to /api/status within 6 seconds", False)
            return
        check("Server started and responded to /api/status within 6 seconds", True)

        base = f"http://localhost:{port}"
        status, data = _http("POST", f"{base}/api/auth/signup",
                             body={"org_name": "Integration Test Co", "email": "it@test.com",
                                  "password": "testpassword123"})
        check("Signup returns 201 with a token", status == 201 and "token" in data)
        token = data["token"]

        status, _ = _http("POST", f"{base}/api/auth/signup",
                          body={"org_name": "Dup", "email": "it@test.com", "password": "testpassword123"})
        check("Duplicate signup is rejected with 409", status == 409)

        status, data = _http("POST", f"{base}/api/auth/login",
                             body={"email": "it@test.com", "password": "wrongpassword"})
        check("Login with wrong password returns 401", status == 401)

        status, data = _http("GET", f"{base}/api/auth/me", token=token)
        check("Authenticated /me returns the correct email", status == 200 and data["user"]["email"] == "it@test.com")

        status, _ = _http("GET", f"{base}/api/auth/me")
        check("Unauthenticated /me is rejected with 401", status == 401)

        status, data = _http("POST", f"{base}/api/simulate/reference-mission", token=token, body={})
        check("Reference-mission simulation returns 200 with real physics results",
              status == 200 and abs(data["final_altitude_km"] - 300.0) < 1.0)

        status, data = _http("POST", f"{base}/api/simulate/peg-insertion", token=token,
                             body={"r0_alt_km": 180, "v0_m_s": 7300, "gamma0_deg": 3.0, "m0_kg": 4000,
                                  "thrust_n": 90000, "isp_s": 320, "target_alt_km": 300})
        check("PEG insertion simulation converges with tiny insertion error",
              status == 200 and abs(data["insertion_error"]["radius_error_m"]) < 100.0)

        status, data = _http("POST", f"{base}/api/apikeys", token=token, body={"label": "integration test"})
        check("API key creation returns 201 with a plaintext key", status == 201 and data["api_key"].startswith("agnc_"))
        api_key = data["api_key"]
        key_id = data["id"]

        status, data = _http("POST", f"{base}/api/simulate/ascent-profile", api_key=api_key, body={})
        check("A simulation call authenticated via API key (not JWT) succeeds", status == 200)

        status, _ = _http("DELETE", f"{base}/api/apikeys/{key_id}", token=token)
        check("API key revocation succeeds", status == 200)

        status, _ = _http("POST", f"{base}/api/simulate/ascent-profile", api_key=api_key, body={})
        check("A revoked API key is rejected with 401", status == 401)

        status, data = _http("GET", f"{base}/api/billing/usage", token=token)
        check("Billing usage endpoint reports the correct simulation count",
              status == 200 and data["simulations_this_month"] == 3)   # reference + peg + ascent(apikey)

    finally:
        proc.terminate()
        proc.wait(timeout=5)


if __name__ == "__main__":
    print("Running AscentGNC platform verification suite...\n")
    test_password_hash_verify_round_trip()
    test_password_hash_uses_unique_salt_per_call()
    test_jwt_round_trip_and_tamper_detection()
    test_api_key_generation_and_hashing()
    test_db_user_and_org_lifecycle()
    test_db_api_key_lifecycle()
    test_db_simulation_run_logging_and_quota_counting()
    test_full_http_integration()

    print(f"\n{PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    if FAIL > 0:
        sys.exit(1)
