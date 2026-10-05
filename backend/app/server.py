"""
server.py

Minimal HTTP API server built on Python's stdlib http.server -- chosen
specifically so this entire platform MVP runs and is testable with
ZERO external pip packages beyond PyJWT (for session tokens) and the
simulation engine's own numpy/scipy. This is a deliberate choice for a
pre-revenue MVP: it has no install friction, and a stdlib-only
HTTP server comfortably handles a handful of pilot customers.

SCALING PATH (see README.md "Scaling to production" for detail): once
there is real concurrent load, swap this file for FastAPI + Uvicorn
(ASGI, async, battle-tested) -- auth.py, db.py, and engine_bridge.py
underneath do NOT need to change; only this routing layer does.

Endpoints:
  POST /api/auth/signup          {org_name, email, password} -> {token}
  POST /api/auth/login           {email, password}           -> {token}
  GET  /api/auth/me              (Bearer token)               -> {user, org}
  POST /api/apikeys              (Bearer token) {label}       -> {plaintext_key}  (shown once)
  GET  /api/apikeys              (Bearer token)               -> [key metadata]
  DELETE /api/apikeys/<id>       (Bearer token)               -> {revoked: true}
  POST /api/billing/upgrade      (Bearer token) {plan}        -> see billing.py (Stripe/Razorpay skeleton)
  GET  /api/billing/usage        (Bearer token)               -> {run_count, plan, quota}
  GET  /api/status                                             -> {engine ready?}
  POST /api/simulate/reference-mission   (Bearer token OR API key) -> real simulation result
  POST /api/simulate/peg-insertion       (Bearer token OR API key) -> real simulation result
  POST /api/simulate/ascent-profile      (Bearer token OR API key) -> real simulation result
  POST /api/simulate/anomaly-comparison  (Bearer token OR API key) -> real simulation result
  POST /api/simulate/monte-carlo         (Bearer token OR API key) -> real simulation result

Auth accepted two ways, matching real SaaS convention:
  - "Authorization: Bearer <jwt>"      -- interactive users (web frontend)
  - "Authorization: ApiKey <key>"      -- programmatic/CI integration
"""

import json
import sys
import os
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db
import auth
import engine_bridge as eb
import billing

PLAN_QUOTAS = {"free": 50, "pro": 2000, "enterprise": float("inf")}

_PLATFORM_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FRONTEND_DIR = os.path.join(_PLATFORM_ROOT, "frontend")


class ApiError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message


def _authenticate(handler) -> dict:
    """Returns {'user_id', 'org_id', 'role'} or raises ApiError(401, ...)."""
    header = handler.headers.get("Authorization", "")
    if header.startswith("Bearer "):
        token = header[len("Bearer "):]
        try:
            payload = auth.verify_jwt(token)
        except Exception:
            raise ApiError(401, "Invalid or expired session token")
        return {"user_id": int(payload["sub"]), "org_id": payload["org_id"], "role": payload["role"]}
    if header.startswith("ApiKey "):
        key = header[len("ApiKey "):]
        key_hash = auth.hash_api_key(key)
        row = db.get_api_key_by_hash(key_hash)
        if row is None:
            raise ApiError(401, "Invalid or revoked API key")
        return {"user_id": None, "org_id": row["org_id"], "role": "api"}
    raise ApiError(401, "Missing Authorization header (expected 'Bearer <token>' or 'ApiKey <key>')")


def _enforce_quota(org_id: int):
    org = db.get_organization(org_id)
    quota = PLAN_QUOTAS.get(org["plan"], PLAN_QUOTAS["free"])
    if quota == float("inf"):
        return
    used = db.count_simulation_runs(org_id, since_timestamp=_month_start_timestamp())
    if used >= quota:
        raise ApiError(402, f"Monthly simulation quota ({quota}) reached for the '{org['plan']}' plan. "
                            f"Upgrade your plan to continue.")


def _month_start_timestamp() -> float:
    import time
    now = time.gmtime()
    return time.mktime((now.tm_year, now.tm_mon, 1, 0, 0, 0, 0, 0, 0))


def _run_logged_simulation(handler, auth_ctx, sim_type: str, params: dict, fn):
    _enforce_quota(auth_ctx["org_id"])
    result = fn(**params)
    db.log_simulation_run(auth_ctx["org_id"], auth_ctx["user_id"], sim_type, params, result)
    return result


ROUTES = {}


def route(method, path):
    def decorator(fn):
        ROUTES[(method, path)] = fn
        return fn
    return decorator


@route("POST", "/api/auth/signup")
def handle_signup(handler, body):
    org_name = body.get("org_name", "").strip()
    email = body.get("email", "").strip()
    password = body.get("password", "")
    if not org_name or not email or len(password) < 8:
        raise ApiError(400, "org_name, email, and a password of at least 8 characters are required")
    if db.get_user_by_email(email) is not None:
        raise ApiError(409, "An account with this email already exists")

    org_id = db.create_organization(org_name, plan="free")
    password_hash, salt = auth.hash_password(password)
    user_id = db.create_user(org_id, email, password_hash, salt, role="admin")
    token = auth.issue_jwt(user_id, org_id, "admin")
    return 201, {"token": token, "org_id": org_id, "user_id": user_id}


@route("POST", "/api/auth/login")
def handle_login(handler, body):
    email = body.get("email", "").strip()
    password = body.get("password", "")
    user = db.get_user_by_email(email)
    if user is None or not auth.verify_password(password, user["password_hash"], user["password_salt"]):
        raise ApiError(401, "Invalid email or password")
    token = auth.issue_jwt(user["id"], user["org_id"], user["role"])
    return 200, {"token": token, "org_id": user["org_id"], "user_id": user["id"]}


@route("GET", "/api/auth/me")
def handle_me(handler, body):
    ctx = _authenticate(handler)
    user = db.get_user_by_id(ctx["user_id"]) if ctx["user_id"] else None
    org = db.get_organization(ctx["org_id"])
    return 200, {
        "user": {"id": user["id"], "email": user["email"], "role": user["role"]} if user else None,
        "organization": {"id": org["id"], "name": org["name"], "plan": org["plan"]},
    }


@route("POST", "/api/apikeys")
def handle_create_apikey(handler, body):
    ctx = _authenticate(handler)
    if ctx["role"] != "admin":
        raise ApiError(403, "Only an org admin can create API keys")
    label = body.get("label", "")
    plaintext, key_hash = auth.generate_api_key()
    key_id = db.create_api_key(ctx["org_id"], key_hash, label)
    return 201, {"id": key_id, "api_key": plaintext,
                "warning": "This key is shown once and cannot be retrieved again. Store it securely."}


@route("GET", "/api/apikeys")
def handle_list_apikeys(handler, body):
    ctx = _authenticate(handler)
    return 200, {"keys": db.list_api_keys(ctx["org_id"])}


@route("DELETE", "/api/apikeys/{id}")
def handle_revoke_apikey(handler, body, key_id):
    ctx = _authenticate(handler)
    if ctx["role"] != "admin":
        raise ApiError(403, "Only an org admin can revoke API keys")
    changed = db.revoke_api_key(int(key_id), ctx["org_id"])
    if not changed:
        raise ApiError(404, "API key not found in your organization")
    return 200, {"revoked": True}


@route("GET", "/api/billing/usage")
def handle_billing_usage(handler, body):
    ctx = _authenticate(handler)
    org = db.get_organization(ctx["org_id"])
    used = db.count_simulation_runs(ctx["org_id"], since_timestamp=_month_start_timestamp())
    quota = PLAN_QUOTAS.get(org["plan"], PLAN_QUOTAS["free"])
    return 200, {"plan": org["plan"], "simulations_this_month": used,
                "quota": None if quota == float("inf") else quota}


@route("POST", "/api/billing/upgrade")
def handle_billing_upgrade(handler, body):
    ctx = _authenticate(handler)
    if ctx["role"] != "admin":
        raise ApiError(403, "Only an org admin can change the billing plan")
    return 200, billing.create_checkout_session(ctx["org_id"], body.get("plan", "pro"))


@route("POST", "/api/billing/webhook")
def handle_billing_webhook(handler, body):
    return 200, billing.handle_webhook_event(body)


@route("GET", "/api/status")
def handle_status(handler, body):
    return 200, eb.engine_status()


@route("POST", "/api/simulate/reference-mission")
def handle_sim_reference_mission(handler, body):
    ctx = _authenticate(handler)
    result = _run_logged_simulation(handler, ctx, "reference_mission", {}, eb.run_reference_mission_demo)
    return 200, result


@route("POST", "/api/simulate/peg-insertion")
def handle_sim_peg(handler, body):
    ctx = _authenticate(handler)
    params = {
        "r0_alt_km": float(body.get("r0_alt_km", 180.0)), "v0_m_s": float(body.get("v0_m_s", 7300.0)),
        "gamma0_deg": float(body.get("gamma0_deg", 3.0)), "m0_kg": float(body.get("m0_kg", 4000.0)),
        "thrust_n": float(body.get("thrust_n", 90000.0)), "isp_s": float(body.get("isp_s", 320.0)),
        "target_alt_km": float(body.get("target_alt_km", 300.0)),
    }
    result = _run_logged_simulation(handler, ctx, "peg_insertion", params, eb.run_peg_insertion)
    return 200, result


@route("POST", "/api/simulate/ascent-profile")
def handle_sim_ascent(handler, body):
    ctx = _authenticate(handler)
    params = {
        "m0_kg": float(body.get("m0_kg", 60000.0)), "m_dry_kg": float(body.get("m_dry_kg", 12000.0)),
        "thrust_n": float(body.get("thrust_n", 850000.0)), "isp_s": float(body.get("isp_s", 285.0)),
        "drag_coeff": float(body.get("drag_coeff", 0.3)), "ref_area_m2": float(body.get("ref_area_m2", 2.0)),
        "kick_altitude_m": float(body.get("kick_altitude_m", 1000.0)),
        "kick_angle_deg": float(body.get("kick_angle_deg", 6.0)),
    }
    result = _run_logged_simulation(handler, ctx, "ascent_profile", params, eb.run_ascent_profile)
    return 200, result


@route("POST", "/api/simulate/anomaly-comparison")
def handle_sim_anomaly(handler, body):
    ctx = _authenticate(handler)
    params = {
        "r0_alt_km": float(body.get("r0_alt_km", 300.0)), "v0_m_s": float(body.get("v0_m_s", 7550.0)),
        "gamma0_deg": float(body.get("gamma0_deg", 0.3)), "m0_kg": float(body.get("m0_kg", 4000.0)),
        "thrust_n": float(body.get("thrust_n", 90000.0)), "isp_s": float(body.get("isp_s", 320.0)),
        "target_alt_km": float(body.get("target_alt_km", 356.2)),
        "anomaly_start_s": float(body.get("anomaly_start_s", 5.0)),
    }
    result = _run_logged_simulation(handler, ctx, "anomaly_comparison", params, eb.run_anomaly_comparison)
    return 200, result


@route("POST", "/api/simulate/monte-carlo")
def handle_sim_monte_carlo(handler, body):
    ctx = _authenticate(handler)
    params = {
        "r0_alt_km": float(body.get("r0_alt_km", 180.0)), "v0_m_s": float(body.get("v0_m_s", 7300.0)),
        "gamma0_deg": float(body.get("gamma0_deg", 3.0)), "m0_kg": float(body.get("m0_kg", 4000.0)),
        "thrust_n": float(body.get("thrust_n", 90000.0)), "isp_s": float(body.get("isp_s", 320.0)),
        "target_alt_km": float(body.get("target_alt_km", 300.0)),
        "n_trials": int(body.get("n_trials", 50)),
    }
    result = _run_logged_simulation(handler, ctx, "monte_carlo", params, eb.run_monte_carlo_dispersion)
    return 200, result


def _match_route(method, path):
    """Supports simple {param} path segments (used by the apikey-delete route)."""
    if (method, path) in ROUTES:
        return ROUTES[(method, path)], ()
    for (m, pattern), fn in ROUTES.items():
        if m != method or "{" not in pattern:
            continue
        pattern_parts = pattern.strip("/").split("/")
        path_parts = path.strip("/").split("/")
        if len(pattern_parts) != len(path_parts):
            continue
        params = []
        matched = True
        for pp, rp in zip(pattern_parts, path_parts):
            if pp.startswith("{") and pp.endswith("}"):
                params.append(rp)
            elif pp != rp:
                matched = False
                break
        if matched:
            return fn, tuple(params)
    return None, ()


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status_code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            raise ApiError(400, "Request body is not valid JSON")

    def _serve_static(self, url_path):
        """
        Serves frontend/index.html (and any other file under frontend/)
        for GET requests that aren't an API route — lets the whole
        platform run from a single `python3 server.py` command instead
        of needing a separate static-file server for the frontend.
        """
        rel_path = url_path.lstrip("/") or "index.html"
        full_path = os.path.normpath(os.path.join(FRONTEND_DIR, rel_path))
        if not full_path.startswith(FRONTEND_DIR) or not os.path.isfile(full_path):
            full_path = os.path.join(FRONTEND_DIR, "index.html")
            if not os.path.isfile(full_path):
                self._send_json(404, {"error": "Frontend not found"})
                return
        content_type, _ = mimetypes.guess_type(full_path)
        with open(full_path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _dispatch(self, method):
        parsed = urlparse(self.path)
        fn, params = _match_route(method, parsed.path)
        if fn is None:
            if method == "GET" and not parsed.path.startswith("/api/"):
                self._serve_static(parsed.path)
                return
            self._send_json(404, {"error": f"No such route: {method} {parsed.path}"})
            return
        try:
            body = self._read_body() if method in ("POST", "PUT") else {}
            status_code, payload = fn(self, body, *params)
            self._send_json(status_code, payload)
        except ApiError as e:
            self._send_json(e.status_code, {"error": e.message})
        except Exception as e:
            self._send_json(500, {"error": f"Internal server error: {e}"})

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_DELETE(self):
        self._dispatch("DELETE")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.end_headers()

    def log_message(self, format, *args):
        pass   # quiet by default; swap for real logging in production


def main():
    db.init_db()
    port = int(os.environ.get("PORT", 8000))
    httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"AscentGNC platform API listening on http://0.0.0.0:{port}")
    print(f"Engine status: {eb.engine_status()}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
