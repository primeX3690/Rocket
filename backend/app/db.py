"""
db.py

SQLite data layer for the AscentGNC platform MVP. Uses stdlib sqlite3
directly (no ORM) so this is fully testable in any Python environment
with zero extra dependencies. See README.md's "Scaling to production"
section for the SQLAlchemy + Postgres migration path once there are
real paying customers and concurrent-write load matters.

Schema supports multi-tenancy from day one: every user belongs to an
Organization, every API key belongs to an Organization, and every
simulation run is logged against (org, user) for usage auditing and
future usage-based billing.
"""

import sqlite3
import time
import json
import os
import contextlib

DB_PATH = os.environ.get(
    "ASCENTGNC_DB_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "data", "platform.db"),
)


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextlib.contextmanager
def _session():
    """
    Ensures every connection is closed exactly once, even if the query
    raises (e.g. a UNIQUE constraint IntegrityError) -- an earlier
    version of this module left the connection open on any exception,
    which held SQLite's write lock and caused a cascading
    'database is locked' failure on the very next call. Rolls back on
    exception, commits on success.
    """
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_connection()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS organizations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        plan TEXT NOT NULL DEFAULT 'free',
        created_at REAL NOT NULL
    );

    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        org_id INTEGER NOT NULL REFERENCES organizations(id),
        email TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        password_salt TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'admin',
        created_at REAL NOT NULL
    );

    CREATE TABLE IF NOT EXISTS api_keys (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        org_id INTEGER NOT NULL REFERENCES organizations(id),
        key_hash TEXT NOT NULL UNIQUE,
        label TEXT,
        created_at REAL NOT NULL,
        revoked INTEGER NOT NULL DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS simulation_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        org_id INTEGER NOT NULL REFERENCES organizations(id),
        user_id INTEGER REFERENCES users(id),
        sim_type TEXT NOT NULL,
        params_json TEXT NOT NULL,
        result_json TEXT NOT NULL,
        created_at REAL NOT NULL
    );
    """)
    conn.commit()
    conn.close()


def create_organization(name: str, plan: str = "free") -> int:
    with _session() as conn:
        cur = conn.execute("INSERT INTO organizations (name, plan, created_at) VALUES (?, ?, ?)",
                          (name, plan, time.time()))
        return cur.lastrowid


def create_user(org_id: int, email: str, password_hash: str, password_salt: str, role: str = "admin") -> int:
    with _session() as conn:
        cur = conn.execute(
            "INSERT INTO users (org_id, email, password_hash, password_salt, role, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (org_id, email.lower().strip(), password_hash, password_salt, role, time.time()),
        )
        return cur.lastrowid


def get_user_by_email(email: str):
    with _session() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email.lower().strip(),)).fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: int):
    with _session() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


def get_organization(org_id: int):
    with _session() as conn:
        row = conn.execute("SELECT * FROM organizations WHERE id = ?", (org_id,)).fetchone()
        return dict(row) if row else None


def create_api_key(org_id: int, key_hash: str, label: str = "") -> int:
    with _session() as conn:
        cur = conn.execute(
            "INSERT INTO api_keys (org_id, key_hash, label, created_at) VALUES (?, ?, ?, ?)",
            (org_id, key_hash, label, time.time()),
        )
        return cur.lastrowid


def get_api_key_by_hash(key_hash: str):
    with _session() as conn:
        row = conn.execute("SELECT * FROM api_keys WHERE key_hash = ? AND revoked = 0", (key_hash,)).fetchone()
        return dict(row) if row else None


def revoke_api_key(key_id: int, org_id: int) -> bool:
    with _session() as conn:
        cur = conn.execute("UPDATE api_keys SET revoked = 1 WHERE id = ? AND org_id = ?", (key_id, org_id))
        return cur.rowcount > 0


def list_api_keys(org_id: int):
    with _session() as conn:
        rows = conn.execute("SELECT id, label, created_at, revoked FROM api_keys WHERE org_id = ?",
                            (org_id,)).fetchall()
        return [dict(r) for r in rows]


def log_simulation_run(org_id: int, user_id: int, sim_type: str, params: dict, result: dict) -> int:
    with _session() as conn:
        cur = conn.execute(
            "INSERT INTO simulation_runs (org_id, user_id, sim_type, params_json, result_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (org_id, user_id, sim_type, json.dumps(params), json.dumps(result), time.time()),
        )
        return cur.lastrowid


def count_simulation_runs(org_id: int, since_timestamp: float = 0.0) -> int:
    """Used for usage-based billing / plan quota enforcement."""
    with _session() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as c FROM simulation_runs WHERE org_id = ? AND created_at >= ?",
            (org_id, since_timestamp),
        ).fetchone()
        return row["c"]


def list_recent_runs(org_id: int, limit: int = 20):
    with _session() as conn:
        rows = conn.execute(
            "SELECT id, sim_type, created_at FROM simulation_runs WHERE org_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (org_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]
