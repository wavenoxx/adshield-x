"""
AdShield-X :: app/db.py
-----------------------
SQLite persistence for the console: accounts, scan jobs, per-click verdicts and
an append-only audit log.

Everything an invalid-traffic dispute needs has to survive the request that
produced it, so verdicts are written with the reason codes that justified them
and the threshold that was in force at the time. A verdict without its reasons
is not evidence.
"""

from __future__ import annotations
import json, os, sqlite3, secrets
from datetime import datetime, timezone
from flask import g

DB_PATH = os.environ.get(
    "ADSHIELD_DB",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "outputs", "adshield.db"))

SCHEMA = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS users (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  username      TEXT    NOT NULL UNIQUE,
  password_hash TEXT    NOT NULL,
  api_key       TEXT    NOT NULL UNIQUE,
  role          TEXT    NOT NULL DEFAULT 'analyst',
  created_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS scans (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id      INTEGER NOT NULL REFERENCES users(id),
  source       TEXT    NOT NULL,            -- upload | sample | manual | api
  label        TEXT,                        -- filename or free-text note
  n_clicks     INTEGER NOT NULL,
  n_blocked    INTEGER NOT NULL,
  n_escalated  INTEGER NOT NULL,
  budget_held  REAL    NOT NULL,
  threshold    REAL    NOT NULL,
  cpc          REAL    NOT NULL,
  n_correct    INTEGER,                     -- null when the log has no labels
  created_at   TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS verdicts (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  scan_id      INTEGER NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
  row_no       INTEGER NOT NULL,
  probability  REAL    NOT NULL,
  supervised   REAL    NOT NULL,
  escalated    INTEGER NOT NULL,
  blocked      INTEGER NOT NULL,
  truth        TEXT,
  reasons      TEXT    NOT NULL             -- JSON array of reason codes
);
CREATE INDEX IF NOT EXISTS ix_verdicts_scan ON verdicts(scan_id);

CREATE TABLE IF NOT EXISTS audit (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id    INTEGER,
  action     TEXT NOT NULL,
  detail     TEXT,
  created_at TEXT NOT NULL
);
"""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_db():
    if "db" not in g:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


def close_db(_=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db(app):
    """Create the schema and seed the demo account if the table is empty."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    n = con.execute("SELECT COUNT(*) c FROM users").fetchone()["c"]
    if n == 0:
        from werkzeug.security import generate_password_hash
        con.execute(
            "INSERT INTO users (username, password_hash, api_key, role, created_at)"
            " VALUES (?,?,?,?,?)",
            ("admin", generate_password_hash("admin"),
             "ak_" + secrets.token_hex(16), "admin", now()))
        app.logger.info("seeded demo account admin/admin")
    con.commit()
    con.close()
    app.teardown_appcontext(close_db)


# ---------------------------------------------------------------------------
def create_user(username, password, role="analyst"):
    from werkzeug.security import generate_password_hash
    db = get_db()
    key = "ak_" + secrets.token_hex(16)
    cur = db.execute(
        "INSERT INTO users (username, password_hash, api_key, role, created_at)"
        " VALUES (?,?,?,?,?)",
        (username, generate_password_hash(password), key, role, now()))
    db.commit()
    return cur.lastrowid


def find_user(username):
    return get_db().execute("SELECT * FROM users WHERE username = ?",
                            (username,)).fetchone()


def user_by_id(uid):
    return get_db().execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()


def user_by_key(key):
    return get_db().execute("SELECT * FROM users WHERE api_key = ?",
                            (key,)).fetchone()


def save_scan(user_id, source, label, summary, records):
    db = get_db()
    cur = db.execute(
        "INSERT INTO scans (user_id, source, label, n_clicks, n_blocked,"
        " n_escalated, budget_held, threshold, cpc, n_correct, created_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (user_id, source, label, summary["scanned"], summary["blocked"],
         summary["escalated"], summary["budget_held"], summary["threshold"],
         summary["cpc"], summary.get("correct"), now()))
    sid = cur.lastrowid
    db.executemany(
        "INSERT INTO verdicts (scan_id, row_no, probability, supervised,"
        " escalated, blocked, truth, reasons) VALUES (?,?,?,?,?,?,?,?)",
        [(sid, r["row"], r["probability"], r["supervised"],
          int(r["escalated"]), int(r["blocked"]), r.get("truth"),
          json.dumps(r["reasons"])) for r in records])
    db.commit()
    return sid


def list_scans(user_id, limit=40):
    return get_db().execute(
        "SELECT * FROM scans WHERE user_id = ? ORDER BY id DESC LIMIT ?",
        (user_id, limit)).fetchall()


def get_scan(scan_id, user_id):
    return get_db().execute(
        "SELECT * FROM scans WHERE id = ? AND user_id = ?",
        (scan_id, user_id)).fetchone()


def get_verdicts(scan_id):
    rows = get_db().execute(
        "SELECT * FROM verdicts WHERE scan_id = ? ORDER BY row_no", (scan_id,)
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["reasons"] = json.loads(d["reasons"])
        out.append(d)
    return out


def totals(user_id):
    r = get_db().execute(
        "SELECT COUNT(*) scans, COALESCE(SUM(n_clicks),0) clicks,"
        " COALESCE(SUM(n_blocked),0) blocked,"
        " COALESCE(SUM(n_escalated),0) escalated,"
        " COALESCE(SUM(budget_held),0) budget FROM scans WHERE user_id = ?",
        (user_id,)).fetchone()
    return dict(r)


def audit(user_id, action, detail=""):
    db = get_db()
    db.execute("INSERT INTO audit (user_id, action, detail, created_at)"
               " VALUES (?,?,?,?)", (user_id, action, detail, now()))
    db.commit()


def recent_audit(limit=25):
    return get_db().execute(
        "SELECT a.*, u.username FROM audit a LEFT JOIN users u ON u.id=a.user_id"
        " ORDER BY a.id DESC LIMIT ?", (limit,)).fetchall()
