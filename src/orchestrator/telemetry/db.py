"""Routing telemetry + bake-off results — SQLite helpers.

An external runtime router may write to the SAME database from its own
process (e.g. a Node service embedding the routing table). Schema is declared
in schema.sql (alongside this file) so every writer reads one canonical
source.

Public surface
--------------
- `init_db(path=None)`                  apply schema; idempotent
- `db_path()`                           resolve the active DB path
- `log_decision(...)`                   write a routing decision row
- `log_bakeoff_result(...)`             write a bake-off cell row
- `slot_winner(slot)`                   most-recent winner for a slot (or None)

Environment variable overrides
------------------------------
`ORCHESTRATOR_DB_PATH` — point every writer (Python and any external router)
at a temp path for tests. If unset, defaults to
`~/.agent-orchestrator/telemetry.sqlite`.

Privacy note
------------
The DB persists raw prompt/response content: `routing_decisions` keeps
message excerpts and `routed_call_samples` keeps full sampled input/output
text. Treat the file accordingly. To relocate it, set ORCHESTRATOR_DB_PATH
(or pass `init_db(path=...)`) before any telemetry call.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

# Resolve at call-time, not import-time, so tests can monkey-patch the env.
DEFAULT_DB_PATH = Path.home() / ".agent-orchestrator" / "telemetry.sqlite"
SCHEMA_SQL = Path(__file__).resolve().parent / "schema.sql"


def db_path() -> Path:
    """Resolve the active DB path. Respects ORCHESTRATOR_DB_PATH env var."""
    override = os.environ.get("ORCHESTRATOR_DB_PATH")
    if override:
        return Path(override).expanduser()
    return DEFAULT_DB_PATH


def _connect() -> sqlite3.Connection:
    """Open the DB, applying schema if needed.

    Keeps the connection short-lived — fine for our write volume. WAL gives
    safer concurrent reads when an external router process also writes.
    """
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    # WAL — small benefit on a personal machine, but free.
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db(path: Path | None = None) -> Path:
    """Apply schema.sql to the DB. Returns the resolved path.

    Pass `path` to override env / default. Idempotent — safe to call repeatedly.

    NOTE: passing `path` writes it into ``os.environ["ORCHESTRATOR_DB_PATH"]``,
    and that mutation PERSISTS for the rest of the process — every subsequent
    telemetry call (from any module) resolves to that path until the variable
    is changed or removed.

    Also migrates DBs created by older schema versions in place (currently:
    adds `routing_decisions.app_name` when missing).
    """
    if path is not None:
        os.environ["ORCHESTRATOR_DB_PATH"] = str(path)
    sql = SCHEMA_SQL.read_text(encoding="utf-8")
    conn = _connect()
    try:
        # Migration guard — must run BEFORE schema.sql: app_name landed after
        # the first release, and `CREATE TABLE IF NOT EXISTS` won't touch an
        # existing table. ALTER first so schema.sql's index on app_name can
        # apply. New DBs (empty table_info) get the column from schema.sql.
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(routing_decisions)").fetchall()}
        if cols and "app_name" not in cols:
            conn.execute("ALTER TABLE routing_decisions ADD COLUMN app_name TEXT")
        conn.executescript(sql)
        conn.commit()
    finally:
        conn.close()
    return db_path()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_decision(
    *,
    session_id: str,
    message_excerpt: str,
    classified_slot: str | None,
    selected_model: str,
    app_name: str | None = None,
    fallback_used: bool = False,
    latency_ms: int | None = None,
    cost_usd: float | None = None,
    user_feedback: str | None = None,
    timestamp: str | None = None,
) -> int:
    """Insert a routing_decisions row. Returns the new row id.

    An external runtime router may write the same shape. Schema is shared.

    Pass `app_name` so per-app audits can scope to this row — rows written
    without it (NULL) are excluded from app-filtered audit queries.
    """
    init_db()  # cheap; idempotent
    conn = _connect()
    try:
        cur = conn.execute(
            """
            INSERT INTO routing_decisions
                (timestamp, session_id, app_name, message_excerpt, classified_slot,
                 selected_model, fallback_used, latency_ms, cost_usd, user_feedback)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp or _now_iso(),
                session_id,
                app_name,
                message_excerpt[:2000],  # cap excerpt size
                classified_slot,
                selected_model,
                1 if fallback_used else 0,
                latency_ms,
                cost_usd,
                user_feedback,
            ),
        )
        conn.commit()
        return int(cur.lastrowid or 0)
    finally:
        conn.close()


def log_bakeoff_result(
    *,
    slot: str,
    candidate_model: str,
    example_id: str,
    output: str,
    judge_model: str,
    rubric_scores: dict,
    mean_score: float,
    latency_ms: int | None,
    cost_usd: float | None,
    baked_at: str,
) -> int:
    """Insert a bakeoff_results row. Returns the new row id."""
    init_db()
    conn = _connect()
    try:
        cur = conn.execute(
            """
            INSERT INTO bakeoff_results
                (slot, candidate_model, example_id, output,
                 judge_model, rubric_scores, mean_score, latency_ms, cost_usd, baked_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                slot,
                candidate_model,
                example_id,
                output[:20000],  # cap output length
                judge_model,
                json.dumps(rubric_scores, sort_keys=True),
                float(mean_score),
                latency_ms,
                cost_usd,
                baked_at,
            ),
        )
        conn.commit()
        return int(cur.lastrowid or 0)
    finally:
        conn.close()


def slot_winner(slot: str) -> str | None:
    """Look up the highest-scoring model for `slot` from the most recent bake.

    "Most recent bake" = max baked_at; among rows in that bake, the model
    with highest mean of mean_score. Returns None if no bake-off has happened.

    WARNING: slots are a single GLOBAL namespace shared by every app writing
    to this DB — `bakeoff_results` carries no app_name column. If two apps
    reuse the same slot name, their bake-off rows mix and the "winner" is
    computed over the combined data. When isolation matters, use per-app slot
    names (e.g. "myapp.summarize") or point each app at its own DB via
    ORCHESTRATOR_DB_PATH.
    """
    init_db()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT MAX(baked_at) AS most_recent FROM bakeoff_results WHERE slot = ?",
            (slot,),
        ).fetchone()
        if not row or row["most_recent"] is None:
            return None
        most_recent = row["most_recent"]
        winners = conn.execute(
            """
            SELECT candidate_model, AVG(mean_score) AS score
            FROM bakeoff_results
            WHERE slot = ? AND baked_at = ?
            GROUP BY candidate_model
            ORDER BY score DESC
            LIMIT 1
            """,
            (slot, most_recent),
        ).fetchone()
        return winners["candidate_model"] if winners else None
    finally:
        conn.close()


def log_sample(
    *,
    sample_id: str,
    app_name: str,
    slot: str,
    candidate_model: str,
    input_text: str,
    output_text: str,
    latency_ms: int,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    sampled_for: str = "audit",
    routed_at: str | None = None,
) -> str:
    """Insert a routed_call_samples row. Returns the sample_id.

    An external runtime router may write this same shape to the shared DB from
    its own process — the Python signature here mirrors the SQL contract so
    fixtures + router-side writes match exactly.

    Caps `input_text` at 8000 chars and `output_text` at 20_000 chars to keep
    SQLite row size reasonable. Sampled inputs longer than that are rare;
    truncating is preferable to refusing the write.
    """
    init_db()
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO routed_call_samples
                (sample_id, routed_at, app_name, slot, candidate_model,
                 input_text, output_text, latency_ms, input_tokens,
                 output_tokens, sampled_for)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sample_id,
                routed_at or _now_iso(),
                app_name,
                slot,
                candidate_model,
                input_text[:8000],
                output_text[:20_000],
                int(latency_ms),
                input_tokens,
                output_tokens,
                sampled_for,
            ),
        )
        conn.commit()
        return sample_id
    finally:
        conn.close()


def samples_for_audit(
    *,
    app_name: str,
    slot: str | None = None,
    since_iso: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Return routed_call_samples rows for an app, optionally filtered by slot/time.

    Used by `orchestrator.audit.quality` to build the judge batch from the
    lookback window. Returns a list of dicts (one per row).
    """
    init_db()
    conn = _connect()
    try:
        sql = "SELECT * FROM routed_call_samples WHERE app_name = ?"
        params: list = [app_name]
        if slot is not None:
            sql += " AND slot = ?"
            params.append(slot)
        if since_iso is not None:
            sql += " AND routed_at >= ?"
            params.append(since_iso)
        sql += " ORDER BY routed_at ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def routing_decisions_for_audit(
    *,
    app_name: str | None,
    slots: list[str] | None,
    since_iso: str,
) -> list[dict]:
    """Return routing_decisions rows for the audit window.

    Args:
        app_name: If provided, only return rows logged for this app. Legacy
            rows written before the app_name column existed (NULL app_name)
            cannot be attributed to any app, so a filtered query EXCLUDES
            them and emits a reduced stderr warning with the excluded count.
            Pass None to include every row regardless of app.
        slots: If provided, only return rows whose classified_slot is in this list.
        since_iso: ISO timestamp string lower bound (inclusive).

    Returns:
        List of dicts (one per row).
    """
    init_db()  # also migrates legacy DBs so app_name always exists
    conn = _connect()
    try:
        slot_clause = ""
        slot_params: list = []
        if slots:
            placeholders = ",".join(["?"] * len(slots))
            slot_clause = f" AND classified_slot IN ({placeholders})"
            slot_params = list(slots)

        sql = "SELECT * FROM routing_decisions WHERE timestamp >= ?"
        params: list = [since_iso]
        if app_name is not None:
            sql += " AND app_name = ?"
            params.append(app_name)
            # Legacy NULL-app_name rows in the same window can't be scoped to
            # an app. Excluding them silently would hide a coverage gap, so
            # report how many were skipped.
            legacy = conn.execute(
                "SELECT COUNT(*) FROM routing_decisions"
                " WHERE timestamp >= ? AND app_name IS NULL" + slot_clause,
                [since_iso, *slot_params],
            ).fetchone()[0]
            if legacy:
                import sys as _sys
                _sys.stderr.write(
                    f"[telemetry] {legacy} legacy routing_decisions row(s) in this "
                    f"window have no app_name and were excluded from the "
                    f"app={app_name!r} audit.\n"
                )
        sql += slot_clause
        params.extend(slot_params)
        sql += " ORDER BY timestamp ASC"
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def graded_decisions_for(slot: str, model: str) -> list[dict]:
    """Return user-graded routing decisions for a (slot, model) pair.

    Used by the DSPy Phase-2 compile step (Deliverable 6). Filters to rows
    where user_feedback is non-NULL.
    """
    init_db()
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT timestamp, message_excerpt, user_feedback, latency_ms, cost_usd
            FROM routing_decisions
            WHERE classified_slot = ? AND selected_model = ?
              AND user_feedback IS NOT NULL
            ORDER BY timestamp ASC
            """,
            (slot, model),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
