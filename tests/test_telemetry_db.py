"""Tests for the SQLite telemetry helpers.

Each test uses a per-function tmp_path DB via the ORCHESTRATOR_DB_PATH env var
so the user's real ~/.agent-orchestrator DB is never touched.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from orchestrator.telemetry import db as tdb


@pytest.fixture
def tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Yield a fresh DB path for each test."""
    p = tmp_path / "test-telemetry.sqlite"
    monkeypatch.setenv("ORCHESTRATOR_DB_PATH", str(p))
    tdb.init_db()
    return p


def test_init_db_creates_tables(tmp_db: Path):
    conn = sqlite3.connect(tmp_db)
    try:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        ]
    finally:
        conn.close()
    assert "routing_decisions" in tables
    assert "bakeoff_results" in tables


def test_log_decision_inserts_row(tmp_db: Path):
    row_id = tdb.log_decision(
        session_id="sess-1",
        message_excerpt="hello world",
        classified_slot="entity_extraction",
        selected_model="anthropic/claude-haiku-4-5",
        fallback_used=False,
        latency_ms=120,
        cost_usd=0.0009,
    )
    assert row_id >= 1

    conn = sqlite3.connect(tmp_db)
    try:
        row = conn.execute(
            "SELECT session_id, message_excerpt, classified_slot, selected_model, "
            "fallback_used, latency_ms, cost_usd FROM routing_decisions WHERE id = ?",
            (row_id,),
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert row[0] == "sess-1"
    assert row[1] == "hello world"
    assert row[2] == "entity_extraction"
    assert row[3] == "anthropic/claude-haiku-4-5"
    assert row[4] == 0  # fallback_used = False → 0
    assert row[5] == 120
    assert row[6] == pytest.approx(0.0009)


def test_log_bakeoff_result_serializes_rubric_as_json(tmp_db: Path):
    row_id = tdb.log_bakeoff_result(
        slot="entity_extraction",
        candidate_model="anthropic/claude-haiku-4-5",
        example_id="ex-00",
        output="Acme Corp, Widget Inc",
        judge_model="anthropic/claude-opus-4-7",
        rubric_scores={"recall": 5, "precision": 4, "format_compliance": 5},
        mean_score=4.667,
        latency_ms=850,
        cost_usd=0.0012,
        baked_at="2026-05-25T22:00:00+00:00",
    )
    assert row_id >= 1

    conn = sqlite3.connect(tmp_db)
    try:
        row = conn.execute(
            "SELECT slot, rubric_scores, mean_score FROM bakeoff_results WHERE id = ?",
            (row_id,),
        ).fetchone()
    finally:
        conn.close()
    import json
    assert row is not None
    assert row[0] == "entity_extraction"
    parsed = json.loads(row[1])
    assert parsed == {"format_compliance": 5, "precision": 4, "recall": 5}
    assert row[2] == pytest.approx(4.667)


def test_slot_winner_returns_highest_mean(tmp_db: Path):
    # Two bake-offs at different baked_at; only the latest one is considered.
    tdb.log_bakeoff_result(
        slot="entity_extraction",
        candidate_model="loser-model",
        example_id="ex-00",
        output="...",
        judge_model="anthropic/claude-opus-4-7",
        rubric_scores={"r": 5},
        mean_score=5.0,
        latency_ms=100,
        cost_usd=0.0,
        baked_at="2026-01-01T00:00:00+00:00",  # stale bake
    )
    tdb.log_bakeoff_result(
        slot="entity_extraction",
        candidate_model="winner-model",
        example_id="ex-00",
        output="...",
        judge_model="anthropic/claude-opus-4-7",
        rubric_scores={"r": 4.2},
        mean_score=4.2,
        latency_ms=100,
        cost_usd=0.0,
        baked_at="2026-05-25T00:00:00+00:00",  # current bake
    )
    tdb.log_bakeoff_result(
        slot="entity_extraction",
        candidate_model="winner-model",
        example_id="ex-01",
        output="...",
        judge_model="anthropic/claude-opus-4-7",
        rubric_scores={"r": 4.5},
        mean_score=4.5,
        latency_ms=100,
        cost_usd=0.0,
        baked_at="2026-05-25T00:00:00+00:00",
    )
    tdb.log_bakeoff_result(
        slot="entity_extraction",
        candidate_model="runner-up",
        example_id="ex-00",
        output="...",
        judge_model="anthropic/claude-opus-4-7",
        rubric_scores={"r": 3.5},
        mean_score=3.5,
        latency_ms=100,
        cost_usd=0.0,
        baked_at="2026-05-25T00:00:00+00:00",
    )

    assert tdb.slot_winner("entity_extraction") == "winner-model"


def test_slot_winner_returns_none_when_no_data(tmp_db: Path):
    assert tdb.slot_winner("entity_extraction") is None


def test_db_path_respects_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    custom = tmp_path / "custom-location.sqlite"
    monkeypatch.setenv("ORCHESTRATOR_DB_PATH", str(custom))
    assert tdb.db_path() == custom


def test_default_db_path_is_agent_orchestrator_dotdir(monkeypatch: pytest.MonkeyPatch):
    """Regression: the default must live in this project's own dotdir —
    it used to point inside another product's dotdir."""
    monkeypatch.delenv("ORCHESTRATOR_DB_PATH", raising=False)
    assert tdb.db_path() == Path.home() / ".agent-orchestrator" / "telemetry.sqlite"


# --- per-app scoping (app_name) ---------------------------------------------


def test_log_decision_records_app_name_and_audit_filters_by_it(tmp_db: Path):
    """Two apps writing to the same DB must not contaminate each other's audit."""
    tdb.log_decision(
        session_id="a1",
        message_excerpt="from app a",
        classified_slot="entity_extraction",
        selected_model="m",
        app_name="app-a",
        timestamp="2026-06-01T00:00:00+00:00",
    )
    tdb.log_decision(
        session_id="b1",
        message_excerpt="from app b",
        classified_slot="entity_extraction",
        selected_model="m",
        app_name="app-b",
        timestamp="2026-06-01T00:00:01+00:00",
    )
    rows_a = tdb.routing_decisions_for_audit(
        app_name="app-a", slots=None, since_iso="2026-01-01"
    )
    assert [r["session_id"] for r in rows_a] == ["a1"]
    rows_b = tdb.routing_decisions_for_audit(
        app_name="app-b", slots=None, since_iso="2026-01-01"
    )
    assert [r["session_id"] for r in rows_b] == ["b1"]
    # No filter → everything.
    rows_all = tdb.routing_decisions_for_audit(
        app_name=None, slots=None, since_iso="2026-01-01"
    )
    assert {r["session_id"] for r in rows_all} == {"a1", "b1"}


# Pre-app_name routing_decisions shape, verbatim — used to fake a legacy DB.
_LEGACY_ROUTING_DECISIONS_SQL = """
CREATE TABLE routing_decisions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp TEXT NOT NULL,
  session_id TEXT NOT NULL,
  message_excerpt TEXT NOT NULL,
  classified_slot TEXT,
  selected_model TEXT NOT NULL,
  fallback_used INTEGER DEFAULT 0,
  latency_ms INTEGER,
  cost_usd REAL,
  user_feedback TEXT
);
"""


def test_init_db_migrates_legacy_db_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    """A DB created before the app_name column existed must upgrade in place:
    column added, legacy rows readable (NULL app_name), filtered audits
    exclude them with a warning instead of returning other apps' rows."""
    p = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(p)
    try:
        conn.executescript(_LEGACY_ROUTING_DECISIONS_SQL)
        conn.execute(
            "INSERT INTO routing_decisions "
            "(timestamp, session_id, message_excerpt, classified_slot, selected_model) "
            "VALUES ('2026-06-01T00:00:00+00:00', 'legacy-1', 'old row', "
            "'entity_extraction', 'm')"
        )
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setenv("ORCHESTRATOR_DB_PATH", str(p))
    tdb.init_db()

    conn = sqlite3.connect(p)
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(routing_decisions)")}
    finally:
        conn.close()
    assert "app_name" in cols

    # Unfiltered query still surfaces the legacy row, app_name NULL.
    rows = tdb.routing_decisions_for_audit(app_name=None, slots=None, since_iso="2020-01-01")
    assert len(rows) == 1
    assert rows[0]["session_id"] == "legacy-1"
    assert rows[0]["app_name"] is None

    # Filtered query excludes it — and says so on stderr.
    rows = tdb.routing_decisions_for_audit(app_name="app-a", slots=None, since_iso="2020-01-01")
    assert rows == []
    err = capsys.readouterr().err
    assert "legacy" in err
    assert "app_name" in err

    # New writes after migration carry app_name and round-trip.
    tdb.log_decision(
        session_id="new-1",
        message_excerpt="post-migration",
        classified_slot="entity_extraction",
        selected_model="m",
        app_name="app-a",
        timestamp="2026-06-02T00:00:00+00:00",
    )
    rows = tdb.routing_decisions_for_audit(app_name="app-a", slots=None, since_iso="2020-01-01")
    assert [r["session_id"] for r in rows] == ["new-1"]


def test_graded_decisions_for_filters_correctly(tmp_db: Path):
    # Three rows: one matching, one wrong slot, one with no feedback.
    tdb.log_decision(
        session_id="s1",
        message_excerpt="extract entities",
        classified_slot="entity_extraction",
        selected_model="ollama/qwen3:14b",
        user_feedback="good",
    )
    tdb.log_decision(
        session_id="s2",
        message_excerpt="extract entities wrong slot",
        classified_slot="summary_synthesis",
        selected_model="ollama/qwen3:14b",
        user_feedback="good",
    )
    tdb.log_decision(
        session_id="s3",
        message_excerpt="ungraded",
        classified_slot="entity_extraction",
        selected_model="ollama/qwen3:14b",
        user_feedback=None,
    )
    matches = tdb.graded_decisions_for("entity_extraction", "ollama/qwen3:14b")
    assert len(matches) == 1
    assert matches[0]["message_excerpt"] == "extract entities"
    assert matches[0]["user_feedback"] == "good"
