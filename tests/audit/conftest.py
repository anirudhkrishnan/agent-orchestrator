"""Shared fixtures for audit tests.

Two recurring needs across the suite:
  * a fresh tmp telemetry DB per test, so a developer's real telemetry DB is
    never touched and tests don't leak state into each other.
  * a sample routing.json + audit-config YAML pair, since most audit
    functions read both.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from orchestrator.audit import AuditConfig, PricingEntry, PricingTable
from orchestrator.telemetry import db as tdb


@pytest.fixture
def tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Fresh SQLite DB per test."""
    p = tmp_path / "audit-test-telemetry.sqlite"
    monkeypatch.setenv("ORCHESTRATOR_DB_PATH", str(p))
    tdb.init_db()
    return p


@pytest.fixture
def routing_json(tmp_path: Path) -> Path:
    """A minimal routing.json the audit can read for expected models + baselines.

    Note: every fixture slot has ``last_baked_at`` set, otherwise the
    completeness gate (per the primitive's 2026-05-26 rule) would refuse
    to audit and the per-slot tests below could never reach their
    assertions. Tests for the completeness gate itself live in
    ``test_correctness_completeness.py``.
    """
    routing = {
        "_README": "Test fixture.",
        "entity_extraction": {
            "model": "ollama/gemma4:e4b",
            "fallback_model": "ollama/qwen3:8b",
            "judge_model": "claude-opus-4-7-interactive-session",
            "last_baked_at": "2026-05-26T20:03:40.592124+00:00",
            "quality_pct_of_judge": 98.0,
        },
        "relevance_triage": {
            "model": "ollama/qwen3:8b",
            "fallback_model": "ollama/gemma4:e4b",
            "judge_model": "claude-opus-4-7-interactive-session",
            "last_baked_at": "2026-05-26T20:03:40.592124+00:00",
            "quality_pct_of_judge": 86.0,
        },
        "summary_synthesis": {
            "model": "ollama/qwen3:8b",
            "fallback_model": "ollama/qwen3.5:9b",
            "judge_model": "claude-opus-4-7-interactive-session",
            "last_baked_at": "2026-05-26T20:03:40.592124+00:00",
            "quality_pct_of_judge": 88.0,
        },
    }
    p = tmp_path / "routing.json"
    p.write_text(json.dumps(routing, indent=2))
    return p


@pytest.fixture
def audit_cfg(routing_json: Path, tmp_path: Path) -> AuditConfig:
    """A populated AuditConfig pointing at the fixture routing.json."""
    return AuditConfig(
        app_name="test-app",
        slots_in_scope=["entity_extraction", "relevance_triage", "summary_synthesis"],
        routing_json_path=routing_json,
        warn_threshold_pct=95.0,
        rebake_threshold_pct=80.0,
        sample_rate=0.05,
        lookback_days=7,
        max_samples_per_slot=10,
        judge_model="claude-opus-4-7-interactive-session",
        pricing=PricingTable(
            frontier_model="anthropic/claude-opus-4-7",
            entries=[
                PricingEntry(
                    model="anthropic/claude-opus-4-7",
                    input_usd_per_1m=15.0,
                    output_usd_per_1m=75.0,
                ),
                PricingEntry(
                    model="ollama/qwen3:8b",
                    input_usd_per_1m=0.0,
                    output_usd_per_1m=0.0,
                ),
                PricingEntry(
                    model="ollama/gemma4:e4b",
                    input_usd_per_1m=0.0,
                    output_usd_per_1m=0.0,
                ),
                PricingEntry(
                    model="ollama/qwen3.5:9b",
                    input_usd_per_1m=0.0,
                    output_usd_per_1m=0.0,
                ),
            ],
        ),
        out_dir=tmp_path / "audit-out",
        max_fallback_rate_pct=5.0,
        max_error_rate_pct=2.0,
    )


@pytest.fixture
def fixed_now() -> datetime:
    """A stable "now" for tests that build telemetry with explicit timestamps.

    Anchored one hour behind the wall clock (NOT an absolute date): the CLI
    uses the real ``datetime.now(UTC)`` with a 7-day lookback, so rows seeded
    at an absolute date would age out of the window and rot the suite.
    """
    return (datetime.now(timezone.utc) - timedelta(hours=1)).replace(microsecond=0)


def seed_decisions(
    *,
    slot: str,
    selected_model: str,
    n: int,
    base_time: datetime,
    app_name: str = "test-app",
    cost_usd: float = 0.0,
    latency_ms: int = 1500,
    fallback_used: bool = False,
    error: bool = False,
) -> None:
    """Helper used across tests to seed routing_decisions rows.

    The function is exported via conftest so individual tests don't have to
    duplicate the kwargs. `app_name` defaults to the fixture config's app —
    rows without it are excluded from app-scoped audit queries.
    """
    for i in range(n):
        ts = base_time.replace(microsecond=i * 1000).isoformat()
        tdb.log_decision(
            session_id=f"sess-{slot}-{i}",
            message_excerpt=f"msg {i} for {slot}",
            classified_slot=slot,
            selected_model=selected_model,
            app_name=app_name,
            fallback_used=fallback_used,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            user_feedback="error" if error else None,
            timestamp=ts,
        )


def seed_samples(
    *,
    app_name: str,
    slot: str,
    candidate_model: str,
    n: int,
    base_time: datetime,
) -> list[str]:
    """Seed routed_call_samples rows. Returns the list of sample_ids written."""
    ids: list[str] = []
    for i in range(n):
        sid = f"{slot}-{i}"
        tdb.log_sample(
            sample_id=sid,
            app_name=app_name,
            slot=slot,
            candidate_model=candidate_model,
            input_text=f"sample input {i} for {slot}",
            output_text=f"sample output {i} for {slot}",
            latency_ms=1200,
            input_tokens=500,
            output_tokens=1500,
            routed_at=base_time.replace(microsecond=i * 1000).isoformat(),
        )
        ids.append(sid)
    return ids


def write_judge_scores(path: Path, *, scores_by_sample: dict[str, float]) -> None:
    """Write a judge-scores.json the audit's finalize step can read."""
    arr = [
        {"sample_id": sid, "mean_quality_score": score, "notes": f"test note for {sid}"}
        for sid, score in scores_by_sample.items()
    ]
    path.write_text(json.dumps(arr, indent=2))


def write_yaml_config(path: Path, cfg: AuditConfig) -> None:
    """Persist an AuditConfig as YAML for the CLI tests."""
    data = json.loads(cfg.model_dump_json())  # round-trip handles Path → str
    path.write_text(yaml.safe_dump(data, sort_keys=False))
