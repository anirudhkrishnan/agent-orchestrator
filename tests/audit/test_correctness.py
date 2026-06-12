"""Tests for orchestrator.audit.correctness — verify expectations vs reality."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from orchestrator.audit import AuditConfig, run_correctness_audit

from .conftest import seed_decisions


def test_all_primary_matches_routing(audit_cfg: AuditConfig, tmp_db: Path, fixed_now: datetime):
    """Happy path: all routed calls used the configured primary model."""
    seed_decisions(
        slot="entity_extraction",
        selected_model="ollama/gemma4:e4b",
        n=10,
        base_time=fixed_now,
    )
    seed_decisions(
        slot="relevance_triage",
        selected_model="ollama/qwen3:8b",
        n=10,
        base_time=fixed_now,
    )
    seed_decisions(
        slot="summary_synthesis",
        selected_model="ollama/qwen3:8b",
        n=10,
        base_time=fixed_now,
    )
    report = run_correctness_audit(audit_cfg, now=fixed_now + timedelta(hours=5))
    assert report.overall_pass
    # All slots have all calls hitting the primary.
    for s in report.slots:
        assert s.n_with_expected_primary == 10
        assert s.alarms == []


def test_unexpected_model_triggers_alarm(audit_cfg: AuditConfig, tmp_db: Path, fixed_now: datetime):
    """Calls that used a model not in routing.json raise an UNEXPECTED MODEL alarm."""
    # 9 calls on the right model, 1 on a wrong model.
    seed_decisions(
        slot="entity_extraction",
        selected_model="ollama/gemma4:e4b",
        n=9,
        base_time=fixed_now,
    )
    seed_decisions(
        slot="entity_extraction",
        selected_model="anthropic/claude-opus-4-7",
        n=1,
        base_time=fixed_now,
    )
    # Seed traffic for the other in-scope slots so they don't trip NO TRAFFIC.
    seed_decisions(
        slot="relevance_triage",
        selected_model="ollama/qwen3:8b",
        n=5,
        base_time=fixed_now,
    )
    seed_decisions(
        slot="summary_synthesis",
        selected_model="ollama/qwen3:8b",
        n=5,
        base_time=fixed_now,
    )
    report = run_correctness_audit(audit_cfg, now=fixed_now + timedelta(hours=5))
    assert not report.overall_pass
    entity_slot = next(s for s in report.slots if s.slot == "entity_extraction")
    assert entity_slot.n_with_unexpected_model == 1
    assert any("UNEXPECTED MODEL" in a for a in entity_slot.alarms)
    assert "anthropic/claude-opus-4-7" in entity_slot.unexpected_models


def test_no_traffic_triggers_alarm(audit_cfg: AuditConfig, tmp_db: Path, fixed_now: datetime):
    """A slot in scope that received zero calls is alarmed."""
    # Only seed two of three slots.
    seed_decisions(
        slot="entity_extraction",
        selected_model="ollama/gemma4:e4b",
        n=5,
        base_time=fixed_now,
    )
    seed_decisions(
        slot="relevance_triage",
        selected_model="ollama/qwen3:8b",
        n=5,
        base_time=fixed_now,
    )
    report = run_correctness_audit(audit_cfg, now=fixed_now + timedelta(hours=5))
    assert not report.overall_pass
    summary_slot = next(s for s in report.slots if s.slot == "summary_synthesis")
    assert summary_slot.n_calls == 0
    assert any("NO TRAFFIC" in a for a in summary_slot.alarms)


def test_high_fallback_rate_alarm(audit_cfg: AuditConfig, tmp_db: Path, fixed_now: datetime):
    """When fallback_rate exceeds max_fallback_rate_pct, alarm."""
    # 8 primary, 2 fallback for entity_extraction = 20% fallback (above 5% default)
    seed_decisions(
        slot="entity_extraction",
        selected_model="ollama/gemma4:e4b",
        n=8,
        base_time=fixed_now,
    )
    seed_decisions(
        slot="entity_extraction",
        selected_model="ollama/qwen3:8b",
        n=2,
        base_time=fixed_now,
        fallback_used=True,
    )
    # Healthy traffic on the other two slots so they don't alarm.
    seed_decisions(slot="relevance_triage", selected_model="ollama/qwen3:8b", n=5, base_time=fixed_now)
    seed_decisions(slot="summary_synthesis", selected_model="ollama/qwen3:8b", n=5, base_time=fixed_now)
    report = run_correctness_audit(audit_cfg, now=fixed_now + timedelta(hours=5))
    entity_slot = next(s for s in report.slots if s.slot == "entity_extraction")
    assert entity_slot.fallback_rate_pct == 20.0
    assert any("HIGH FALLBACK RATE" in a for a in entity_slot.alarms)


def test_high_error_rate_alarm(audit_cfg: AuditConfig, tmp_db: Path, fixed_now: datetime):
    """Calls with user_feedback='error' get counted toward error_rate."""
    seed_decisions(
        slot="entity_extraction",
        selected_model="ollama/gemma4:e4b",
        n=8,
        base_time=fixed_now,
    )
    # 2 errored on primary → 20% error rate
    seed_decisions(
        slot="entity_extraction",
        selected_model="ollama/gemma4:e4b",
        n=2,
        base_time=fixed_now,
        error=True,
    )
    seed_decisions(slot="relevance_triage", selected_model="ollama/qwen3:8b", n=5, base_time=fixed_now)
    seed_decisions(slot="summary_synthesis", selected_model="ollama/qwen3:8b", n=5, base_time=fixed_now)
    report = run_correctness_audit(audit_cfg, now=fixed_now + timedelta(hours=5))
    entity_slot = next(s for s in report.slots if s.slot == "entity_extraction")
    assert entity_slot.error_rate_pct == 20.0
    assert any("HIGH ERROR RATE" in a for a in entity_slot.alarms)


def test_routing_json_missing_raises(audit_cfg: AuditConfig, tmp_db: Path, fixed_now: datetime, tmp_path: Path):
    """When routing.json doesn't exist, FileNotFoundError surfaces."""
    audit_cfg.routing_json_path = tmp_path / "nope.json"
    import pytest

    with pytest.raises(FileNotFoundError):
        run_correctness_audit(audit_cfg, now=fixed_now)


def test_window_excludes_older_decisions(audit_cfg: AuditConfig, tmp_db: Path):
    """Decisions older than lookback_days don't enter the audit."""
    # Window default = 7 days. Seed something 30 days old + something today.
    old = datetime(2026, 4, 26, tzinfo=timezone.utc)
    now = datetime(2026, 5, 26, tzinfo=timezone.utc)
    seed_decisions(
        slot="entity_extraction",
        selected_model="anthropic/claude-opus-4-7",
        n=100,
        base_time=old,
    )
    seed_decisions(
        slot="entity_extraction",
        selected_model="ollama/gemma4:e4b",
        n=3,
        base_time=now.replace(hour=10),
    )
    seed_decisions(slot="relevance_triage", selected_model="ollama/qwen3:8b", n=3, base_time=now)
    seed_decisions(slot="summary_synthesis", selected_model="ollama/qwen3:8b", n=3, base_time=now)
    report = run_correctness_audit(audit_cfg, now=now.replace(hour=23))
    entity_slot = next(s for s in report.slots if s.slot == "entity_extraction")
    # Only the 3 recent calls — the 100 stale opus calls are out of window.
    assert entity_slot.n_calls == 3
    assert entity_slot.n_with_unexpected_model == 0
