"""Tests for the completeness gate in `correctness.run_correctness_audit`.

Per the primitive's 2026-05-26 rule, every slot in `slots_in_scope` must be
fully baked in `routing.json` (non-null `last_baked_at`, not
`queue-for-human` without a measured fallback) before the audit can
produce any verdict. These tests guard that gate stays functional — without
it, the audit silently produces a "100% on the measured subset" verdict
that hides the slots still flowing to Claude.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from orchestrator.audit import AuditConfig, run_correctness_audit
from orchestrator.audit.correctness import check_scope_completeness


def test_check_completeness_passes_on_fully_baked(routing_json: Path):
    """When every slot has last_baked_at + a real model, return empty alarms."""
    routing = json.loads(routing_json.read_text())
    alarms = check_scope_completeness(
        ["entity_extraction", "relevance_triage", "summary_synthesis"],
        routing,
    )
    assert alarms == []


def test_check_completeness_alarms_on_null_last_baked_at(tmp_path: Path):
    """A slot present in routing.json but with last_baked_at=None alarms."""
    routing = {
        "entity_extraction": {
            "model": "ollama/gemma4:e4b",
            "last_baked_at": None,
        },
    }
    alarms = check_scope_completeness(["entity_extraction"], routing)
    assert len(alarms) == 1
    assert "INCOMPLETE_SCOPE" in alarms[0]
    assert "entity_extraction" in alarms[0]
    assert "last_baked_at=null" in alarms[0] or "never measured" in alarms[0]


def test_check_completeness_alarms_on_queue_for_claude_code_unbaked(tmp_path: Path):
    """queue-for-human WITH null last_baked_at = no measured fallback = alarm."""
    routing = {
        "pattern_classification": {
            "model": "queue-for-human",
            "last_baked_at": None,
        },
    }
    alarms = check_scope_completeness(["pattern_classification"], routing)
    assert len(alarms) == 1
    assert "INCOMPLETE_SCOPE" in alarms[0]
    assert "queue-for-human" in alarms[0]


def test_check_completeness_allows_queue_for_claude_code_with_measured_fallback(tmp_path: Path):
    """queue-for-human WITH a baked fallback is OK — the bake-off ran,
    the verdict was "no candidate hit the bar", and the system has documented
    that. Not an incomplete-scope alarm; the slot has been measured."""
    routing = {
        "structured_report": {
            "model": "queue-for-human",
            "fallback_model": "ollama/qwen3.5:9b",
            "last_baked_at": "2026-05-26T22:00:00+00:00",
        },
    }
    alarms = check_scope_completeness(["structured_report"], routing)
    assert alarms == []


def test_check_completeness_alarms_on_missing_from_routing(tmp_path: Path):
    """A slot in scope but absent from routing.json = scope drift = alarm."""
    routing = {}  # empty
    alarms = check_scope_completeness(["html_rendering"], routing)
    assert len(alarms) == 1
    assert "not present in routing.json" in alarms[0]


def test_run_audit_short_circuits_on_incomplete_scope(
    audit_cfg: AuditConfig,
    tmp_db: Path,
    fixed_now: datetime,
    tmp_path: Path,
):
    """The full audit short-circuits before reading telemetry when the
    completeness gate fails. Per-slot table is empty; overall_pass is False."""
    # Add a fourth slot to scope that doesn't exist in routing.json.
    audit_cfg.slots_in_scope = [
        "entity_extraction",
        "relevance_triage",
        "summary_synthesis",
        "pattern_classification",  # not in routing fixture
    ]
    report = run_correctness_audit(audit_cfg, now=fixed_now + timedelta(hours=5))
    assert not report.overall_pass
    assert report.slots == []  # short-circuited
    assert len(report.incomplete_scope_alarms) == 1
    assert "pattern_classification" in report.incomplete_scope_alarms[0]


def test_alarm_count_includes_incomplete_scope_alarms(
    audit_cfg: AuditConfig,
    tmp_db: Path,
    fixed_now: datetime,
):
    """alarm_count should aggregate both per-slot and incomplete-scope alarms."""
    audit_cfg.slots_in_scope = ["entity_extraction", "missing_slot"]
    report = run_correctness_audit(audit_cfg, now=fixed_now + timedelta(hours=5))
    # Short-circuited: no per-slot alarms, but the incomplete-scope alarm counts.
    assert report.alarm_count == 1
    assert not report.overall_pass
