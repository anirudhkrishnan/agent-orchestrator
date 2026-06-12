"""Tests for orchestrator.audit.effects — frontier counterfactual + savings."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from orchestrator.audit import AuditConfig, PricingEntry, compute_effects_report
from orchestrator.audit.effects import _estimate_tokens_from_cost, _percentile

from .conftest import seed_decisions


def test_estimate_tokens_zero_cost_returns_zero(audit_cfg: AuditConfig):
    in_tok, out_tok = _estimate_tokens_from_cost(0.0, audit_cfg.pricing, "ollama/qwen3:8b")
    assert (in_tok, out_tok) == (0.0, 0.0)


def test_estimate_tokens_unknown_model_returns_zero(audit_cfg: AuditConfig):
    in_tok, out_tok = _estimate_tokens_from_cost(0.10, audit_cfg.pricing, "unknown/model")
    assert (in_tok, out_tok) == (0.0, 0.0)


def test_estimate_tokens_back_solves_from_cost(audit_cfg: AuditConfig):
    """Given known rates, tokens should be reproducible from cost."""
    # Opus rates: $15/$75 per 1M. Blended (1:3) → $15 + 3*$75 = $240 per 1M input.
    # For $0.024 cost → 100 input tokens, 300 output tokens.
    in_tok, out_tok = _estimate_tokens_from_cost(0.024, audit_cfg.pricing, "anthropic/claude-opus-4-7")
    assert in_tok == pytest.approx(100.0, rel=0.01)
    assert out_tok == pytest.approx(300.0, rel=0.01)


def test_percentile_handles_empty_list():
    assert _percentile([], 50) == 0.0


def test_percentile_basic():
    assert _percentile([1, 2, 3, 4, 5], 50) == pytest.approx(3.0)
    assert _percentile([1, 2, 3, 4, 5], 100) == pytest.approx(5.0)
    assert _percentile([1, 2, 3, 4, 5], 0) == pytest.approx(1.0)


def test_local_model_full_displacement(audit_cfg: AuditConfig, tmp_db: Path, fixed_now: datetime):
    """All calls on local Ollama → ~100% savings vs Opus counterfactual."""
    seed_decisions(
        slot="entity_extraction",
        selected_model="ollama/gemma4:e4b",
        n=10,
        base_time=fixed_now,
        cost_usd=0.0,
        latency_ms=1200,
    )
    # Seed empty slots so they don't break aggregation.
    report = compute_effects_report(audit_cfg, now=fixed_now + timedelta(hours=5))
    assert report.total_actual_cost_usd == 0.0
    # Counterfactual should be > 0 (nominal 500/1500 estimate per call * 10 calls).
    assert report.total_counterfactual_cost_usd > 0.0
    assert report.overall_savings_pct == pytest.approx(100.0)


def test_frontier_calls_show_zero_savings(audit_cfg: AuditConfig, tmp_db: Path, fixed_now: datetime):
    """Calls that already used Opus don't generate phantom displacement."""
    seed_decisions(
        slot="entity_extraction",
        selected_model="anthropic/claude-opus-4-7",
        n=10,
        base_time=fixed_now,
        cost_usd=0.05,
        latency_ms=800,
    )
    report = compute_effects_report(audit_cfg, now=fixed_now + timedelta(hours=5))
    # Actual and counterfactual both = $0.50 → 0% savings.
    assert report.total_actual_cost_usd == pytest.approx(0.50)
    assert report.total_counterfactual_cost_usd == pytest.approx(0.50)
    assert report.overall_savings_pct == pytest.approx(0.0)


def test_mixed_traffic_partial_savings(audit_cfg: AuditConfig, tmp_db: Path, fixed_now: datetime):
    """Mixed local + frontier traffic → partial savings."""
    # 10 local calls, 5 frontier calls.
    seed_decisions(
        slot="entity_extraction",
        selected_model="ollama/gemma4:e4b",
        n=10,
        base_time=fixed_now,
        cost_usd=0.0,
    )
    seed_decisions(
        slot="relevance_triage",
        selected_model="anthropic/claude-opus-4-7",
        n=5,
        base_time=fixed_now,
        cost_usd=0.10,
    )
    report = compute_effects_report(audit_cfg, now=fixed_now + timedelta(hours=5))
    assert report.total_calls == 15
    assert 0.0 < report.overall_savings_pct < 100.0


def test_per_slot_latency_percentiles(audit_cfg: AuditConfig, tmp_db: Path, fixed_now: datetime):
    """p50 and p95 are computed per slot."""
    # Latencies for entity_extraction: 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000
    for i, lat in enumerate([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]):
        seed_decisions(
            slot="entity_extraction",
            selected_model="ollama/gemma4:e4b",
            n=1,
            base_time=fixed_now.replace(microsecond=i * 1000),
            latency_ms=lat,
        )
    report = compute_effects_report(audit_cfg, now=fixed_now + timedelta(hours=5))
    entity_slot = next(s for s in report.slots if s.slot == "entity_extraction")
    assert entity_slot.p50_latency_ms == pytest.approx(550.0)  # median of 10 ints
    # p95 of 1-10 (scaled by 100) ≈ 9.55 * 100 = 955
    assert 900 < entity_slot.p95_latency_ms <= 1000


def test_empty_window_returns_zero_savings(audit_cfg: AuditConfig, tmp_db: Path, fixed_now: datetime):
    """No traffic → zero savings, zero calls, no crash."""
    report = compute_effects_report(audit_cfg, now=fixed_now)
    assert report.total_calls == 0
    assert report.overall_savings_pct == 0.0
    # All per-slot rows have n_calls=0 and savings=0.
    for s in report.slots:
        assert s.n_calls == 0
        assert s.savings_pct == 0.0


def test_total_savings_signed_when_router_costs_more(
    audit_cfg: AuditConfig, tmp_db: Path, fixed_now: datetime
):
    """A routed model pricier than the frontier must show NEGATIVE savings.

    Regression: total_savings_usd used to be floored at $0, hiding the case
    where the router actually cost more than the counterfactual."""
    cfg = audit_cfg.model_copy(deep=True)
    cfg.pricing.entries.append(
        PricingEntry(model="expensive/model", input_usd_per_1m=150.0, output_usd_per_1m=750.0)
    )
    seed_decisions(
        slot="entity_extraction",
        selected_model="expensive/model",
        n=5,
        base_time=fixed_now,
        cost_usd=0.10,
    )
    report = compute_effects_report(cfg, now=fixed_now + timedelta(hours=5))
    assert report.total_savings_usd < 0.0
    assert report.total_savings_usd == pytest.approx(
        report.total_counterfactual_cost_usd - report.total_actual_cost_usd
    )
    assert report.overall_savings_pct < 0.0


def test_paid_model_without_logged_cost_excluded_from_savings(
    audit_cfg: AuditConfig, tmp_db: Path, fixed_now: datetime
):
    """A KNOWN paid model whose calls carry no logged cost must NOT fall into
    the free-local branch and fabricate 100% savings — it lands in the
    cost-unverified bucket with zero claimed displacement."""
    cfg = audit_cfg.model_copy(deep=True)
    cfg.pricing.entries.append(
        PricingEntry(
            model="anthropic/claude-sonnet-4-6", input_usd_per_1m=3.0, output_usd_per_1m=15.0
        )
    )
    seed_decisions(
        slot="entity_extraction",
        selected_model="anthropic/claude-sonnet-4-6",
        n=5,
        base_time=fixed_now,
        cost_usd=0.0,  # paid model, but the plugin logged nothing
    )
    report = compute_effects_report(cfg, now=fixed_now + timedelta(hours=5))
    assert report.total_counterfactual_cost_usd == 0.0
    assert report.total_savings_usd == 0.0
    assert report.overall_savings_pct == 0.0
    assert report.cost_unverified_models == ["anthropic/claude-sonnet-4-6"]


def test_unpriced_model_flagged_on_report(
    audit_cfg: AuditConfig, tmp_db: Path, fixed_now: datetime
):
    """Models absent from the pricing table travel on the EffectsReport (not
    stderr-only) and claim zero displacement."""
    seed_decisions(
        slot="entity_extraction",
        selected_model="mystery/model",
        n=3,
        base_time=fixed_now,
        cost_usd=0.05,
    )
    report = compute_effects_report(audit_cfg, now=fixed_now + timedelta(hours=5))
    assert report.unpriced_models == ["mystery/model"]
    # Counted at actual cost on both sides → zero claimed displacement.
    assert report.total_actual_cost_usd == pytest.approx(0.15)
    assert report.total_counterfactual_cost_usd == pytest.approx(0.15)
    assert report.total_savings_usd == pytest.approx(0.0)


def test_success_rate_reflects_errors(audit_cfg: AuditConfig, tmp_db: Path, fixed_now: datetime):
    """user_feedback='error' rows depress success_rate."""
    seed_decisions(
        slot="entity_extraction",
        selected_model="ollama/gemma4:e4b",
        n=8,
        base_time=fixed_now,
    )
    seed_decisions(
        slot="entity_extraction",
        selected_model="ollama/gemma4:e4b",
        n=2,
        base_time=fixed_now,
        error=True,
    )
    report = compute_effects_report(audit_cfg, now=fixed_now + timedelta(hours=5))
    entity_slot = next(s for s in report.slots if s.slot == "entity_extraction")
    assert entity_slot.success_rate_pct == pytest.approx(80.0)  # 8/10 ok
