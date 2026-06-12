"""Tests for orchestrator.audit.report.compose_audit_report."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from orchestrator.audit import (
    AuditConfig,
    PricingEntry,
    compose_audit_report,
    compute_effects_report,
    finalize_quality,
    prepare_quality_batch,
    run_correctness_audit,
)

from .conftest import seed_decisions, seed_samples, write_judge_scores


def test_report_renders_all_sections(
    audit_cfg: AuditConfig, tmp_db: Path, tmp_path: Path, fixed_now: datetime
):
    """End-to-end: seed data → run all three audits → compose report."""
    # 10 healthy calls per slot.
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
    # 3 samples on entity_extraction → judge scores 98.
    sids = seed_samples(
        app_name="test-app",
        slot="entity_extraction",
        candidate_model="ollama/gemma4:e4b",
        n=3,
        base_time=fixed_now,
    )

    out_dir = tmp_path / "run-dir"
    prepare_quality_batch(audit_cfg, out_dir=out_dir, now=fixed_now + timedelta(hours=5))
    write_judge_scores(out_dir / "judge-scores.json", scores_by_sample={s: 98.0 for s in sids})

    correctness = run_correctness_audit(audit_cfg, now=fixed_now + timedelta(hours=5))
    effects = compute_effects_report(audit_cfg, now=fixed_now + timedelta(hours=5))
    quality = finalize_quality(audit_cfg, out_dir=out_dir, now=fixed_now + timedelta(hours=5))

    report_path = out_dir / "AUDIT-REPORT.md"
    compose_audit_report(
        cfg=audit_cfg,
        correctness=correctness,
        effects=effects,
        quality=quality,
        out_path=report_path,
    )
    body = report_path.read_text()
    # Front matter
    assert "Audit Report — `test-app`" in body
    # Headline
    assert "Headline" in body
    assert "Reduced" in body
    # Sections
    assert "## Correctness" in body
    assert "## Effects" in body
    assert "## Quality drift" in body
    assert "## Recommendations" in body
    # Slot present in tables
    assert "`entity_extraction`" in body
    # Quality should be 'ok' since 98 matches baseline 98.
    assert " ok " in body or "(ok)" in body


def test_report_no_traffic_headline(
    audit_cfg: AuditConfig, tmp_db: Path, tmp_path: Path, fixed_now: datetime
):
    """When there are no calls at all, the headline says so explicitly."""
    out_dir = tmp_path / "run-dir"
    prepare_quality_batch(audit_cfg, out_dir=out_dir, now=fixed_now)
    write_judge_scores(out_dir / "judge-scores.json", scores_by_sample={})

    correctness = run_correctness_audit(audit_cfg, now=fixed_now)
    effects = compute_effects_report(audit_cfg, now=fixed_now)
    quality = finalize_quality(audit_cfg, out_dir=out_dir, now=fixed_now)

    report_path = out_dir / "AUDIT-REPORT.md"
    compose_audit_report(
        cfg=audit_cfg,
        correctness=correctness,
        effects=effects,
        quality=quality,
        out_path=report_path,
    )
    body = report_path.read_text()
    assert "No traffic" in body


def test_report_surfaces_effects_accounting_flags(
    audit_cfg: AuditConfig, tmp_db: Path, tmp_path: Path, fixed_now: datetime
):
    """Unpriced + cost-unverified models must appear in the report markdown —
    a stderr-only warning dies with the terminal scrollback."""
    cfg = audit_cfg.model_copy(deep=True)
    cfg.pricing.entries.append(
        PricingEntry(
            model="anthropic/claude-sonnet-4-6", input_usd_per_1m=3.0, output_usd_per_1m=15.0
        )
    )
    # mystery/model is absent from the pricing table; sonnet is a paid model
    # whose calls carry no logged cost.
    seed_decisions(
        slot="entity_extraction", selected_model="mystery/model",
        n=2, base_time=fixed_now, cost_usd=0.05,
    )
    seed_decisions(
        slot="relevance_triage", selected_model="anthropic/claude-sonnet-4-6",
        n=2, base_time=fixed_now, cost_usd=0.0,
    )
    seed_decisions(slot="summary_synthesis", selected_model="ollama/qwen3:8b", n=2, base_time=fixed_now)

    out_dir = tmp_path / "run-dir"
    prepare_quality_batch(cfg, out_dir=out_dir, now=fixed_now + timedelta(hours=5))
    write_judge_scores(out_dir / "judge-scores.json", scores_by_sample={})

    correctness = run_correctness_audit(cfg, now=fixed_now + timedelta(hours=5))
    effects = compute_effects_report(cfg, now=fixed_now + timedelta(hours=5))
    quality = finalize_quality(cfg, out_dir=out_dir, now=fixed_now + timedelta(hours=5))

    compose_audit_report(
        cfg=cfg,
        correctness=correctness,
        effects=effects,
        quality=quality,
        out_path=out_dir / "AUDIT-REPORT.md",
    )
    body = (out_dir / "AUDIT-REPORT.md").read_text()
    assert "missing from the pricing table" in body
    assert "`mystery/model`" in body
    assert "paid models with no logged cost" in body
    assert "`anthropic/claude-sonnet-4-6`" in body


def test_report_rebake_recommendation_surfaces(
    audit_cfg: AuditConfig, tmp_db: Path, tmp_path: Path, fixed_now: datetime
):
    """When a slot needs rebake, the Recommendations section names it."""
    seed_decisions(
        slot="entity_extraction", selected_model="ollama/gemma4:e4b",
        n=5, base_time=fixed_now,
    )
    seed_decisions(slot="relevance_triage", selected_model="ollama/qwen3:8b", n=5, base_time=fixed_now)
    seed_decisions(slot="summary_synthesis", selected_model="ollama/qwen3:8b", n=5, base_time=fixed_now)
    sids = seed_samples(
        app_name="test-app", slot="entity_extraction",
        candidate_model="ollama/gemma4:e4b", n=3, base_time=fixed_now,
    )
    out_dir = tmp_path / "run-dir"
    prepare_quality_batch(audit_cfg, out_dir=out_dir, now=fixed_now + timedelta(hours=5))
    # Score very low → triggers rebake.
    write_judge_scores(out_dir / "judge-scores.json", scores_by_sample={s: 50.0 for s in sids})

    correctness = run_correctness_audit(audit_cfg, now=fixed_now + timedelta(hours=5))
    effects = compute_effects_report(audit_cfg, now=fixed_now + timedelta(hours=5))
    quality = finalize_quality(audit_cfg, out_dir=out_dir, now=fixed_now + timedelta(hours=5))

    compose_audit_report(
        cfg=audit_cfg,
        correctness=correctness,
        effects=effects,
        quality=quality,
        out_path=out_dir / "AUDIT-REPORT.md",
    )
    body = (out_dir / "AUDIT-REPORT.md").read_text()
    assert "RE-BAKE" in body
    assert "`entity_extraction`" in body
