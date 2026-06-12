"""Tests for orchestrator.audit.quality — sampling-based drift detection."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from orchestrator.audit import AuditConfig, finalize_quality, prepare_quality_batch
from orchestrator.audit.quality import _classify_alert

from .conftest import seed_samples, write_judge_scores


def test_classify_alert_no_samples():
    assert _classify_alert(
        None,
        current_pct_of_judge=None,
        warn_threshold=95.0,
        rebake_threshold=80.0,
        n_samples=0,
    ) == "no_samples"


def test_classify_alert_unknown():
    """n_samples > 0 but no baseline."""
    assert _classify_alert(
        None,
        current_pct_of_judge=None,
        warn_threshold=95.0,
        rebake_threshold=80.0,
        n_samples=5,
    ) == "unknown"


def test_classify_alert_ok():
    """Current >= warn → ok."""
    assert _classify_alert(
        0.0,
        current_pct_of_judge=98.0,
        warn_threshold=95.0,
        rebake_threshold=80.0,
        n_samples=5,
    ) == "ok"


def test_classify_alert_warn():
    """warn > current >= rebake → warn."""
    assert _classify_alert(
        -5.0,
        current_pct_of_judge=90.0,
        warn_threshold=95.0,
        rebake_threshold=80.0,
        n_samples=5,
    ) == "warn"


def test_classify_alert_rebake():
    """current < rebake → rebake."""
    assert _classify_alert(
        -20.0,
        current_pct_of_judge=75.0,
        warn_threshold=95.0,
        rebake_threshold=80.0,
        n_samples=5,
    ) == "rebake"


def test_prepare_batch_with_no_samples(audit_cfg: AuditConfig, tmp_db: Path, tmp_path: Path, fixed_now: datetime):
    """No samples in DB → batch is written but items list is empty."""
    out_dir = tmp_path / "audit-run"
    batch_path = prepare_quality_batch(
        audit_cfg,
        out_dir=out_dir,
        now=fixed_now,
    )
    assert batch_path.exists()
    import json
    batch = json.loads(batch_path.read_text())
    assert batch["items"] == []
    assert batch["app_name"] == "test-app"


def test_prepare_batch_caps_per_slot(audit_cfg: AuditConfig, tmp_db: Path, tmp_path: Path, fixed_now: datetime):
    """Slot with > max_samples_per_slot is capped."""
    # max_samples_per_slot=10 in fixture; sample_rate=1.0 wants all 20.
    cfg = audit_cfg.model_copy(update={"sample_rate": 1.0})
    seed_samples(
        app_name="test-app",
        slot="entity_extraction",
        candidate_model="ollama/gemma4:e4b",
        n=20,
        base_time=fixed_now,
    )
    out_dir = tmp_path / "audit-run"
    batch_path = prepare_quality_batch(
        cfg,
        out_dir=out_dir,
        now=fixed_now + timedelta(hours=5),
    )
    import json
    batch = json.loads(batch_path.read_text())
    entity_items = [i for i in batch["items"] if i["slot"] == "entity_extraction"]
    assert len(entity_items) == 10  # capped


def test_prepare_batch_applies_sample_rate(audit_cfg: AuditConfig, tmp_db: Path, tmp_path: Path, fixed_now: datetime):
    """sample_rate is honored: ~5% of 20 samples → ceil(1.0) = 1 judged item.

    Regression: sample_rate used to be a dead config knob — the batch always
    took max_samples_per_slot regardless of the configured fraction."""
    # Fixture sample_rate=0.05; 20 samples → 1 item (not the 10-cap).
    seed_samples(
        app_name="test-app",
        slot="entity_extraction",
        candidate_model="ollama/gemma4:e4b",
        n=20,
        base_time=fixed_now,
    )
    out_dir = tmp_path / "audit-run"
    batch_path = prepare_quality_batch(
        audit_cfg,
        out_dir=out_dir,
        now=fixed_now + timedelta(hours=5),
    )
    import json
    batch = json.loads(batch_path.read_text())
    entity_items = [i for i in batch["items"] if i["slot"] == "entity_extraction"]
    assert len(entity_items) == 1


def test_prepare_batch_min_one_sample_when_traffic_exists(
    audit_cfg: AuditConfig, tmp_db: Path, tmp_path: Path, fixed_now: datetime
):
    """A slot with ANY samples gets at least one judged item, however small
    sample_rate * n is — a live slot must never be silently unjudged."""
    # 3 samples at 5% → 0.15 raw → still 1 item.
    seed_samples(
        app_name="test-app",
        slot="entity_extraction",
        candidate_model="ollama/gemma4:e4b",
        n=3,
        base_time=fixed_now,
    )
    out_dir = tmp_path / "audit-run"
    batch_path = prepare_quality_batch(
        audit_cfg,
        out_dir=out_dir,
        now=fixed_now + timedelta(hours=5),
    )
    import json
    batch = json.loads(batch_path.read_text())
    entity_items = [i for i in batch["items"] if i["slot"] == "entity_extraction"]
    assert len(entity_items) == 1


def test_prepare_batch_takes_most_recent_samples(
    audit_cfg: AuditConfig, tmp_db: Path, tmp_path: Path, fixed_now: datetime
):
    """The judge batch holds the NEWEST samples per slot, not the oldest —
    recent traffic is the honest drift signal (a fresh regression hides if
    the judge only ever re-scores week-old calls)."""
    cfg = audit_cfg.model_copy(update={"sample_rate": 1.0})  # want 20, cap 10
    sids = seed_samples(
        app_name="test-app",
        slot="entity_extraction",
        candidate_model="ollama/gemma4:e4b",
        n=20,
        base_time=fixed_now,
    )
    out_dir = tmp_path / "audit-run"
    batch_path = prepare_quality_batch(
        cfg,
        out_dir=out_dir,
        now=fixed_now + timedelta(hours=5),
    )
    import json
    batch = json.loads(batch_path.read_text())
    batch_ids = [i["sample_id"] for i in batch["items"] if i["slot"] == "entity_extraction"]
    # seed_samples writes ascending routed_at → the last 10 ids are the newest.
    assert batch_ids == sids[-10:]


def test_finalize_quality_computes_drift(
    audit_cfg: AuditConfig, tmp_db: Path, tmp_path: Path, fixed_now: datetime
):
    """Full prepare → score → finalize roundtrip computes per-slot drift."""
    # sample_rate=1.0 so all 3 seeded samples land in the batch — this test
    # exercises the drift math, not the sampling (covered above).
    cfg = audit_cfg.model_copy(update={"sample_rate": 1.0})
    # Seed 3 samples on entity_extraction. Baseline in routing.json is 98%.
    sids = seed_samples(
        app_name="test-app",
        slot="entity_extraction",
        candidate_model="ollama/gemma4:e4b",
        n=3,
        base_time=fixed_now,
    )
    out_dir = tmp_path / "audit-run"
    prepare_quality_batch(cfg, out_dir=out_dir, now=fixed_now + timedelta(hours=5))
    # Judge gives all 3 samples 90 → drift = 90 - 98 = -8 → current = 90.
    write_judge_scores(
        out_dir / "judge-scores.json",
        scores_by_sample={s: 90.0 for s in sids},
    )
    quality = finalize_quality(cfg, out_dir=out_dir, now=fixed_now + timedelta(hours=5))
    entity_slot = next(s for s in quality.slots if s.slot == "entity_extraction")
    assert entity_slot.n_samples == 3
    assert entity_slot.mean_quality == 90.0
    assert entity_slot.baseline_pct_of_judge == 98.0
    assert entity_slot.drift_pct == -8.0
    # Current 90 is below warn (95) but above rebake (80) → warn.
    assert entity_slot.alert == "warn"


def test_finalize_quality_triggers_rebake_below_threshold(
    audit_cfg: AuditConfig, tmp_db: Path, tmp_path: Path, fixed_now: datetime
):
    """Current quality below rebake_threshold_pct → alert == 'rebake'."""
    sids = seed_samples(
        app_name="test-app",
        slot="entity_extraction",
        candidate_model="ollama/gemma4:e4b",
        n=5,
        base_time=fixed_now,
    )
    out_dir = tmp_path / "audit-run"
    prepare_quality_batch(audit_cfg, out_dir=out_dir, now=fixed_now + timedelta(hours=5))
    # Judge scores all 5 samples 70 → below 80 rebake threshold.
    write_judge_scores(out_dir / "judge-scores.json", scores_by_sample={s: 70.0 for s in sids})
    quality = finalize_quality(audit_cfg, out_dir=out_dir, now=fixed_now + timedelta(hours=5))
    entity_slot = next(s for s in quality.slots if s.slot == "entity_extraction")
    assert entity_slot.alert == "rebake"
    assert "entity_extraction" in quality.needs_rebake()


def test_finalize_quality_handles_no_samples_slot(
    audit_cfg: AuditConfig, tmp_db: Path, tmp_path: Path, fixed_now: datetime
):
    """Slot with no samples gets alert=='no_samples' (not crash)."""
    # Seed samples only for entity_extraction.
    sids = seed_samples(
        app_name="test-app",
        slot="entity_extraction",
        candidate_model="ollama/gemma4:e4b",
        n=2,
        base_time=fixed_now,
    )
    out_dir = tmp_path / "audit-run"
    prepare_quality_batch(audit_cfg, out_dir=out_dir, now=fixed_now + timedelta(hours=5))
    write_judge_scores(out_dir / "judge-scores.json", scores_by_sample={s: 95.0 for s in sids})
    quality = finalize_quality(audit_cfg, out_dir=out_dir, now=fixed_now + timedelta(hours=5))
    triage_slot = next(s for s in quality.slots if s.slot == "relevance_triage")
    assert triage_slot.n_samples == 0
    assert triage_slot.alert == "no_samples"


def test_finalize_quality_missing_baseline_marks_unknown(
    audit_cfg: AuditConfig, tmp_db: Path, tmp_path: Path, fixed_now: datetime
):
    """A sampled slot with no quality_pct_of_judge in routing.json classifies
    'unknown' and is listed by missing_baseline() — the unverified path the
    CLI turns into exit 5 (it must never read as a healthy pass)."""
    import json
    routing = json.loads(audit_cfg.routing_json_path.read_text())
    del routing["entity_extraction"]["quality_pct_of_judge"]
    audit_cfg.routing_json_path.write_text(json.dumps(routing))
    sids = seed_samples(
        app_name="test-app",
        slot="entity_extraction",
        candidate_model="ollama/gemma4:e4b",
        n=2,
        base_time=fixed_now,
    )
    out_dir = tmp_path / "audit-run"
    prepare_quality_batch(audit_cfg, out_dir=out_dir, now=fixed_now + timedelta(hours=5))
    write_judge_scores(out_dir / "judge-scores.json", scores_by_sample={s: 95.0 for s in sids})
    quality = finalize_quality(audit_cfg, out_dir=out_dir, now=fixed_now + timedelta(hours=5))
    entity_slot = next(s for s in quality.slots if s.slot == "entity_extraction")
    assert entity_slot.alert == "unknown"
    assert "entity_extraction" in quality.missing_baseline()
    assert "entity_extraction" in quality.unverified()


def test_finalize_quality_missing_judge_scores_raises(
    audit_cfg: AuditConfig, tmp_db: Path, tmp_path: Path, fixed_now: datetime
):
    """No judge-scores.json yet → FileNotFoundError on finalize."""
    out_dir = tmp_path / "audit-run"
    out_dir.mkdir()
    # Write a batch but no scores.
    prepare_quality_batch(audit_cfg, out_dir=out_dir, now=fixed_now)
    import pytest

    with pytest.raises(FileNotFoundError):
        finalize_quality(audit_cfg, out_dir=out_dir, now=fixed_now)


def test_overall_quality_pct_of_baseline(
    audit_cfg: AuditConfig, tmp_db: Path, tmp_path: Path, fixed_now: datetime
):
    """Headline Y aggregates across slots that have both samples and baseline."""
    # Two slots, both with samples + baselines.
    sids_a = seed_samples(
        app_name="test-app", slot="entity_extraction",
        candidate_model="ollama/gemma4:e4b", n=2, base_time=fixed_now,
    )
    sids_b = seed_samples(
        app_name="test-app", slot="relevance_triage",
        candidate_model="ollama/qwen3:8b", n=2, base_time=fixed_now,
    )
    out_dir = tmp_path / "audit-run"
    prepare_quality_batch(audit_cfg, out_dir=out_dir, now=fixed_now + timedelta(hours=5))
    write_judge_scores(
        out_dir / "judge-scores.json",
        scores_by_sample={
            sids_a[0]: 98.0, sids_a[1]: 98.0,  # baseline 98, drift 0
            sids_b[0]: 86.0, sids_b[1]: 86.0,  # baseline 86, drift 0
        },
    )
    quality = finalize_quality(audit_cfg, out_dir=out_dir, now=fixed_now + timedelta(hours=5))
    # Headline = mean of (98 + 86) / 2 = 92.0
    assert quality.overall_quality_pct_of_baseline == 92.0
