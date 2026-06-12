"""Tests for orchestrator.audit.cli — the user-facing entry point."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


from orchestrator.audit.cli import main as audit_main
from orchestrator.audit.config import load_audit_config

from .conftest import seed_decisions, seed_samples, write_judge_scores, write_yaml_config


def test_init_scaffolds_yaml(tmp_path: Path, capsys):
    """`audit init` writes a YAML the loader can read back."""
    out = tmp_path / "cfg.yaml"
    rc = audit_main(["init", "--app", "news-digest", "--out", str(out)])
    assert rc == 0
    out_text = capsys.readouterr().out
    assert "wrote skeleton" in out_text
    cfg = load_audit_config(out)
    assert cfg.app_name == "news-digest"


def test_init_refuses_to_overwrite(tmp_path: Path, capsys):
    out = tmp_path / "cfg.yaml"
    audit_main(["init", "--app", "x", "--out", str(out)])
    rc = audit_main(["init", "--app", "x", "--out", str(out)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "already exists" in err


def test_run_writes_preliminary_report_and_batch(
    tmp_path: Path, tmp_db: Path, fixed_now: datetime, capsys, audit_cfg
):
    """`audit run` produces the preliminary report + judge-batch.json."""
    cfg_path = tmp_path / "audit-cfg.yaml"
    write_yaml_config(cfg_path, audit_cfg)
    # Seed minimal traffic so the report has content.
    seed_decisions(
        slot="entity_extraction", selected_model="ollama/gemma4:e4b",
        n=3, base_time=fixed_now,
    )
    seed_decisions(slot="relevance_triage", selected_model="ollama/qwen3:8b", n=3, base_time=fixed_now)
    seed_decisions(slot="summary_synthesis", selected_model="ollama/qwen3:8b", n=3, base_time=fixed_now)

    rc = audit_main(["run", "--app", "test-app", "--config", str(cfg_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "READY FOR JUDGE" in out
    # Find the run dir under out_dir/test-app/.
    base = audit_cfg.out_dir / "test-app"
    runs = list(base.iterdir())
    assert len(runs) == 1
    run_dir = runs[0]
    assert (run_dir / "judge-batch.json").exists()
    assert (run_dir / "AUDIT-REPORT.md").exists()
    body = (run_dir / "AUDIT-REPORT.md").read_text()
    # Quality section should be present but flagged pending.
    assert "## Quality drift" in body


def test_run_refuses_app_name_mismatch(tmp_path: Path, capsys, audit_cfg):
    cfg_path = tmp_path / "audit-cfg.yaml"
    write_yaml_config(cfg_path, audit_cfg)
    rc = audit_main(["run", "--app", "different-app", "--config", str(cfg_path)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "does not match" in err


def test_finalize_picks_latest_run(
    tmp_path: Path, tmp_db: Path, fixed_now: datetime, audit_cfg, capsys
):
    """`audit finalize` discovers the latest run dir and rewrites the report."""
    cfg_path = tmp_path / "audit-cfg.yaml"
    write_yaml_config(cfg_path, audit_cfg)
    seed_decisions(
        slot="entity_extraction", selected_model="ollama/gemma4:e4b",
        n=3, base_time=fixed_now,
    )
    seed_decisions(slot="relevance_triage", selected_model="ollama/qwen3:8b", n=3, base_time=fixed_now)
    seed_decisions(slot="summary_synthesis", selected_model="ollama/qwen3:8b", n=3, base_time=fixed_now)
    # Seed samples for EVERY in-scope slot — otherwise an unsampled slot is
    # correctly flagged 'no_samples' (unverified) and finalize returns exit 5.
    sids = []
    for slot, model in [
        ("entity_extraction", "ollama/gemma4:e4b"),
        ("relevance_triage", "ollama/qwen3:8b"),
        ("summary_synthesis", "ollama/qwen3:8b"),
    ]:
        sids += seed_samples(
            app_name="test-app", slot=slot, candidate_model=model,
            n=2, base_time=fixed_now,
        )

    # Phase 1.
    audit_main(["run", "--app", "test-app", "--config", str(cfg_path)])

    # Judge writes scores.
    run_dir = next(iter((audit_cfg.out_dir / "test-app").iterdir()))
    write_judge_scores(run_dir / "judge-scores.json", scores_by_sample={s: 98.0 for s in sids})

    # Phase 2 — all slots sampled + high quality → clean pass (exit 0).
    rc = audit_main(["finalize", "--app", "test-app", "--config", str(cfg_path)])
    assert rc == 0
    body = (run_dir / "AUDIT-REPORT.md").read_text()
    # Quality section should now show non-empty samples for entity_extraction.
    assert "entity_extraction" in body
    assert "ok" in body.lower()


def test_finalize_exit5_on_unverified_slot(
    tmp_path: Path, tmp_db: Path, fixed_now: datetime, audit_cfg, capsys
):
    """A baked, in-scope slot with ZERO sampled calls is 'unverified', not
    'healthy' — finalize must return non-zero (5) so automation can't read a
    silent green (RCA stress-test HIGH)."""
    cfg_path = tmp_path / "audit-cfg.yaml"
    write_yaml_config(cfg_path, audit_cfg)
    for slot, model in [
        ("entity_extraction", "ollama/gemma4:e4b"),
        ("relevance_triage", "ollama/qwen3:8b"),
        ("summary_synthesis", "ollama/qwen3:8b"),
    ]:
        seed_decisions(slot=slot, selected_model=model, n=3, base_time=fixed_now)
    # Sample + score ONLY entity_extraction; the other two stay unsampled.
    sids = seed_samples(
        app_name="test-app", slot="entity_extraction",
        candidate_model="ollama/gemma4:e4b", n=2, base_time=fixed_now,
    )
    audit_main(["run", "--app", "test-app", "--config", str(cfg_path)])
    run_dir = next(iter((audit_cfg.out_dir / "test-app").iterdir()))
    write_judge_scores(run_dir / "judge-scores.json", scores_by_sample={s: 98.0 for s in sids})
    rc = audit_main(["finalize", "--app", "test-app", "--config", str(cfg_path)])
    assert rc == 5
    assert "INSUFFICIENT SAMPLES" in capsys.readouterr().err


def test_finalize_exit5_on_missing_baseline(
    tmp_path: Path, tmp_db: Path, fixed_now: datetime, audit_cfg, capsys
):
    """A baked, in-scope slot with samples but NO quality_pct_of_judge baseline
    in routing.json cannot have its drift assessed — that's 'unverified', not
    'healthy'. finalize must exit 5 (the couldn't-verify exit), not 0."""
    # Strip the baseline from one slot; it stays baked so the completeness
    # gate still passes.
    routing = json.loads(audit_cfg.routing_json_path.read_text())
    del routing["summary_synthesis"]["quality_pct_of_judge"]
    audit_cfg.routing_json_path.write_text(json.dumps(routing))

    cfg_path = tmp_path / "audit-cfg.yaml"
    write_yaml_config(cfg_path, audit_cfg)
    sids = []
    for slot, model in [
        ("entity_extraction", "ollama/gemma4:e4b"),
        ("relevance_triage", "ollama/qwen3:8b"),
        ("summary_synthesis", "ollama/qwen3:8b"),
    ]:
        seed_decisions(slot=slot, selected_model=model, n=3, base_time=fixed_now)
        sids += seed_samples(
            app_name="test-app", slot=slot, candidate_model=model,
            n=2, base_time=fixed_now,
        )
    audit_main(["run", "--app", "test-app", "--config", str(cfg_path)])
    run_dir = next(iter((audit_cfg.out_dir / "test-app").iterdir()))
    write_judge_scores(run_dir / "judge-scores.json", scores_by_sample={s: 98.0 for s in sids})
    rc = audit_main(["finalize", "--app", "test-app", "--config", str(cfg_path)])
    assert rc == 5
    assert "NO BASELINE" in capsys.readouterr().err
    # The report names the slot as unverified instead of letting an "n/a" row
    # read as healthy.
    body = (run_dir / "AUDIT-REPORT.md").read_text()
    assert "UNVERIFIED" in body
    assert "summary_synthesis" in body


def test_finalize_without_run_fails_cleanly(tmp_path: Path, audit_cfg, capsys):
    cfg_path = tmp_path / "audit-cfg.yaml"
    write_yaml_config(cfg_path, audit_cfg)
    rc = audit_main(["finalize", "--app", "test-app", "--config", str(cfg_path)])
    assert rc == 2


def test_finalize_without_judge_scores_fails(
    tmp_path: Path, tmp_db: Path, audit_cfg, fixed_now, capsys
):
    """Run created, but no judge-scores yet → exit 2 with a clear message."""
    cfg_path = tmp_path / "audit-cfg.yaml"
    write_yaml_config(cfg_path, audit_cfg)
    seed_decisions(slot="entity_extraction", selected_model="ollama/gemma4:e4b", n=2, base_time=fixed_now)
    seed_decisions(slot="relevance_triage", selected_model="ollama/qwen3:8b", n=2, base_time=fixed_now)
    seed_decisions(slot="summary_synthesis", selected_model="ollama/qwen3:8b", n=2, base_time=fixed_now)
    audit_main(["run", "--app", "test-app", "--config", str(cfg_path)])
    rc = audit_main(["finalize", "--app", "test-app", "--config", str(cfg_path)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "judge-scores" in err
