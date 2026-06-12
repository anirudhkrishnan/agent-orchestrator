"""Tests for the improve CLI: data-dir resolution, audit-scores wiring, and
actionable errors for corrupt state files."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from orchestrator.improve.cli import main
from orchestrator.telemetry import db as tdb


@pytest.fixture
def tmp_env(tmp_path: Path, monkeypatch):
    """Isolated telemetry DB + cwd; no ORCHESTRATOR_DATA_DIR leaking in."""
    monkeypatch.setenv("ORCHESTRATOR_DB_PATH", str(tmp_path / "t.sqlite"))
    monkeypatch.delenv("ORCHESTRATOR_DATA_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    tdb.init_db()
    return tmp_path


# ----------------------------- data-dir resolution -------------------------

def test_data_dir_defaults_to_cwd(tmp_env: Path):
    # Installed console scripts must NOT resolve data into the package install
    # location (site-packages) — the default is ./data under the cwd.
    assert main(["harvest", "--app", "demo"]) == 0
    assert (tmp_env / "data" / "improve" / "staged-scenarios.json").exists()
    assert (tmp_env / "data" / "improve" / "rebake-queue.json").exists()


def test_data_dir_env_override(tmp_env: Path, tmp_path_factory, monkeypatch):
    env_dir = tmp_path_factory.mktemp("envdata")
    monkeypatch.setenv("ORCHESTRATOR_DATA_DIR", str(env_dir))
    assert main(["harvest", "--app", "demo"]) == 0
    assert (env_dir / "improve" / "staged-scenarios.json").exists()
    assert not (tmp_env / "data").exists()  # cwd default not used


def test_data_dir_flag_beats_env(tmp_env: Path, tmp_path_factory, monkeypatch):
    env_dir = tmp_path_factory.mktemp("envdata")
    flag_dir = tmp_path_factory.mktemp("flagdata")
    monkeypatch.setenv("ORCHESTRATOR_DATA_DIR", str(env_dir))
    assert main(["harvest", "--app", "demo", "--data-dir", str(flag_dir)]) == 0
    assert (flag_dir / "improve" / "staged-scenarios.json").exists()
    assert not (env_dir / "improve").exists()


# ----------------------------- audit-scores wiring -------------------------

def test_harvest_audit_scores_flag(tmp_env: Path):
    # --audit-scores wires the audit-quality judge scores into Loop A's second
    # failure source (it is unreachable from the CLI without this).
    now = datetime.now(timezone.utc)  # CLI harvest windows off real "now"
    tdb.log_sample(sample_id="smp-1", app_name="demo", slot="summary_synthesis",
                   candidate_model="ollama/qwen3:8b", input_text="summarize this",
                   output_text="meh", latency_ms=50, routed_at=now.isoformat())
    scores_file = tmp_env / "judge-scores.json"
    scores_file.write_text(json.dumps(
        [{"sample_id": "smp-1", "mean_quality_score": 41.5, "notes": "weak"}]))
    assert main(["harvest", "--app", "demo", "--audit-scores", str(scores_file)]) == 0
    staged = json.loads(
        (tmp_env / "data" / "improve" / "staged-scenarios.json").read_text()
    )["candidate_scenarios"]
    assert len(staged) == 1
    assert staged[0]["provenance"]["reason"] == "below_rebake_threshold"
    assert staged[0]["provenance"]["failure_id"] == "routed_call_samples:smp-1"


# ----------------------------- corrupt state files -------------------------

def test_corrupt_staging_file_is_actionable(tmp_env: Path, capsys):
    bad = tmp_env / "data" / "improve" / "staged-scenarios.json"
    bad.parent.mkdir(parents=True)
    bad.write_text("{not json")
    rc = main(["harvest", "--app", "demo"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "staged-scenarios.json" in err   # which file
    assert "invalid JSON" in err            # what's wrong
    assert "re-run harvest" in err          # how to reset


def test_corrupt_routing_file_is_actionable(tmp_env: Path, capsys):
    (tmp_env / "data").mkdir()
    (tmp_env / "data" / "routing.json").write_text('{"slot1":')
    rc = main(["detect-models"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "routing.json" in err
    assert "invalid JSON" in err


def test_corrupt_audit_scores_is_actionable(tmp_env: Path, capsys):
    bad = tmp_env / "judge-scores.json"
    bad.write_text("[1, 2,")
    rc = main(["harvest", "--app", "demo", "--audit-scores", str(bad)])
    assert rc == 2
    assert "invalid JSON" in capsys.readouterr().err


# ----------------------------- gate-check ----------------------------------

def test_gate_check_fails_on_degenerate_run(tmp_env: Path, capsys):
    # An empty batch + empty scores must FAIL the gate — never print the
    # "data is safe" verdict.
    rd = tmp_env / "run"
    rd.mkdir()
    (rd / "judge-batch.json").write_text("{}")
    (rd / "judge-scores.json").write_text("[]")
    rc = main(["gate-check", "--run-dir", str(rd)])
    out = capsys.readouterr()
    assert rc == 3
    assert "FAIL" in out.err
    assert "data is safe" not in out.out
