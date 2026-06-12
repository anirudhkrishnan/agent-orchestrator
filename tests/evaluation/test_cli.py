"""CLI smoke tests for orchestrator.evaluation.

Covers the new init-baselines and prepare-batch subcommands. The `run`
command requires a live Ollama instance and is covered by integration
testing, not unit tests; we exercise the CLI parser and the wiring of the
disk-mediated steps.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from orchestrator.evaluation.batch import BaselinesFile
from orchestrator.evaluation.cli import _tally_sample_errors, main
from orchestrator.evaluation.runner import CandidateRun, CandidateSample


def _seed_run_dir(tmp_path: Path) -> Path:
    """Minimal run directory for CLI tests — manifest + one candidate run."""
    run_dir = tmp_path / "2026-05-26T00-00-00Z"
    run_dir.mkdir()
    manifest = {
        "started_at": "2026-05-26T00:00:00+00:00",
        "completed_at": "2026-05-26T00:05:00+00:00",
        "ollama_url": "http://localhost:11434",
        "keep_alive_seconds": 1800,
        "candidates": ["ollama/qwen3:8b"],
        "tasks": [
            {
                "id": "tk1",
                "description": "Task 1.",
                "system_prompt": "sys",
                "max_response_tokens": 256,
                "quality_dimensions": [
                    {"name": "accuracy", "description": "x", "weight": 1.0},
                ],
                "scenarios": [
                    {"id": "s1", "input": "in", "notes": None, "expected_output_shape": None},
                    {"id": "s2", "input": "in2", "notes": None, "expected_output_shape": None},
                ],
            }
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    cand_dir = run_dir / "ollama-qwen3-8b"
    cand_dir.mkdir()
    for sid in ("s1", "s2"):
        run = CandidateRun(
            candidate="ollama/qwen3:8b",
            task_id="tk1",
            scenario_id=sid,
            output_text="output",
            latency_ms=1_500,
            error=None,
            completed_at=datetime.now(timezone.utc),
        )
        (cand_dir / f"tk1_{sid}.json").write_text(run.model_dump_json())
    return run_dir


# --- help / no-arg ----------------------------------------------------------


def test_cli_no_args_prints_help_returns_zero(capsys):
    """Running with no subcommand prints help and exits 0 (parity with argparse default)."""
    rc = main([])
    assert rc == 0
    out = capsys.readouterr().out
    # The four subcommands should all be mentioned in the help text.
    assert "run" in out
    assert "init-baselines" in out
    assert "prepare-batch" in out
    assert "finalize" in out


def test_cli_help_flag(capsys):
    """`--help` exits via SystemExit but with code 0."""
    try:
        main(["--help"])
    except SystemExit as e:
        assert e.code == 0
    out = capsys.readouterr().out
    assert "init-baselines" in out
    assert "prepare-batch" in out


# --- init-baselines --------------------------------------------------------


def test_cli_init_baselines_creates_skeleton(tmp_path: Path, capsys):
    run_dir = _seed_run_dir(tmp_path)
    rc = main(["init-baselines", str(run_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "wrote skeleton" in out
    skeleton_path = run_dir / "baselines.json"
    assert skeleton_path.exists()
    parsed = BaselinesFile.model_validate_json(skeleton_path.read_text())
    # All keys present and empty.
    assert parsed.baselines == {"tk1": {"s1": "", "s2": ""}}


def test_cli_init_baselines_missing_run_dir_returns_2(tmp_path: Path, capsys):
    rc = main(["init-baselines", str(tmp_path / "does-not-exist")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "does not exist" in err


def test_cli_init_baselines_refuses_existing_without_overwrite(tmp_path: Path, capsys):
    run_dir = _seed_run_dir(tmp_path)
    assert main(["init-baselines", str(run_dir)]) == 0
    rc = main(["init-baselines", str(run_dir)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "already exists" in err


def test_cli_init_baselines_overwrite_succeeds(tmp_path: Path):
    run_dir = _seed_run_dir(tmp_path)
    assert main(["init-baselines", str(run_dir)]) == 0
    # Mutate the file, then re-run with --overwrite.
    p = run_dir / "baselines.json"
    data = json.loads(p.read_text())
    data["baselines"]["tk1"]["s1"] = "filled in"
    p.write_text(json.dumps(data))
    assert main(["init-baselines", str(run_dir), "--overwrite"]) == 0
    parsed = BaselinesFile.model_validate_json(p.read_text())
    assert parsed.baselines["tk1"]["s1"] == ""


def test_cli_init_baselines_records_custom_judge(tmp_path: Path):
    run_dir = _seed_run_dir(tmp_path)
    assert main(["init-baselines", str(run_dir), "--judge", "my-custom-judge"]) == 0
    parsed = BaselinesFile.model_validate_json((run_dir / "baselines.json").read_text())
    assert parsed.judge_name == "my-custom-judge"


# --- prepare-batch ---------------------------------------------------------


def test_cli_prepare_batch_writes_judge_batch(tmp_path: Path, capsys):
    run_dir = _seed_run_dir(tmp_path)
    rc = main(["prepare-batch", str(run_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "READY FOR JUDGE" in out
    assert (run_dir / "judge-batch.json").exists()


def test_cli_prepare_batch_missing_run_dir_returns_2(tmp_path: Path, capsys):
    rc = main(["prepare-batch", str(tmp_path / "missing")])
    assert rc == 2


def test_cli_prepare_batch_flags_no_baselines_when_absent(tmp_path: Path, capsys):
    run_dir = _seed_run_dir(tmp_path)
    rc = main(["prepare-batch", str(run_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "no baselines.json found" in out


def test_cli_prepare_batch_acknowledges_baselines_when_present(tmp_path: Path, capsys):
    run_dir = _seed_run_dir(tmp_path)
    # Hand-roll a filled baselines.json.
    baselines = BaselinesFile(
        judge_name="frontier-judge",
        produced_at="2026-05-26T01:00:00+00:00",
        baselines={"tk1": {"s1": "answer1", "s2": "answer2"}},
    )
    (run_dir / "baselines.json").write_text(baselines.model_dump_json(indent=2))
    rc = main(["prepare-batch", str(run_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "baselines included" in out


# --- run (error tally) ------------------------------------------------------


_TASKS_YAML = """\
- id: tk1
  description: Task 1.
  system_prompt: sys
  max_response_tokens: 256
  quality_dimensions:
    - name: accuracy
      description: x
      weight: 1.0
  scenarios:
    - id: s1
      input: in
"""


def _seed_sampled_run_dir(tmp_path: Path, errors: list[str | None]) -> Path:
    """Run dir with a single cell whose N samples carry the given errors."""
    run_dir = tmp_path / "runs" / "2026-06-12T00-00-00Z"
    cand_dir = run_dir / "ollama-qwen3-8b"
    cand_dir.mkdir(parents=True)
    run = CandidateRun(
        candidate="ollama/qwen3:8b",
        task_id="tk1",
        scenario_id="s1",
        samples=[
            CandidateSample(
                output_text="" if err else "out",
                latency_ms=0 if err else 1_000,
                error=err,
            )
            for err in errors
        ],
        output_text="" if errors[0] else "out",
        latency_ms=0 if errors[0] else 1_000,
        error=errors[0],
        completed_at=datetime.now(timezone.utc),
    )
    (cand_dir / "tk1_s1.json").write_text(run.model_dump_json())
    return run_dir


def test_tally_sample_errors_counts_per_sample_and_legacy():
    """Per-sample errors are each counted once; legacy (no-samples) cells fall
    back to the flat error field."""
    sampled = CandidateRun(
        candidate="ollama/qwen3:8b",
        task_id="tk1",
        scenario_id="s1",
        samples=[
            CandidateSample(output_text="out", latency_ms=1_000, error=None),
            CandidateSample(output_text="", latency_ms=0, error="TimeoutError: x"),
        ],
        completed_at=datetime.now(timezone.utc),
    )
    legacy = CandidateRun(
        candidate="ollama/qwen3:8b",
        task_id="tk1",
        scenario_id="s2",
        output_text="",
        latency_ms=0,
        error="model_load_failed: boom",
        completed_at=datetime.now(timezone.utc),
    )
    assert _tally_sample_errors([sampled, legacy]) == (3, 2)


def test_cli_run_exits_1_when_all_samples_errored(tmp_path: Path, capsys, monkeypatch):
    """Regression: `run` used to print 'Phase 1 complete' + exit 0 even when
    every model call failed (e.g. Ollama down)."""
    tasks_yaml = tmp_path / "tasks.yaml"
    tasks_yaml.write_text(_TASKS_YAML)
    run_dir = _seed_sampled_run_dir(tmp_path, ["TimeoutError: x"] * 3)
    monkeypatch.setattr(
        "orchestrator.evaluation.cli.run_evaluation_sync", lambda **kwargs: run_dir
    )
    rc = main([
        "run", "--tasks", str(tasks_yaml),
        "--candidates", "ollama/qwen3:8b",
        "--out-dir", str(tmp_path / "runs"),
    ])
    assert rc == 1
    captured = capsys.readouterr()
    assert "sample errors: 3/3" in captured.out
    assert "every sample in this run errored" in captured.err
    assert "Phase 1 complete" not in captured.out


def test_cli_run_warns_on_partial_errors_but_succeeds(tmp_path: Path, capsys, monkeypatch):
    """Some (not all) samples errored → loud warning, exit 0, count printed."""
    tasks_yaml = tmp_path / "tasks.yaml"
    tasks_yaml.write_text(_TASKS_YAML)
    run_dir = _seed_sampled_run_dir(tmp_path, ["TimeoutError: x", None, None])
    monkeypatch.setattr(
        "orchestrator.evaluation.cli.run_evaluation_sync", lambda **kwargs: run_dir
    )
    rc = main([
        "run", "--tasks", str(tasks_yaml),
        "--candidates", "ollama/qwen3:8b",
        "--out-dir", str(tmp_path / "runs"),
    ])
    assert rc == 0
    captured = capsys.readouterr()
    assert "sample errors: 1/3" in captured.out
    assert "WARNING" in captured.err
    assert "Phase 1 complete" in captured.out


def test_cli_run_clean_run_reports_zero_errors(tmp_path: Path, capsys, monkeypatch):
    tasks_yaml = tmp_path / "tasks.yaml"
    tasks_yaml.write_text(_TASKS_YAML)
    run_dir = _seed_sampled_run_dir(tmp_path, [None, None, None])
    monkeypatch.setattr(
        "orchestrator.evaluation.cli.run_evaluation_sync", lambda **kwargs: run_dir
    )
    rc = main([
        "run", "--tasks", str(tasks_yaml),
        "--candidates", "ollama/qwen3:8b",
        "--out-dir", str(tmp_path / "runs"),
    ])
    assert rc == 0
    captured = capsys.readouterr()
    assert "sample errors: 0/3" in captured.out
    assert "WARNING" not in captured.err
    assert "Phase 1 complete" in captured.out


# --- finalize --------------------------------------------------------------


def test_cli_finalize_missing_run_dir_returns_2(tmp_path: Path, capsys):
    rc = main(["finalize", str(tmp_path / "missing")])
    assert rc == 2


def test_cli_finalize_full_flow(tmp_path: Path):
    """End-to-end: seed run, init-baselines, prepare-batch, write scores, finalize."""
    run_dir = _seed_run_dir(tmp_path)
    # Skip init-baselines and write a filled baselines.json directly.
    baselines = BaselinesFile(
        judge_name="frontier-judge",
        produced_at="2026-05-26T01:00:00+00:00",
        baselines={"tk1": {"s1": "answer1", "s2": "answer2"}},
    )
    (run_dir / "baselines.json").write_text(baselines.model_dump_json(indent=2))
    assert main(["prepare-batch", str(run_dir)]) == 0
    # Synthesize judge scores (v2 shape with baselines).
    judge_scores = [
        {
            "item_id": "ollama/qwen3:8b::tk1::s1",
            "baseline_scores": {
                "scores": {"accuracy": 95},
                "mean_quality_score": 95.0,
                "notes": "clean",
            },
            "candidate_scores": {
                "scores": {"accuracy": 80},
                "mean_quality_score": 80.0,
                "notes": "ok",
            },
        },
        {
            "item_id": "ollama/qwen3:8b::tk1::s2",
            "baseline_scores": {
                "scores": {"accuracy": 90},
                "mean_quality_score": 90.0,
                "notes": "good",
            },
            "candidate_scores": {
                "scores": {"accuracy": 75},
                "mean_quality_score": 75.0,
                "notes": "decent",
            },
        },
    ]
    (run_dir / "judge-scores.json").write_text(json.dumps(judge_scores))
    rc = main(["finalize", str(run_dir)])
    assert rc == 0
    body = (run_dir / "REPORT.md").read_text()
    assert "% of Judge" in body
    assert "Overall Delegation Matrix" in body


def _seed_scores_without_batch(tmp_path: Path) -> Path:
    """Run dir with judge-scores.json present but NO judge-batch.json."""
    run_dir = _seed_run_dir(tmp_path)
    judge_scores = [
        {
            "item_id": f"ollama/qwen3:8b::tk1::{sid}",
            "scores": {"accuracy": 80},
            "mean_quality_score": 80.0,
            "notes": "ok",
        }
        for sid in ("s1", "s2")
    ]
    (run_dir / "judge-scores.json").write_text(json.dumps(judge_scores))
    assert not (run_dir / "judge-batch.json").exists()
    return run_dir


def test_cli_finalize_judge_pending_exits_2_no_traceback(tmp_path: Path, capsys):
    """Regression: finalize on a run dir whose judge step hasn't produced
    judge-scores.json used to crash with a raw FileNotFoundError traceback —
    the exact state a new user reaches following the quickstart top-to-bottom.
    Now it must exit 2 with a message pointing at the judge step."""
    run_dir = _seed_run_dir(tmp_path)
    assert not (run_dir / "judge-scores.json").exists()
    rc = main(["finalize", str(run_dir)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "judge-scores.json missing" in err
    assert "judge" in err
    assert not (run_dir / "REPORT.md").exists()


def test_cli_finalize_fails_closed_when_judge_batch_missing(tmp_path: Path, capsys):
    """Regression: the integrity gate used to be silently SKIPPED when
    judge-scores.json existed without judge-batch.json — unverifiable scores
    sailed straight into REPORT.md. Now finalize must refuse."""
    run_dir = _seed_scores_without_batch(tmp_path)
    rc = main(["finalize", str(run_dir)])
    assert rc == 3
    err = capsys.readouterr().err
    assert "cannot verify score integrity" in err
    assert "judge-batch.json" in err
    assert not (run_dir / "REPORT.md").exists()


def test_cli_finalize_skip_integrity_warns_and_proceeds(tmp_path: Path, capsys):
    """--skip-integrity overrides the fail-closed gate with a loud warning."""
    run_dir = _seed_scores_without_batch(tmp_path)
    rc = main(["finalize", str(run_dir), "--skip-integrity"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "--skip-integrity" in err
    assert (run_dir / "REPORT.md").exists()
