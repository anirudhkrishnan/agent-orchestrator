"""Tests for evaluation.batch — judge batch preparation, baselines skeleton."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from orchestrator.evaluation.batch import (
    BaselinesFile,
    JudgeBatch,
    init_baselines_skeleton,
    prepare_judge_batch,
)
from orchestrator.evaluation.runner import CandidateRun


def _seed_run_dir(tmp_path: Path) -> Path:
    """Build a complete synthetic run dir on disk: manifest + 2 candidate runs."""
    run_dir = tmp_path / "2026-05-26T00-00-00Z"
    run_dir.mkdir()

    manifest = {
        "started_at": "2026-05-26T00:00:00+00:00",
        "completed_at": "2026-05-26T00:05:00+00:00",
        "ollama_url": "http://localhost:11434",
        "keep_alive_seconds": 1800,
        "candidates": ["ollama/qwen3:8b", "ollama/gemma4:e4b"],
        "tasks": [
            {
                "id": "tk1",
                "description": "Task 1 description",
                "system_prompt": "You are a test assistant.",
                "max_response_tokens": 256,
                "quality_dimensions": [
                    {"name": "accuracy", "description": "accurate?", "weight": 1.0},
                ],
                "scenarios": [
                    {
                        "id": "s1",
                        "input": "What is 2+2?",
                        "notes": "trivial math",
                        "expected_output_shape": "single number",
                    }
                ],
            }
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Write two CandidateRun JSON files, one per candidate.
    for cand, dir_name, out, latency in [
        ("ollama/qwen3:8b", "ollama-qwen3-8b", "4", 1_500),
        ("ollama/gemma4:e4b", "ollama-gemma4-e4b", "four", 5_000),
    ]:
        run = CandidateRun(
            candidate=cand,
            task_id="tk1",
            scenario_id="s1",
            output_text=out,
            latency_ms=latency,
            error=None,
            completed_at=datetime.now(timezone.utc),
        )
        cand_dir = run_dir / dir_name
        cand_dir.mkdir()
        (cand_dir / "tk1_s1.json").write_text(run.model_dump_json(indent=2))
    return run_dir


# --- prepare_judge_batch (no baselines) ----------------------------------


def test_prepare_judge_batch_writes_judge_batch_json(tmp_path: Path):
    run_dir = _seed_run_dir(tmp_path)
    batch_path = prepare_judge_batch(run_dir)
    assert batch_path == run_dir / "judge-batch.json"
    assert batch_path.exists()
    batch = JudgeBatch.model_validate_json(batch_path.read_text())
    # 1 task × 1 scenario × 2 candidates = 2 items.
    # With N=5 sampling (post-2026-05-26), item_ids carry a per-sample
    # suffix. Single-sample legacy runs get "::sample-0".
    assert len(batch.items) == 2
    item_ids = sorted(i.item_id for i in batch.items)
    assert item_ids == [
        "ollama/gemma4:e4b::tk1::s1::sample-0",
        "ollama/qwen3:8b::tk1::s1::sample-0",
    ]


def test_prepare_judge_batch_carries_scenario_and_task_metadata(tmp_path: Path):
    run_dir = _seed_run_dir(tmp_path)
    batch_path = prepare_judge_batch(run_dir)
    batch = JudgeBatch.model_validate_json(batch_path.read_text())
    item = batch.items[0]
    assert item.task_description == "Task 1 description"
    assert item.scenario_input == "What is 2+2?"
    assert item.expected_output_shape == "single number"
    assert item.scenario_notes == "trivial math"
    assert len(item.quality_dimensions) == 1
    assert item.quality_dimensions[0].name == "accuracy"


def test_prepare_judge_batch_without_baselines_uses_no_baseline_instructions(tmp_path: Path):
    run_dir = _seed_run_dir(tmp_path)
    batch_path = prepare_judge_batch(run_dir)
    batch = JudgeBatch.model_validate_json(batch_path.read_text())
    instr = batch.instructions_for_judge
    assert "judge-scores.json" in instr
    assert "0-100" in instr
    assert "speed" in instr.lower()
    assert "mean_quality_score" in instr
    # No-baseline variant flags it explicitly.
    assert "NO baselines" in instr
    assert batch.baselines_present is False
    # baseline_output should be None on every item.
    for item in batch.items:
        assert item.baseline_output is None


def test_prepare_judge_batch_records_judge_model(tmp_path: Path):
    run_dir = _seed_run_dir(tmp_path)
    batch_path = prepare_judge_batch(run_dir, judge_model="custom-judge-id")
    batch = JudgeBatch.model_validate_json(batch_path.read_text())
    assert batch.judge_model == "custom-judge-id"


def test_prepare_judge_batch_raises_on_missing_manifest(tmp_path: Path):
    run_dir = tmp_path / "empty-run"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        prepare_judge_batch(run_dir)


def test_prepare_judge_batch_equal_sized_groups_get_distinct_item_ids(tmp_path: Path):
    """Regression: two equal-sized dedup groups within one cell must NOT
    collide on item_id. Samples [A,A,B,B] used to emit two items both named
    '::group-of-2', blinding the score-integrity gate (one score silently
    overwrote the other in its per-item index)."""
    run_dir = _seed_run_dir(tmp_path)
    cell_path = run_dir / "ollama-qwen3-8b" / "tk1_s1.json"
    run = CandidateRun(
        candidate="ollama/qwen3:8b",
        task_id="tk1",
        scenario_id="s1",
        samples=[
            {"output_text": "A", "latency_ms": 1_000, "error": None},
            {"output_text": "A", "latency_ms": 1_100, "error": None},
            {"output_text": "B", "latency_ms": 1_200, "error": None},
            {"output_text": "B", "latency_ms": 1_300, "error": None},
        ],
        output_text="A",
        latency_ms=1_000,
        error=None,
        completed_at=datetime.now(timezone.utc),
    )
    cell_path.write_text(run.model_dump_json(indent=2))
    batch_path = prepare_judge_batch(run_dir)
    batch = JudgeBatch.model_validate_json(batch_path.read_text())
    qwen_ids = [i.item_id for i in batch.items if i.candidate == "ollama/qwen3:8b"]
    assert len(qwen_ids) == 2
    assert len(set(qwen_ids)) == 2, f"colliding item_ids: {qwen_ids}"
    # Each id carries the group size + the representative sample index.
    assert sorted(qwen_ids) == [
        "ollama/qwen3:8b::tk1::s1::group-of-2-s0",
        "ollama/qwen3:8b::tk1::s1::group-of-2-s2",
    ]
    # sample_count still reflects the multiplicity for aggregation weighting.
    for item in batch.items:
        if item.candidate == "ollama/qwen3:8b":
            assert item.sample_count == 2


# --- init_baselines_skeleton ---------------------------------------------


def test_init_baselines_writes_skeleton_with_all_keys(tmp_path: Path):
    run_dir = _seed_run_dir(tmp_path)
    path = init_baselines_skeleton(run_dir)
    assert path == run_dir / "baselines.json"
    assert path.exists()
    parsed = BaselinesFile.model_validate_json(path.read_text())
    assert parsed.judge_name == "interactive-judge-session"
    # Every (task, scenario) key present with an empty string value.
    assert parsed.baselines == {"tk1": {"s1": ""}}


def test_init_baselines_records_custom_judge_name(tmp_path: Path):
    run_dir = _seed_run_dir(tmp_path)
    path = init_baselines_skeleton(run_dir, judge_name="my-judge")
    parsed = BaselinesFile.model_validate_json(path.read_text())
    assert parsed.judge_name == "my-judge"


def test_init_baselines_refuses_to_overwrite_by_default(tmp_path: Path):
    run_dir = _seed_run_dir(tmp_path)
    init_baselines_skeleton(run_dir)
    with pytest.raises(FileExistsError, match="already exists"):
        init_baselines_skeleton(run_dir)


def test_init_baselines_overwrites_when_flag_set(tmp_path: Path):
    run_dir = _seed_run_dir(tmp_path)
    init_baselines_skeleton(run_dir)
    # User fills in a value, then we overwrite — should clear back to skeleton.
    p = run_dir / "baselines.json"
    raw = json.loads(p.read_text())
    raw["baselines"]["tk1"]["s1"] = "the answer is 4"
    p.write_text(json.dumps(raw))
    init_baselines_skeleton(run_dir, overwrite=True)
    parsed = BaselinesFile.model_validate_json(p.read_text())
    assert parsed.baselines["tk1"]["s1"] == ""


def test_init_baselines_raises_on_missing_manifest(tmp_path: Path):
    run_dir = tmp_path / "empty-run"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        init_baselines_skeleton(run_dir)


# --- prepare_judge_batch WITH baselines ---------------------------------


def test_prepare_judge_batch_attaches_baselines_when_present(tmp_path: Path):
    run_dir = _seed_run_dir(tmp_path)
    # Seed a filled-in baselines.json.
    baselines = BaselinesFile(
        judge_name="frontier-judge",
        produced_at="2026-05-26T01:00:00+00:00",
        baselines={"tk1": {"s1": "4"}},
    )
    (run_dir / "baselines.json").write_text(baselines.model_dump_json(indent=2))
    batch_path = prepare_judge_batch(run_dir)
    batch = JudgeBatch.model_validate_json(batch_path.read_text())
    # Both candidates' items should carry the baseline.
    assert batch.baselines_present is True
    assert len(batch.items) == 2
    for item in batch.items:
        assert item.baseline_output == "4"


def test_prepare_judge_batch_uses_with_baseline_instructions(tmp_path: Path):
    run_dir = _seed_run_dir(tmp_path)
    baselines = BaselinesFile(
        judge_name="frontier-judge",
        produced_at="2026-05-26T01:00:00+00:00",
        baselines={"tk1": {"s1": "4"}},
    )
    (run_dir / "baselines.json").write_text(baselines.model_dump_json(indent=2))
    batch_path = prepare_judge_batch(run_dir)
    batch = JudgeBatch.model_validate_json(batch_path.read_text())
    instr = batch.instructions_for_judge
    # The with-baseline variant has the two-scores schema and the baseline language.
    assert "baseline_scores" in instr
    assert "candidate_scores" in instr
    assert "INCLUDES BASELINES" in instr
    assert "0-100" in instr


def test_prepare_judge_batch_empty_baseline_string_treated_as_unfilled(tmp_path: Path):
    """Empty strings in baselines.json are skeleton slots, not real baselines.

    The batch should not flag these items as having a baseline (the judge
    hasn't actually produced one yet).
    """
    run_dir = _seed_run_dir(tmp_path)
    baselines = BaselinesFile(
        judge_name="frontier-judge",
        produced_at="2026-05-26T01:00:00+00:00",
        baselines={"tk1": {"s1": ""}},  # unfilled skeleton entry
    )
    (run_dir / "baselines.json").write_text(baselines.model_dump_json(indent=2))
    batch_path = prepare_judge_batch(run_dir)
    batch = JudgeBatch.model_validate_json(batch_path.read_text())
    # File exists, but no item has a real baseline → flip to no-baseline mode.
    assert batch.baselines_present is False
    for item in batch.items:
        assert item.baseline_output is None
    assert "NO baselines" in batch.instructions_for_judge


def test_prepare_judge_batch_raises_on_malformed_baselines(tmp_path: Path):
    run_dir = _seed_run_dir(tmp_path)
    (run_dir / "baselines.json").write_text("not valid json {")
    with pytest.raises(ValueError, match="Malformed baselines"):
        prepare_judge_batch(run_dir)
