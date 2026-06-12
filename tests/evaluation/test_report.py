"""End-to-end tests for evaluation.report.generate_report on synthetic data.

Covers both v1 (no-baseline) and v2 (baseline-present) judge-scores shapes.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from orchestrator.evaluation.report import generate_report
from orchestrator.evaluation.runner import CandidateRun


def _seed_full_run(tmp_path: Path) -> Path:
    """Synthesize a complete run dir: manifest + runs + judge-scores.json (v1 shape, no baseline)."""
    run_dir = tmp_path / "2026-05-26T00-00-00Z"
    run_dir.mkdir()
    manifest = {
        "started_at": "2026-05-26T00:00:00+00:00",
        "completed_at": "2026-05-26T00:05:00+00:00",
        "ollama_url": "http://localhost:11434",
        "keep_alive_seconds": 1800,
        "judge_model": "frontier-judge",
        "candidates": ["ollama/qwen3:8b", "ollama/gemma4:e4b"],
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
                    {"id": "s1", "input": "in", "notes": None, "expected_output_shape": None}
                ],
            }
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))

    for cand, dir_name, latency in [
        ("ollama/qwen3:8b", "ollama-qwen3-8b", 1_500),
        ("ollama/gemma4:e4b", "ollama-gemma4-e4b", 12_000),
    ]:
        run = CandidateRun(
            candidate=cand,
            task_id="tk1",
            scenario_id="s1",
            output_text="x",
            latency_ms=latency,
            error=None,
            completed_at=datetime.now(timezone.utc),
        )
        (run_dir / dir_name).mkdir()
        (run_dir / dir_name / "tk1_s1.json").write_text(run.model_dump_json())

    # Judge scores: qwen3:8b gets 80, gemma4:e4b gets 100 (v1 flat shape, no baselines).
    judge_scores = [
        {
            "item_id": "ollama/qwen3:8b::tk1::s1",
            "scores": {"accuracy": 80},
            "mean_quality_score": 80.0,
            "notes": "good but missed one nuance",
        },
        {
            "item_id": "ollama/gemma4:e4b::tk1::s1",
            "scores": {"accuracy": 100},
            "mean_quality_score": 100.0,
            "notes": "excellent",
        },
    ]
    (run_dir / "judge-scores.json").write_text(json.dumps(judge_scores))
    return run_dir


def _seed_full_run_with_baselines(tmp_path: Path) -> Path:
    """Synthesize a complete run dir with v2 baseline-aware judge-scores."""
    run_dir = tmp_path / "2026-05-26T01-00-00Z"
    run_dir.mkdir()
    manifest = {
        "started_at": "2026-05-26T01:00:00+00:00",
        "completed_at": "2026-05-26T01:05:00+00:00",
        "ollama_url": "http://localhost:11434",
        "keep_alive_seconds": 1800,
        "judge_model": "frontier-judge",
        "candidates": ["ollama/qwen3:8b", "ollama/gemma4:e4b"],
        "tasks": [
            {
                "id": "entity_extraction",
                "description": "Pull named entities.",
                "system_prompt": "sys",
                "max_response_tokens": 256,
                "quality_dimensions": [
                    {"name": "accuracy", "description": "x", "weight": 1.0},
                ],
                "scenarios": [
                    {"id": "s1", "input": "in", "notes": None, "expected_output_shape": None}
                ],
            }
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))

    for cand, dir_name, latency in [
        ("ollama/qwen3:8b", "ollama-qwen3-8b", 1_500),
        ("ollama/gemma4:e4b", "ollama-gemma4-e4b", 12_000),
    ]:
        run = CandidateRun(
            candidate=cand,
            task_id="entity_extraction",
            scenario_id="s1",
            output_text="x",
            latency_ms=latency,
            error=None,
            completed_at=datetime.now(timezone.utc),
        )
        (run_dir / dir_name).mkdir()
        (run_dir / dir_name / "entity_extraction_s1.json").write_text(run.model_dump_json())

    # v2 nested shape with baseline + candidate scores.
    # Judge scores its own baseline at 97; qwen3 86, gemma 50.
    # → qwen3 = 88.7% of judge, gemma = 51.5% of judge.
    judge_scores = [
        {
            "item_id": "ollama/qwen3:8b::entity_extraction::s1",
            "baseline_scores": {
                "scores": {"accuracy": 97},
                "mean_quality_score": 97.0,
                "notes": "baseline got 4 of 5 entities cleanly",
            },
            "candidate_scores": {
                "scores": {"accuracy": 86},
                "mean_quality_score": 86.0,
                "notes": "missed one entity; otherwise clean",
            },
        },
        {
            "item_id": "ollama/gemma4:e4b::entity_extraction::s1",
            "baseline_scores": {
                "scores": {"accuracy": 97},
                "mean_quality_score": 97.0,
                "notes": "baseline same as above",
            },
            "candidate_scores": {
                "scores": {"accuracy": 50},
                "mean_quality_score": 50.0,
                "notes": "missed three entities",
            },
        },
    ]
    (run_dir / "judge-scores.json").write_text(json.dumps(judge_scores))
    return run_dir


# --- v1 (no baseline) tests ----------------------------------------------


def test_generate_report_writes_markdown(tmp_path: Path):
    run_dir = _seed_full_run(tmp_path)
    report_path = generate_report(run_dir)
    assert report_path == run_dir / "REPORT.md"
    body = report_path.read_text()
    # Header bits.
    assert "Evaluation Report" in body
    assert run_dir.name in body
    # Per-task table.
    assert "## Per-task leaderboards" in body
    assert "### tk1" in body
    # Both candidates appear.
    assert "ollama/qwen3:8b" in body
    assert "ollama/gemma4:e4b" in body
    # Routing block exists.
    assert "## Suggested routing.json update" in body
    assert "```json" in body


def test_generate_report_omits_baseline_column_when_absent(tmp_path: Path):
    run_dir = _seed_full_run(tmp_path)
    generate_report(run_dir)
    body = (run_dir / "REPORT.md").read_text()
    # The column header should NOT include "% of Judge".
    assert "% of Judge" not in body
    # And the delegation matrix should not appear without baselines.
    assert "Overall Delegation Matrix" not in body
    # Baselines: absent should be called out.
    assert "Baselines:** absent" in body


def test_generate_report_picks_winner_balancing_quality_and_speed(tmp_path: Path):
    """qwen3:8b: quality 80 + speed 90 (1.5s) → combined 0.7*80+0.3*90 = 83
    gemma4:e4b: quality 100 + speed 60 (12s in 10-20s bin) → combined 0.7*100+0.3*60 = 88
    Default weighting still picks gemma4.
    """
    run_dir = _seed_full_run(tmp_path)
    generate_report(run_dir)
    body = (run_dir / "REPORT.md").read_text()
    assert "**Winner:** `ollama/gemma4:e4b`" in body


def test_generate_report_routing_update_uses_winner(tmp_path: Path):
    run_dir = _seed_full_run(tmp_path)
    generate_report(run_dir)
    body = (run_dir / "REPORT.md").read_text()
    start = body.index("```json")
    end = body.index("```", start + 7)
    json_block = body[start + 7 : end].strip()
    parsed = json.loads(json_block)
    assert "tk1" in parsed
    assert parsed["tk1"]["model"] == "ollama/gemma4:e4b"


def test_generate_report_quality_weight_can_flip_winner(tmp_path: Path):
    """At quality_weight=1.0 (speed ignored), gemma's higher quality always wins."""
    run_dir = _seed_full_run(tmp_path)
    generate_report(run_dir, quality_weight=1.0)
    body = (run_dir / "REPORT.md").read_text()
    assert "**Winner:** `ollama/gemma4:e4b`" in body


def test_generate_report_quality_weight_zero_picks_fastest(tmp_path: Path):
    """At quality_weight=0.0, qwen3:8b at 1.5s (speed=90) beats gemma at 12s (speed=60)."""
    run_dir = _seed_full_run(tmp_path)
    generate_report(run_dir, quality_weight=0.0)
    body = (run_dir / "REPORT.md").read_text()
    assert "**Winner:** `ollama/qwen3:8b`" in body


# --- stdev / ⚡ / routing stdev (README Aggregation contract) ---------------


def _seed_run_with_variance(tmp_path: Path) -> Path:
    """1 task × 2 scenarios. qwen scores 60 & 90 (stdev 15 → ⚡); gemma scores
    80 & 82 (stdev 1 → no flag)."""
    run_dir = tmp_path / "2026-05-26T02-00-00Z"
    run_dir.mkdir()
    manifest = {
        "started_at": "2026-05-26T02:00:00+00:00",
        "completed_at": "2026-05-26T02:05:00+00:00",
        "ollama_url": "http://localhost:11434",
        "keep_alive_seconds": 1800,
        "judge_model": "judge-x",
        "candidates": ["ollama/qwen3:8b", "ollama/gemma4:e4b"],
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
    for cand, dir_name in [
        ("ollama/qwen3:8b", "ollama-qwen3-8b"),
        ("ollama/gemma4:e4b", "ollama-gemma4-e4b"),
    ]:
        (run_dir / dir_name).mkdir()
        for sid in ("s1", "s2"):
            run = CandidateRun(
                candidate=cand,
                task_id="tk1",
                scenario_id=sid,
                output_text="x",
                latency_ms=1_500,
                error=None,
                completed_at=datetime.now(timezone.utc),
            )
            (run_dir / dir_name / f"tk1_{sid}.json").write_text(run.model_dump_json())
    judge_scores = []
    for cand, qualities in [
        ("ollama/qwen3:8b", [60.0, 90.0]),
        ("ollama/gemma4:e4b", [80.0, 82.0]),
    ]:
        for sid, q in zip(("s1", "s2"), qualities):
            judge_scores.append(
                {
                    "item_id": f"{cand}::tk1::{sid}",
                    "scores": {"accuracy": q},
                    "mean_quality_score": q,
                    "notes": "n",
                }
            )
    (run_dir / "judge-scores.json").write_text(json.dumps(judge_scores))
    return run_dir


def test_generate_report_renders_stdev_column_with_flag(tmp_path: Path):
    """Leaderboard carries a Stdev column; stdev > 10 is ⚡-flagged."""
    run_dir = _seed_run_with_variance(tmp_path)
    generate_report(run_dir)
    body = (run_dir / "REPORT.md").read_text()
    assert "| Candidate | Quality | Stdev | Speed | Combined |" in body
    # qwen: pstdev([60, 90]) = 15 → flagged.
    qwen_row = next(ln for ln in body.splitlines() if ln.startswith("| ollama/qwen3:8b"))
    assert "15.0 ⚡" in qwen_row
    # gemma: pstdev([80, 82]) = 1 → NOT flagged.
    gemma_row = next(ln for ln in body.splitlines() if ln.startswith("| ollama/gemma4:e4b"))
    assert "| 1.0 " in gemma_row
    assert "⚡" not in gemma_row


def test_generate_report_routing_block_includes_stdev(tmp_path: Path):
    """The routing.json suggestion carries stdev_quality for the winner."""
    run_dir = _seed_run_with_variance(tmp_path)
    generate_report(run_dir)
    body = (run_dir / "REPORT.md").read_text()
    start = body.index("```json")
    end = body.index("```", start + 7)
    parsed = json.loads(body[start + 7 : end].strip())
    assert "stdev_quality" in parsed["tk1"]


# --- v2 (with baseline) tests --------------------------------------------


def test_generate_report_includes_baseline_column_when_present(tmp_path: Path):
    run_dir = _seed_full_run_with_baselines(tmp_path)
    generate_report(run_dir)
    body = (run_dir / "REPORT.md").read_text()
    # Per-task leaderboard now has the "% of Judge" column.
    assert "% of Judge" in body
    # And the delegation matrix.
    assert "Overall Delegation Matrix" in body
    # Mention baselines present in the header.
    assert "Baselines:** present" in body


def test_generate_report_computes_pct_of_judge_correctly(tmp_path: Path):
    """qwen3:8b = 86/97 ≈ 88.7% → rounds to 89%.

    Scans inside the per-task leaderboard section to avoid matching the
    candidates list in the report header.
    """
    run_dir = _seed_full_run_with_baselines(tmp_path)
    generate_report(run_dir)
    body = (run_dir / "REPORT.md").read_text()
    # Slice to the per-task section so we don't pick up header artifacts.
    section_start = body.index("## Per-task leaderboards")
    section = body[section_start:]
    # Locate qwen3 row in the table; expect "89%" with rounding (86/97 ≈ 88.66 → 89).
    qwen_row_idx = section.index("ollama/qwen3:8b")
    qwen_row = section[qwen_row_idx : qwen_row_idx + 200]
    assert "89%" in qwen_row
    # Gemma row: 50/97 ≈ 51.5% → 52%.
    gemma_row_idx = section.index("ollama/gemma4:e4b")
    gemma_row = section[gemma_row_idx : gemma_row_idx + 200]
    assert "52%" in gemma_row


def test_generate_report_delegation_matrix_classifies_actions(tmp_path: Path):
    """Winner is qwen (combined 0.7*86 + 0.3*90 = 87.2). 89% of judge → Delegate freely."""
    run_dir = _seed_full_run_with_baselines(tmp_path)
    generate_report(run_dir)
    body = (run_dir / "REPORT.md").read_text()
    # The delegation matrix should classify the winner of entity_extraction as
    # "Delegate freely" (winner qwen at 89% of judge, above 80% threshold).
    matrix_idx = body.index("Overall Delegation Matrix")
    matrix_section = body[matrix_idx:]
    assert "Delegate freely" in matrix_section
    assert "entity_extraction" in matrix_section


def test_generate_report_delegation_callout_present(tmp_path: Path):
    """A one-line _Insight:_ callout follows each per-task leaderboard."""
    run_dir = _seed_full_run_with_baselines(tmp_path)
    generate_report(run_dir)
    body = (run_dir / "REPORT.md").read_text()
    assert "_Insight:" in body


def test_generate_report_routing_update_includes_pct_of_judge(tmp_path: Path):
    """When baselines present, the routing.json suggestion gets a quality_pct_of_judge key."""
    run_dir = _seed_full_run_with_baselines(tmp_path)
    generate_report(run_dir)
    body = (run_dir / "REPORT.md").read_text()
    start = body.index("```json")
    end = body.index("```", start + 7)
    json_block = body[start + 7 : end].strip()
    parsed = json.loads(json_block)
    assert "entity_extraction" in parsed
    assert "quality_pct_of_judge" in parsed["entity_extraction"]
