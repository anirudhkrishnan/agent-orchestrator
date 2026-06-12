"""Unit tests for the tiered routing module.

All tests use synthetic in-memory data — no disk I/O, no network.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from orchestrator.tiered.cli import main as cli_main
from orchestrator.tiered.dry_run import (
    COST_WEIGHTS,
    analyze_workflow,
    classify_call,
    render_dry_run_report,
)
from orchestrator.tiered.routing_table import (
    _agg,
    _pick,
    build_routing_table,
    load_frontier_per_task,
    load_oss_per_task,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic quality dicts
# ---------------------------------------------------------------------------

def _make_frontier_q(
    task: str,
    scn: str,
    mid: float,
    low: float,
) -> dict:
    """Build a minimal frontier_q dict for one cell."""
    return {
        (task, scn, "mid_frontier"): {"median": mid, "stdev": 0.0, "n": 5},
        (task, scn, "low_frontier"): {"median": low, "stdev": 0.0, "n": 5},
    }


def _make_oss_q(task: str, scn: str, val: float, model: str = "ollama/test:7b") -> dict:
    return {
        (task, scn, "oss"): {"median": val, "stdev": 0.0, "model": model, "n": 5},
    }


# ---------------------------------------------------------------------------
# classify_call
# ---------------------------------------------------------------------------

class TestClassifyCall:
    """Tests for the per-call routing logic."""

    def test_tiered_routes_to_oss_when_both_clear(self):
        fq = _make_frontier_q("t", "s", mid=97.0, low=96.0)
        oq = _make_oss_q("t", "s", 96.0)
        tier, label, quality, weight = classify_call(
            "t", "s", fq, oq, t1_threshold=95.0, t2_threshold=95.0, mode="tiered"
        )
        assert tier == 2
        assert weight == COST_WEIGHTS["oss"]
        assert quality >= 95.0

    def test_tiered_falls_back_to_low_frontier_when_oss_below_threshold(self):
        fq = _make_frontier_q("t", "s", mid=97.0, low=96.0)
        oq = _make_oss_q("t", "s", 80.0)
        tier, label, quality, weight = classify_call(
            "t", "s", fq, oq, t1_threshold=95.0, t2_threshold=95.0, mode="tiered"
        )
        assert tier == 1
        assert label == "low_frontier"
        assert weight == COST_WEIGHTS["low_frontier"]

    def test_tiered_falls_back_to_mid_frontier_when_low_below_threshold(self):
        fq = _make_frontier_q("t", "s", mid=97.0, low=90.0)
        oq = _make_oss_q("t", "s", 80.0)
        tier, label, quality, weight = classify_call(
            "t", "s", fq, oq, t1_threshold=95.0, t2_threshold=95.0, mode="tiered"
        )
        assert tier == 1
        assert label == "mid_frontier"
        assert weight == COST_WEIGHTS["mid_frontier"]

    def test_tiered_stays_on_top_frontier_when_all_below_threshold(self):
        fq = _make_frontier_q("t", "s", mid=85.0, low=80.0)
        oq = _make_oss_q("t", "s", 75.0)
        tier, label, quality, weight = classify_call(
            "t", "s", fq, oq, t1_threshold=95.0, t2_threshold=95.0, mode="tiered"
        )
        assert tier == 0
        assert label == "top_frontier"
        assert weight == COST_WEIGHTS["top_frontier"]

    def test_frontier_only_never_picks_oss(self):
        fq = _make_frontier_q("t", "s", mid=80.0, low=97.0)
        oq = _make_oss_q("t", "s", 99.0)
        tier, label, quality, weight = classify_call(
            "t", "s", fq, oq, t1_threshold=95.0, t2_threshold=95.0, mode="frontier_only"
        )
        assert tier != 2
        assert weight != COST_WEIGHTS["oss"]

    def test_oss_only_picks_oss_when_it_clears_threshold(self):
        fq = _make_frontier_q("t", "s", mid=99.0, low=99.0)
        oq = _make_oss_q("t", "s", 96.0)
        tier, label, quality, weight = classify_call(
            "t", "s", fq, oq, t1_threshold=95.0, t2_threshold=95.0, mode="oss_only"
        )
        assert tier == 2
        assert weight == COST_WEIGHTS["oss"]

    def test_oss_only_falls_back_to_top_frontier_when_oss_below_threshold(self):
        fq = _make_frontier_q("t", "s", mid=99.0, low=99.0)
        oq = _make_oss_q("t", "s", 80.0)
        tier, label, quality, weight = classify_call(
            "t", "s", fq, oq, t1_threshold=95.0, t2_threshold=95.0, mode="oss_only"
        )
        assert tier == 0
        assert weight == COST_WEIGHTS["top_frontier"]

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            classify_call("t", "s", {}, {}, 95.0, 95.0, mode="bogus")

    def test_oss_only_gates_on_t2_threshold_not_t1(self):
        # OSS delegation is a Tier-2 decision: an OSS score that clears
        # t2_threshold must delegate even when t1_threshold is stricter.
        fq = _make_frontier_q("t", "s", mid=99.0, low=99.0)
        oq = _make_oss_q("t", "s", 95.0)
        tier, label, quality, weight = classify_call(
            "t", "s", fq, oq, t1_threshold=99.0, t2_threshold=90.0, mode="oss_only"
        )
        assert tier == 2
        assert weight == COST_WEIGHTS["oss"]

    def test_oss_only_respects_strict_t2_threshold(self):
        # Conversely, a lenient t1_threshold must NOT leak into the OSS gate.
        fq = _make_frontier_q("t", "s", mid=99.0, low=99.0)
        oq = _make_oss_q("t", "s", 95.0)
        tier, label, quality, weight = classify_call(
            "t", "s", fq, oq, t1_threshold=90.0, t2_threshold=99.0, mode="oss_only"
        )
        assert tier == 0
        assert label == "top_frontier"


# ---------------------------------------------------------------------------
# analyze_workflow
# ---------------------------------------------------------------------------

class TestAnalyzeWorkflow:
    """Tests for per-workflow cost/quality analysis."""

    def _make_perfect_data(self, task: str, scn: str):
        fq = _make_frontier_q(task, scn, mid=97.0, low=96.0)
        oq = _make_oss_q(task, scn, 96.0)
        return fq, oq

    def test_returns_all_three_modes(self):
        fq, oq = self._make_perfect_data("entity_extraction", "scn-01")
        calls = [("entity_extraction", "scn-01")]
        result = analyze_workflow(calls, "test", fq, oq)
        assert set(result.keys()) == {"tiered", "frontier_only", "oss_only"}

    def test_oss_only_has_highest_savings(self):
        fq = _make_frontier_q("entity_extraction", "scn-01", mid=96.0, low=96.0)
        oq = _make_oss_q("entity_extraction", "scn-01", 96.0)
        calls = [("entity_extraction", "scn-01")]
        r = analyze_workflow(calls, "test", fq, oq)
        # OSS cost weight = 0.0, so savings should be 100%
        assert r["oss_only"]["cost_saved_pct"] == pytest.approx(100.0)

    def test_frontier_only_no_oss_delegation(self):
        fq = _make_frontier_q("entity_extraction", "scn-01", mid=97.0, low=96.0)
        oq = _make_oss_q("entity_extraction", "scn-01", 99.0)
        calls = [("entity_extraction", "scn-01")]
        r = analyze_workflow(calls, "test", fq, oq)
        assert r["frontier_only"]["tier_counts"][2] == 0

    def test_min_slot_quality_reflects_worst_call(self):
        fq = {
            ("t1", "s", "low_frontier"): {"median": 97.0, "stdev": 0.0, "n": 5},
            ("t1", "s", "mid_frontier"): {"median": 97.0, "stdev": 0.0, "n": 5},
            ("t2", "s", "low_frontier"): {"median": 70.0, "stdev": 0.0, "n": 5},
            ("t2", "s", "mid_frontier"): {"median": 70.0, "stdev": 0.0, "n": 5},
        }
        oq = {
            ("t1", "s", "oss"): {"median": 96.0, "stdev": 0.0, "model": "ollama/x", "n": 5},
            ("t2", "s", "oss"): {"median": 65.0, "stdev": 0.0, "model": "ollama/x", "n": 5},
        }
        calls = [("t1", "s"), ("t2", "s")]
        tokens = {"t1": {"in": 100, "out": 50}, "t2": {"in": 100, "out": 50}}
        r = analyze_workflow(calls, "test", fq, oq, tokens_per_task=tokens)
        # In tiered mode, t2 stays on top_frontier (scores too low); quality = 100
        # In oss_only, t2 falls back to top_frontier → 100. t1 → OSS 96.
        assert r["tiered"]["min_slot_quality"] <= 100.0
        assert r["tiered"]["n_calls"] == 2

    def test_cost_saved_zero_when_all_on_top_frontier(self):
        # No data → everything falls back to top_frontier
        calls = [("entity_extraction", "scn-01")]
        r = analyze_workflow(calls, "test", {}, {})
        for mode in ("tiered", "frontier_only", "oss_only"):
            assert r[mode]["cost_saved_pct"] == pytest.approx(0.0)

    def test_delegated_quality_100_when_nothing_delegated(self):
        calls = [("entity_extraction", "scn-01")]
        r = analyze_workflow(calls, "test", {}, {})
        for mode in ("tiered", "frontier_only", "oss_only"):
            assert r[mode]["n_delegated"] == 0
            # When nothing delegated, delegated_quality_pct defaults to 100.
            assert r[mode]["delegated_quality_pct"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# render_dry_run_report
# ---------------------------------------------------------------------------

class TestRenderReport:
    """Tests that the Markdown report is well-formed."""

    def _simple_analyses(self):
        fq = _make_frontier_q("entity_extraction", "scn-01", mid=97.0, low=96.0)
        oq = _make_oss_q("entity_extraction", "scn-01", 96.0)
        calls = [("entity_extraction", "scn-01")]
        return analyze_workflow(calls, "test_wf", fq, oq), calls

    def test_report_contains_all_three_mode_rows(self):
        analyses, calls = self._simple_analyses()
        md = render_dry_run_report("test_wf", calls, analyses)
        assert "Tiered (quality + cost)" in md
        assert "Frontier-only (max quality)" in md
        assert "OSS-only (max savings)" in md

    def test_report_contains_recommendation_section(self):
        analyses, calls = self._simple_analyses()
        md = render_dry_run_report("test_wf", calls, analyses)
        assert "## Recommendation" in md

    def test_report_contains_blended_avg_caveat(self):
        analyses, calls = self._simple_analyses()
        md = render_dry_run_report("test_wf", calls, analyses)
        assert "rewards NOT delegating" in md

    def test_report_mentions_worst_slot(self):
        analyses, calls = self._simple_analyses()
        md = render_dry_run_report("test_wf", calls, analyses)
        assert "Worst slot" in md


# ---------------------------------------------------------------------------
# routing_table helpers
# ---------------------------------------------------------------------------

class TestAgg:
    def test_empty(self):
        assert _agg([]) == (0.0, 0.0, 0.0)

    def test_single(self):
        med, mn, sd = _agg([80.0])
        assert med == 80.0
        assert mn == 80.0
        assert sd == 0.0

    def test_multiple(self):
        vals = [90.0, 80.0, 95.0]
        med, mn, sd = _agg(vals)
        assert med == pytest.approx(statistics.median(vals), abs=0.1)
        assert mn == 80.0


class TestPick:
    def test_tiered_prefers_oss_when_all_clear(self):
        frontier = {
            "mid_frontier": {"quality_worst_scenario": 97.0},
            "low_frontier": {"quality_worst_scenario": 96.0},
        }
        oss = {"model": "ollama/x", "quality_worst_scenario": 96.0}
        result = _pick("tiered", frontier, oss, threshold=95.0)
        assert result["tier"] == 2
        assert result["cost_weight"] == COST_WEIGHTS["oss"]

    def test_frontier_only_never_returns_tier2(self):
        frontier = {
            "mid_frontier": {"quality_worst_scenario": 97.0},
            "low_frontier": {"quality_worst_scenario": 96.0},
        }
        oss = {"model": "ollama/x", "quality_worst_scenario": 99.0}
        result = _pick("frontier_only", frontier, oss, threshold=95.0)
        assert result["tier"] != 2

    def test_oss_only_returns_tier2_when_oss_clears(self):
        frontier = {}
        oss = {"model": "ollama/x", "quality_worst_scenario": 96.0}
        result = _pick("oss_only", frontier, oss, threshold=95.0)
        assert result["tier"] == 2

    def test_oss_only_fallback_when_oss_below_threshold(self):
        frontier = {}
        oss = {"model": "ollama/x", "quality_worst_scenario": 80.0}
        result = _pick("oss_only", frontier, oss, threshold=95.0)
        assert result["tier"] == 0

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            _pick("bad", {}, None, 95.0)


# ---------------------------------------------------------------------------
# build_routing_table (integration — uses example run dirs)
# ---------------------------------------------------------------------------

class TestBuildRoutingTable:
    """Integration test against the committed example data."""

    def test_builds_table_with_correct_slots(self):
        repo_root = Path(__file__).parent.parent.parent
        oss_dir      = repo_root / "examples" / "example-run-oss"
        frontier_dir = repo_root / "examples" / "example-run-frontier"

        table = build_routing_table(oss_dir, frontier_dir)

        assert "slots" in table
        assert "_README" in table
        # All 6 example tasks should appear.
        expected_tasks = {
            "entity_extraction", "relevance_triage", "sentiment_classification",
            "schema_extraction", "summary_synthesis", "document_qa",
        }
        assert expected_tasks.issubset(set(table["slots"].keys()))

    def test_each_slot_has_all_three_mode_picks(self):
        repo_root = Path(__file__).parent.parent.parent
        oss_dir      = repo_root / "examples" / "example-run-oss"
        frontier_dir = repo_root / "examples" / "example-run-frontier"

        table = build_routing_table(oss_dir, frontier_dir)

        for task_id, slot in table["slots"].items():
            picks = slot["picks_at_default_threshold"]
            assert set(picks.keys()) == {"tiered", "frontier_only", "oss_only"}, task_id
            for mode, pick in picks.items():
                assert "tier" in pick, f"{task_id}/{mode}"
                assert "cost_weight" in pick, f"{task_id}/{mode}"

    def test_table_is_json_serialisable(self):
        repo_root = Path(__file__).parent.parent.parent
        oss_dir      = repo_root / "examples" / "example-run-oss"
        frontier_dir = repo_root / "examples" / "example-run-frontier"

        table = build_routing_table(oss_dir, frontier_dir)
        serialised = json.dumps(table)
        assert json.loads(serialised) == table


# ---------------------------------------------------------------------------
# routing_table loaders — quality_median must be the true pooled median
# ---------------------------------------------------------------------------

def _write_synthetic_run_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Write OSS + frontier run dirs where mean(scenario medians) != pooled median.

    Per-sample scores:
        scn-01 -> [90, 90, 90]   (median 90)
        scn-02 -> [100, 100, 60] (median 100)

    mean of scenario medians = 95.0, but the pooled median of all six
    samples is 90.0 — the two aggregates are deliberately distinct so a
    mislabeled mean fails the assertions below.
    """
    per_scn = {"scn-01": [90.0, 90.0, 90.0], "scn-02": [100.0, 100.0, 60.0]}

    # --- OSS run dir (judge-batch.json + judge-scores.json) ---
    oss_dir = tmp_path / "run-oss"
    oss_dir.mkdir()
    items, oss_scores = [], []
    for scn_id, vals in per_scn.items():
        for i, v in enumerate(vals):
            item_id = f"{scn_id}-{i}"
            items.append({
                "item_id":      item_id,
                "task_id":      "task_a",
                "scenario_id":  scn_id,
                "candidate":    "ollama/test:7b",
                "sample_count": 1,
            })
            oss_scores.append({
                "item_id":          item_id,
                "candidate_scores": {"mean_quality_score": v},
            })
    (oss_dir / "judge-batch.json").write_text(json.dumps({"items": items}))
    (oss_dir / "judge-scores.json").write_text(json.dumps(oss_scores))

    # --- Frontier run dir (judge-scores.json only) ---
    frontier_dir = tmp_path / "run-frontier"
    frontier_dir.mkdir()
    frontier_scores = []
    for scn_id, vals in per_scn.items():
        for i, v in enumerate(vals):
            frontier_scores.append({
                "item_id": f"anthropic/claude-sonnet-4-6::task_a::{scn_id}::sample-{i}",
                "candidate_scores": {"mean_quality_score": v},
            })
    (frontier_dir / "judge-scores.json").write_text(json.dumps(frontier_scores))

    return oss_dir, frontier_dir


class TestQualityMedianIsPooledMedian:
    """quality_median must be the pooled per-sample median, never mean(scn medians)."""

    POOLED_MEDIAN = 90.0          # median of [60, 90, 90, 90, 100, 100]
    MEAN_OF_SCN_MEDIANS = 95.0    # mean of [90, 100] — the old, wrong value

    def test_oss_loader_reports_pooled_median(self, tmp_path):
        oss_dir, _ = _write_synthetic_run_dirs(tmp_path)
        result = load_oss_per_task(oss_dir)
        assert result["task_a"]["quality_median"] == pytest.approx(self.POOLED_MEDIAN)
        assert result["task_a"]["quality_median"] != pytest.approx(self.MEAN_OF_SCN_MEDIANS)
        # The worst-scenario gate is unaffected by the median fix.
        assert result["task_a"]["quality_worst_scenario"] == pytest.approx(90.0)

    def test_frontier_loader_reports_pooled_median(self, tmp_path):
        _, frontier_dir = _write_synthetic_run_dirs(tmp_path)
        result = load_frontier_per_task(frontier_dir)
        cell = result["task_a"]["mid_frontier"]
        assert cell["quality_median"] == pytest.approx(self.POOLED_MEDIAN)
        assert cell["quality_median"] != pytest.approx(self.MEAN_OF_SCN_MEDIANS)
        assert cell["quality_worst_scenario"] == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# CLI guards — missing/empty run dirs must exit 1, not report confidently
# ---------------------------------------------------------------------------

class TestCliNoQualityData:
    """Both commands must fail loudly when there is no quality data to load."""

    ERROR_SNIPPET = "Error: no quality data found in"

    def _assert_exits_1(self, argv: list[str], capsys) -> None:
        with pytest.raises(SystemExit) as exc:
            cli_main(argv)
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert self.ERROR_SNIPPET in err
        assert "judge-scores.json" in err

    def test_dry_run_missing_dirs_exit_1(self, tmp_path, capsys):
        self._assert_exits_1(
            ["dry-run", str(tmp_path / "no-oss"), str(tmp_path / "no-frontier")],
            capsys,
        )

    def test_dry_run_empty_dirs_exit_1(self, tmp_path, capsys):
        oss_dir = tmp_path / "oss"
        frontier_dir = tmp_path / "frontier"
        oss_dir.mkdir()
        frontier_dir.mkdir()
        self._assert_exits_1(["dry-run", str(oss_dir), str(frontier_dir)], capsys)

    def test_build_table_missing_dirs_exit_1(self, tmp_path, capsys):
        out = tmp_path / "routing-tiered.json"
        self._assert_exits_1(
            [
                "build-table",
                str(tmp_path / "no-oss"), str(tmp_path / "no-frontier"),
                "--out", str(out),
            ],
            capsys,
        )
        assert not out.exists()

    def test_build_table_empty_dirs_exit_1(self, tmp_path, capsys):
        oss_dir = tmp_path / "oss"
        frontier_dir = tmp_path / "frontier"
        oss_dir.mkdir()
        frontier_dir.mkdir()
        out = tmp_path / "routing-tiered.json"
        self._assert_exits_1(
            ["build-table", str(oss_dir), str(frontier_dir), "--out", str(out)],
            capsys,
        )
        assert not out.exists()

    def test_commands_still_succeed_with_real_data(self, tmp_path, capsys):
        oss_dir, frontier_dir = _write_synthetic_run_dirs(tmp_path)
        out = tmp_path / "routing-tiered.json"
        cli_main(["build-table", str(oss_dir), str(frontier_dir), "--out", str(out)])
        assert out.exists()


# ---------------------------------------------------------------------------
# Package docstring — Quickstart import lines must actually work
# ---------------------------------------------------------------------------

class TestQuickstartDocstring:
    def test_quickstart_imports_execute(self):
        import orchestrator.tiered as tiered_pkg

        import_lines = [
            line.strip()
            for line in (tiered_pkg.__doc__ or "").splitlines()
            if line.strip().startswith("from orchestrator")
        ]
        assert import_lines, "Quickstart docstring should contain import examples"
        for line in import_lines:
            exec(line, {})  # raises ImportError if the docstring lies
