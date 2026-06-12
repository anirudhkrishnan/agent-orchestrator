"""Pure-function tests for evaluation.scoring (0-100 scale, baseline support)."""

import pytest

from orchestrator.evaluation.scoring import (
    DELEGATE_FREELY_BASELINE_PCT,
    DELEGATE_WITH_MONITOR_BASELINE_PCT,
    aggregate_per_task_candidate,
    combined_score,
    degradation_callout,
    delegation_tier,
    quality_pct_of_baseline,
    speed_score_from_latency,
    winner_per_task,
)


# --- speed_score_from_latency --------------------------------------------


def test_speed_score_zero_latency_is_max():
    assert speed_score_from_latency(0) == 100.0


def test_speed_score_just_under_1s_is_100():
    assert speed_score_from_latency(999) == 100.0


def test_speed_score_exactly_1s_is_90():
    """Exactly 1000ms falls into the 90 bin (strict less-than at 1s)."""
    assert speed_score_from_latency(1_000) == 90.0


def test_speed_score_exactly_3s_is_80():
    assert speed_score_from_latency(3_000) == 80.0


def test_speed_score_exactly_5s_is_70():
    assert speed_score_from_latency(5_000) == 70.0


def test_speed_score_exactly_10s_is_60():
    assert speed_score_from_latency(10_000) == 60.0


def test_speed_score_exactly_20s_is_50():
    assert speed_score_from_latency(20_000) == 50.0


def test_speed_score_exactly_30s_is_40():
    assert speed_score_from_latency(30_000) == 40.0


def test_speed_score_exactly_45s_is_30():
    assert speed_score_from_latency(45_000) == 30.0


def test_speed_score_exactly_60s_is_20():
    assert speed_score_from_latency(60_000) == 20.0


def test_speed_score_exactly_90s_is_10():
    assert speed_score_from_latency(90_000) == 10.0


def test_speed_score_very_slow_is_10():
    assert speed_score_from_latency(120_000) == 10.0


def test_speed_score_boundaries_full_table():
    """One golden test that walks every bin's boundary."""
    cases = [
        (0, 100.0),
        (999, 100.0),
        (1_000, 90.0),
        (2_999, 90.0),
        (3_000, 80.0),
        (4_999, 80.0),
        (5_000, 70.0),
        (9_999, 70.0),
        (10_000, 60.0),
        (19_999, 60.0),
        (20_000, 50.0),
        (29_999, 50.0),
        (30_000, 40.0),
        (44_999, 40.0),
        (45_000, 30.0),
        (59_999, 30.0),
        (60_000, 20.0),
        (89_999, 20.0),
        (90_000, 10.0),
        (999_999, 10.0),
    ]
    for ms, expected in cases:
        assert speed_score_from_latency(ms) == expected, f"latency={ms}ms"


# --- combined_score -------------------------------------------------------


def test_combined_score_default_weighting_on_0_100():
    """Default 0.7 quality / 0.3 speed on the 0-100 scale."""
    assert combined_score(80.0, 100.0) == pytest.approx(0.7 * 80.0 + 0.3 * 100.0)


def test_combined_score_all_quality():
    assert combined_score(80.0, 100.0, quality_weight=1.0) == 80.0


def test_combined_score_all_speed():
    assert combined_score(80.0, 100.0, quality_weight=0.0) == 100.0


def test_combined_score_balanced():
    assert combined_score(40.0, 80.0, quality_weight=0.5) == 60.0


def test_combined_score_rejects_out_of_range_weight():
    with pytest.raises(ValueError, match="quality_weight"):
        combined_score(50.0, 50.0, quality_weight=1.5)
    with pytest.raises(ValueError, match="quality_weight"):
        combined_score(50.0, 50.0, quality_weight=-0.1)


# --- quality_pct_of_baseline ---------------------------------------------


def test_quality_pct_of_baseline_basic_ratio():
    """86 / 97 * 100 ≈ 88.7%."""
    assert quality_pct_of_baseline(86.0, 97.0) == pytest.approx(88.659793, abs=1e-3)


def test_quality_pct_of_baseline_equal_returns_100():
    assert quality_pct_of_baseline(90.0, 90.0) == pytest.approx(100.0)


def test_quality_pct_of_baseline_zero_baseline_returns_none():
    """Degenerate baseline (judge scored its own gold ~0) → unscoreable cell.
    Returns None (excluded / 'no baseline'), NOT 100% — a zero baseline must
    never silently promote a candidate to 'Delegate freely' (RCA 2026-05-28)."""
    assert quality_pct_of_baseline(0.0, 0.0) is None
    assert quality_pct_of_baseline(50.0, 0.0) is None


def test_quality_pct_of_baseline_negative_baseline_returns_none():
    """Negative baseline treated same as zero → None, not 100."""
    assert quality_pct_of_baseline(50.0, -5.0) is None


def test_quality_pct_of_baseline_can_exceed_100():
    """Candidate outperforming baseline is allowed (with a clamp at 200)."""
    pct = quality_pct_of_baseline(95.0, 80.0)
    assert pct == pytest.approx(118.75, abs=1e-3)


def test_quality_pct_of_baseline_clamps_at_200():
    """Even if candidate is 3x baseline, clamp at 200 to keep scale tame."""
    assert quality_pct_of_baseline(300.0, 50.0) == 200.0


def test_quality_pct_of_baseline_zero_candidate():
    assert quality_pct_of_baseline(0.0, 80.0) == 0.0


# --- delegation_tier ------------------------------------------------------


def test_delegation_tier_freely_at_threshold():
    assert delegation_tier(DELEGATE_FREELY_BASELINE_PCT) == "Delegate freely"
    assert delegation_tier(95.0) == "Delegate freely"


def test_delegation_tier_with_monitor_band():
    assert delegation_tier(DELEGATE_WITH_MONITOR_BASELINE_PCT) == "Delegate with monitor"
    assert delegation_tier(70.0) == "Delegate with monitor"
    assert delegation_tier(79.999) == "Delegate with monitor"


def test_delegation_tier_keep_on_judge_band():
    assert delegation_tier(50.0) == "Keep on judge"
    assert delegation_tier(0.0) == "Keep on judge"


def test_delegation_tier_no_baseline_returns_sentinel():
    assert delegation_tier(None) == "No baseline"


# --- degradation_callout --------------------------------------------------


def test_degradation_callout_strong_candidate_mentions_delegation():
    msg = degradation_callout(
        combined=85.0,
        baseline_pct=94.0,
        candidate="gemma4:e4b",
        judge_model="frontier-judge",
        task_id="entity_extraction",
        p50_latency_ms=2_000,
        baseline_p50_latency_ms=None,
    )
    assert "strong candidate" in msg
    assert "entity_extraction" in msg
    assert "94%" in msg


def test_degradation_callout_significant_degradation():
    msg = degradation_callout(
        combined=40.0,
        baseline_pct=45.0,
        candidate="qwen3:8b",
        judge_model="frontier-judge",
        task_id="summary_synthesis",
        p50_latency_ms=30_000,
        baseline_p50_latency_ms=None,
    )
    assert "significant degradation" in msg
    assert "prefer judge" in msg


def test_degradation_callout_acceptable_band():
    msg = degradation_callout(
        combined=65.0,
        baseline_pct=70.0,
        candidate="qwen3:8b",
        judge_model="frontier-judge",
        task_id="relevance_triage",
        p50_latency_ms=8_000,
        baseline_p50_latency_ms=None,
    )
    assert "acceptable degradation" in msg


def test_degradation_callout_no_baseline_falls_back():
    msg = degradation_callout(
        combined=85.0,
        baseline_pct=None,
        candidate="gemma4:e4b",
        judge_model="frontier-judge",
        task_id="entity_extraction",
        p50_latency_ms=2_000,
        baseline_p50_latency_ms=None,
    )
    assert "no baseline available" in msg


# --- aggregate_per_task_candidate ----------------------------------------


def _judge_item_v1(candidate: str, task: str, scn: str, q: float, notes: str = "") -> dict:
    """v1 (flat) shape — for backward-compat tests."""
    return {
        "item_id": f"{candidate}::{task}::{scn}",
        "scores": {"quality": q},
        "mean_quality_score": q,
        "notes": notes,
    }


def _judge_item_v2(
    candidate: str,
    task: str,
    scn: str,
    candidate_q: float,
    baseline_q: float | None = None,
    candidate_notes: str = "",
    baseline_notes: str = "",
) -> dict:
    """v2 (nested baseline+candidate) shape."""
    out: dict = {
        "item_id": f"{candidate}::{task}::{scn}",
        "candidate_scores": {
            "scores": {"quality": candidate_q},
            "mean_quality_score": candidate_q,
            "notes": candidate_notes,
        },
    }
    if baseline_q is not None:
        out["baseline_scores"] = {
            "scores": {"quality": baseline_q},
            "mean_quality_score": baseline_q,
            "notes": baseline_notes,
        }
    return out


def _run_item(candidate: str, task: str, scn: str, latency: int, error: str | None = None) -> dict:
    return {
        "candidate": candidate,
        "task_id": task,
        "scenario_id": scn,
        "output_text": "" if error else "out",
        "latency_ms": latency,
        "error": error,
        "completed_at": "2026-05-26T00:00:00+00:00",
    }


def _run_item_sampled(
    candidate: str, task: str, scn: str, samples: list[tuple[int, str | None]]
) -> dict:
    """N>1 cell: `samples` is a list of (latency_ms, error) tuples."""
    return {
        "candidate": candidate,
        "task_id": task,
        "scenario_id": scn,
        "samples": [
            {"output_text": "" if err else f"out-{i}", "latency_ms": lat, "error": err}
            for i, (lat, err) in enumerate(samples)
        ],
        # Back-compat flat fields mirror samples[0].
        "output_text": "" if samples[0][1] else "out-0",
        "latency_ms": samples[0][0],
        "error": samples[0][1],
        "completed_at": "2026-05-26T00:00:00+00:00",
    }


def test_aggregate_v1_shape_single_candidate_single_task():
    """v1 (flat) backward compat — baseline fields should be None."""
    judge = [
        _judge_item_v1("ollama/qwen3:8b", "tk", "s1", 80.0, "good"),
        _judge_item_v1("ollama/qwen3:8b", "tk", "s2", 100.0, "great"),
    ]
    runs = [
        _run_item("ollama/qwen3:8b", "tk", "s1", 1_500),
        _run_item("ollama/qwen3:8b", "tk", "s2", 5_000),
    ]
    agg = aggregate_per_task_candidate(judge, runs)
    metrics = agg[("tk", "ollama/qwen3:8b")]
    assert metrics["mean_quality"] == pytest.approx(90.0)
    # speed: 1500ms->90, 5000ms->70 (>=5s); mean = 80
    assert metrics["mean_speed"] == pytest.approx(80.0)
    # combined default 0.7*90+0.3*80 = 87
    assert metrics["mean_combined"] == pytest.approx(87.0)
    # p50 of [1500, 5000] = 3250
    assert metrics["p50_latency_ms"] == pytest.approx(3250.0)
    assert metrics["n_scenarios"] == 2
    assert metrics["n_errors"] == 0
    # No baseline in v1 shape
    assert metrics["mean_baseline_quality"] is None
    assert metrics["quality_pct_of_baseline"] is None


def test_aggregate_v2_shape_with_baseline():
    """v2 shape: judge scores both baseline + candidate; pct computed."""
    judge = [
        _judge_item_v2("ollama/qwen3:8b", "tk", "s1", candidate_q=86.0, baseline_q=97.0),
        _judge_item_v2("ollama/qwen3:8b", "tk", "s2", candidate_q=80.0, baseline_q=90.0),
    ]
    runs = [
        _run_item("ollama/qwen3:8b", "tk", "s1", 1_500),
        _run_item("ollama/qwen3:8b", "tk", "s2", 2_500),
    ]
    agg = aggregate_per_task_candidate(judge, runs)
    metrics = agg[("tk", "ollama/qwen3:8b")]
    assert metrics["mean_quality"] == pytest.approx(83.0)
    assert metrics["mean_baseline_quality"] == pytest.approx(93.5)
    # 83 / 93.5 * 100 ≈ 88.77%
    assert metrics["quality_pct_of_baseline"] == pytest.approx(83.0 / 93.5 * 100.0, abs=1e-3)


def test_aggregate_v2_without_baseline_section_is_none():
    """v2 shape but no baseline_scores attached → baseline metrics None."""
    judge = [
        _judge_item_v2("ollama/qwen3:8b", "tk", "s1", candidate_q=86.0, baseline_q=None),
    ]
    runs = [_run_item("ollama/qwen3:8b", "tk", "s1", 1_500)]
    agg = aggregate_per_task_candidate(judge, runs)
    metrics = agg[("tk", "ollama/qwen3:8b")]
    assert metrics["mean_quality"] == pytest.approx(86.0)
    assert metrics["mean_baseline_quality"] is None
    assert metrics["quality_pct_of_baseline"] is None


def test_aggregate_multiple_candidates_separated_by_key():
    judge = [
        _judge_item_v1("ollama/qwen3:8b", "tk", "s1", 80.0),
        _judge_item_v1("ollama/gemma4:e4b", "tk", "s1", 60.0),
    ]
    runs = [
        _run_item("ollama/qwen3:8b", "tk", "s1", 1_500),
        _run_item("ollama/gemma4:e4b", "tk", "s1", 15_000),
    ]
    agg = aggregate_per_task_candidate(judge, runs)
    assert ("tk", "ollama/qwen3:8b") in agg
    assert ("tk", "ollama/gemma4:e4b") in agg
    # qwen: quality 80, speed 90 (1.5s); gemma: quality 60, speed 60 (15s falls in 10-20s -> 60)
    assert agg[("tk", "ollama/qwen3:8b")]["mean_quality"] == pytest.approx(80.0)
    assert agg[("tk", "ollama/gemma4:e4b")]["mean_speed"] == pytest.approx(60.0)


def test_aggregate_counts_errors():
    judge = [_judge_item_v1("ollama/qwen3:8b", "tk", "s1", 0.0, "errored")]
    runs = [_run_item("ollama/qwen3:8b", "tk", "s1", 0, error="TimeoutError: x")]
    agg = aggregate_per_task_candidate(judge, runs)
    metrics = agg[("tk", "ollama/qwen3:8b")]
    assert metrics["n_errors"] == 1


def test_aggregate_raises_on_orphan_judge_score():
    judge = [_judge_item_v1("ollama/qwen3:8b", "tk", "s1", 80.0)]
    runs = []
    with pytest.raises(ValueError, match="no.*matching"):
        aggregate_per_task_candidate(judge, runs)


def test_aggregate_raises_on_malformed_item_id():
    judge = [{"item_id": "bad-shape", "scores": {}, "mean_quality_score": 0, "notes": ""}]
    runs = []
    with pytest.raises(ValueError, match="Malformed item_id"):
        aggregate_per_task_candidate(judge, runs)


def test_aggregate_skips_baseline_only_candidate_when_id_provided():
    """When baseline_candidate_id is set, baseline-only judge rows aren't
    expected to have matching CandidateRuns — the aggregator should skip them."""
    judge = [
        _judge_item_v1("ollama/qwen3:8b", "tk", "s1", 80.0),
        # Baseline pseudo-candidate, no matching run.
        _judge_item_v1("judge-baseline/opus", "tk", "s1", 95.0),
    ]
    runs = [_run_item("ollama/qwen3:8b", "tk", "s1", 1_500)]
    agg = aggregate_per_task_candidate(
        judge, runs, baseline_candidate_id="judge-baseline/opus"
    )
    # Only the real candidate appears in the result.
    assert list(agg.keys()) == [("tk", "ollama/qwen3:8b")]


def test_aggregate_min_max_quality_keys():
    """Aggregation exposes min/max quality bracketing the observed range."""
    judge = [
        _judge_item_v1("ollama/qwen3:8b", "tk", "s1", 60.0),
        _judge_item_v1("ollama/qwen3:8b", "tk", "s2", 90.0),
    ]
    runs = [
        _run_item("ollama/qwen3:8b", "tk", "s1", 1_500),
        _run_item("ollama/qwen3:8b", "tk", "s2", 1_500),
    ]
    agg = aggregate_per_task_candidate(judge, runs)
    metrics = agg[("tk", "ollama/qwen3:8b")]
    assert metrics["min_quality"] == pytest.approx(60.0)
    assert metrics["max_quality"] == pytest.approx(90.0)


# --- aggregation over N>1 samples (per-sample latency/error data) ----------


def _judge_item_v2_suffixed(
    candidate: str, task: str, scn: str, suffix: str, q: float
) -> dict:
    """v2 item with an explicit 4th item_id segment (sample-k / group-of-n-sk)."""
    return {
        "item_id": f"{candidate}::{task}::{scn}::{suffix}",
        "candidate_scores": {
            "scores": {"quality": q},
            "mean_quality_score": q,
            "notes": "",
        },
    }


def test_aggregate_fully_errored_cell_gets_speed_zero_not_100():
    """Regression: every sample errored (latency 0) → mean_speed must be 0.

    Before the fix, the cell's back-compat latency_ms=0 was scored
    speed_score_from_latency(0)=100, so a model that produced NOTHING got
    mean_speed=100 / combined=30 and won latency tiebreaks."""
    judge = [
        # 5 errored samples → 5 distinct judge items (errors never dedupe).
        _judge_item_v2_suffixed("ollama/qwen3:8b", "tk", "s1", f"sample-{k}", 0.0)
        for k in range(5)
    ]
    runs = [
        _run_item_sampled(
            "ollama/qwen3:8b", "tk", "s1", [(0, "TimeoutError: x")] * 5
        )
    ]
    agg = aggregate_per_task_candidate(judge, runs)
    metrics = agg[("tk", "ollama/qwen3:8b")]
    assert metrics["mean_speed"] == 0.0
    assert metrics["mean_combined"] == 0.0
    assert metrics["p50_latency_ms"] == 0.0


def test_aggregate_p50_latency_over_all_samples():
    """Regression: p50 latency must cover ALL N samples, not just samples[0].

    5 samples at distinct latencies; before the fix only samples[0]'s 1000ms
    was visible (appended once per judged item)."""
    latencies = [1_000, 2_000, 3_000, 4_000, 100_000]
    judge = [
        _judge_item_v2_suffixed("ollama/qwen3:8b", "tk", "s1", f"sample-{k}", 80.0)
        for k in range(5)
    ]
    runs = [
        _run_item_sampled(
            "ollama/qwen3:8b", "tk", "s1", [(lat, None) for lat in latencies]
        )
    ]
    agg = aggregate_per_task_candidate(judge, runs)
    metrics = agg[("tk", "ollama/qwen3:8b")]
    assert metrics["p50_latency_ms"] == pytest.approx(3_000.0)
    # speeds: 1000→90, 2000→90, 3000→80, 4000→80, 100000→10 → mean 70
    assert metrics["mean_speed"] == pytest.approx(70.0)


def test_aggregate_n_errors_counts_each_errored_sample_exactly_once():
    """Regression: 2 of 5 samples errored → n_errors == 2.

    Before the fix only samples[0]'s error was visible — counted once per
    judged item (over-count) while sample 1-4 errors were invisible."""
    # Sample 0 errors; sample 1 errors; samples 2-4 are fine.
    samples = [(0, "TimeoutError: x"), (0, "empty_output"), (1_000, None), (2_000, None), (3_000, None)]
    judge = [
        _judge_item_v2_suffixed("ollama/qwen3:8b", "tk", "s1", f"sample-{k}", q)
        for k, q in enumerate([0.0, 0.0, 80.0, 80.0, 80.0])
    ]
    runs = [_run_item_sampled("ollama/qwen3:8b", "tk", "s1", samples)]
    agg = aggregate_per_task_candidate(judge, runs)
    metrics = agg[("tk", "ollama/qwen3:8b")]
    assert metrics["n_errors"] == 2
    # Errored samples are excluded from latency stats: p50 of [1000,2000,3000].
    assert metrics["p50_latency_ms"] == pytest.approx(2_000.0)


def test_aggregate_group_of_n_with_rep_index_weights_samples():
    """The disambiguated 'group-of-{n}-s{k}' suffix still carries weight n.

    Two equal-sized groups in one cell (samples [A,A,B,B]) produce distinct
    item_ids; each must weight its score by 2 so the cell aggregates over 4
    samples."""
    judge = [
        _judge_item_v2_suffixed("ollama/qwen3:8b", "tk", "s1", "group-of-2-s0", 60.0),
        _judge_item_v2_suffixed("ollama/qwen3:8b", "tk", "s1", "group-of-2-s2", 90.0),
    ]
    runs = [
        _run_item_sampled("ollama/qwen3:8b", "tk", "s1", [(1_000, None)] * 4)
    ]
    agg = aggregate_per_task_candidate(judge, runs)
    metrics = agg[("tk", "ollama/qwen3:8b")]
    assert metrics["n_samples"] == 4
    # median of [60, 60, 90, 90] = 75
    assert metrics["median_quality"] == pytest.approx(75.0)
    # Legacy bare "group-of-{n}" suffix (old batches) still parses.
    judge_legacy = [
        _judge_item_v2_suffixed("ollama/qwen3:8b", "tk", "s1", "group-of-4", 70.0),
    ]
    agg2 = aggregate_per_task_candidate(judge_legacy, runs)
    assert agg2[("tk", "ollama/qwen3:8b")]["n_samples"] == 4


# --- winner_per_task ------------------------------------------------------


def test_winner_per_task_highest_combined():
    aggregated = {
        ("tk1", "a"): {
            "mean_quality": 80.0, "mean_speed": 100.0, "mean_combined": 86.0,
            "p50_latency_ms": 1000, "n_scenarios": 5, "n_errors": 0, "notes": [],
            "mean_baseline_quality": None, "quality_pct_of_baseline": None,
            "baseline_notes": [],
        },
        ("tk1", "b"): {
            "mean_quality": 100.0, "mean_speed": 60.0, "mean_combined": 88.0,
            "p50_latency_ms": 20_000, "n_scenarios": 5, "n_errors": 0, "notes": [],
            "mean_baseline_quality": None, "quality_pct_of_baseline": None,
            "baseline_notes": [],
        },
    }
    winners = winner_per_task(aggregated)
    assert winners == {"tk1": "b"}


def test_winner_per_task_tiebreak_on_quality_then_latency():
    aggregated = {
        ("tk1", "a"): {
            "mean_quality": 90.0, "mean_speed": 80.0, "mean_combined": 87.0,
            "p50_latency_ms": 3000, "n_scenarios": 5, "n_errors": 0, "notes": [],
            "mean_baseline_quality": None, "quality_pct_of_baseline": None,
            "baseline_notes": [],
        },
        ("tk1", "b"): {
            "mean_quality": 80.0, "mean_speed": 100.0, "mean_combined": 87.0,
            "p50_latency_ms": 1000, "n_scenarios": 5, "n_errors": 0, "notes": [],
            "mean_baseline_quality": None, "quality_pct_of_baseline": None,
            "baseline_notes": [],
        },
    }
    # Combined tied → prefer higher quality (a).
    winners = winner_per_task(aggregated)
    assert winners == {"tk1": "a"}


def test_winner_per_task_multiple_tasks():
    aggregated = {
        ("tk1", "a"): {"mean_quality": 100, "mean_speed": 100, "mean_combined": 100, "p50_latency_ms": 1000, "n_scenarios": 1, "n_errors": 0, "notes": [], "mean_baseline_quality": None, "quality_pct_of_baseline": None, "baseline_notes": []},
        ("tk1", "b"): {"mean_quality": 60, "mean_speed": 100, "mean_combined": 72, "p50_latency_ms": 1000, "n_scenarios": 1, "n_errors": 0, "notes": [], "mean_baseline_quality": None, "quality_pct_of_baseline": None, "baseline_notes": []},
        ("tk2", "a"): {"mean_quality": 40, "mean_speed": 100, "mean_combined": 58, "p50_latency_ms": 1000, "n_scenarios": 1, "n_errors": 0, "notes": [], "mean_baseline_quality": None, "quality_pct_of_baseline": None, "baseline_notes": []},
        ("tk2", "b"): {"mean_quality": 80, "mean_speed": 100, "mean_combined": 86, "p50_latency_ms": 1000, "n_scenarios": 1, "n_errors": 0, "notes": [], "mean_baseline_quality": None, "quality_pct_of_baseline": None, "baseline_notes": []},
    }
    winners = winner_per_task(aggregated)
    assert winners == {"tk1": "a", "tk2": "b"}
