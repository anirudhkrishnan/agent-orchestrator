"""Scoring helpers — speed binning, combined score, baseline relativization, aggregation.

All functions in this module are pure (no I/O, no globals). The judge supplies
per-item quality scores via `judge-scores.json`; the framework adds the speed
dimension from `latency_ms` and combines them into a single ranking signal.

Scoring scale (v2): 0-100 on every axis. The earlier 1-5 scale flattened most
candidates into the 3.5-4.5 band; 0-100 gives enough granularity to see real
degradation between, say, a 92% candidate and a 78% candidate.

Baseline relativization (v2): the judge produces a gold-standard answer for
each (task, scenario) BEFORE scoring candidates, then scores both the baseline
and each candidate on the same rubric. The framework computes
`quality_pct_of_baseline = candidate_quality / baseline_quality * 100`, giving
a "what fraction of the judge's own answer can this local model reproduce"
signal — the load-bearing insight for delegation decisions.

Default weighting (quality 0.7 / speed 0.3) reflects the framework's bias:
a slow correct answer beats a fast wrong one for the delegation use case.
CLI exposes `--quality-weight` for tuning per-evaluation.
"""

from __future__ import annotations

import re
import statistics
import sys
from collections import defaultdict


# --- Delegation tiers ------------------------------------------------------
#
# Thresholds drive both the per-task callouts and the final delegation matrix.
# Tuned for this framework: "delegate freely" requires the
# local model to nearly match the judge (≥80% of judge quality) AND to be
# reasonably competent in absolute terms (combined ≥70). The middle band is
# "delegate with monitor" — the model is good enough to try, but worth
# spot-checking outputs. Below 60% of judge means a noticeable quality cliff;
# keep work on the judge.

DELEGATE_FREELY_BASELINE_PCT = 80.0
DELEGATE_WITH_MONITOR_BASELINE_PCT = 60.0
STRONG_CANDIDATE_COMBINED = 80.0
STRONG_CANDIDATE_BASELINE_PCT = 90.0


# --- Speed binning ---------------------------------------------------------


def speed_score_from_latency(latency_ms: int) -> float:
    """Bin a latency measurement to a 0-100 speed score.

    Boundaries (chosen by spec, more granular than the v1 1-5 scale):
        <  1 s  -> 100  (instant feel)
        <  3 s  -> 90   (snappy)
        <  5 s  -> 80   (fast)
        < 10 s  -> 70   (acceptable)
        < 20 s  -> 60   (noticeable)
        < 30 s  -> 50   (slow)
        < 45 s  -> 40   (sluggish)
        < 60 s  -> 30   (annoying)
        < 90 s  -> 20   (painful)
        >= 90 s -> 10   (unusable)

    Boundary semantics are strictly less-than: exactly 1000ms scores 90, exactly
    3000ms scores 80, etc. This matters for the tests at the boundary values.
    """
    if latency_ms < 1_000:
        return 100.0
    if latency_ms < 3_000:
        return 90.0
    if latency_ms < 5_000:
        return 80.0
    if latency_ms < 10_000:
        return 70.0
    if latency_ms < 20_000:
        return 60.0
    if latency_ms < 30_000:
        return 50.0
    if latency_ms < 45_000:
        return 40.0
    if latency_ms < 60_000:
        return 30.0
    if latency_ms < 90_000:
        return 20.0
    return 10.0


def combined_score(quality: float, speed: float, *, quality_weight: float = 0.7) -> float:
    """Linear combination of quality (judge) and speed (latency bin).

    Args:
        quality: Weighted quality mean produced by the judge, on 0-100.
        speed: 0-100 speed score from `speed_score_from_latency`.
        quality_weight: Fraction in [0,1] giving quality's share. Default 0.7
            means quality counts 70%, speed 30%.

    Returns:
        Weighted combined score on the same 0-100 scale.
    """
    if not 0.0 <= quality_weight <= 1.0:
        raise ValueError(f"quality_weight must be in [0,1]; got {quality_weight}")
    return quality_weight * quality + (1.0 - quality_weight) * speed


# --- Baseline relativization ----------------------------------------------


def quality_pct_of_baseline(candidate_quality: float, baseline_quality: float) -> float | None:
    """Return `candidate / baseline * 100`, clamped to [0, 200], or None.

    The "% of judge" signal — the load-bearing insight for delegation
    decisions. A candidate at 90% of baseline is materially different from a
    candidate at 50%, even if their absolute scores look close on the 0-100
    scale.

    Edge cases:
        * baseline_quality <= 0: return None. A degenerate baseline (the judge
          scored its OWN gold answer ~0) means the scenario is unscoreable —
          there is no meaningful ratio, and the cell must be excluded / shown
          as "no baseline" rather than silently promoted to 100%. Logged to
          stderr because this usually indicates a malformed baseline.
        * candidate exceeds baseline: result can exceed 100 (and is clamped at
          200 to avoid a runaway scale). Rare but possible — the judge isn't
          omniscient; a smaller model might genuinely outperform on a narrow
          scenario. A warning is logged when this happens.

    Args:
        candidate_quality: Candidate's mean quality on 0-100.
        baseline_quality: Baseline (judge) mean quality on 0-100.

    Returns:
        Percentage in [0, 200], or None when baseline_quality is <= 0
        (degenerate baseline — no meaningful ratio).
    """
    if baseline_quality <= 0:
        # A degenerate baseline (the judge scored its OWN gold answer ~0) means
        # the scenario is unscoreable, NOT that the candidate is perfect. Return
        # None so it's excluded / shown as "no baseline" — never silently
        # promoted to 100% "Delegate freely" (RCA stress-test MED).
        sys.stderr.write(
            f"[evaluation] WARN: baseline_quality={baseline_quality} (<=0) for a "
            f"cell; excluding from % -of-judge (returning None, not 100).\n"
        )
        return None
    pct = candidate_quality / baseline_quality * 100.0
    if pct > 100.0:
        sys.stderr.write(
            f"[evaluation] NOTE: candidate quality {candidate_quality} exceeds "
            f"baseline {baseline_quality} ({pct:.1f}% of baseline). Rare; verify the "
            f"judge's baseline answer.\n"
        )
    return min(pct, 200.0)


# --- Delegation tier classification ---------------------------------------


def delegation_tier(baseline_pct: float | None) -> str:
    """Bucket a `% of judge` value into a delegation recommendation.

    Args:
        baseline_pct: % of judge quality, or None if no baseline available.

    Returns:
        One of: "Delegate freely", "Delegate with monitor", "Keep on judge",
        "No baseline" (when baseline_pct is None).
    """
    if baseline_pct is None:
        return "No baseline"
    if baseline_pct >= DELEGATE_FREELY_BASELINE_PCT:
        return "Delegate freely"
    if baseline_pct >= DELEGATE_WITH_MONITOR_BASELINE_PCT:
        return "Delegate with monitor"
    return "Keep on judge"


def degradation_callout(
    combined: float,
    baseline_pct: float | None,
    candidate: str,
    judge_model: str,
    task_id: str,
    p50_latency_ms: float,
    baseline_p50_latency_ms: float | None,
) -> str:
    """Produce a one-line prose callout for the leaderboard footer.

    Mechanically derived from `combined` and `baseline_pct`; never hand-tuned.
    The four tiers (in priority order):

        1. STRONG: combined >= 80 AND baseline_pct >= 90
        2. SOLID:  combined >= 70 AND baseline_pct >= 80
        3. ACCEPTABLE: baseline_pct >= 60 (regardless of combined)
        4. DEGRADED:   baseline_pct < 60 → "prefer judge"

    If no baseline is available, falls back to a combined-only callout.

    Args:
        combined: Candidate's mean_combined score (0-100).
        baseline_pct: Candidate's % of judge quality, or None.
        candidate: Candidate id (e.g. "ollama/gemma4:e4b") for the prose.
        judge_model: Judge id, for the prose.
        task_id: Task id, for the prose.
        p50_latency_ms: Candidate's p50 latency.
        baseline_p50_latency_ms: Baseline's p50 latency (None if not measured).

    Returns:
        A single line of prose (no trailing newline).
    """
    speed_note = ""
    if baseline_p50_latency_ms and baseline_p50_latency_ms > 0 and p50_latency_ms > 0:
        ratio = baseline_p50_latency_ms / p50_latency_ms
        if ratio >= 1.5:
            speed_note = f" with {ratio:.1f}× faster latency"
        elif ratio <= 0.67:
            speed_note = f" but {1.0 / ratio:.1f}× slower"

    if baseline_pct is None:
        if combined >= STRONG_CANDIDATE_COMBINED:
            return f"Insight: {candidate} scores combined {combined:.0f} on {task_id} (no baseline available — re-run with baselines)."
        return f"Insight: {candidate} scores combined {combined:.0f} on {task_id} (no baseline available)."

    if combined >= STRONG_CANDIDATE_COMBINED and baseline_pct >= STRONG_CANDIDATE_BASELINE_PCT:
        return (
            f"Insight: {candidate} achieves {baseline_pct:.0f}% of {judge_model} quality on "
            f"{task_id}{speed_note} — strong candidate for delegation."
        )
    if combined >= 70 and baseline_pct >= 80:
        return (
            f"Insight: {candidate} achieves {baseline_pct:.0f}% of {judge_model} quality on "
            f"{task_id}{speed_note} — solid candidate, delegate with light monitoring."
        )
    if baseline_pct >= DELEGATE_WITH_MONITOR_BASELINE_PCT:
        return (
            f"Insight: {candidate} achieves {baseline_pct:.0f}% of {judge_model} quality on "
            f"{task_id} — acceptable degradation, monitor outputs."
        )
    return (
        f"Insight: {candidate} achieves only {baseline_pct:.0f}% of {judge_model} quality on "
        f"{task_id} — significant degradation; prefer judge for this task."
    )


# --- Aggregation -----------------------------------------------------------


def aggregate_per_task_candidate(
    judge_scores: list[dict],
    runner_outputs: list[dict],
    *,
    quality_weight: float = 0.7,
    baseline_candidate_id: str | None = None,
) -> dict[tuple[str, str], dict]:
    """Group scores by (task_id, candidate) and compute means + p50 latency.

    Accepts judge scores in either v1 shape (flat `scores`/`mean_quality_score`)
    or v2 shape (`candidate_scores` / `baseline_scores` nested objects). The v2
    shape carries enough information to compute `quality_pct_of_baseline`
    inline; the v1 shape produces None for that field.

    Args:
        judge_scores: List of judge-score objects. Each object MUST contain
            `item_id` (format `"{candidate}::{task_id}::{scenario_id}"`). For
            v2 (baseline-enabled) it should also contain `candidate_scores`
            and (optionally) `baseline_scores`, each shaped as
            `{"scores": {...}, "mean_quality_score": float, "notes": "..."}`.
            For v1 backward-compat the object may carry `scores` /
            `mean_quality_score` / `notes` at top level.
        runner_outputs: List of CandidateRun-dict objects with `candidate`,
            `task_id`, `scenario_id`, and either a `samples` list (N>1 runs;
            each sample carries its own `latency_ms` / `error`) or flat
            `latency_ms` / `error` fields (legacy N=1 runs).
        quality_weight: Forwarded to `combined_score`.
        baseline_candidate_id: When provided, judge_scores entries whose
            item_id begins with this candidate are treated as baselines and
            excluded from the regular aggregation (their baseline_scores show
            up under separate baseline-aggregation keys). Typical value:
            `"judge-baseline/<judge-name>"`.

    Returns:
        dict keyed by `(task_id, candidate)` with aggregated metrics::

            {
              "mean_quality": float,           # 0-100 (median; back-compat alias)
              "min_quality": float,            # lowest sample quality in the cell group
              "max_quality": float,            # highest sample quality in the cell group
              "mean_speed": float,             # 0-100 (0 when every sample errored)
              "mean_combined": float,          # 0-100
              "mean_baseline_quality": float | None,   # 0-100, judge scoring own answer
              "quality_pct_of_baseline": float | None, # candidate/baseline * 100
              "p50_latency_ms": float,
              "n_scenarios": int,
              "n_errors": int,                 # per-sample errors, each counted once
              "notes": list[str],              # judge notes for the candidate
              "baseline_notes": list[str],     # judge notes on its own baseline
            }

    Latency/speed stats are computed over ALL samples in each cell (read from
    the cell's `samples` list), not just the back-compat `samples[0]` copy.
    Errored samples are excluded — their `latency_ms` is typically 0 (transport
    failure), and `speed_score_from_latency(0)` is 100, so counting them would
    award a fully-errored cell top speed marks. A bucket whose samples ALL
    errored therefore gets `mean_speed` 0.0, not 100.0.
    """
    # Index runner outputs by (candidate, task_id, scenario_id) for O(1) lookup.
    runs_by_key: dict[tuple[str, str, str], dict] = {}
    for r in runner_outputs:
        key = (r["candidate"], r["task_id"], r["scenario_id"])
        runs_by_key[key] = r

    # Bucket judge scores by (task_id, candidate).
    buckets: dict[tuple[str, str], dict] = defaultdict(
        lambda: {
            "qualities": [],
            "baseline_qualities": [],
            "latencies": [],
            "n_errors": 0,
            "notes": [],
            "baseline_notes": [],
        }
    )
    # Latency/error stats come from the runner's per-sample data and must be
    # consumed exactly ONCE per cell — multiple judged items (sample-k /
    # group-of-n dedupe) map back to the same cell, and re-adding the cell's
    # samples per judged item would double-count errors and skew p50.
    consumed_cells: set[tuple[str, str, str]] = set()
    for js in judge_scores:
        # item_id = "{candidate}::{task_id}::{scenario_id}" with an OPTIONAL 4th
        # segment for N>1 sampling: "::sample-{k}" or "::group-of-{n}-s{k}" (added
        # by the dedup in prepare_judge_batch). Take the first three as the cell key;
        # multiple judged samples per cell aggregate into the same bucket, which
        # is exactly the per-cell quality distribution we want for the leaderboard.
        parts = js["item_id"].split("::")
        if len(parts) < 3:
            raise ValueError(
                f"Malformed item_id {js['item_id']!r}; expected at least "
                f"'candidate::task_id::scenario_id'."
            )
        candidate, task_id, scenario_id = parts[0], parts[1], parts[2]
        # N>1 sampling: the dedup in prepare_judge_batch collapses identical
        # outputs into one item with a "::group-of-{n}-s{k}" suffix; distinct
        # outputs get "::sample-{k}". `weight` = how many raw samples this judged
        # item represents, so aggregation is SAMPLE-weighted, not dedup-group-
        # weighted (a group-of-5 must count as 5, not 1).
        weight = _sample_weight(parts[3] if len(parts) >= 4 else "")

        # If this is a baseline-only entry (no candidate to match against),
        # the runner won't have produced a CandidateRun for it; skip the
        # lookup. Baseline-only rows enter aggregation via the candidate
        # rows' nested `baseline_scores`, not as their own bucket.
        if baseline_candidate_id is not None and candidate == baseline_candidate_id:
            continue

        run = runs_by_key.get((candidate, task_id, scenario_id))
        if run is None:
            # Judge scored an item with no matching runner output — surface,
            # don't silently drop. Likely a stale judge file vs run directory.
            raise ValueError(
                f"judge-scores.json references item {js['item_id']!r} with no "
                f"matching CandidateRun. Re-prepare the batch."
            )
        bucket = buckets[(task_id, candidate)]

        # v2 shape: nested candidate_scores / baseline_scores.
        # `qualities` is expanded by `weight` so the list is sample-weighted —
        # median/mean/stdev over it reflect all N samples, not deduped groups.
        if "candidate_scores" in js:
            cs = js["candidate_scores"]
            bucket["qualities"].extend([float(cs["mean_quality_score"])] * weight)
            bucket["notes"].append(cs.get("notes", ""))
            if "baseline_scores" in js and js["baseline_scores"] is not None:
                bs = js["baseline_scores"]
                bucket["baseline_qualities"].extend([float(bs["mean_quality_score"])] * weight)
                bucket["baseline_notes"].append(bs.get("notes", ""))
        else:
            # v1 shape: flat scores/mean_quality_score/notes.
            bucket["qualities"].extend([float(js["mean_quality_score"])] * weight)
            bucket["notes"].append(js.get("notes", ""))

        cell = (candidate, task_id, scenario_id)
        if cell not in consumed_cells:
            consumed_cells.add(cell)
            sample_latencies, sample_errors = _cell_sample_stats(run)
            bucket["latencies"].extend(sample_latencies)
            bucket["n_errors"] += sample_errors

    out: dict[tuple[str, str], dict] = {}
    for (task_id, candidate), b in buckets.items():
        qualities: list[float] = b["qualities"]          # sample-weighted
        baseline_qualities: list[float] = b["baseline_qualities"]
        latencies: list[int] = b["latencies"]            # non-errored samples only
        speeds = [speed_score_from_latency(lat) for lat in latencies]
        # MEDIAN is the headline (robust to tail samples — what the docs +
        # routing.json's "p50" promise). MEAN + STDEV reported alongside; stdev
        # is the N>1 stability signal. MIN/MAX bracket the observed range.
        median_q = statistics.median(qualities) if qualities else 0.0
        mean_q = statistics.mean(qualities) if qualities else 0.0
        stdev_q = statistics.pstdev(qualities) if len(qualities) > 1 else 0.0
        min_q = min(qualities) if qualities else 0.0
        max_q = max(qualities) if qualities else 0.0
        # Empty `latencies` means every sample errored → speed 0, NOT 100.
        mean_s = statistics.mean(speeds) if speeds else 0.0
        combined = combined_score(median_q, mean_s, quality_weight=quality_weight)
        p50_lat = statistics.median(latencies) if latencies else 0.0

        median_baseline_q: float | None = None
        pct_of_baseline: float | None = None
        if baseline_qualities:
            median_baseline_q = statistics.median(baseline_qualities)
            # pct uses MEDIAN vs MEDIAN; returns None on a degenerate (<=0)
            # baseline instead of a misleading 100% (RCA stress-test MED).
            pct_of_baseline = quality_pct_of_baseline(median_q, median_baseline_q)

        out[(task_id, candidate)] = {
            # Headline = median. `mean_quality` kept (now sample-weighted) for
            # continuity; `stdev_quality` is the new stability signal.
            "median_quality": median_q,
            "mean_quality": median_q,  # back-compat alias → median (the p50 the label always claimed)
            "mean_quality_arithmetic": mean_q,
            "stdev_quality": stdev_q,
            "min_quality": min_q,
            "max_quality": max_q,
            "mean_speed": mean_s,
            "mean_combined": combined,
            "median_baseline_quality": median_baseline_q,
            "mean_baseline_quality": median_baseline_q,  # back-compat alias
            "quality_pct_of_baseline": pct_of_baseline,
            "p50_latency_ms": float(p50_lat),
            "n_samples": len(qualities),
            "n_scenarios": len(qualities),  # back-compat alias (now sample count)
            "n_errors": b["n_errors"],
            "notes": b["notes"],
            "baseline_notes": b["baseline_notes"],
        }
    return out


def _cell_sample_stats(run: dict) -> tuple[list[int], int]:
    """Extract (non-errored sample latencies, per-sample error count) for a cell.

    Cells written with N>1 sampling carry a `samples` list with per-sample
    `latency_ms` / `error`; legacy N=1 cells carry flat top-level fields, which
    are treated as a single sample. Errored samples are EXCLUDED from the
    latency list — an errored sample's latency_ms is typically 0 (transport
    failure), and `speed_score_from_latency(0)` is 100, so including it would
    award a fully-errored cell top speed marks.
    """
    samples = run.get("samples") or [
        {"latency_ms": run.get("latency_ms", 0), "error": run.get("error")}
    ]
    latencies = [int(s["latency_ms"]) for s in samples if not s.get("error")]
    n_errors = sum(1 for s in samples if s.get("error"))
    return latencies, n_errors


def _sample_weight(suffix: str) -> int:
    """How many raw samples a deduped judged item represents.

    item_id 4th segment is "group-of-{n}-s{k}" (n identical samples collapsed,
    representative sample index k) or "sample-{k}" (one distinct sample) or ""
    (legacy single-sample run). The bare "group-of-{n}" form from older batches
    is also accepted.
    """
    m = re.match(r"group-of-(\d+)", suffix)
    if m:
        return max(1, int(m.group(1)))
    return 1


def winner_per_task(
    aggregated: dict[tuple[str, str], dict],
) -> dict[str, str]:
    """For each task, return the candidate with the highest mean_combined.

    Tiebreak: higher mean_quality, then lower p50_latency_ms.
    """
    by_task: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for (task_id, candidate), metrics in aggregated.items():
        by_task[task_id].append((candidate, metrics))

    winners: dict[str, str] = {}
    for task_id, entries in by_task.items():
        entries.sort(
            key=lambda kv: (
                -kv[1]["mean_combined"],
                -kv[1]["mean_quality"],
                kv[1]["p50_latency_ms"],
            )
        )
        winners[task_id] = entries[0][0]
    return winners
