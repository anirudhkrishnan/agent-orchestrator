"""Tiered orchestration dry-run — 3-mode comparison.

Three orchestration modes compared side-by-side for any workflow:

  Mode A — Tiered: top-frontier → mid-frontier (if ≥T1 quality) → local OSS
    (if ≥T2 quality). Best balance of quality preservation + cost reduction.
  Mode B — Frontier-only: top-frontier → mid/low-frontier, never OSS. Best
    when quality is paramount.
  Mode C — OSS-only: top-frontier → OSS directly, never mid-frontier. Best
    when maximum cost reduction matters.

Cost weights (illustrative output-price ratios, 2026-05-26):
  Top-frontier (Opus-class):       1.0    — reference
  Mid-frontier (Sonnet-class):     0.20   — 5× cheaper
  Low-frontier (Haiku-class):      0.0667 — 15× cheaper
  Local OSS (Ollama/etc.):         0.0    — electricity only

These are relative to the top-frontier output price and are intended as
illustrative order-of-magnitude ratios, not exact billing figures. Replace
COST_WEIGHTS with your provider's actual rates.

Human-escalation sentinel: ``queue-for-human``.  If a slot falls below the
quality threshold in all modes, the routing table records ``queue-for-human``
(not a model name) so callers know to escalate that call to a person.

Tier numbering:
  Tier 0 — top-frontier (Opus-class)
  Tier 1 — mid/low-frontier (Sonnet/Haiku-class)
  Tier 2 — local OSS
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Cost weights
# ---------------------------------------------------------------------------
# Illustrative output-price ratios relative to the top-frontier tier.
# Sourced from Anthropic public pricing, 2026-05-26 (output token rate):
#   top-frontier (Opus 4.7):   $75/1M  → 1.0
#   mid-frontier (Sonnet 4.6): $15/1M  → 0.20
#   low-frontier (Haiku 4.5):  $5/1M   → 0.0667
#   local OSS (Ollama):        $0/1M   → 0.0 (marginal; electricity only)
#
# Replace with your provider's actual rates if these drift.
COST_WEIGHTS: dict[str, float] = {
    "top_frontier":  1.0,
    "mid_frontier":  0.20,
    "low_frontier":  0.0667,
    "oss":           0.0,
}

# Alias labels used in routing-table model names → tier bucket.
# anthropic/* → inferred from model name; ollama/* → oss.
SENTINEL_HUMAN = "queue-for-human"

# ---------------------------------------------------------------------------
# Generic example workflow definitions
# ---------------------------------------------------------------------------
# Task IDs match those in data/evaluation/tasks-example.yaml.
# Approximate token volumes per call (input + output tokens).  Replace with
# your application's profiling data.
TOKENS_PER_TASK: dict[str, dict[str, int]] = {
    "entity_extraction":        {"in": 180,  "out": 80},
    "relevance_triage":         {"in": 220,  "out": 140},
    "sentiment_classification": {"in": 200,  "out": 100},
    "schema_extraction":        {"in": 280,  "out": 180},
    "summary_synthesis":        {"in": 380,  "out": 250},
    "document_qa":              {"in": 220,  "out": 120},
}

# Plausible generic pipeline definitions: a list of (task_id, scenario_id)
# call-sequences.  These represent two stylized workflows over the example
# task set — replace with your application's real call graphs.
EXAMPLE_WORKFLOWS: dict[str, list[tuple[str, str]]] = {
    # ingest_pipeline: classify + extract on each incoming document,
    # 3 scenarios worth of calls.
    "ingest_pipeline": [
        ("entity_extraction",        "scn-01"),
        ("relevance_triage",         "scn-01"),
        ("sentiment_classification", "scn-01"),
        ("entity_extraction",        "scn-02"),
        ("relevance_triage",         "scn-02"),
        ("sentiment_classification", "scn-02"),
        ("entity_extraction",        "scn-03"),
        ("relevance_triage",         "scn-03"),
        ("sentiment_classification", "scn-03"),
        ("schema_extraction",        "scn-01"),
        ("schema_extraction",        "scn-02"),
    ],
    # report_pipeline: synthesise and answer questions over assembled docs.
    "report_pipeline": [
        ("summary_synthesis", "scn-01"),
        ("summary_synthesis", "scn-02"),
        ("summary_synthesis", "scn-03"),
        ("document_qa",       "scn-01"),
        ("document_qa",       "scn-02"),
        ("document_qa",       "scn-03"),
        ("schema_extraction", "scn-03"),
        ("entity_extraction", "scn-01"),
    ],
}


# ---------------------------------------------------------------------------
# Quality loading helpers
# ---------------------------------------------------------------------------

def _model_tier_label(model_id: str) -> str | None:
    """Map a model id from judge-scores.json to a tier label.

    Returns ``"mid_frontier"`` / ``"low_frontier"`` for anthropic/* models,
    ``"oss"`` for ollama/*, or ``None`` to skip (e.g. top-frontier/opus rows).
    """
    if "sonnet" in model_id.lower():
        return "mid_frontier"
    if "haiku" in model_id.lower():
        return "low_frontier"
    if model_id.startswith("ollama/"):
        return "oss"
    return None


def load_frontier_quality(frontier_run_dir: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Load per-cell quality for mid/low-frontier models from judge-scores.json.

    Reads ``{frontier_run_dir}/judge-scores.json``.  Each item_id has the form::

        anthropic/claude-sonnet-4-6::task_id::scenario_id::sample-N

    Returns a dict keyed by ``(task_id, scenario_id, tier_label)`` where
    ``tier_label`` is ``"mid_frontier"`` or ``"low_frontier"``.  Value::

        {"median": float, "stdev": float, "n": int}
    """
    import json

    scores_path = frontier_run_dir / "judge-scores.json"
    if not scores_path.exists():
        return {}
    scores: list[dict] = json.loads(scores_path.read_text())

    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for s in scores:
        parts = s["item_id"].split("::")
        if len(parts) < 3:
            continue
        model_id, task_id, scenario_id = parts[0], parts[1], parts[2]
        label = _model_tier_label(model_id)
        if label not in ("mid_frontier", "low_frontier"):
            continue
        cq = s.get("candidate_scores", {}).get("mean_quality_score")
        if cq is None:
            cq = s.get("mean_quality_score")
        if cq is not None:
            grouped[(task_id, scenario_id, label)].append(float(cq))

    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, vals in grouped.items():
        result[key] = {
            "median": statistics.median(vals),
            "stdev":  statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "n":      len(vals),
        }
    return result


def load_oss_quality(oss_run_dir: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Load per-cell quality for the best OSS candidate from judge-scores.json.

    Reads ``{oss_run_dir}/judge-batch.json`` + ``judge-scores.json``.

    Returns a dict keyed by ``(task_id, scenario_id, "oss")``::

        {"median": float, "stdev": float, "model": str, "n": int}
    """
    import json

    batch_path  = oss_run_dir / "judge-batch.json"
    scores_path = oss_run_dir / "judge-scores.json"
    if not batch_path.exists() or not scores_path.exists():
        return {}

    batch:  dict = json.loads(batch_path.read_text())
    scores: list[dict] = json.loads(scores_path.read_text())

    # Index batch items by item_id for fast lookup.
    items_by_id: dict[str, dict] = {i["item_id"]: i for i in batch.get("items", [])}

    # (task_id, scenario_id, candidate) -> [quality scores]
    cell_quality: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for s in scores:
        item = items_by_id.get(s["item_id"])
        if item is None:
            continue
        task_id     = item["task_id"]
        scenario_id = item["scenario_id"]
        candidate   = item["candidate"]
        if not candidate.startswith("ollama/"):
            continue
        cq = s.get("candidate_scores", {}).get("mean_quality_score")
        if cq is None:
            cq = s.get("mean_quality_score")
        if cq is None:
            continue
        weight = item.get("sample_count", 1)
        for _ in range(max(1, weight)):
            cell_quality[(task_id, scenario_id, candidate)].append(float(cq))

    # Per (task, scn): pick the OSS candidate with the highest median.
    # Group by (task, scn) first.
    by_cell: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (task_id, scn_id, cand), vals in cell_quality.items():
        by_cell[(task_id, scn_id)][cand].extend(vals)

    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for (task_id, scn_id), per_cand in by_cell.items():
        best_cand:   str | None = None
        best_median: float = -1.0
        for cand, vals in per_cand.items():
            m = statistics.median(vals)
            if m > best_median:
                best_median = m
                best_cand   = cand
        if best_cand is None:
            continue
        vals = per_cand[best_cand]
        result[(task_id, scn_id, "oss")] = {
            "median": best_median,
            "stdev":  statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "model":  best_cand,
            "n":      len(vals),
        }
    return result


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def classify_call(
    task_id: str,
    scenario_id: str,
    frontier_q: dict[tuple[str, str, str], dict],
    oss_q: dict[tuple[str, str, str], dict],
    t1_threshold: float,
    t2_threshold: float,
    mode: str,
) -> tuple[int, str, float, float]:
    """Determine which model a call routes to under ``mode``.

    Returns:
        ``(tier, model_label, quality_pct, cost_weight)``

        * tier 0 = top-frontier (Opus-class)
        * tier 1 = mid/low-frontier (Sonnet/Haiku-class)
        * tier 2 = local OSS
        * model_label is a human-readable name or ``SENTINEL_HUMAN``.
    """
    mid_q  = frontier_q.get((task_id, scenario_id, "mid_frontier"),  {}).get("median", 0.0)
    low_q  = frontier_q.get((task_id, scenario_id, "low_frontier"),  {}).get("median", 0.0)
    oss_d  = oss_q.get((task_id, scenario_id, "oss"),  {})
    oss_qv = oss_d.get("median", 0.0)
    oss_model = oss_d.get("model", SENTINEL_HUMAN)

    if mode == "tiered":
        # Tier 1: pick cheapest frontier model that clears t1_threshold.
        if low_q >= t1_threshold:
            tier1_label, tier1_q, tier1_w = "low_frontier",  low_q,  COST_WEIGHTS["low_frontier"]
        elif mid_q >= t1_threshold:
            tier1_label, tier1_q, tier1_w = "mid_frontier",  mid_q,  COST_WEIGHTS["mid_frontier"]
        else:
            tier1_label, tier1_q, tier1_w = "top_frontier", 100.0, COST_WEIGHTS["top_frontier"]

        # Tier 2: take OSS if it also clears t2_threshold.  (No relative-to-
        # tier-1 ratio gate: tier-1 quality is capped at 100, so any OSS value
        # clearing the absolute threshold would clear the ratio too.)
        if oss_qv >= t2_threshold:
            return (2, oss_model, oss_qv, COST_WEIGHTS["oss"])
        tier_num = {"top_frontier": 0, "mid_frontier": 1, "low_frontier": 1}[tier1_label]
        return (tier_num, tier1_label, tier1_q, tier1_w)

    elif mode == "frontier_only":
        if low_q >= t1_threshold:
            return (1, "low_frontier",  low_q,  COST_WEIGHTS["low_frontier"])
        if mid_q >= t1_threshold:
            return (1, "mid_frontier",  mid_q,  COST_WEIGHTS["mid_frontier"])
        return (0, "top_frontier", 100.0, COST_WEIGHTS["top_frontier"])

    elif mode == "oss_only":
        # OSS delegation is a Tier-2 decision, so it gates on t2_threshold.
        if oss_qv >= t2_threshold:
            return (2, oss_model, oss_qv, COST_WEIGHTS["oss"])
        return (0, "top_frontier", 100.0, COST_WEIGHTS["top_frontier"])

    raise ValueError(f"Unknown mode: {mode!r}. Expected 'tiered', 'frontier_only', or 'oss_only'.")


# ---------------------------------------------------------------------------
# Workflow analysis
# ---------------------------------------------------------------------------

def analyze_workflow(
    workflow_calls: list[tuple[str, str]],
    name: str,
    frontier_q: dict[tuple[str, str, str], dict],
    oss_q: dict[tuple[str, str, str], dict],
    *,
    t1_threshold: float = 95.0,
    t2_threshold: float = 95.0,
    tokens_per_task: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    """For one workflow, compute quality + cost-weighted savings under each mode.

    Args:
        workflow_calls: Sequence of ``(task_id, scenario_id)`` pairs.
        name: Human-readable workflow name (for display only).
        frontier_q: Output of :func:`load_frontier_quality`.
        oss_q: Output of :func:`load_oss_quality`.
        t1_threshold: Min quality (0-100) for delegating to Tier 1 (frontier).
        t2_threshold: Min quality (0-100) for delegating to Tier 2 (OSS).
        tokens_per_task: Override token volume per task.  Defaults to
            :data:`TOKENS_PER_TASK`.

    Returns:
        Dict keyed by mode (``"tiered"``, ``"frontier_only"``, ``"oss_only"``),
        each with::

            {
              "rows": [...],           # per-call detail
              "n_calls": int,
              "cost_saved_pct": float, # % of Opus token-cost avoided
              "avg_quality_pct": float, # blended avg (biased — see note)
              "delegated_quality_pct": float, # mean quality of calls moved off top tier
              "n_delegated": int,
              "min_slot_quality": float, # worst single call (downside-risk)
              "tier_counts": {0: int, 1: int, 2: int},
              "thresholds": (t1_threshold, t2_threshold),
            }
    """
    toks_lookup = tokens_per_task if tokens_per_task is not None else TOKENS_PER_TASK
    results: dict[str, Any] = {}

    for mode in ("tiered", "frontier_only", "oss_only"):
        rows: list[dict] = []
        total_counterfactual = 0.0  # sum of (tokens × top_frontier weight = tokens)
        total_actual         = 0.0
        total_quality_sum    = 0.0
        delegated_qualities: list[float] = []
        min_slot_quality = 100.0
        tier_counts = {0: 0, 1: 0, 2: 0}

        for task_id, scn_id in workflow_calls:
            toks = toks_lookup.get(task_id, {"in": 200, "out": 100})
            token_volume = toks["in"] + toks["out"]

            tier, label, quality_pct, cost_weight = classify_call(
                task_id, scn_id, frontier_q, oss_q, t1_threshold, t2_threshold, mode,
            )
            counterfactual = token_volume * COST_WEIGHTS["top_frontier"]
            actual         = token_volume * cost_weight

            total_counterfactual += counterfactual
            total_actual         += actual
            total_quality_sum    += quality_pct
            if tier > 0:
                delegated_qualities.append(quality_pct)
            min_slot_quality = min(min_slot_quality, quality_pct)
            tier_counts[tier] += 1

            call_savings = (
                (counterfactual - actual) / counterfactual * 100
                if counterfactual else 0.0
            )
            rows.append({
                "task":                  task_id,
                "scenario":              scn_id,
                "tier":                  tier,
                "model":                 label,
                "quality_pct":           quality_pct,
                "cost_weight":           cost_weight,
                "tokens":                token_volume,
                "savings_pct_for_call":  call_savings,
            })

        n_calls = len(workflow_calls)
        cost_saved_pct = (
            (total_counterfactual - total_actual) / total_counterfactual * 100
            if total_counterfactual else 0.0
        )
        avg_quality_pct = total_quality_sum / n_calls if n_calls else 0.0
        n_delegated = len(delegated_qualities)
        delegated_quality_pct = (
            sum(delegated_qualities) / n_delegated if n_delegated else 100.0
        )

        results[mode] = {
            "rows":                  rows,
            "n_calls":               n_calls,
            "cost_saved_pct":        cost_saved_pct,
            "avg_quality_pct":       avg_quality_pct,
            "delegated_quality_pct": delegated_quality_pct,
            "n_delegated":           n_delegated,
            "min_slot_quality":      min_slot_quality,
            "tier_counts":           tier_counts,
            "thresholds":            (t1_threshold, t2_threshold),
        }

    return results


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def render_dry_run_report(
    workflow_name: str,
    workflow_calls: list[tuple[str, str]],
    analyses: dict[str, Any],
    *,
    threshold: float = 95.0,
) -> str:
    """Render a Markdown dry-run report.

    Args:
        workflow_name: Display name of the workflow.
        workflow_calls: The call sequence analysed.
        analyses: Output of :func:`analyze_workflow`.
        threshold: Quality threshold used (for the recommendation section).

    Returns:
        Multi-line Markdown string.
    """
    lines: list[str] = []
    p = lines.append

    p(f"# Dry-Run Comparison — `{workflow_name}`\n")
    p(f"- **Generated:** {datetime.now(timezone.utc).isoformat()}")
    p(f"- **Calls in workflow:** {len(workflow_calls)}")
    p(f"- **Threshold:** {threshold:.0f}% (configurable via --threshold)\n")

    p(
        "_Three orchestration modes compared on this workflow's exact call sequence. "
        "Quality is the median across N=5 samples per cell. Cost-weighted savings uses "
        "top-frontier = 1.0 reference, mid-frontier = 0.20, low-frontier = 0.0667, "
        "OSS = 0.0 (illustrative output-price ratios)._\n"
    )

    # Headline comparison table
    p("## Headline comparison\n")
    p("| Mode | Cost-weighted savings | Delegated quality | Worst slot | Blended avg | Tier 0/1/2 |")
    p("|---|---:|---:|---:|---:|---|")
    for mode_key, label in [
        ("tiered",        "Tiered (quality + cost)"),
        ("frontier_only", "Frontier-only (max quality)"),
        ("oss_only",      "OSS-only (max savings)"),
    ]:
        a = analyses[mode_key]
        t = a["tier_counts"]
        p(
            f"| {label} "
            f"| {a['cost_saved_pct']:.1f}% "
            f"| {a['delegated_quality_pct']:.1f}% (n={a['n_delegated']}) "
            f"| {a['min_slot_quality']:.1f}% "
            f"| {a['avg_quality_pct']:.1f}% "
            f"| {t[0]} / {t[1]} / {t[2]} |"
        )
    p("")

    p("_**Read the metrics in this order:**_")
    p(
        "- **Cost-weighted savings** — the headline. "
        "% of the top-frontier token-cost counterfactual avoided."
    )
    p(
        "- **Delegated quality** — mean quality of ONLY the calls actually moved "
        "off top-frontier (n = how many). "
        "This is the honest \"how good is the cheaper model when we use it\" number."
    )
    p(
        "- **Worst slot** — the single lowest-quality call. "
        "The downside-risk number; watch this against your threshold."
    )
    p(
        "- **Blended avg** — average across ALL calls, counting kept-on-top-frontier "
        "calls as 100%. "
        "**This number rewards NOT delegating** (a mode that delegates nothing "
        "scores 100%), so it is NOT comparable across modes. Shown last, for "
        "continuity only."
    )
    p("")
    p("_Tier 0 = top-frontier (Opus-class) | Tier 1 = mid/low-frontier "
      "(Sonnet/Haiku-class) | Tier 2 = local OSS_\n")

    # Per-slot detail per mode
    for mode_key, label in [
        ("tiered",        "Tiered"),
        ("frontier_only", "Frontier-only"),
        ("oss_only",      "OSS-only"),
    ]:
        a = analyses[mode_key]
        p(f"## {label} mode — per-call detail\n")
        p("| # | Task | Scenario | Routes to | Tier | Quality | Cost-saved |")
        p("|---:|---|---|---|:--:|---:|---:|")
        for i, r in enumerate(a["rows"], 1):
            p(
                f"| {i} | `{r['task']}` | {r['scenario']} "
                f"| `{r['model']}` | T{r['tier']} "
                f"| {r['quality_pct']:.1f}% "
                f"| {r['savings_pct_for_call']:.1f}% |"
            )
        p("")

    # Recommendation — honest criterion: among modes whose WORST slot clears
    # the threshold, pick the one with the most cost savings.
    clearing = {k: v for k, v in analyses.items() if v["min_slot_quality"] >= threshold}
    if clearing:
        best_key = max(clearing, key=lambda k: clearing[k]["cost_saved_pct"])
        basis = (
            f"its worst slot ({clearing[best_key]['min_slot_quality']:.1f}%) "
            f"clears the {threshold:.0f}% threshold AND it saves the most"
        )
    else:
        best_key = max(analyses, key=lambda k: analyses[k]["min_slot_quality"])
        basis = (
            f"NO mode keeps every call >= {threshold:.0f}% — this mode has the "
            f"best worst-slot ({analyses[best_key]['min_slot_quality']:.1f}%). "
            f"Either lower the threshold or keep the sub-threshold slot(s) on "
            f"top-frontier explicitly"
        )

    b = analyses[best_key]
    p("## Recommendation\n")
    p(
        f"**Mode `{best_key}`** — {basis}. "
        f"Cost-weighted savings {b['cost_saved_pct']:.1f}%, "
        f"delegated-call quality {b['delegated_quality_pct']:.1f}% "
        f"(n={b['n_delegated']}), worst slot {b['min_slot_quality']:.1f}%.\n"
    )
    p(
        "_Recommendation keys off **worst-slot >= threshold**, not blended average — "
        "a single bad slot is the real risk in a production pipeline, and the "
        "blended average would hide it._\n"
    )

    p("## Sensitivity\n")
    p(
        "To adjust your threshold: lower it (e.g. 90%) to route more calls to "
        "cheaper tiers (more savings, more quality risk). Raise it (e.g. 98%) "
        "to keep more calls at higher tiers (less savings, less quality risk).\n"
    )

    return "\n".join(lines)
