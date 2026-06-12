"""Generate routing-tiered.json — machine-readable tiered routing table.

For each task slot, records per-tier quality metrics and per-mode picks at a
configurable threshold.  The per-mode picks use the WORST-scenario quality as
the gate (downside-risk, not average) — so a slot that scores 97% on four
scenarios but 82% on one will be gated by the 82%.

Output shape::

    {
      "_README": "...",
      "_generated_default_threshold_pct": 95.0,
      "slots": {
        "<task_id>": {
          "top_frontier": {"quality_median": 100.0, "note": "reference"},
          "mid_frontier": {"quality_median": float, "quality_worst_scenario": float, "stdev": float},
          "low_frontier": {"quality_median": float, "quality_worst_scenario": float, "stdev": float},
          "oss": {"model": str, "quality_median": float, "quality_worst_scenario": float, "stdev": float},
          "picks_at_default_threshold": {
            "tiered":        {"tier": int, "model": str, "gate_quality_worst_scenario": float, "cost_weight": float},
            "frontier_only": {...},
            "oss_only":      {...},
          }
        }
      }
    }
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .dry_run import (
    COST_WEIGHTS,
    SENTINEL_HUMAN,
    _model_tier_label,
)

DEFAULT_THRESHOLD: float = 95.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _agg(values: list[float]) -> tuple[float, float, float]:
    """Return (median, min, pstdev) for a list of per-sample quality scores."""
    if not values:
        return (0.0, 0.0, 0.0)
    return (
        round(statistics.median(values), 1),
        round(min(values), 1),
        round(statistics.pstdev(values) if len(values) > 1 else 0.0, 1),
    )


def load_oss_per_task(
    oss_run_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Per task: best OSS candidate by median quality, with worst-scenario gate.

    Returns::

        { task_id: {"model": str, "quality_median": float,
                    "quality_worst_scenario": float, "stdev": float} }
    """
    batch_path  = oss_run_dir / "judge-batch.json"
    scores_path = oss_run_dir / "judge-scores.json"
    if not batch_path.exists() or not scores_path.exists():
        return {}

    batch  = json.loads(batch_path.read_text())
    scores = json.loads(scores_path.read_text())

    items_by_id: dict[str, dict] = {i["item_id"]: i for i in batch.get("items", [])}

    # (task, scn, cand) -> [quality samples]
    cell: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for s in scores:
        item = items_by_id.get(s["item_id"])
        if item is None:
            continue
        cand = item["candidate"]
        if not cand.startswith("ollama/"):
            continue
        task_id     = item["task_id"]
        scenario_id = item["scenario_id"]
        cq = s.get("candidate_scores", {}).get("mean_quality_score")
        if cq is None:
            cq = s.get("mean_quality_score")
        if cq is None:
            continue
        weight = max(1, item.get("sample_count", 1))
        for _ in range(weight):
            cell[(task_id, scenario_id, cand)].append(float(cq))

    # Per (task, cand): list of per-scenario medians + global sample pool
    by_task_cand_scn_medians: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_task_cand_pool:        dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (task_id, scn_id, cand), vals in cell.items():
        by_task_cand_scn_medians[task_id][cand].append(statistics.median(vals))
        by_task_cand_pool[task_id][cand].extend(vals)

    result: dict[str, dict[str, Any]] = {}
    for task_id, cand_scn_medians in by_task_cand_scn_medians.items():
        best_cand, best_mean = None, -1.0
        for cand, scn_medians in cand_scn_medians.items():
            m = statistics.mean(scn_medians)
            if m > best_mean:
                best_mean, best_cand = m, cand
        if best_cand is None:
            continue
        scn_medians = cand_scn_medians[best_cand]
        pool        = by_task_cand_pool[task_id][best_cand]
        med, _, sd  = _agg(pool)
        result[task_id] = {
            "model":                  best_cand,
            # True pooled sample median — NOT the mean of per-scenario medians
            # (best_mean above is only the candidate-selection criterion).
            "quality_median":         med,
            "quality_worst_scenario": round(min(scn_medians), 1),
            "stdev":                  sd,
        }
    return result


def load_frontier_per_task(
    frontier_run_dir: Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Per task: aggregated quality for mid/low-frontier models.

    Returns::

        { task_id: {
            "mid_frontier": {"quality_median": float, "quality_worst_scenario": float, "stdev": float},
            "low_frontier": {...},
          }
        }
    """
    scores_path = frontier_run_dir / "judge-scores.json"
    if not scores_path.exists():
        return {}
    scores = json.loads(scores_path.read_text())

    # (task, scn, label) -> [quality samples]
    cell: dict[tuple[str, str, str], list[float]] = defaultdict(list)
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
            cell[(task_id, scenario_id, label)].append(float(cq))

    # Per task, per label: pool of all samples + per-scenario medians
    by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(lambda: {"scn_medians": [], "pool": []})
    )
    for (task_id, scn_id, label), vals in cell.items():
        by_task[task_id][label]["scn_medians"].append(statistics.median(vals))
        by_task[task_id][label]["pool"].extend(vals)

    result: dict[str, dict[str, dict[str, Any]]] = {}
    for task_id, labels in by_task.items():
        result[task_id] = {}
        for label, d in labels.items():
            med, _, sd = _agg(d["pool"])
            result[task_id][label] = {
                # True pooled sample median — NOT a mean of per-scenario medians.
                "quality_median":         med,
                "quality_worst_scenario": round(min(d["scn_medians"]), 1),
                "stdev":                  sd,
            }
    return result


# ---------------------------------------------------------------------------
# Tier pick logic (worst-scenario gate)
# ---------------------------------------------------------------------------

def _pick(
    mode: str,
    frontier: dict[str, dict[str, Any]],  # label -> {quality_median, quality_worst_scenario, ...}
    oss: dict[str, Any] | None,
    threshold: float,
) -> dict[str, Any]:
    """Return the tier pick for a single task slot under a given mode.

    Gate uses WORST-scenario quality (downside risk).  Within Tier 1,
    low_frontier (cheaper) is preferred over mid_frontier.
    """
    mid_q  = (frontier.get("mid_frontier") or {}).get("quality_worst_scenario", 0.0)
    low_q  = (frontier.get("low_frontier") or {}).get("quality_worst_scenario", 0.0)
    oss_q  = (oss or {}).get("quality_worst_scenario", 0.0)
    oss_model = (oss or {}).get("model", SENTINEL_HUMAN)

    # Cheapest frontier tier that clears the threshold.
    if low_q >= threshold:
        tier1_tier, tier1_model, tier1_weight, tier1_q = (
            1, "low_frontier", COST_WEIGHTS["low_frontier"], low_q,
        )
    elif mid_q >= threshold:
        tier1_tier, tier1_model, tier1_weight, tier1_q = (
            1, "mid_frontier", COST_WEIGHTS["mid_frontier"], mid_q,
        )
    else:
        tier1_tier, tier1_model, tier1_weight, tier1_q = (
            0, "top_frontier", COST_WEIGHTS["top_frontier"], 100.0,
        )

    if mode == "frontier_only":
        return {
            "tier":                      tier1_tier,
            "model":                     tier1_model,
            "gate_quality_worst_scenario": tier1_q,
            "cost_weight":               tier1_weight,
        }
    if mode == "oss_only":
        if oss_q >= threshold:
            return {
                "tier":                      2,
                "model":                     oss_model,
                "gate_quality_worst_scenario": oss_q,
                "cost_weight":               COST_WEIGHTS["oss"],
            }
        return {
            "tier":                      0,
            "model":                     "top_frontier",
            "gate_quality_worst_scenario": 100.0,
            "cost_weight":               COST_WEIGHTS["top_frontier"],
        }
    if mode == "tiered":
        # Prefer OSS if it clears the threshold.  (No relative-to-tier-1 ratio
        # gate: tier-1 quality is capped at 100, so any OSS value clearing the
        # absolute threshold would clear the ratio too.)
        if oss_q >= threshold:
            return {
                "tier":                      2,
                "model":                     oss_model,
                "gate_quality_worst_scenario": oss_q,
                "cost_weight":               COST_WEIGHTS["oss"],
            }
        return {
            "tier":                      tier1_tier,
            "model":                     tier1_model,
            "gate_quality_worst_scenario": tier1_q,
            "cost_weight":               tier1_weight,
        }
    raise ValueError(f"Unknown mode: {mode!r}")


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_routing_table(
    oss_run_dir: Path,
    frontier_run_dir: Path,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Build the routing table dict (not written to disk — caller decides path).

    Args:
        oss_run_dir: Path to the OSS bake-off run dir.
        frontier_run_dir: Path to the frontier bake-off run dir.
        threshold: Quality gate (0-100). Defaults to 95.

    Returns:
        The full routing table dict (serialise with json.dumps).
    """
    oss      = load_oss_per_task(oss_run_dir)
    frontier = load_frontier_per_task(frontier_run_dir)
    all_tasks = sorted(set(oss) | set(frontier))

    slots: dict[str, Any] = {}
    for task_id in all_tasks:
        fr  = frontier.get(task_id, {})
        os_ = oss.get(task_id)
        picks: dict[str, Any] = {}
        for mode in ("tiered", "frontier_only", "oss_only"):
            picks[mode] = _pick(mode, fr, os_, threshold)

        slots[task_id] = {
            "top_frontier": {
                "quality_median": 100.0,
                "note": "reference (judge / baseline tier)",
            },
            "mid_frontier": fr.get("mid_frontier", {"quality_median": None, "note": "not evaluated"}),
            "low_frontier": fr.get("low_frontier", {"quality_median": None, "note": "not evaluated"}),
            "oss": os_ or {"model": None, "note": "not evaluated"},
            "picks_at_default_threshold": picks,
        }

    table: dict[str, Any] = {
        "_README": (
            "Machine-readable TIERED routing table. Generated by "
            "orchestrator.tiered.routing_table from OSS + frontier bake-off runs. "
            "Gate uses WORST-scenario quality (downside risk), not average. "
            "Threshold is configurable via --threshold; "
            f"default {threshold}%. "
            "Cost weights (illustrative Anthropic output-price ratios 2026-05-26): "
            "top_frontier 1.0, mid_frontier 0.20, low_frontier 0.0667, oss 0.0. "
            "Human-escalation sentinel: queue-for-human."
        ),
        "_generated_default_threshold_pct": threshold,
        "slots": slots,
    }
    return table
