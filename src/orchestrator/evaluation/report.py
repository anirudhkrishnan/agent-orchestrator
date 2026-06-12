"""Markdown report generation for a completed evaluation run.

Reads `manifest.json`, `judge-scores.json`, and every per-cell CandidateRun
JSON from the run directory; writes `REPORT.md` with leaderboards, winners,
a paste-ready routing.json update block, and (when baselines are present) a
delegation-matrix summary.

The report is the human-readable end-state of an evaluation; you can re-skim
a year-old run and reconstruct what won and why.

v2 additions:

* `% of Judge` column on per-task leaderboards (only when baselines present).
* Per-task degradation callout — one-line prose mechanically derived from
  combined score + baseline pct.
* Overall delegation matrix at the bottom — Recommended Model / % of Judge /
  Combined / Action ("Delegate freely" / "Delegate with monitor" / "Keep on
  judge") per task.
"""

from __future__ import annotations

import json
from pathlib import Path

from .batch import _load_runs_from_dir
from .scoring import (
    aggregate_per_task_candidate,
    degradation_callout,
    delegation_tier,
    winner_per_task,
)


def _load_judge_scores(run_dir: Path) -> list[dict]:
    """Read `judge-scores.json`. Raises if absent — the judge must run first."""
    p = run_dir / "judge-scores.json"
    if not p.exists():
        raise FileNotFoundError(
            f"judge-scores.json missing at {p}. Run the judge step first: read "
            f"judge-batch.json and write scores to {p}."
        )
    data = json.loads(p.read_text())
    if not isinstance(data, list):
        raise ValueError(
            f"judge-scores.json must be a JSON array; got {type(data).__name__}."
        )
    return data


def _runs_to_dict_list(runs: list) -> list[dict]:
    """Convert CandidateRun pydantic objects to plain dicts for `aggregate_*`."""
    return [r.model_dump() for r in runs]


def _format_stdev(metrics: dict) -> str:
    """Render the Stdev cell: per-cell quality std-dev, ⚡-flagged above 10.

    Std-dev over N=5 samples is a coarse stability signal; > 10 points on the
    0-100 scale means the median is shaky — flag it so the reader treats the
    routing suggestion with caution.
    """
    stdev = metrics.get("stdev_quality", 0.0) or 0.0
    flag = " ⚡" if stdev > 10 else ""
    return f"{stdev:.1f}{flag}"


def _format_leaderboard_row(
    candidate: str,
    metrics: dict,
    notes_preview: str,
    *,
    show_baseline_pct: bool,
) -> str:
    """Render one row of the per-task leaderboard table.

    Columns when baseline is shown:
        Candidate | Quality | Stdev | Speed | Combined | % of Judge | p50 latency | Notes
    Columns when no baseline:
        Candidate | Quality | Stdev | Speed | Combined | p50 latency | Errors | Notes
    """
    if show_baseline_pct:
        pct = metrics["quality_pct_of_baseline"]
        pct_str = f"{pct:.0f}%" if pct is not None else "—"
        return (
            f"| {candidate} "
            f"| {metrics['mean_quality']:.1f} "
            f"| {_format_stdev(metrics)} "
            f"| {metrics['mean_speed']:.1f} "
            f"| {metrics['mean_combined']:.1f} "
            f"| {pct_str} "
            f"| {metrics['p50_latency_ms']:.0f} "
            f"| {notes_preview} |"
        )
    return (
        f"| {candidate} "
        f"| {metrics['mean_quality']:.1f} "
        f"| {_format_stdev(metrics)} "
        f"| {metrics['mean_speed']:.1f} "
        f"| {metrics['mean_combined']:.1f} "
        f"| {metrics['p50_latency_ms']:.0f} "
        f"| {metrics['n_errors']} "
        f"| {notes_preview} |"
    )


def _aggregate_notes(notes: list[str], max_chars: int = 140) -> str:
    """Compact judge notes for the report table.

    Joins per-scenario notes with `; ` and truncates to `max_chars` characters
    so the table stays readable. Pipe and newline characters are stripped so
    they don't break the markdown table.
    """
    joined = "; ".join(n.strip() for n in notes if n.strip())
    joined = joined.replace("|", "/").replace("\n", " ")
    if len(joined) > max_chars:
        joined = joined[: max_chars - 1] + "…"
    return joined or "—"


def _baseline_present(aggregated: dict[tuple[str, str], dict]) -> bool:
    """True if any aggregated row carries a baseline-pct value."""
    return any(m.get("quality_pct_of_baseline") is not None for m in aggregated.values())


def generate_report(
    run_dir: Path,
    *,
    quality_weight: float = 0.7,
) -> Path:
    """Generate `REPORT.md` in the run directory.

    Args:
        run_dir: Path to a completed run (must contain manifest.json,
            judge-scores.json, and per-cell CandidateRun JSON files).
        quality_weight: Forwarded to `aggregate_per_task_candidate`.

    Returns:
        Path to the written `REPORT.md`.
    """
    manifest = json.loads((run_dir / "manifest.json").read_text())
    judge_scores = _load_judge_scores(run_dir)
    runs = _load_runs_from_dir(run_dir)

    aggregated = aggregate_per_task_candidate(
        judge_scores,
        _runs_to_dict_list(runs),
        quality_weight=quality_weight,
    )
    winners = winner_per_task(aggregated)
    show_baseline = _baseline_present(aggregated)
    # Judge provenance: authoritative source is judge-batch.json (set at
    # prepare-batch via --judge); manifest is a fallback. Only fall back to a
    # hardcoded default with a loud warning — never silently bake the wrong
    # judge id into routing.json (RCA stress-test HIGH).
    judge_model = manifest.get("judge_model")
    batch_path = run_dir / "judge-batch.json"
    if batch_path.exists():
        jm = json.loads(batch_path.read_text()).get("judge_model")
        if jm:
            judge_model = jm
    if not judge_model:
        judge_model = "interactive-judge-session"
        import sys as _sys
        _sys.stderr.write(
            "[evaluation] WARN: judge_model not found in judge-batch.json or "
            "manifest.json; defaulting to "
            f"{judge_model!r}. Verify the report's judge attribution.\n"
        )

    # Group aggregated metrics by task for per-task tables.
    by_task: dict[str, list[tuple[str, dict]]] = {}
    for (task_id, candidate), metrics in aggregated.items():
        by_task.setdefault(task_id, []).append((candidate, metrics))
    for task_id in by_task:
        # Sort within task by combined desc, then quality desc, then p50 asc.
        by_task[task_id].sort(
            key=lambda kv: (
                -kv[1]["mean_combined"],
                -kv[1]["mean_quality"],
                kv[1]["p50_latency_ms"],
            )
        )

    lines: list[str] = []
    lines.append(f"# Evaluation Report — {run_dir.name}")
    lines.append("")
    lines.append(f"- **Started:** {manifest['started_at']}")
    lines.append(f"- **Completed:** {manifest['completed_at']}")
    lines.append(f"- **Candidates:** {', '.join(manifest['candidates'])}")
    lines.append(f"- **Tasks:** {', '.join(t['id'] for t in manifest['tasks'])}")
    lines.append(f"- **Judge:** `{judge_model}`")
    lines.append("- **Scoring scale:** 0-100 per axis (quality, speed, combined).")
    lines.append(f"- **Quality weight:** {quality_weight:.2f} (speed weight = {1 - quality_weight:.2f})")
    if show_baseline:
        lines.append("- **Baselines:** present — candidates scored both absolutely and as % of judge quality.")
    else:
        lines.append("- **Baselines:** absent — % of judge column omitted. To enable, run `init-baselines`, fill them in, and re-run `prepare-batch`.")
    lines.append("")
    lines.append("Frontier judge sits OUTSIDE the candidate pool. Quality scores")
    lines.append("come from the interactive judge (`judge-scores.json`); speed is")
    lines.append("binned from measured `latency_ms` per cell.")
    lines.append("")

    # --- Per-task leaderboards ----------------------------------------------
    lines.append("## Per-task leaderboards")
    lines.append("")
    for task_meta in manifest["tasks"]:
        task_id = task_meta["id"]
        entries = by_task.get(task_id, [])
        winner = winners.get(task_id, "—")
        lines.append(f"### {task_id}")
        lines.append("")
        lines.append(f"_{task_meta['description']}_")
        lines.append("")
        if not entries:
            lines.append("(no scored candidates)")
            lines.append("")
            continue
        if show_baseline:
            lines.append("| Candidate | Quality | Stdev | Speed | Combined | % of Judge | p50 ms | Notes |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
        else:
            lines.append("| Candidate | Quality | Stdev | Speed | Combined | p50 ms | Errors | Notes |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
        for candidate, metrics in entries:
            notes_preview = _aggregate_notes(metrics["notes"])
            lines.append(
                _format_leaderboard_row(
                    candidate, metrics, notes_preview, show_baseline_pct=show_baseline
                )
            )
        lines.append("")
        lines.append(f"**Winner:** `{winner}`")

        # Degradation callout — one-line prose derived mechanically from the
        # winner's metrics. Always emitted; reads as "no baseline" when
        # baselines aren't available.
        winner_metrics = next((m for c, m in entries if c == winner), None)
        if winner_metrics is not None:
            callout = degradation_callout(
                combined=winner_metrics["mean_combined"],
                baseline_pct=winner_metrics["quality_pct_of_baseline"],
                candidate=winner,
                judge_model=judge_model,
                task_id=task_id,
                p50_latency_ms=winner_metrics["p50_latency_ms"],
                baseline_p50_latency_ms=None,  # baseline latency isn't measured
            )
            lines.append("")
            lines.append(f"_{callout}_")
        lines.append("")

    # --- Routing recommendation --------------------------------------------
    lines.append("## Suggested routing.json update")
    lines.append("")
    lines.append("Paste into `data/routing.json`, merging by task id:")
    lines.append("")
    lines.append("```json")
    routing_update: dict[str, dict] = {}
    for task_id, candidate in winners.items():
        metrics = aggregated[(task_id, candidate)]
        entry = {
            "model": candidate,
            "p50_quality": round(metrics["mean_quality"], 2),
            "stdev_quality": round(metrics.get("stdev_quality", 0.0) or 0.0, 2),
            "p50_combined": round(metrics["mean_combined"], 2),
            "p50_latency_ms": round(metrics["p50_latency_ms"], 0),
            "judge_model": judge_model,
            "last_evaluated_at": manifest["completed_at"],
        }
        if metrics["quality_pct_of_baseline"] is not None:
            entry["quality_pct_of_judge"] = round(metrics["quality_pct_of_baseline"], 1)
        routing_update[task_id] = entry
    lines.append(json.dumps(routing_update, indent=2))
    lines.append("```")
    lines.append("")

    # --- Insights (best-effort, prose) -------------------------------------
    lines.append("## Aggregated insights")
    lines.append("")
    # Winner frequency — does one model dominate?
    win_counts: dict[str, int] = {}
    for c in winners.values():
        win_counts[c] = win_counts.get(c, 0) + 1
    if win_counts:
        ranked = sorted(win_counts.items(), key=lambda kv: -kv[1])
        top_candidate, top_count = ranked[0]
        lines.append(
            f"- `{top_candidate}` wins {top_count}/{len(winners)} task(s)."
        )
    # Per-candidate combined-score average across tasks.
    cross_task: dict[str, list[float]] = {}
    for (_t, c), m in aggregated.items():
        cross_task.setdefault(c, []).append(m["mean_combined"])
    if cross_task:
        avg_combined = {
            c: sum(scores) / len(scores) for c, scores in cross_task.items()
        }
        ordered = sorted(avg_combined.items(), key=lambda kv: -kv[1])
        lines.append("- Mean combined score across tasks:")
        for c, s in ordered:
            lines.append(f"  - `{c}`: {s:.1f}")
    # Error-rate flag
    err_models: list[str] = []
    err_by_candidate: dict[str, int] = {}
    for (_t, c), m in aggregated.items():
        err_by_candidate[c] = err_by_candidate.get(c, 0) + m["n_errors"]
    for c, n in err_by_candidate.items():
        if n > 0:
            err_models.append(f"`{c}` ({n} error(s))")
    if err_models:
        lines.append(f"- Errors recorded: {', '.join(err_models)}.")
    lines.append("")

    # --- Overall delegation matrix (only when baselines present) -----------
    if show_baseline:
        lines.append("## Overall Delegation Matrix")
        lines.append("")
        lines.append(
            f"Mechanical recommendation per task based on % of `{judge_model}` "
            f"quality. Thresholds: ≥ 80% = delegate freely, 60-80% = "
            f"delegate with monitor, < 60% = keep on judge."
        )
        lines.append("")
        lines.append("| Task | Recommended Model | % of Judge Quality | Combined Score | Action |")
        lines.append("|---|---|---:|---:|---|")
        for task_meta in manifest["tasks"]:
            task_id = task_meta["id"]
            winner = winners.get(task_id)
            if winner is None:
                continue
            metrics = aggregated[(task_id, winner)]
            pct = metrics["quality_pct_of_baseline"]
            pct_str = f"{pct:.0f}%" if pct is not None else "—"
            action = delegation_tier(pct)
            lines.append(
                f"| {task_id} | `{winner}` | {pct_str} | "
                f"{metrics['mean_combined']:.1f} | {action} |"
            )
        lines.append("")

    out_path = run_dir / "REPORT.md"
    out_path.write_text("\n".join(lines) + "\n")
    return out_path
