"""Compose CorrectnessReport + EffectsReport + QualityReport into AUDIT-REPORT.md.

The markdown follows a 4-section layout::

  1. Front matter (app, window, judge, thresholds)
  2. HEADLINE — the one sentence the reader actually needs
  3. Correctness section (pass/fail per check)
  4. Effects section (savings, latency, success rate)
  5. Quality section (per-slot drift table + alert tier)
  6. Recommendations (which slots need re-bake)

The headline is intentionally first-on-the-page below the front matter — the
report exists to surface ONE number per app per week: "reduced Opus usage by
X% staying at Y% of Opus quality."
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .config import AuditConfig
from .correctness import CorrectnessReport
from .effects import EffectsReport
from .quality import QualityReport


# Glyphs are reused across sections; centralized so tests can match on them.
_GLYPH_OK = "PASS"
_GLYPH_FAIL = "FAIL"
_GLYPH_WARN = "WARN"


def _headline(effects: EffectsReport, quality: QualityReport) -> str:
    """Return the one-sentence headline.

    Three cases:
      * Both X and Y available    → full sentence
      * X but no Y (no samples)   → cost claim only, flag Y as TBD
      * Neither                   → diagnostic — usually empty window
    """
    x_pct = effects.overall_savings_pct
    y_pct = quality.overall_quality_pct_of_baseline

    if effects.total_calls == 0:
        return (
            "**No traffic in this window** — the audit can't compute savings or "
            "drift. Verify the router-plugin is wired + writing telemetry, then "
            "re-run after the next batch of routed calls."
        )

    if y_pct is None:
        return (
            f"**Reduced `{effects.frontier_model}` usage by {x_pct:.1f}%** "
            f"(USD cost vs. counterfactual). Quality drift unknown — no samples "
            f"were scored by the judge yet (run `prepare_quality_batch` and the "
            f"judge step to fill in the second half of the headline)."
        )

    return (
        f"**Reduced `{effects.frontier_model}` usage by {x_pct:.1f}%** "
        f"while staying at **{y_pct:.1f}%** of `{effects.frontier_model}` "
        f"quality across {effects.total_calls} routed call(s) in the window."
    )


def _format_correctness_section(report: CorrectnessReport) -> list[str]:
    lines: list[str] = []
    lines.append("## Correctness")
    lines.append("")
    glyph = _GLYPH_OK if report.overall_pass else _GLYPH_FAIL
    lines.append(
        f"- Overall: **{glyph}** — {report.alarm_count} alarm(s) across "
        f"{len(report.slots)} slot(s)."
    )
    lines.append(f"- Routing source: `{report.routing_json_path}`")
    lines.append("")

    # Completeness gate — surface BEFORE the per-slot table so the reader
    # sees the structural problem first. If incomplete_scope_alarms is
    # non-empty, the per-slot table is empty (short-circuited in correctness.py).
    if report.incomplete_scope_alarms:
        lines.append("### 🛑 Incomplete scope — audit cannot complete")
        lines.append("")
        lines.append(
            "The completeness rule requires every slot in `slots_in_scope` "
            "to be measured (`last_baked_at` set; `queue-for-human` only "
            "with a documented bake-off verdict). The audit is short-circuited "
            "until every slot below is either baked or removed from scope."
        )
        lines.append("")
        for a in report.incomplete_scope_alarms:
            lines.append(f"- {a}")
        lines.append("")
        lines.append(
            "**Unblock:** author scenarios for the missing slots in the eval "
            "tasks YAML, run `python -m orchestrator.evaluation run …`, "
            "update `routing.json` with the winner, then re-run this audit."
        )
        lines.append("")
        return lines  # no per-slot table to render when scope is incomplete
    lines.append(
        "| Slot | Calls | Primary OK | Fallback% | Error% | Status |"
    )
    lines.append("|---|---:|---:|---:|---:|---|")
    for s in report.slots:
        status = _GLYPH_OK if not s.alarms else _GLYPH_FAIL
        primary_ok = (
            f"{s.n_with_expected_primary}/{s.n_calls}" if s.n_calls else "0/0"
        )
        lines.append(
            f"| `{s.slot}` | {s.n_calls} | {primary_ok} | "
            f"{s.fallback_rate_pct:.1f}% | {s.error_rate_pct:.1f}% | {status} |"
        )
    lines.append("")
    # Per-slot alarm detail — collapse no-alarm slots to keep the report tight.
    any_alarms = any(s.alarms for s in report.slots)
    if any_alarms:
        lines.append("### Correctness alarms")
        lines.append("")
        for s in report.slots:
            if not s.alarms:
                continue
            lines.append(f"- `{s.slot}`:")
            for a in s.alarms:
                lines.append(f"  - {a}")
        lines.append("")
    return lines


def _format_effects_section(report: EffectsReport) -> list[str]:
    lines: list[str] = []
    lines.append("## Effects (cost + latency)")
    lines.append("")
    lines.append(
        f"- Frontier counterfactual model: `{report.frontier_model}`"
    )
    lines.append(
        f"- Total calls in window: **{report.total_calls}**"
    )
    lines.append(
        f"- Actual spend: **${report.total_actual_cost_usd:.4f}**"
    )
    lines.append(
        f"- Counterfactual spend (had we used frontier for everything): "
        f"**${report.total_counterfactual_cost_usd:.4f}**"
    )
    lines.append(
        f"- Savings: **${report.total_savings_usd:.4f}** "
        f"(**{report.overall_savings_pct:.1f}%** displacement)"
    )
    # Accounting caveats — these models' calls claim ZERO displacement, so the
    # savings figure above understates rather than fabricates. Surfaced here
    # (not just stderr) so the flag survives into the artifact people read.
    if report.unpriced_models:
        models = ", ".join(f"`{m}`" for m in report.unpriced_models)
        lines.append(
            f"- **WARN — models missing from the pricing table:** {models}. "
            f"Their calls are counted at actual cost with zero claimed "
            f"displacement. Add pricing entries for accurate accounting."
        )
    if report.cost_unverified_models:
        models = ", ".join(f"`{m}`" for m in report.cost_unverified_models)
        lines.append(
            f"- **WARN — paid models with no logged cost:** {models}. "
            f"Savings for these calls are unverifiable and excluded from the "
            f"claimed savings. Fix the plugin's cost logging."
        )
    lines.append("")
    lines.append(
        "| Slot | Calls | Actual $ | Counter $ | Savings% | p50 ms | p95 ms | Success% |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in report.slots:
        lines.append(
            f"| `{s.slot}` | {s.n_calls} | "
            f"${s.actual_cost_usd:.4f} | ${s.counterfactual_cost_usd:.4f} | "
            f"{s.savings_pct:.1f}% | {s.p50_latency_ms:.0f} | "
            f"{s.p95_latency_ms:.0f} | {s.success_rate_pct:.1f}% |"
        )
    lines.append("")
    lines.append(
        "_Counterfactual cost is a coarse estimate: when the candidate is a "
        "local model (zero recorded cost), the audit assumes a nominal "
        "500-input / 1500-output call. Refine by adding token columns to "
        "`routing_decisions`._"
    )
    lines.append("")
    return lines


def _format_quality_section(report: QualityReport) -> list[str]:
    lines: list[str] = []
    lines.append("## Quality drift (sampled)")
    lines.append("")
    lines.append(
        f"- Judge: `{report.judge_model}`"
    )
    lines.append(
        f"- WARN below: **{report.warn_threshold_pct:.1f}%** of frontier"
    )
    lines.append(
        f"- RE-BAKE below: **{report.rebake_threshold_pct:.1f}%** of frontier"
    )
    lines.append("")
    lines.append(
        "| Slot | Samples | Current Mean | Baseline %-of-judge | Drift | Alert |"
    )
    lines.append("|---|---:|---:|---:|---:|---|")
    for s in report.slots:
        baseline_str = (
            f"{s.baseline_pct_of_judge:.1f}%" if s.baseline_pct_of_judge is not None else "—"
        )
        drift_str = (
            f"{s.drift_pct:+.1f}" if s.drift_pct is not None else "—"
        )
        alert_emoji = {
            "ok": _GLYPH_OK,
            "warn": _GLYPH_WARN,
            "rebake": _GLYPH_FAIL,
            "unknown": "n/a",
            "no_samples": "n/a",
        }.get(s.alert, s.alert)
        current_str = (
            f"{s.mean_quality:.1f}" if s.n_samples else "—"
        )
        lines.append(
            f"| `{s.slot}` | {s.n_samples} | {current_str} | {baseline_str} | "
            f"{drift_str} | {alert_emoji} ({s.alert}) |"
        )
    lines.append("")
    # UNVERIFIED slots — 'unknown' (no baseline) and 'no_samples' both mean the
    # audit could NOT assess drift; finalize exits non-zero for them. Spell
    # that out so an "n/a" row can't be skim-read as healthy.
    missing_baseline = [s for s in report.slots if s.alert == "unknown"]
    for s in missing_baseline:
        lines.append(
            f"- **UNVERIFIED:** `{s.slot}` has no `quality_pct_of_judge` "
            f"baseline in routing.json — drift cannot be assessed. The slot "
            f"counts as unverified (non-zero exit), not healthy."
        )
    if missing_baseline:
        lines.append("")
    if not any(s.n_samples for s in report.slots):
        lines.append(
            "_No samples were available for any slot in this window. The router "
            "plugin's sampling sink isn't wired yet — see the TODO in the schema._"
        )
        lines.append("")
    return lines


def _format_recommendations(quality: QualityReport, correctness: CorrectnessReport) -> list[str]:
    """Surface the actionable subset at the end."""
    lines: list[str] = []
    lines.append("## Recommendations")
    lines.append("")
    rebake_slots = quality.needs_rebake()
    warn_slots = quality.needs_warn()
    if rebake_slots:
        lines.append(
            f"- **RE-BAKE:** {', '.join(f'`{s}`' for s in rebake_slots)} — "
            f"quality below {quality.rebake_threshold_pct}% of frontier. "
            f"Re-run `python -m orchestrator.evaluation run` to refresh winners."
        )
    if warn_slots:
        lines.append(
            f"- **WARN:** {', '.join(f'`{s}`' for s in warn_slots)} — "
            f"quality between {quality.rebake_threshold_pct}% and "
            f"{quality.warn_threshold_pct}% of frontier. Monitor; consider "
            f"re-bake in the next cycle if trend continues."
        )
    correctness_fails = [s for s in correctness.slots if s.alarms]
    if correctness_fails:
        lines.append(
            f"- **CORRECTNESS:** investigate "
            f"{', '.join(f'`{s.slot}`' for s in correctness_fails)} — "
            f"see Correctness alarms above."
        )
    unverified_slots = quality.unverified()
    if unverified_slots:
        lines.append(
            f"- **UNVERIFIED:** {', '.join(f'`{s}`' for s in unverified_slots)} — "
            f"no samples and/or no baseline to assess drift against. These are "
            f"failures (`finalize` exits non-zero), not passes."
        )
    if not (rebake_slots or warn_slots or correctness_fails or unverified_slots):
        lines.append(
            "- All checks pass at current thresholds. No action required."
        )
    lines.append("")
    return lines


def compose_audit_report(
    *,
    cfg: AuditConfig,
    correctness: CorrectnessReport,
    effects: EffectsReport,
    quality: QualityReport,
    out_path: Path,
    generated_at: datetime | None = None,
) -> Path:
    """Write AUDIT-REPORT.md at `out_path`.

    Args:
        cfg: AuditConfig — front-matter source.
        correctness: Result of `run_correctness_audit`.
        effects: Result of `compute_effects_report`.
        quality: Result of `finalize_quality`.
        out_path: Where to write the markdown.
        generated_at: Override for the timestamp in the front matter.

    Returns:
        Path to the written file.
    """
    generated_at = generated_at or datetime.now().astimezone()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"# Audit Report — `{cfg.app_name}`")
    lines.append("")
    lines.append(f"- **Generated:** {generated_at.isoformat()}")
    lines.append(f"- **Window:** {correctness.window_start_iso} → {correctness.window_end_iso}")
    lines.append(f"- **Lookback:** {cfg.lookback_days} day(s)")
    lines.append(f"- **Judge:** `{cfg.judge_model}`")
    lines.append(f"- **Slots in scope:** {', '.join(f'`{s}`' for s in cfg.slots_in_scope)}")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(_headline(effects, quality))
    lines.append("")
    lines.extend(_format_correctness_section(correctness))
    lines.extend(_format_effects_section(effects))
    lines.extend(_format_quality_section(quality))
    lines.extend(_format_recommendations(quality, correctness))

    out_path.write_text("\n".join(lines) + "\n")
    return out_path
