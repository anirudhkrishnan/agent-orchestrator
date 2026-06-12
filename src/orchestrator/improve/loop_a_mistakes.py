"""Loop A — learn from the system's OWN mistakes.

The closed loop: production calls that went badly become tomorrow's eval
scenarios, and their slots get queued for re-bake. This is active-learning /
hard-example mining — the eval corpus grows from real failures instead of
staying frozen at authoring time.

Two failure sources, both from telemetry:
  1. Negative user feedback on a routed call (routing_decisions.user_feedback).
  2. A sampled routed call the audit judge scored below the slot's re-bake line
     (routed_call_samples + audit quality scores).

What it does NOT do: auto-append to tasks-v1.yaml. New scenarios must satisfy
the scenario-realism rule (real provenance, reviewed). So Loop A STAGES
candidate scenarios — each carrying the production call as provenance — into a
review file, and writes a re-bake queue. A human (or a gated promote step)
moves staged scenarios into the live task set. Self-improving, but not
self-deceiving (see guard.py — the hard rule).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from orchestrator.telemetry import db as tdb

from .state import load_json_state

# Feedback strings that mark a call as a failure worth learning from.
DEFAULT_BAD_FEEDBACK = ("bad", "thumbs_down", "wrong", "reject", "error")

# Recovery hint for corrupt Loop A state files (staging / queue).
_STATE_FIX = ("move the corrupt file aside (or delete it) and re-run harvest; "
              "it will be rebuilt from the current lookback window")


@dataclass
class HarvestedFailure:
    """One production call worth turning into a regression scenario."""

    slot: str
    model: str
    reason: str           # "negative_feedback" | "below_rebake_threshold"
    detail: str           # the feedback value, or the score vs threshold
    input_excerpt: str    # the production input (scenario seed)
    output_excerpt: str   # what the model produced (for the reviewer)
    observed_at: str
    source: str           # "routing_decisions" | "routed_call_samples"
    failure_id: str = ""  # stable root identity (for cross-run queue dedup)


@dataclass
class HarvestReport:
    app_name: str
    window_start_iso: str
    failures: list[HarvestedFailure] = field(default_factory=list)
    slots_to_rebake: list[str] = field(default_factory=list)


def _decision_failure_id(row: dict) -> str:
    """Stable identity of a routing_decisions failure across harvest runs.

    Daily harvests over a multi-day lookback window re-see the same rows; the
    re-bake queue dedups on this id so a failure is counted once, ever. Prefer
    the DB row id; fall back to (timestamp, excerpt) for id-less rows."""
    rid = row.get("id")
    if rid is not None:
        return f"routing_decisions:{rid}"
    return f"routing_decisions:{row.get('timestamp', '')}:{(row.get('message_excerpt') or '')[:80]}"


def harvest_failures(
    app_name: str,
    *,
    now: datetime | None = None,
    lookback_days: int = 7,
    bad_feedback_values: tuple[str, ...] = DEFAULT_BAD_FEEDBACK,
    sample_scores: dict[str, float] | None = None,
    rebake_threshold_pct: float = 80.0,
) -> HarvestReport:
    """Collect failed production calls in the window.

    Args:
        app_name: app to scope to.
        now / lookback_days: window.
        bad_feedback_values: user_feedback strings that count as failures.
        sample_scores: optional {sample_id: judged_quality_0_100} from an audit
            quality run. Samples scoring below `rebake_threshold_pct` are harvested.
            The CLI wires this from `harvest --audit-scores <judge-scores.json>`.
        rebake_threshold_pct: the below-this-is-a-failure line for sampled scores.
            Applied per-sample (one bad call is one hard example), unlike the
            audit's slot-level drift threshold of the same name.

    Returns:
        HarvestReport with failures + the distinct slots they implicate.
    """
    now = now or datetime.now(timezone.utc)
    window_start = (now - timedelta(days=lookback_days)).isoformat()
    failures: list[HarvestedFailure] = []

    # Source 1: negative user feedback on routed decisions.
    rows = tdb.routing_decisions_for_audit(app_name=app_name, slots=None, since_iso=window_start)
    for r in rows:
        fb = (r.get("user_feedback") or "").strip().lower()
        if fb and fb in bad_feedback_values:
            failures.append(HarvestedFailure(
                slot=r.get("classified_slot") or "(unclassified)",
                model=r.get("selected_model") or "(unknown)",
                reason="negative_feedback",
                detail=f"user_feedback={fb!r}",
                input_excerpt=(r.get("message_excerpt") or "")[:2000],
                output_excerpt="",
                observed_at=r.get("timestamp") or "",
                source="routing_decisions",
                failure_id=_decision_failure_id(r),
            ))

    # Source 2: sampled calls the audit judge scored below the re-bake line.
    if sample_scores:
        samples = tdb.samples_for_audit(app_name=app_name, slot=None, since_iso=window_start)
        for s in samples:
            sid = s.get("sample_id")
            q = sample_scores.get(sid)
            if q is not None and q < rebake_threshold_pct:
                failures.append(HarvestedFailure(
                    slot=s.get("slot") or "(unclassified)",
                    model=s.get("candidate_model") or "(unknown)",
                    reason="below_rebake_threshold",
                    detail=f"judged {q:.1f} < {rebake_threshold_pct:.0f}",
                    input_excerpt=(s.get("input_text") or "")[:2000],
                    output_excerpt=(s.get("output_text") or "")[:2000],
                    observed_at=s.get("routed_at") or "",
                    source="routed_call_samples",
                    failure_id=f"routed_call_samples:{sid}",
                ))

    slots = sorted({f.slot for f in failures if f.slot and not f.slot.startswith("(")})
    return HarvestReport(
        app_name=app_name,
        window_start_iso=window_start,
        failures=failures,
        slots_to_rebake=slots,
    )


def stage_scenarios(report: HarvestReport, staging_path: Path) -> Path:
    """Write harvested failures as REVIEW-gated candidate scenarios.

    The output is a staging file (not tasks-v1.yaml). Each entry carries the
    production call as provenance, satisfying the scenario-realism rule. A human
    or a gated promote step reviews + moves these into the live task set.
    """
    staging_path = Path(staging_path)
    staging_path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if staging_path.exists():
        existing = load_json_state(
            staging_path, expect=dict, how_to_fix=_STATE_FIX
        ).get("candidate_scenarios", [])
    seen = {(c["slot"], c["input"]) for c in existing}
    for f in report.failures:
        key = (f.slot, f.input_excerpt)
        if key in seen or not f.input_excerpt:
            continue
        existing.append({
            "slot": f.slot,
            "input": f.input_excerpt,
            "provenance": {
                "source": f.source, "model": f.model, "reason": f.reason,
                "detail": f.detail, "observed_at": f.observed_at,
                "failure_id": f.failure_id,
            },
            "status": "needs_review",
            "note": "Harvested by Loop A from a real production failure. Review, "
                    "add an expected-output note, then promote into tasks-v1.yaml.",
        })
        seen.add(key)
    staging_path.write_text(json.dumps(
        {"_README": "Loop A staging — review-gated candidate scenarios from real "
                    "production failures. Promote into tasks-v1.yaml after review.",
         "candidate_scenarios": existing}, indent=2) + "\n")
    return staging_path


def write_rebake_queue(report: HarvestReport, queue_path: Path) -> Path:
    """Merge the implicated slots into the re-bake queue state file.

    Loop B / the human reads this to decide what to re-bake. We only QUEUE —
    we never auto-run a bake-off that mutates routing without the integrity
    gate + (for routing changes) human confirmation.

    Entries dedup on (slot, failure_id) across runs: a daily harvest re-sees
    the same rows for the whole lookback window, so blindly adding per-run
    counts would inflate `count` on every run for the same root failures.
    """
    queue_path = Path(queue_path)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue = {}
    if queue_path.exists():
        queue = load_json_state(queue_path, expect=dict, how_to_fix=_STATE_FIX)
    slots = queue.get("slots", {})
    for slot in report.slots_to_rebake:
        entry = slots.get(slot, {"reasons": [], "first_queued": report.window_start_iso})
        slot_failures = [f for f in report.failures if f.slot == slot]
        known = set(entry.get("failure_ids", []))
        known.update(f.failure_id for f in slot_failures if f.failure_id)
        entry["failure_ids"] = sorted(known)
        entry["count"] = len(known)  # = distinct root failures, ever
        entry["reasons"] = sorted(set(entry["reasons"]) | {f.reason for f in slot_failures})
        entry["app"] = report.app_name
        slots[slot] = entry
    queue["slots"] = slots
    queue["_README"] = ("Loop A re-bake queue. A slot here has accumulated real "
                        "production failures and should be re-baked (new scenarios "
                        "staged alongside). Re-bake never auto-commits routing — "
                        "integrity gate + human confirm required.")
    queue_path.write_text(json.dumps(queue, indent=2) + "\n")
    return queue_path
