"""Quality audit — has the quality drifted since the bake-off baseline?

Two-phase, interactive-judge protocol (same shape as the bake-off):

  Phase A: `prepare_quality_batch(cfg)`
    - Read routed_call_samples for the window (written by the router plugin).
    - Per slot, keep the MOST RECENT ~`sample_rate` fraction of the window's
      samples (at least 1, capped at `max_samples_per_slot`) — recent traffic
      is the honest drift signal.
    - Bundle into `judge-batch.json` with instructions for the judge.
    - Print "READY FOR JUDGE" — an interactive judge session reads + scores.

  Phase B: `finalize_quality(cfg)`
    - Read `judge-scores.json` produced by the judge.
    - Compute current quality % per slot.
    - Compare to `quality_pct_of_judge` baseline in routing.json.
    - Classify into ok / warn / rebake per the per-app thresholds.

Quality scores from the audit are NOT identical to the bake-off's
"quality_pct_of_baseline" — that one was per (task, scenario) and computed
against a judge-authored gold standard for that exact input. The audit asks
the judge to score live samples against the same rubric the bake-off used
for the slot, then compares means. The shapes line up because both speak
"mean_quality_score / 0-100" — small notational gap, same semantic axis.

The audit DOES NOT re-author baselines. Rebaking is the rebake step's job;
the audit's role is to surface drift so a human can DECIDE to rebake.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from orchestrator.telemetry import db as tdb

from .config import AuditConfig


@dataclass
class SlotQuality:
    """Per-slot quality + drift result."""

    slot: str
    n_samples: int
    mean_quality: float
    """Judge-scored 0-100 mean for the candidate over sampled calls."""
    baseline_pct_of_judge: float | None
    """The `quality_pct_of_judge` from routing.json. None if slot isn't baked."""
    drift_pct: float | None
    """current - baseline. Negative means quality dropped."""
    alert: str
    """One of: 'ok', 'warn', 'rebake', 'unknown' (no baseline), 'no_samples'."""
    notes: list[str] = field(default_factory=list)


@dataclass
class QualityReport:
    """Quality audit aggregated across slots."""

    app_name: str
    window_start_iso: str
    window_end_iso: str
    judge_model: str
    warn_threshold_pct: float
    rebake_threshold_pct: float
    slots: list[SlotQuality] = field(default_factory=list)

    @property
    def overall_quality_pct_of_baseline(self) -> float | None:
        """Mean of per-slot mean_quality across slots that have samples + baselines.

        Headline Y value. Returns None if no slot has both.
        """
        usable = [s for s in self.slots if s.baseline_pct_of_judge is not None and s.n_samples > 0]
        if not usable:
            return None
        # We want Y in "% of frontier quality" terms. The slot's baseline_pct_of_judge
        # is the bake-off-time % of frontier; the audit's mean_quality is the current
        # raw 0-100 score the judge gave the candidate. Re-express the current score as
        # a fraction of the baseline raw score (which the bake-off persisted as
        # baseline_pct_of_judge of frontier-100). Drift_pct already encodes the delta;
        # the headline is baseline + drift, clamped to [0, 200].
        values = []
        for s in usable:
            assert s.baseline_pct_of_judge is not None  # type narrowing
            current = s.baseline_pct_of_judge + (s.drift_pct or 0.0)
            values.append(max(0.0, min(200.0, current)))
        return sum(values) / len(values)

    def needs_rebake(self) -> list[str]:
        """Return list of slot names whose alert is 'rebake'."""
        return [s.slot for s in self.slots if s.alert == "rebake"]

    def needs_warn(self) -> list[str]:
        """Return list of slot names whose alert is 'warn'."""
        return [s.slot for s in self.slots if s.alert == "warn"]

    def insufficient_samples(self) -> list[str]:
        """Slots that had zero sampled calls to assess drift.

        A slot with `alert == 'no_samples'` is NOT healthy — it means the audit
        could not verify it at all (it may have silently stopped being judged).
        The CLI surfaces this as a non-zero verdict so "unverified" can't be
        mistaken for "passing" (RCA stress-test HIGH)."""
        return [s.slot for s in self.slots if s.alert == "no_samples"]

    def missing_baseline(self) -> list[str]:
        """Slots that were sampled but have no baseline to compare against.

        A slot with `alert == 'unknown'` is baked + in scope but routing.json
        carries no `quality_pct_of_judge` for it — drift CANNOT be assessed.
        Same verdict class as `insufficient_samples`: unverified, not healthy.
        The CLI folds these into the non-zero "couldn't verify" exit."""
        return [s.slot for s in self.slots if s.alert == "unknown"]

    def unverified(self) -> list[str]:
        """Union of `insufficient_samples()` + `missing_baseline()`."""
        return self.insufficient_samples() + self.missing_baseline()


# --- Judge batch shape (mirrors evaluation/batch.py vocabulary) ------------


_AUDIT_JUDGE_INSTRUCTIONS = """\
You are the AUDIT judge for the orchestration engine.

This batch contains LIVE routed-call samples taken by the router plugin
over the last {lookback_days} day(s). Score each candidate output on a
0-100 quality scale — the same scale the bake-off uses, so the resulting
mean is directly comparable to `quality_pct_of_judge` in routing.json.

For each item in `items[]`, produce one JSON object with this exact shape:

    {{
      "sample_id": "<copy from input>",
      "mean_quality_score": <float 0-100>,
      "notes": "<1-sentence rationale, plain prose>"
    }}

Scoring guidance:
  - Score the OUTPUT against what a competent answer to INPUT_TEXT looks
    like for this `slot`. The slot name carries the task type (e.g.
    `entity_extraction` → did it extract the right entities? `relevance_triage`
    → did it correctly mark relevance?).
  - If the output is empty / errored / clearly broken, score 0.
  - This is a LIGHTWEIGHT audit — don't over-think individual scenarios.
    The goal is to see drift trends, not to author new ground truth.

Write the full array of result objects (one per item) to:

    {scores_path}

as a JSON array (NOT JSONL). After writing, run:

    python -m orchestrator.audit finalize --app {app_name} --config {config_path}

Number of items to score: {n_items}.
"""


def _window_start_iso(now: datetime, lookback_days: int) -> str:
    return (now - timedelta(days=lookback_days)).isoformat()


def prepare_quality_batch(
    cfg: AuditConfig,
    *,
    out_dir: Path,
    config_path: Path | None = None,
    now: datetime | None = None,
) -> Path:
    """Phase A — bundle sampled calls into a judge batch.

    Args:
        cfg: Loaded AuditConfig.
        out_dir: Directory the judge-batch.json + judge-scores.json land in.
            Created if missing.
        config_path: Recorded in the judge instructions so the judge knows
            which file to point `finalize` at. Optional; if None, the
            instructions say "<your config path>".
        now: Override for "now" (defaults to UTC now).

    Returns:
        Path to the written judge-batch.json.

    Notes:
        If no samples exist for a slot, that slot is silently skipped in the
        batch. The finalize step will mark it `no_samples` in the report.
    """
    now = now or datetime.now(timezone.utc)
    window_start = _window_start_iso(now, cfg.lookback_days)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items: list[dict] = []
    for slot in cfg.slots_in_scope:
        samples = tdb.samples_for_audit(
            app_name=cfg.app_name,
            slot=slot,
            since_iso=window_start,
        )
        if samples:
            # Judge ~sample_rate of the window's samples — at least 1 so a
            # live slot is never silently unjudged, capped so the batch stays
            # manageable. Take the MOST RECENT samples (rows arrive ASC by
            # routed_at): the newest traffic is the honest drift signal —
            # judging the oldest would mask a recent regression.
            n_target = min(
                cfg.max_samples_per_slot,
                max(1, math.ceil(cfg.sample_rate * len(samples))),
            )
            samples = samples[-n_target:]
        for s in samples:
            items.append(
                {
                    "sample_id": s["sample_id"],
                    "slot": s["slot"],
                    "candidate_model": s["candidate_model"],
                    "input_text": s["input_text"],
                    "output_text": s["output_text"],
                    "latency_ms": s["latency_ms"],
                }
            )

    scores_path = out_dir / "judge-scores.json"
    instructions = _AUDIT_JUDGE_INSTRUCTIONS.format(
        lookback_days=cfg.lookback_days,
        scores_path=scores_path,
        app_name=cfg.app_name,
        config_path=str(config_path) if config_path else "<your config path>",
        n_items=len(items),
    )

    batch = {
        "app_name": cfg.app_name,
        "judge_model": cfg.judge_model,
        "window_start": window_start,
        "window_end": now.isoformat(),
        "instructions_for_judge": instructions,
        "items": items,
    }
    batch_path = out_dir / "judge-batch.json"
    batch_path.write_text(json.dumps(batch, indent=2) + "\n")
    return batch_path


def _load_judge_scores(out_dir: Path) -> list[dict]:
    p = out_dir / "judge-scores.json"
    if not p.exists():
        raise FileNotFoundError(
            f"judge-scores.json missing at {p}. The judge step hasn't completed yet; "
            f"have the judge read judge-batch.json from the same directory and write "
            f"scores to this path before running finalize."
        )
    data = json.loads(p.read_text())
    if not isinstance(data, list):
        raise ValueError(
            f"judge-scores.json must be a JSON array; got {type(data).__name__}."
        )
    return data


def _load_baseline_pct_of_judge(routing_json_path: Path) -> dict[str, float | None]:
    """Read `quality_pct_of_judge` per slot from routing.json.

    Slots without a quality_pct_of_judge (NOT YET BAKED) get None.
    """
    raw = json.loads(Path(routing_json_path).read_text())
    out: dict[str, float | None] = {}
    for slot, entry in raw.items():
        if not isinstance(entry, dict) or slot.startswith("_"):
            continue
        v = entry.get("quality_pct_of_judge")
        out[slot] = float(v) if isinstance(v, (int, float)) else None
    return out


def _classify_alert(
    drift_pct: float | None,
    *,
    current_pct_of_judge: float | None,
    warn_threshold: float,
    rebake_threshold: float,
    n_samples: int,
) -> str:
    """Bucket a slot into ok/warn/rebake/unknown/no_samples.

    Rule precedence:
      1. n_samples == 0           → 'no_samples'
      2. baseline absent          → 'unknown'
      3. current < rebake         → 'rebake' (deepest concern wins)
      4. current < warn           → 'warn'
      5. otherwise                → 'ok'

    `current` is `baseline + drift` clamped to [0, 200]. Using `current`
    rather than `drift` directly keeps the thresholds in absolute
    "% of frontier" terms, not deltas.
    """
    if n_samples == 0:
        return "no_samples"
    if current_pct_of_judge is None:
        return "unknown"
    if current_pct_of_judge < rebake_threshold:
        return "rebake"
    if current_pct_of_judge < warn_threshold:
        return "warn"
    return "ok"


def finalize_quality(
    cfg: AuditConfig,
    *,
    out_dir: Path,
    config_anchor: Path | None = None,
    now: datetime | None = None,
) -> QualityReport:
    """Phase B — read judge scores and compute drift.

    Args:
        cfg: Loaded AuditConfig.
        out_dir: Directory containing judge-batch.json + judge-scores.json.
        config_anchor: Directory the AuditConfig was loaded from — used to
            resolve relative routing_json_path.
        now: Override for "now".

    Returns:
        QualityReport ready for composition into AUDIT-REPORT.md.

    Raises:
        FileNotFoundError if judge-scores.json is missing.
    """
    now = now or datetime.now(timezone.utc)
    window_start = _window_start_iso(now, cfg.lookback_days)
    out_dir = Path(out_dir)

    routing_path = cfg.routing_json_path
    if not routing_path.is_absolute() and config_anchor is not None:
        routing_path = (config_anchor / routing_path).resolve()
    baselines = _load_baseline_pct_of_judge(routing_path)

    scores = _load_judge_scores(out_dir)
    # Map sample_id → slot via the batch file so finalize doesn't have to
    # re-query the DB. The batch is small (≤ max_samples_per_slot * n_slots).
    batch_path = out_dir / "judge-batch.json"
    if not batch_path.exists():
        raise FileNotFoundError(
            f"judge-batch.json missing at {batch_path}. The `run` step must have "
            f"produced this; re-run if the file was deleted."
        )
    batch = json.loads(batch_path.read_text())
    slot_by_sample: dict[str, str] = {it["sample_id"]: it["slot"] for it in batch["items"]}

    qualities_by_slot: dict[str, list[float]] = defaultdict(list)
    notes_by_slot: dict[str, list[str]] = defaultdict(list)
    for s in scores:
        sid = s.get("sample_id")
        if sid is None:
            continue
        slot = slot_by_sample.get(sid)
        if slot is None:
            # Score for a sample not in our batch — stale judge file vs out_dir.
            # Skip rather than crash; the report counts what we matched.
            continue
        qualities_by_slot[slot].append(float(s.get("mean_quality_score", 0.0)))
        n = s.get("notes")
        if n:
            notes_by_slot[slot].append(str(n))

    slot_qualities: list[SlotQuality] = []
    for slot in cfg.slots_in_scope:
        qs = qualities_by_slot.get(slot, [])
        baseline = baselines.get(slot)
        n = len(qs)
        if n == 0:
            slot_qualities.append(
                SlotQuality(
                    slot=slot,
                    n_samples=0,
                    mean_quality=0.0,
                    baseline_pct_of_judge=baseline,
                    drift_pct=None,
                    alert=_classify_alert(
                        None,
                        current_pct_of_judge=None,
                        warn_threshold=cfg.warn_threshold_pct,
                        rebake_threshold=cfg.rebake_threshold_pct,
                        n_samples=0,
                    ),
                    notes=[],
                )
            )
            continue
        mean_q = statistics.mean(qs)
        # Express current quality as `% of frontier` for direct comparison
        # to baseline. The judge scored against a competent-output rubric,
        # so the candidate's raw mean is already on a 0-100 scale where 100
        # would equal the frontier's typical answer. Drift = current - baseline.
        if baseline is not None:
            current_pct_of_judge: float | None = mean_q
            drift = mean_q - baseline
        else:
            current_pct_of_judge = None
            drift = None
        alert = _classify_alert(
            drift,
            current_pct_of_judge=current_pct_of_judge,
            warn_threshold=cfg.warn_threshold_pct,
            rebake_threshold=cfg.rebake_threshold_pct,
            n_samples=n,
        )
        slot_qualities.append(
            SlotQuality(
                slot=slot,
                n_samples=n,
                mean_quality=mean_q,
                baseline_pct_of_judge=baseline,
                drift_pct=drift,
                alert=alert,
                notes=notes_by_slot.get(slot, []),
            )
        )

    return QualityReport(
        app_name=cfg.app_name,
        window_start_iso=window_start,
        window_end_iso=now.isoformat(),
        judge_model=cfg.judge_model,
        warn_threshold_pct=cfg.warn_threshold_pct,
        rebake_threshold_pct=cfg.rebake_threshold_pct,
        slots=slot_qualities,
    )
