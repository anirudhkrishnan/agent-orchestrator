"""Score-integrity gate — detect "stamped" (recycled) judge scores.

Why this exists
---------------
Incident (2026-05-28): an N=5 re-score was shipped FAKED — a single N=1 score
was copied onto all 5 samples of every cell, so distinct sample outputs
received identical scores and the per-cell variance (the whole point of N=5)
was zero by construction. Entry count + schema validity passed, so the
shortcut shipped as "done."

Root cause: enablement without enforcement. N=5 enabled resampling; nothing
enforced that the resampling was actually *judged*. This module is the
structural mitigation (M2): it asserts the invariant **score variance must
track output variance** — if genuinely distinct candidate outputs systematically
receive identical scores, the scores were stamped, not judged.

How it works
------------
For each (candidate, task, scenario) CELL:
  - Collect (candidate_output, mean_quality_score) for every judged item.
  - A cell is "multi-output" if it contains >= 2 DISTINCT output strings.
    (Deterministic tasks dedup to one output per cell → single-output → excluded,
    so they never false-positive.)
  - A multi-output cell is "zero-variance" if all its scores are identical.

Run-level verdict:
  - frac = zero_variance_multi_output_cells / multi_output_cells
  - If there are enough multi-output cells to be meaningful
    (>= min_multi_output_cells) AND frac > zero_variance_fail_frac → FAIL.

Genuine judging produces *some* zero-variance cells (two outputs can both
legitimately earn 100), so this is a run-level fraction, not a per-cell hard
fail. Stamping produces frac ≈ 1.0; honest judging is well below the default
0.7 threshold.

Pure module: no I/O. Callers (CLI `finalize`) read the files and pass dicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SuspiciousCell:
    """A multi-output cell whose distinct outputs all got the identical score."""

    candidate: str
    task_id: str
    scenario_id: str
    n_distinct_outputs: int
    score: float


@dataclass
class EmptyScoredCell:
    """A cell containing an empty/whitespace output that got a non-zero score.

    The thinking-mode failure (a reasoning model emits only hidden think-tokens →
    empty user-facing output) dedups to a SINGLE distinct output, so it slips
    past the variance check. If such an empty output is scored as real quality,
    the bake-off credits a model for producing nothing. Detected per item: a
    cell that mixes real outputs with an empty-scored-nonzero sample is still
    flagged."""

    candidate: str
    task_id: str
    scenario_id: str
    score: float


@dataclass
class ScoreIntegrityReport:
    """Result of the integrity check.

    `passed` is False when the run looks stamped OR an empty output was scored
    as real. `finalize` treats passed=False as a hard error (overridable).
    """

    passed: bool
    multi_output_cells: int
    zero_variance_cells: int
    zero_variance_fraction: float
    threshold: float
    min_multi_output_cells: int
    suspicious_cells: list[SuspiciousCell] = field(default_factory=list)
    empty_scored_cells: list[EmptyScoredCell] = field(default_factory=list)
    note: str = ""

    def summary(self) -> str:
        if self.passed:
            verdict = "PASS"
        else:
            reasons = []
            if self.zero_variance_fraction > self.threshold and self.multi_output_cells >= self.min_multi_output_cells:
                reasons.append("STAMPED scores")
            if self.empty_scored_cells:
                reasons.append(f"{len(self.empty_scored_cells)} empty-output cell(s) scored as real")
            verdict = "FAIL — " + "; ".join(reasons)
        return (
            f"[score-integrity] {verdict}: "
            f"{self.zero_variance_cells}/{self.multi_output_cells} multi-output "
            f"cells zero-variance ({self.zero_variance_fraction:.0%}; fail above "
            f"{self.threshold:.0%}); {len(self.empty_scored_cells)} empty-scored cell(s)."
        )


def _cell_key(item: dict) -> tuple[str, str, str]:
    """(candidate, task_id, scenario_id) — prefer explicit fields, fall back to
    parsing item_id so the check also works on minimal score-only records."""
    cand = item.get("candidate")
    task = item.get("task_id")
    scn = item.get("scenario_id")
    if cand and task and scn:
        return (cand, task, scn)
    parts = str(item.get("item_id", "")).split("::")
    if len(parts) >= 3:
        return (parts[0], parts[1], parts[2])
    raise ValueError(
        f"Cannot determine cell for item {item.get('item_id')!r}: "
        f"missing candidate/task_id/scenario_id and unparseable item_id."
    )


def check_score_integrity(
    batch_items: list[dict],
    scores: list[dict],
    *,
    zero_variance_fail_frac: float = 0.7,
    min_multi_output_cells: int = 5,
) -> ScoreIntegrityReport:
    """Assert score variance tracks output variance.

    Args:
        batch_items: judge-batch `items[]` (need item_id + candidate_output +
            candidate/task_id/scenario_id).
        scores: judge-scores entries. Need item_id plus a quality score in
            either the v2 shape (`candidate_scores.mean_quality_score`) or the
            v1 flat shape (top-level `mean_quality_score`).
        zero_variance_fail_frac: fraction of multi-output cells allowed to be
            zero-variance before the run is judged stamped. Default 0.7.
        min_multi_output_cells: don't fire below this many multi-output cells —
            too little signal (e.g. an all-deterministic task set).

    Returns:
        ScoreIntegrityReport. `passed=False` ⇒ stamped-score signature detected.
    """
    output_by_id: dict[str, str] = {}
    cell_by_id: dict[str, tuple[str, str, str]] = {}
    for it in batch_items:
        iid = it.get("item_id")
        if iid is None:
            continue
        output_by_id[iid] = it.get("candidate_output", "")
        cell_by_id[iid] = _cell_key(it)

    score_by_id: dict[str, float] = {}
    for s in scores:
        iid = s.get("item_id")
        if iid is None:
            continue
        # v2 nests the quality under candidate_scores; v1 carries
        # mean_quality_score flat at the top level. Accept both — a v1 file
        # must not be misread as all-None (false positives / TypeError).
        cs = s.get("candidate_scores")
        if isinstance(cs, dict) and cs.get("mean_quality_score") is not None:
            score = cs["mean_quality_score"]
        else:
            score = s.get("mean_quality_score")
        if score is None:
            continue  # unscored item — nothing to verify against
        score_by_id[iid] = float(score)

    # Group (output, score) pairs by cell.
    cells: dict[tuple[str, str, str], list[tuple[str, float]]] = {}
    for iid, cell in cell_by_id.items():
        if iid not in score_by_id:
            continue
        cells.setdefault(cell, []).append((output_by_id.get(iid, ""), score_by_id[iid]))

    multi = 0
    zero_var = 0
    suspicious: list[SuspiciousCell] = []
    empty_scored: list[EmptyScoredCell] = []
    for cell, pairs in cells.items():
        # Empty-output check, PER ITEM (not only when the whole cell is empty):
        # any blank/whitespace output that earned a non-zero score means the
        # judge credited nothing as quality — the thinking-mode bug that the
        # variance check can't see. A cell mixing real outputs with an
        # empty-scored-nonzero sample must still be flagged.
        empty_nonzero = [sc for o, sc in pairs if not (o and o.strip()) and sc > 0]
        if empty_nonzero:
            empty_scored.append(
                EmptyScoredCell(
                    candidate=cell[0], task_id=cell[1], scenario_id=cell[2],
                    score=max(empty_nonzero),
                )
            )

        # Variance (stamping) check runs over NON-EMPTY outputs only. Empty/
        # errored samples legitimately score 0, and including them would inject
        # artificial score variance that masks stamping on the remaining real
        # samples of the cell.
        real_pairs = [(o, sc) for o, sc in pairs if o and o.strip()]
        distinct_outputs = {o for o, _ in real_pairs}
        if len(distinct_outputs) < 2:
            continue  # single-output (deterministic) or all-empty cell — excluded
        multi += 1
        distinct_scores = {sc for _, sc in real_pairs}
        if len(distinct_scores) == 1:
            zero_var += 1
            suspicious.append(
                SuspiciousCell(
                    candidate=cell[0], task_id=cell[1], scenario_id=cell[2],
                    n_distinct_outputs=len(distinct_outputs),
                    score=next(iter(distinct_scores)),
                )
            )

    frac = (zero_var / multi) if multi else 0.0
    enough_signal = multi >= min_multi_output_cells
    stamped = enough_signal and frac > zero_variance_fail_frac
    passed = not stamped and not empty_scored

    notes: list[str] = []
    if stamped:
        notes.append(
            "STAMPED: distinct candidate outputs systematically received identical "
            "scores (see RCA 2026-05-28) — the scoring stage likely recycled prior "
            "scores instead of judging each sample. Re-run judging per-item."
        )
    if empty_scored:
        notes.append(
            f"EMPTY-SCORED: {len(empty_scored)} cell(s) had an empty/whitespace "
            f"candidate output scored as real quality (thinking-mode bug — model "
            f"emitted only hidden reasoning). These must score 0, not pass through."
        )
    if not notes:
        if not enough_signal:
            notes.append(
                f"Only {multi} multi-output cell(s) (< {min_multi_output_cells}); "
                f"too little signal to judge stamping. No empty-scored cells. PASS."
            )
        else:
            notes.append("Score variance tracks output variance + no empty-scored cells — consistent with genuine judging.")

    return ScoreIntegrityReport(
        passed=passed,
        multi_output_cells=multi,
        zero_variance_cells=zero_var,
        zero_variance_fraction=frac,
        threshold=zero_variance_fail_frac,
        min_multi_output_cells=min_multi_output_cells,
        suspicious_cells=sorted(suspicious, key=lambda c: (c.task_id, c.candidate, c.scenario_id)),
        empty_scored_cells=sorted(empty_scored, key=lambda c: (c.task_id, c.candidate, c.scenario_id)),
        note=" ".join(notes),
    )
