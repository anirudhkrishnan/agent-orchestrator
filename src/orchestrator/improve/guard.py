"""The HARD RULE — every self-improving loop must pass the integrity gates
before it is allowed to change anything.

Why
---
Self-improvement amplifies risk. An autonomous loop that re-bakes, adds
scenarios, or rewrites routing on its own will, if unguarded, manufacture
confident-but-wrong results at scale — exactly the N=5 stamped-scoring failure
(RCA 2026-05-28), but now running unattended. So the design constraint for ALL
three loops (A: learn-from-mistakes, B: learn-from-new-models, C: learn-from-
research) is identical and non-negotiable:

    No loop may COMMIT a change derived from eval data that has not passed
    the score-integrity gate (and, where a routing/scope is touched, the
    completeness gate).

This module is the single chokepoint. Every loop wraps its commit step in
`require_integrity(...)` or the `@gated` decorator. A loop that tries to act on
ungated data raises IntegrityGateError and aborts — loudly, never silently.
"""

from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

from orchestrator.evaluation.integrity import check_score_integrity


class IntegrityGateError(RuntimeError):
    """Raised when a self-improving loop tries to act on data that fails the
    integrity gate. Loops must NOT catch-and-ignore this."""


def require_integrity(
    run_dir: Path,
    *,
    zero_variance_fail_frac: float = 0.7,
    allow_override: bool = False,
) -> None:
    """Assert the eval data in `run_dir` passed the score-integrity gate.

    Loads judge-batch.json + judge-scores.json and runs the same stamped-score
    / empty-output check `finalize` uses, plus fail-closed checks for
    degenerate state (corrupt/empty files, scores that match none of the
    batch's item_ids). Raises IntegrityGateError on failure.

    Args:
        run_dir: A completed bake-off run directory.
        zero_variance_fail_frac: forwarded to the gate.
        allow_override: if True, a FAILED variance gate prints a loud warning
            and does not raise. This is the ONLY escape hatch and exists for a
            human-confirmed legitimately-uniform run; loops never set it
            themselves. Degenerate state (missing/corrupt/empty files, zero
            item_id overlap) is NEVER overridable — there is no legitimate
            "uniform" reading of no data.

    Raises:
        IntegrityGateError: if the gate fails (and not overridden), or if the
            required files are missing/corrupt/empty or the scores reference
            none of the batch items (fail-closed: degenerate data is NOT a pass).
    """
    batch_path = run_dir / "judge-batch.json"
    scores_path = run_dir / "judge-scores.json"
    if not batch_path.exists() or not scores_path.exists():
        raise IntegrityGateError(
            f"Integrity gate cannot run: missing judge-batch.json / judge-scores.json "
            f"in {run_dir}. A self-improving loop must NOT act on absent data "
            f"(fail-closed)."
        )

    def _load(path: Path):
        # Corrupt JSON is degenerate state, not a crash: fail the gate with a
        # pointer at the file instead of escaping as a raw traceback.
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise IntegrityGateError(
                f"Integrity gate cannot run: {path.name} in {run_dir} is not valid "
                f"JSON ({e}). Re-run the judging step to regenerate it — do not "
                f"hand-edit run files (fail-closed)."
            ) from e

    batch_doc = _load(batch_path)
    scores = _load(scores_path)
    batch_items = batch_doc.get("items", []) if isinstance(batch_doc, dict) else []
    if not isinstance(scores, list):
        scores = []
    # Degenerate state is fail-closed and NOT overridable: an empty batch, an
    # empty score file, or scores that reference none of the batch's item_ids
    # all mean "this run was never actually judged" — exactly the state a
    # skipped/stamped scoring stage leaves behind.
    if not batch_items:
        raise IntegrityGateError(
            f"INTEGRITY GATE FAILED for {run_dir.name}: judge-batch.json contains no "
            f"items — there is nothing this run can claim to have judged (fail-closed)."
        )
    if not scores:
        raise IntegrityGateError(
            f"INTEGRITY GATE FAILED for {run_dir.name}: judge-scores.json contains no "
            f"scores — the judging step never ran, or wrote an empty file (fail-closed)."
        )
    batch_ids = {it.get("item_id") for it in batch_items if isinstance(it, dict)} - {None}
    score_ids = {s.get("item_id") for s in scores if isinstance(s, dict)} - {None}
    if not (batch_ids & score_ids):
        raise IntegrityGateError(
            f"INTEGRITY GATE FAILED for {run_dir.name}: none of the {len(score_ids)} "
            f"score item_ids match the {len(batch_ids)} batch item_ids — these scores "
            f"belong to a different run (fail-closed)."
        )
    report = check_score_integrity(
        batch_items, scores, zero_variance_fail_frac=zero_variance_fail_frac
    )
    if not report.passed:
        if not allow_override:
            raise IntegrityGateError(
                f"INTEGRITY GATE FAILED for {run_dir.name}: {report.note} "
                f"A self-improving loop may not commit changes derived from this data. "
                f"Re-judge per-item, then retry."
            )
        # The escape hatch must be LOUD: an override is a human taking explicit
        # responsibility for a failed gate, never a silent pass-through.
        print(
            f"WARNING [guard]: INTEGRITY GATE FAILED for {run_dir.name} but was "
            f"OVERRIDDEN (allow_override=True): {report.note} Proceeding on "
            f"human-confirmed override — if no human actually confirmed this run as "
            f"legitimately uniform, stop and re-judge.",
            file=sys.stderr,
        )


def gated(get_run_dir):
    """Decorator: gate a loop's commit function on the integrity of a run dir.

    `get_run_dir` is a callable that, given the same args as the wrapped
    function, returns the run_dir to gate (or None to skip — e.g. Loop B/C
    steps that don't consume eval scores directly).

    Usage:
        @gated(lambda proposal: proposal.run_dir)
        def commit_rebake(proposal): ...
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            run_dir = get_run_dir(*args, **kwargs)
            if run_dir is not None:
                require_integrity(Path(run_dir))
            return fn(*args, **kwargs)
        return wrapper
    return decorator
