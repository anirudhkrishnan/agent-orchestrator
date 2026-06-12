"""Judge batch preparation — turn a run directory into a single JSON the judge reads.

The frontier judge (your judge model, in an interactive session) is OUTSIDE the
candidate pool by design. We don't want a candidate model scoring its own
outputs, and we don't want to spend API tokens on judging — the judge reads
the batch file off disk and writes scores back to disk as part of an
interactive agent session.

This module walks `{run_dir}/` (produced by `runner.run_evaluation`), bundles
each (candidate, task, scenario) cell into a `JudgeBatchItem`, and emits a
single `judge-batch.json` containing all items + plain-prose instructions for
the judge.

v2 additions:

* Baselines: if `{run_dir}/baselines.json` exists, each JudgeBatchItem's
  `baseline_output` field is populated with the judge's gold-standard answer
  for that (task, scenario). The judge then scores BOTH the baseline and the
  candidate per item — giving the framework the `% of judge` signal that
  drives delegation decisions.
* Scoring scale: instructions now describe a 0-100 integer scale per quality
  dimension (was 1-5 in v1), giving more granularity for real degradation.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from .runner import CandidateRun
from .tasks import QualityDimension


class JudgeBatchItem(BaseModel):
    """One cell the judge scores. Bundles candidate output with task context."""

    item_id: str = Field(..., description="Unique within the batch. Stable across re-runs.")
    task_id: str
    task_description: str
    quality_dimensions: list[QualityDimension]
    scenario_id: str
    scenario_input: str
    scenario_notes: str | None
    expected_output_shape: str | None
    candidate: str
    candidate_output: str
    latency_ms: int
    error: str | None
    baseline_output: str | None = Field(
        default=None,
        description=(
            "Judge's gold-standard answer for this (task, scenario), read from "
            "baselines.json. None if no baselines.json exists in the run dir."
        ),
    )
    sample_indices: list[int] = Field(
        default_factory=lambda: [0],
        description=(
            "Indices into the source cell's samples list that this item "
            "represents. With N=5 sampling and dedupe, an item with 5 "
            "identical outputs has sample_indices=[0,1,2,3,4]."
        ),
    )
    sample_count: int = Field(
        default=1,
        description=(
            "Number of raw samples this item represents (len(sample_indices))."
            " The aggregation step weights scores by this count when computing"
            " per-cell median + std-dev."
        ),
    )


class JudgeBatch(BaseModel):
    """The full batch the judge consumes in a single pass."""

    run_id: str = Field(..., description="ISO timestamp of the run; matches the run_dir name.")
    judge_model: str = Field(..., description="Identifier of the judge, e.g. 'interactive-judge-session'.")
    instructions_for_judge: str
    items: list[JudgeBatchItem]
    baselines_present: bool = Field(
        default=False,
        description=(
            "True if baselines.json was present at batch-prep time. Lets the "
            "report module decide whether to render % of judge columns."
        ),
    )


class BaselinesFile(BaseModel):
    """Schema for `{run_dir}/baselines.json`.

    Produced either by the framework's `init-baselines` CLI command (as an
    empty skeleton) or directly by the judge in an interactive session.
    """

    judge_name: str = Field(
        ...,
        description=(
            "Identifier of the judge that produced these baselines. Treated as "
            "a special candidate id of the form 'judge-baseline/<judge_name>'."
        ),
    )
    produced_at: str = Field(
        ...,
        description="ISO timestamp string. When the framework writes a skeleton, this is the skeleton-creation time.",
    )
    baselines: dict[str, dict[str, str]] = Field(
        ...,
        description=(
            "Nested dict: baselines[task_id][scenario_id] = baseline_answer "
            "as a string. Empty strings indicate skeleton entries the judge "
            "still needs to fill in."
        ),
    )


# --- Instructions ---------------------------------------------------------
#
# Built as constants so any tweak to the judge protocol is a single edit.
# Two variants because the protocol differs materially with vs without
# baselines: with-baseline rows require scoring BOTH outputs per item.


_JUDGE_INSTRUCTIONS_NO_BASELINE = """\
You are the evaluation judge for a model bake-off.

Scoring scale: integer 0-100 per quality dimension (0 = wholly wrong / no
signal, 100 = perfect). The 0-100 scale gives enough granularity to see real
degradation between candidates — don't cluster everything in the 70-90 band.

For each item in `items[]`, produce one JSON object with this exact shape:

    {{
      "item_id": "<copy from input>",
      "candidate_scores": {{
        "scores": {{ "<dimension_name>": <int 0-100>, ... }},
        "mean_quality_score": <weighted mean as float>,
        "notes": "<1-2 sentence rationale, plain prose>"
      }}
    }}

Scoring rules:
  - Use the per-task `quality_dimensions` list — the dimension names and
    weights live in each item. Score each named dimension on an integer 0-100
    scale.
  - `mean_quality_score` = sum(score_i * weight_i) across that item's
    dimensions. Weights are already normalized to sum to 1.0; the mean
    therefore lands on the same 0-100 scale.
  - Do NOT score speed — speed is graded automatically from `latency_ms`
    by the framework. Score quality only.
  - If `error` is non-null, score the item as if the candidate produced no
    useful output: all dimensions = 0, notes = "Errored: <one-line summary>".
  - Read `expected_output_shape` and `scenario_notes` as judge-side context —
    they explain what a correct response looks like. They were NOT shown to
    the candidate.

Write the full array of result objects (one per item) to:

    {scores_path}

as a JSON array (NOT JSONL, NOT a wrapper object). After writing, the user
will run `python -m orchestrator.evaluation finalize {run_dir}` to compute
the leaderboard and the routing.json update suggestion.

Number of items to score: {n_items}.

NOTE: this batch has NO baselines (no baselines.json present). For richer
analysis, create one with `python -m orchestrator.evaluation init-baselines
{run_dir}`, fill it in, then re-run `prepare-batch`.
"""


_JUDGE_INSTRUCTIONS_WITH_BASELINE = """\
You are the evaluation judge for a model bake-off.

Scoring scale: integer 0-100 per quality dimension (0 = wholly wrong / no
signal, 100 = perfect). The 0-100 scale gives enough granularity to see real
degradation between candidates — don't cluster everything in the 70-90 band.

THIS BATCH INCLUDES BASELINES. Each item carries TWO outputs:
  * `baseline_output` — YOUR gold-standard answer for that (task, scenario),
    read from baselines.json. You produced this earlier; now score it on the
    same rubric you apply to the candidate.
  * `candidate_output` — the local model's answer for the same scenario.

Score BOTH per item. The framework will compute
`quality_pct_of_baseline = candidate / baseline * 100` — the load-bearing
signal for delegation decisions. Honest baseline scoring is essential: if
your own answer is imperfect on some scenario, score it imperfect. A 92/100
baseline is fine; faking it as 100 inflates every candidate's % of baseline.

For each item in `items[]`, produce one JSON object with this exact shape:

    {{
      "item_id": "<copy from input>",
      "baseline_scores": {{
        "scores": {{ "<dimension_name>": <int 0-100>, ... }},
        "mean_quality_score": <weighted mean as float>,
        "notes": "<1-2 sentence rationale, plain prose>"
      }},
      "candidate_scores": {{
        "scores": {{ "<dimension_name>": <int 0-100>, ... }},
        "mean_quality_score": <weighted mean as float>,
        "notes": "<1-2 sentence rationale, plain prose>"
      }}
    }}

Scoring rules:
  - Use the per-task `quality_dimensions` list — the dimension names and
    weights live in each item. Score each named dimension on an integer 0-100
    scale.
  - `mean_quality_score` = sum(score_i * weight_i) across that item's
    dimensions. Weights are already normalized to sum to 1.0.
  - Do NOT score speed — speed is graded automatically from `latency_ms`.
    Score quality only.
  - If candidate `error` is non-null, score the candidate as if it produced
    no useful output: all dimensions = 0, notes = "Errored: <summary>". The
    baseline still gets a normal score.
  - Read `expected_output_shape` and `scenario_notes` as judge-side context —
    they explain what a correct response looks like.

Write the full array of result objects (one per item) to:

    {scores_path}

as a JSON array (NOT JSONL, NOT a wrapper object). After writing, the user
will run `python -m orchestrator.evaluation finalize {run_dir}` to compute
the leaderboard and the routing.json update suggestion.

Number of items to score: {n_items}.
"""


def _load_runs_from_dir(run_dir: Path) -> list[CandidateRun]:
    """Walk `{run_dir}/*/` and load every CandidateRun JSON.

    Each candidate has its own subdirectory; each subdirectory contains
    `{task_id}_{scenario_id}.json` files. Anything else (manifest.json,
    judge-batch.json, judge-scores.json, baselines.json, REPORT.md) is skipped.
    """
    runs: list[CandidateRun] = []
    for sub in sorted(run_dir.iterdir()):
        if not sub.is_dir():
            continue
        for f in sorted(sub.glob("*.json")):
            try:
                data = json.loads(f.read_text())
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed candidate-run JSON at {f}: {e}") from e
            runs.append(CandidateRun.model_validate(data))
    return runs


def _load_manifest(run_dir: Path) -> dict:
    """Read the manifest written by `runner._write_manifest`."""
    p = run_dir / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(
            f"manifest.json missing in {run_dir}. Was this directory produced by run_evaluation()?"
        )
    return json.loads(p.read_text())


def _load_baselines(run_dir: Path) -> BaselinesFile | None:
    """Read `{run_dir}/baselines.json` if it exists.

    Returns None when the file is absent — callers fall back to a no-baseline
    batch in that case. Raises ValueError on a malformed file (better to fail
    loudly than silently degrade to no-baseline scoring on a typo).
    """
    p = run_dir / "baselines.json"
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed baselines.json at {p}: {e}") from e
    return BaselinesFile.model_validate(raw)


def init_baselines_skeleton(
    run_dir: Path,
    *,
    judge_name: str = "interactive-judge-session",
    overwrite: bool = False,
) -> Path:
    """Write a `baselines.json` skeleton with all (task, scenario) keys empty.

    The judge fills the empty strings in interactively, then runs
    `prepare-batch` to bundle baselines + candidate outputs for scoring.

    Args:
        run_dir: Path to a completed Phase-1 run directory.
        judge_name: Recorded inside the skeleton; matches the `--judge` arg
            from the `run` command.
        overwrite: If True, replace any existing baselines.json. Default False
            preserves judge-authored content (the file is usually filled in by
            hand after the skeleton is created).

    Returns:
        Path to the written `baselines.json`.

    Raises:
        FileNotFoundError if manifest.json is missing.
        FileExistsError if baselines.json already exists and overwrite=False.
    """
    manifest = _load_manifest(run_dir)
    p = run_dir / "baselines.json"
    if p.exists() and not overwrite:
        raise FileExistsError(
            f"baselines.json already exists at {p}. Pass overwrite=True to replace "
            f"(this will discard any judge-filled-in baseline answers)."
        )
    skeleton: dict[str, dict[str, str]] = {}
    for task in manifest["tasks"]:
        skeleton[task["id"]] = {scn["id"]: "" for scn in task["scenarios"]}
    baselines = BaselinesFile(
        judge_name=judge_name,
        produced_at=datetime.now(timezone.utc).isoformat(),
        baselines=skeleton,
    )
    p.write_text(baselines.model_dump_json(indent=2) + "\n")
    return p


def prepare_judge_batch(
    run_dir: Path,
    *,
    judge_model: str = "interactive-judge-session",
) -> Path:
    """Build a JudgeBatch from a completed run directory; write `judge-batch.json`.

    If `{run_dir}/baselines.json` is present, attaches baseline answers to each
    JudgeBatchItem AND switches the judge instructions to the with-baseline
    variant (judge scores both baseline and candidate per item).

    Args:
        run_dir: Path to a single timestamped run directory.
        judge_model: Identifier recorded in the batch + referenced in the
            instructions. Defaults to a generic interactive-session label.

    Returns:
        Path to the written `judge-batch.json`.

    Raises:
        FileNotFoundError if manifest.json is missing.
        ValueError if any CandidateRun JSON is malformed.
    """
    manifest = _load_manifest(run_dir)
    # Index task metadata by id so we can hydrate JudgeBatchItem fields.
    task_index: dict[str, dict] = {t["id"]: t for t in manifest["tasks"]}
    scenario_index: dict[tuple[str, str], dict] = {}
    for t in manifest["tasks"]:
        for s in t["scenarios"]:
            scenario_index[(t["id"], s["id"])] = s

    baselines = _load_baselines(run_dir)
    baselines_present = baselines is not None
    # Index baseline strings by (task_id, scenario_id) for O(1) attach.
    baseline_index: dict[tuple[str, str], str] = {}
    if baselines is not None:
        for tid, scn_map in baselines.baselines.items():
            for sid, ans in scn_map.items():
                # Skip empty strings — those are unfilled skeleton slots; we
                # don't want to push them into the judge batch as if they
                # were real baselines.
                if ans:
                    baseline_index[(tid, sid)] = ans

    runs = _load_runs_from_dir(run_dir)
    items: list[JudgeBatchItem] = []
    for r in runs:
        task_meta = task_index.get(r.task_id)
        if task_meta is None:
            raise ValueError(
                f"Run references task {r.task_id!r} not present in manifest.json. "
                f"manifest tasks: {sorted(task_index)}"
            )
        scn_meta = scenario_index.get((r.task_id, r.scenario_id))
        if scn_meta is None:
            raise ValueError(
                f"Run references unknown scenario {r.task_id}/{r.scenario_id} "
                f"(not in manifest)."
            )
        # N=5 sampling (post-2026-05-26): each cell has multiple samples.
        # Dedupe by output_text within the cell — identical outputs get one
        # judge-batch item that represents the group, with sample_indices
        # listing which raw samples it maps back to. Saves the judge from
        # scoring 5 identical outputs 5 times.
        cell_samples = r.samples if r.samples else [
            # Legacy single-sample runs: synthesize a samples list of length 1
            # from the backward-compat top-level fields.
            type("S", (), {"output_text": r.output_text, "latency_ms": r.latency_ms, "error": r.error})()
        ]
        # Group sample indices by output_text (only consider non-error samples
        # for dedupe; error samples are 1-per-item so they always score).
        groups: dict[str, list[int]] = {}
        for idx, s in enumerate(cell_samples):
            key = s.output_text if not s.error else f"__error_{idx}__"
            groups.setdefault(key, []).append(idx)

        for group_key, idx_list in groups.items():
            primary_idx = idx_list[0]
            sample = cell_samples[primary_idx]
            # If this group represents multiple identical samples, encode
            # that in the item_id so the judge can see the multiplicity
            # without having to score N times. The representative sample
            # index ("-s{k}") disambiguates two equal-sized groups within
            # the same cell — without it, e.g. samples [A,A,B,B] would emit
            # two items both named "::group-of-2", and the colliding ids
            # would blind the score-integrity gate (one score silently
            # overwrites the other in its per-item index).
            multiplicity_suffix = (
                f"::group-of-{len(idx_list)}-s{primary_idx}"
                if len(idx_list) > 1
                else f"::sample-{primary_idx}"
            )
            items.append(
                JudgeBatchItem(
                    item_id=f"{r.candidate}::{r.task_id}::{r.scenario_id}{multiplicity_suffix}",
                    task_id=r.task_id,
                    task_description=task_meta["description"],
                    quality_dimensions=[
                        QualityDimension(**d) for d in task_meta["quality_dimensions"]
                    ],
                    scenario_id=r.scenario_id,
                    scenario_input=scn_meta["input"],
                    scenario_notes=scn_meta.get("notes"),
                    expected_output_shape=scn_meta.get("expected_output_shape"),
                    candidate=r.candidate,
                    candidate_output=sample.output_text,
                    latency_ms=sample.latency_ms,
                    error=sample.error,
                    baseline_output=baseline_index.get((r.task_id, r.scenario_id)),
                    sample_indices=idx_list,
                    sample_count=len(idx_list),
                )
            )

    # Whether any item actually carries a baseline determines which
    # instructions go in the batch.
    any_baseline = any(i.baseline_output for i in items)
    template = (
        _JUDGE_INSTRUCTIONS_WITH_BASELINE
        if any_baseline
        else _JUDGE_INSTRUCTIONS_NO_BASELINE
    )

    run_id = run_dir.name
    scores_path = run_dir / "judge-scores.json"
    instructions = template.format(
        scores_path=scores_path,
        run_dir=run_dir,
        n_items=len(items),
    )
    batch = JudgeBatch(
        run_id=run_id,
        judge_model=judge_model,
        instructions_for_judge=instructions,
        items=items,
        baselines_present=baselines_present and any_baseline,
    )
    out_path = run_dir / "judge-batch.json"
    out_path.write_text(batch.model_dump_json(indent=2) + "\n")
    return out_path
