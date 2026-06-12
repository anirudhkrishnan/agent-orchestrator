"""Frontier-judge-outside-pool evaluation framework.

An evaluation framework for per-application model bake-offs. Design choices::

    judge sits OUTSIDE the candidate pool (no self-judging bias)
    interactive judging (read disk / write disk; no API tokens spent grading)
    weighted quality+speed combined score (0-100 scale)
    YAML-authored task spec (tasks evolve without touching Python)
    candidates run sequentially (RAM-aware; fair latency measurements)
    judge produces baselines first; candidates scored as % of baseline quality

Pipeline::

    1. `runner.run_evaluation`         — load model N → run all scenarios → unload
    2. `batch.init_baselines_skeleton` — write baselines.json template
    3. JUDGE                            — fills baselines.json with gold answers
    4. `batch.prepare_judge_batch`     — bundle baselines + candidates into judge-batch.json
    5. JUDGE                            — reads judge-batch.json, writes scores for both
    6. `report.generate_report`        — REPORT.md + routing.json + delegation matrix

Invocation::

    python -m orchestrator.evaluation run --tasks ... --candidates ... --out-dir ...
    python -m orchestrator.evaluation init-baselines <run-dir>
    # ... judge fills baselines.json ...
    python -m orchestrator.evaluation prepare-batch <run-dir>
    # ... judge writes judge-scores.json ...
    python -m orchestrator.evaluation finalize <run-dir>
"""

from .batch import (
    BaselinesFile,
    JudgeBatch,
    JudgeBatchItem,
    init_baselines_skeleton,
    prepare_judge_batch,
)
from .report import generate_report
from .runner import CandidateRun, run_evaluation, run_evaluation_sync
from .scoring import (
    DELEGATE_FREELY_BASELINE_PCT,
    DELEGATE_WITH_MONITOR_BASELINE_PCT,
    STRONG_CANDIDATE_BASELINE_PCT,
    STRONG_CANDIDATE_COMBINED,
    aggregate_per_task_candidate,
    combined_score,
    degradation_callout,
    delegation_tier,
    quality_pct_of_baseline,
    speed_score_from_latency,
    winner_per_task,
)
from .tasks import QualityDimension, Scenario, TaskSpec, load_tasks_yaml

__all__ = [
    "BaselinesFile",
    "CandidateRun",
    "DELEGATE_FREELY_BASELINE_PCT",
    "DELEGATE_WITH_MONITOR_BASELINE_PCT",
    "JudgeBatch",
    "JudgeBatchItem",
    "QualityDimension",
    "STRONG_CANDIDATE_BASELINE_PCT",
    "STRONG_CANDIDATE_COMBINED",
    "Scenario",
    "TaskSpec",
    "aggregate_per_task_candidate",
    "combined_score",
    "degradation_callout",
    "delegation_tier",
    "generate_report",
    "init_baselines_skeleton",
    "load_tasks_yaml",
    "prepare_judge_batch",
    "quality_pct_of_baseline",
    "run_evaluation",
    "run_evaluation_sync",
    "speed_score_from_latency",
    "winner_per_task",
]
