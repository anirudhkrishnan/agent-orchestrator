"""CLI entry point for the evaluation framework.

Four discrete phases::

    # Phase 1 — run all candidates against all tasks (sequential, RAM-aware)
    python -m orchestrator.evaluation run \\
        --tasks data/evaluation/tasks-example.yaml \\
        --candidates ollama/qwen3:8b ollama/qwen3.5:9b ollama/gemma4:e4b \\
        --judge interactive-judge-session \\
        --out-dir data/evaluation/runs

    # Phase 1.5 — create the baselines.json skeleton
    python -m orchestrator.evaluation init-baselines <run-dir>

    # Phase 2 — judge (your judge model, in an interactive session) fills in
    #          baselines.json with gold-standard answers per (task, scenario)

    # Phase 2.5 — bundle baselines + candidate outputs into judge-batch.json
    python -m orchestrator.evaluation prepare-batch <run-dir>

    # Phase 3 — judge reads judge-batch.json, writes judge-scores.json with
    #          scores for BOTH baseline and candidate per item

    # Phase 4 — produce REPORT.md + per-task winners + delegation matrix
    python -m orchestrator.evaluation finalize <run-dir>

Why split into four phases? — The frontier judge sits OUTSIDE the candidate
pool by design. Instead of API-driven grading, this module passes scoring
through an interactive agent session reading/writing disk. The baseline step
makes the judge produce a gold standard BEFORE seeing candidates, which gives
the framework the "% of judge quality" signal that drives delegation decisions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .batch import _load_runs_from_dir, init_baselines_skeleton, prepare_judge_batch
from .integrity import check_score_integrity
from .report import generate_report
from .runner import run_evaluation_sync
from .tasks import load_tasks_yaml


def build_parser() -> argparse.ArgumentParser:
    """Top-level parser with `run`, `init-baselines`, `prepare-batch`, `finalize` subcommands."""
    parser = argparse.ArgumentParser(
        prog="orchestrator.evaluation",
        description=(
            "Frontier-judge-outside-pool evaluation framework. Phase 1 (`run`) "
            "executes candidates sequentially with RAM-managed Ollama; "
            "`init-baselines` creates a baselines.json skeleton for the judge "
            "to fill in; `prepare-batch` bundles baselines + candidates for "
            "judge scoring; `finalize` produces REPORT.md."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=False)

    # --- run ---------------------------------------------------------------
    p_run = sub.add_parser(
        "run",
        help="Phase 1: execute candidates against all tasks.",
        description=(
            "Run every (candidate × task × scenario) cell sequentially against "
            "local Ollama. Writes per-cell JSON and a manifest into a new "
            "timestamped subdirectory of --out-dir. Does NOT prepare the judge "
            "batch — use `init-baselines` then `prepare-batch` after this."
        ),
    )
    p_run.add_argument(
        "--tasks",
        type=Path,
        required=True,
        help="Path to the tasks YAML file.",
    )
    p_run.add_argument(
        "--candidates",
        nargs="+",
        required=True,
        help="Ordered provider/model ids, e.g. ollama/qwen3:8b ollama/gemma4:e4b.",
    )
    p_run.add_argument(
        "--judge",
        default="interactive-judge-session",
        help="Identifier of the judge to record in judge-batch.json.",
    )
    p_run.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Root directory where the timestamped run dir is created.",
    )
    p_run.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama base URL (default: http://localhost:11434).",
    )
    p_run.add_argument(
        "--keep-alive-seconds",
        type=int,
        default=1800,
        help="How long Ollama keeps a candidate's weights resident (default 1800).",
    )
    p_run.add_argument(
        "--per-call-timeout-s",
        type=float,
        default=600.0,
        help="Per-scenario HTTP timeout in seconds (default 600).",
    )
    p_run.add_argument(
        "--samples-per-cell",
        type=int,
        default=5,
        help=(
            "How many samples to draw per (candidate, task, scenario) cell. "
            "DEFAULT 5 — a pragmatic balance between variance stabilization "
            "and run cost (routing uses the per-cell MEDIAN; std-dev over 5 "
            "samples is a coarse stability signal). Lower to 1 for quick "
            "smoke tests."
        ),
    )
    p_run.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress per-cell progress lines.",
    )

    # --- init-baselines ----------------------------------------------------
    p_init = sub.add_parser(
        "init-baselines",
        help="Phase 1.5: write an empty baselines.json skeleton for the judge to fill.",
        description=(
            "Creates `{run-dir}/baselines.json` with all (task_id, scenario_id) "
            "keys present and empty string values. The judge then fills in "
            "gold-standard answers; `prepare-batch` reads them back."
        ),
    )
    p_init.add_argument(
        "run_dir",
        type=Path,
        help="Path to the timestamped run directory (must contain manifest.json).",
    )
    p_init.add_argument(
        "--judge",
        default="interactive-judge-session",
        help="Judge name recorded in the skeleton (default matches `run`'s default).",
    )
    p_init.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing baselines.json (DESTRUCTIVE — discards judge-filled answers).",
    )

    # --- prepare-batch -----------------------------------------------------
    p_prep = sub.add_parser(
        "prepare-batch",
        help="Phase 2.5: bundle baselines + candidate outputs into judge-batch.json.",
        description=(
            "Walks the run directory and emits judge-batch.json with one item "
            "per (candidate, task, scenario). If baselines.json is present, "
            "each item also carries the baseline answer and the judge "
            "instructions switch to the with-baseline variant."
        ),
    )
    p_prep.add_argument(
        "run_dir",
        type=Path,
        help="Path to the timestamped run directory.",
    )
    p_prep.add_argument(
        "--judge",
        default="interactive-judge-session",
        help="Judge identifier recorded in judge-batch.json.",
    )

    # --- finalize ----------------------------------------------------------
    p_fin = sub.add_parser(
        "finalize",
        help="Phase 4: produce REPORT.md from judge-scores.json.",
        description=(
            "After the judge has written judge-scores.json to the run directory, "
            "run this to compute aggregated metrics, declare per-task winners, "
            "and emit a paste-ready routing.json update block + delegation matrix."
        ),
    )
    p_fin.add_argument(
        "run_dir",
        type=Path,
        help="Path to the timestamped run directory.",
    )
    p_fin.add_argument(
        "--quality-weight",
        type=float,
        default=0.7,
        help="Weight on quality in the combined score (default 0.7).",
    )
    p_fin.add_argument(
        "--allow-stamped-scores",
        action="store_true",
        help=(
            "Override the score-integrity gate. By default `finalize` REFUSES to "
            "produce a report if distinct candidate outputs systematically received "
            "identical scores (the stamped-score signature; see RCA 2026-05-28). "
            "Only set this if you have confirmed the run is legitimately uniform."
        ),
    )
    p_fin.add_argument(
        "--stamp-fail-frac",
        type=float,
        default=0.7,
        help="Fraction of multi-output cells allowed to be zero-variance before the gate fails (default 0.7).",
    )
    p_fin.add_argument(
        "--skip-integrity",
        action="store_true",
        help=(
            "Skip the score-integrity gate entirely (loud warning). Without this, "
            "`finalize` REFUSES to run when judge-scores.json exists but "
            "judge-batch.json is missing — scores can't be verified against "
            "candidate outputs without the batch. Prefer re-running prepare-batch."
        ),
    )

    return parser


def _tally_sample_errors(runs: list) -> tuple[int, int]:
    """Count (total_samples, errored_samples) across CandidateRun cells.

    Cells written with N>1 sampling carry per-sample errors in `samples`;
    legacy N=1 cells fall back to the flat top-level `error` field.
    """
    total = errored = 0
    for r in runs:
        errors = [s.error for s in r.samples] if r.samples else [r.error]
        total += len(errors)
        errored += sum(1 for e in errors if e)
    return total, errored


def _cmd_run(args: argparse.Namespace) -> int:
    """Phase 1 — execute candidates only. Judge prep is done in later phases."""
    tasks = load_tasks_yaml(args.tasks)
    print(
        f"[evaluation] loaded {len(tasks)} task(s) from {args.tasks}: "
        f"{', '.join(t.id for t in tasks)}",
        flush=True,
    )
    run_dir = run_evaluation_sync(
        candidates=args.candidates,
        tasks=tasks,
        out_dir=args.out_dir,
        ollama_url=args.ollama_url,
        keep_alive_seconds=args.keep_alive_seconds,
        log_progress=not args.no_progress,
        per_call_timeout_s=args.per_call_timeout_s,
        samples_per_cell=args.samples_per_cell,
    )
    # Per-sample error tally — "Phase 1 complete" must not paper over a run
    # where model calls failed (e.g. Ollama down → every cell errored but the
    # runner still writes uniform error records and exits cleanly).
    total_samples, errored_samples = _tally_sample_errors(_load_runs_from_dir(run_dir))
    print()
    print(f"[evaluation] sample errors: {errored_samples}/{total_samples}")
    if total_samples and errored_samples == total_samples:
        print(
            "[evaluation] ERROR: every sample in this run errored — no candidate "
            f"produced output. Check that Ollama is reachable at {args.ollama_url} "
            f"and the candidate model names are correct. Error records are at "
            f"{run_dir}.",
            file=sys.stderr,
        )
        return 1
    if errored_samples:
        print(
            f"[evaluation] ⚠️  WARNING: {errored_samples}/{total_samples} sample(s) "
            f"errored; the judge will score them 0. Inspect the cell JSONs in "
            f"{run_dir} before judging.",
            file=sys.stderr,
        )
    print()
    print(f"Phase 1 complete — candidate outputs at {run_dir}")
    print()
    print("Next steps:")
    print(f"  1. python -m orchestrator.evaluation init-baselines {run_dir}")
    print("     → creates baselines.json skeleton for the judge")
    print("  2. Your judge model fills in baselines.json with gold-standard answers")
    print(f"  3. python -m orchestrator.evaluation prepare-batch {run_dir}")
    print("     → bundles baselines + candidate outputs into judge-batch.json")
    print("  4. Judge reads judge-batch.json and writes judge-scores.json")
    print(f"  5. python -m orchestrator.evaluation finalize {run_dir}")
    print("     → generates REPORT.md with delegation matrix")
    return 0


def _cmd_init_baselines(args: argparse.Namespace) -> int:
    """Phase 1.5 — write baselines.json skeleton."""
    if not args.run_dir.exists():
        print(f"[evaluation] run-dir does not exist: {args.run_dir}", file=sys.stderr)
        return 2
    try:
        path = init_baselines_skeleton(
            args.run_dir,
            judge_name=args.judge,
            overwrite=args.overwrite,
        )
    except FileExistsError as e:
        print(f"[evaluation] {e}", file=sys.stderr)
        return 2
    print(f"[evaluation] wrote skeleton at {path}")
    print()
    print("Next: have the judge fill in each (task_id, scenario_id) value with a")
    print("gold-standard answer, then run:")
    print(f"  python -m orchestrator.evaluation prepare-batch {args.run_dir}")
    return 0


def _cmd_prepare_batch(args: argparse.Namespace) -> int:
    """Phase 2.5 — bundle baselines + candidate outputs into judge-batch.json."""
    if not args.run_dir.exists():
        print(f"[evaluation] run-dir does not exist: {args.run_dir}", file=sys.stderr)
        return 2
    batch_path = prepare_judge_batch(args.run_dir, judge_model=args.judge)
    scores_path = args.run_dir / "judge-scores.json"
    baselines_path = args.run_dir / "baselines.json"
    print(f"[evaluation] wrote {batch_path}")
    if baselines_path.exists():
        print(f"[evaluation] baselines included from {baselines_path}")
    else:
        print(
            "[evaluation] no baselines.json found — judge will score candidates only "
            "(no % of judge insight)."
        )
    print()
    print("READY FOR JUDGE")
    print(f"  - Read    : {batch_path}")
    print(f"  - Write to: {scores_path}")
    print(f"  - Then run: python -m orchestrator.evaluation finalize {args.run_dir}")
    return 0


def _cmd_finalize(args: argparse.Namespace) -> int:
    """Phase 4 — generate REPORT.md from judge scores already on disk.

    Runs the score-integrity gate FIRST (M2 mitigation, RCA 2026-05-28): refuses
    to produce a report on stamped/recycled scores unless explicitly overridden.
    """
    if not args.run_dir.exists():
        print(f"[evaluation] run-dir does not exist: {args.run_dir}", file=sys.stderr)
        return 2
    scores_path = args.run_dir / "judge-scores.json"
    if not scores_path.exists():
        print(
            f"[evaluation] judge-scores.json missing in {args.run_dir} — the judge "
            "step hasn't run yet. Have your judge read judge-batch.json and write "
            "judge-scores.json (or use `python -m orchestrator.judge_adapter`), "
            "then re-run finalize.",
            file=sys.stderr,
        )
        return 2

    # --- Score-integrity gate -------------------------------------------------
    batch_path = args.run_dir / "judge-batch.json"
    if args.skip_integrity:
        print(
            "[evaluation] ⚠️  WARNING: --skip-integrity set — the score-integrity "
            "gate is DISABLED. Scores will NOT be verified against candidate "
            "outputs for stamping or empty-output crediting.",
            file=sys.stderr,
        )
    elif scores_path.exists() and not batch_path.exists():
        # Fail CLOSED: scores with no batch to check them against can't be
        # verified. Silently skipping the gate here would wave stamped or
        # hand-edited scores straight through to REPORT.md.
        print(
            "[evaluation] ERROR: cannot verify score integrity (judge-batch.json "
            f"missing in {args.run_dir}).",
            file=sys.stderr,
        )
        print(
            "   judge-scores.json is present but there is no judge-batch.json to "
            "check it against. Re-run `prepare-batch` to regenerate it, or pass "
            "--skip-integrity to finalize anyway (NOT recommended).",
            file=sys.stderr,
        )
        return 3
    if not args.skip_integrity and batch_path.exists() and scores_path.exists():
        batch_items = json.loads(batch_path.read_text()).get("items", [])
        scores = json.loads(scores_path.read_text())
        report = check_score_integrity(
            batch_items, scores, zero_variance_fail_frac=args.stamp_fail_frac
        )
        print(report.summary(), file=sys.stderr)
        if not report.passed and not args.allow_stamped_scores:
            print(file=sys.stderr)
            print("🛑 SCORE-INTEGRITY GATE FAILED — refusing to finalize.", file=sys.stderr)
            print(f"   {report.note}", file=sys.stderr)
            print(
                f"   {len(report.suspicious_cells)} suspicious cell(s). Examples:",
                file=sys.stderr,
            )
            for c in report.suspicious_cells[:5]:
                print(
                    f"     - {c.candidate} :: {c.task_id} :: {c.scenario_id} "
                    f"({c.n_distinct_outputs} distinct outputs, all scored {c.score})",
                    file=sys.stderr,
                )
            print(
                "   Fix: re-run judging per-item. Override (only if truly uniform): "
                "--allow-stamped-scores",
                file=sys.stderr,
            )
            return 3
        if not report.passed and args.allow_stamped_scores:
            print(
                "[evaluation] ⚠️  integrity gate OVERRIDDEN via --allow-stamped-scores",
                file=sys.stderr,
            )

    report_path = generate_report(args.run_dir, quality_weight=args.quality_weight)
    print(f"[evaluation] wrote {report_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Process entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "init-baselines":
        return _cmd_init_baselines(args)
    if args.command == "prepare-batch":
        return _cmd_prepare_batch(args)
    if args.command == "finalize":
        return _cmd_finalize(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
