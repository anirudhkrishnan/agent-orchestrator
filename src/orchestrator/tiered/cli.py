"""CLI for the tiered-routing tooling.

Commands::

    python -m orchestrator.tiered dry-run \\
        <run-dir-oss> <run-dir-frontier> \\
        [--workflow <name>] \\
        [--threshold <float>] \\
        [--out <path>]

    python -m orchestrator.tiered build-table \\
        <run-dir-oss> <run-dir-frontier> \\
        [--threshold <float>] \\
        [--out <path>]

All paths are relative or absolute — no hardcoded locations.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .dry_run import (
    EXAMPLE_WORKFLOWS,
    analyze_workflow,
    load_frontier_quality,
    load_oss_quality,
    render_dry_run_report,
)
from .routing_table import (
    DEFAULT_THRESHOLD,
    build_routing_table,
)


def _fail_no_quality_data(oss_dir: Path, frontier_dir: Path) -> None:
    """Exit 1 with a diagnostic instead of reporting confidently on no data."""
    print(
        f"Error: no quality data found in {oss_dir}, {frontier_dir} — "
        "expected judge-scores.json (+ judge-batch.json for OSS runs)",
        file=sys.stderr,
    )
    sys.exit(1)


def _cmd_dry_run(args: argparse.Namespace) -> None:
    oss_dir      = Path(args.run_dir_oss)
    frontier_dir = Path(args.run_dir_frontier)

    # A missing/typo'd run dir or empty loaders would otherwise produce a
    # confident exit-0 report ("worst slot 100.0%", n=0) — fail loudly instead.
    if not oss_dir.is_dir() or not frontier_dir.is_dir():
        _fail_no_quality_data(oss_dir, frontier_dir)

    print("Loading quality data…", file=sys.stderr)
    frontier_q = load_frontier_quality(frontier_dir)
    oss_q      = load_oss_quality(oss_dir)
    if not frontier_q and not oss_q:
        _fail_no_quality_data(oss_dir, frontier_dir)
    print(
        f"  Frontier cells: {len(frontier_q)}   OSS cells: {len(oss_q)}",
        file=sys.stderr,
    )

    # Determine which workflows to run.
    if args.workflow:
        if args.workflow not in EXAMPLE_WORKFLOWS:
            available = ", ".join(sorted(EXAMPLE_WORKFLOWS))
            print(
                f"Error: unknown workflow {args.workflow!r}. "
                f"Available: {available}",
                file=sys.stderr,
            )
            sys.exit(1)
        workflows = {args.workflow: EXAMPLE_WORKFLOWS[args.workflow]}
    else:
        workflows = EXAMPLE_WORKFLOWS

    threshold = float(args.threshold)
    all_reports: list[str] = []

    for name, calls in workflows.items():
        analyses = analyze_workflow(
            calls, name, frontier_q, oss_q,
            t1_threshold=threshold, t2_threshold=threshold,
        )
        report_md = render_dry_run_report(
            name, calls, analyses, threshold=threshold,
        )
        all_reports.append(report_md)

        # Print a compact summary to stderr so users see progress.
        print(f"\n--- {name} ---", file=sys.stderr)
        for mode in ("tiered", "frontier_only", "oss_only"):
            a = analyses[mode]
            print(
                f"  {mode:15s} "
                f"cost_saved={a['cost_saved_pct']:.1f}%  "
                f"delegated_q={a['delegated_quality_pct']:.1f}% (n={a['n_delegated']})  "
                f"worst={a['min_slot_quality']:.1f}%  "
                f"tiers={a['tier_counts']}",
                file=sys.stderr,
            )

    combined_md = "\n\n---\n\n".join(all_reports)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(combined_md)
        print(f"\nWrote {out_path}", file=sys.stderr)
    else:
        print(combined_md)


def _cmd_build_table(args: argparse.Namespace) -> None:
    oss_dir      = Path(args.run_dir_oss)
    frontier_dir = Path(args.run_dir_frontier)
    threshold    = float(args.threshold)

    # Same guard as dry-run: never write a confidently-empty routing table.
    if not oss_dir.is_dir() or not frontier_dir.is_dir():
        _fail_no_quality_data(oss_dir, frontier_dir)

    print("Building routing table…", file=sys.stderr)
    table = build_routing_table(oss_dir, frontier_dir, threshold=threshold)
    if not table["slots"]:
        _fail_no_quality_data(oss_dir, frontier_dir)

    out_path = Path(args.out) if args.out else Path("routing-tiered.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(table, indent=2) + "\n")
    print(f"Wrote {out_path}  ({len(table['slots'])} slots)", file=sys.stderr)

    for task_id, slot in table["slots"].items():
        pick = slot["picks_at_default_threshold"]["tiered"]
        print(
            f"  {task_id:30s} tiered -> T{pick['tier']} {pick['model']}  "
            f"(gate {pick['gate_quality_worst_scenario']}%)",
            file=sys.stderr,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m orchestrator.tiered",
        description="Tiered routing tooling: dry-run comparison + routing table.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- dry-run ---
    dr = sub.add_parser(
        "dry-run",
        help="3-mode cost/quality comparison for one or all example workflows.",
    )
    dr.add_argument(
        "run_dir_oss",
        metavar="<run-dir-oss>",
        help="Path to an OSS bake-off run directory (contains judge-batch.json + judge-scores.json).",
    )
    dr.add_argument(
        "run_dir_frontier",
        metavar="<run-dir-frontier>",
        help="Path to a frontier bake-off run directory (contains judge-scores.json).",
    )
    dr.add_argument(
        "--workflow",
        default=None,
        metavar="NAME",
        help=(
            f"Workflow to analyse. Choices: {', '.join(sorted(EXAMPLE_WORKFLOWS))}. "
            "Omit to run all."
        ),
    )
    dr.add_argument(
        "--threshold",
        default=DEFAULT_THRESHOLD,
        type=float,
        metavar="FLOAT",
        help=f"Quality threshold for delegation decisions (default: {DEFAULT_THRESHOLD}).",
    )
    dr.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="Write Markdown output to this file instead of stdout.",
    )
    dr.set_defaults(func=_cmd_dry_run)

    # --- build-table ---
    bt = sub.add_parser(
        "build-table",
        help="Generate routing-tiered.json from OSS + frontier bake-off run dirs.",
    )
    bt.add_argument(
        "run_dir_oss",
        metavar="<run-dir-oss>",
        help="Path to an OSS bake-off run directory.",
    )
    bt.add_argument(
        "run_dir_frontier",
        metavar="<run-dir-frontier>",
        help="Path to a frontier bake-off run directory.",
    )
    bt.add_argument(
        "--threshold",
        default=DEFAULT_THRESHOLD,
        type=float,
        metavar="FLOAT",
        help=f"Quality threshold (default: {DEFAULT_THRESHOLD}).",
    )
    bt.add_argument(
        "--out",
        default="routing-tiered.json",
        metavar="PATH",
        help="Output path for the JSON file (default: routing-tiered.json).",
    )
    bt.set_defaults(func=_cmd_build_table)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args   = parser.parse_args(argv)
    args.func(args)
