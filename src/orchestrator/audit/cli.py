"""CLI entry point for the audit engine.

Three subcommands::

    # Scaffold a starter audit-config.yaml for a new app.
    python -m orchestrator.audit init \\
        --app news-digest \\
        --out data/audit/news-digest.yaml

    # Run correctness + effects + prepare quality batch. Writes
    # AUDIT-REPORT.md immediately for the non-quality sections and prints
    # 'READY FOR JUDGE' for the quality batch (interactive judge step).
    python -m orchestrator.audit run \\
        --app news-digest \\
        --config data/audit/news-digest.yaml

    # After the judge writes judge-scores.json: compute drift, update the
    # AUDIT-REPORT.md with the quality section + headline.
    python -m orchestrator.audit finalize \\
        --app news-digest \\
        --config data/audit/news-digest.yaml

The split is identical in shape to the evaluation framework's
`prepare-batch` / `finalize` flow — once your muscle memory knows
"interactive judge sits between two phases", the audit feels familiar.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import init_audit_config_skeleton, load_audit_config
from .correctness import run_correctness_audit
from .effects import compute_effects_report
from .quality import finalize_quality, prepare_quality_batch
from .report import compose_audit_report


def build_parser() -> argparse.ArgumentParser:
    """Top-level parser with `init`, `run`, `finalize` subcommands."""
    parser = argparse.ArgumentParser(
        prog="orchestrator.audit",
        description=(
            "Audit the orchestration engine: verify routing correctness, "
            "quantify frontier-model displacement, and detect quality drift "
            "against the bake-off baseline. Two-phase (run → judge → finalize) "
            "to keep the frontier judge interactive — same pattern as the "
            "evaluation framework."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=False)

    # --- init -----------------------------------------------------------------
    p_init = sub.add_parser(
        "init",
        help="Scaffold a starter audit-config.yaml for an app.",
        description=(
            "Writes a YAML file at --out with sensible defaults. Edit "
            "`slots_in_scope` and (optionally) thresholds + pricing entries, "
            "then run `audit run`."
        ),
    )
    p_init.add_argument("--app", required=True, help="Stable identifier for the app.")
    p_init.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Path to write the new YAML file.",
    )
    p_init.add_argument(
        "--frontier-model",
        default="anthropic/claude-opus-4-7",
        help="Frontier model recorded in pricing.frontier_model.",
    )
    p_init.add_argument(
        "--slots",
        nargs="+",
        default=None,
        help="Slot names to seed (default: the canonical 5 bake-off slots).",
    )
    p_init.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing file at --out (DESTRUCTIVE).",
    )

    # --- run ------------------------------------------------------------------
    p_run = sub.add_parser(
        "run",
        help="Run correctness + effects, prepare the quality batch.",
        description=(
            "Reads telemetry for the configured lookback window; produces a "
            "preliminary AUDIT-REPORT.md (correctness + effects sections "
            "complete, quality marked 'pending'); writes judge-batch.json + "
            "prints 'READY FOR JUDGE'."
        ),
    )
    p_run.add_argument("--app", required=True)
    p_run.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to audit-config YAML.",
    )

    # --- finalize -------------------------------------------------------------
    p_fin = sub.add_parser(
        "finalize",
        help="After the judge writes scores, finalize the audit report.",
        description=(
            "Reads judge-scores.json from the run's out_dir, computes per-slot "
            "drift vs the routing.json baseline, and rewrites AUDIT-REPORT.md "
            "with the quality section populated."
        ),
    )
    p_fin.add_argument("--app", required=True)
    p_fin.add_argument("--config", type=Path, required=True)

    return parser


def _cmd_init(args: argparse.Namespace) -> int:
    try:
        path = init_audit_config_skeleton(
            app_name=args.app,
            out_path=args.out,
            slots=args.slots,
            frontier_model=args.frontier_model,
            overwrite=args.overwrite,
        )
    except FileExistsError as e:
        print(f"[audit] {e}", file=sys.stderr)
        return 2
    print(f"[audit] wrote skeleton at {path}")
    print()
    print("Next steps:")
    print(f"  1. Edit {path} — set `slots_in_scope` to match this app's slots.")
    print("  2. Adjust `warn_threshold_pct` / `rebake_threshold_pct` if needed.")
    print(f"  3. Run: python -m orchestrator.audit run --app {args.app} --config {path}")
    return 0


def _resolve_config_path(args: argparse.Namespace) -> tuple[Path, Path]:
    """Return (config_path, anchor) — both absolute.

    The anchor is the parent directory of the config file; we use it to
    resolve relative paths inside the YAML (e.g. routing_json_path) without
    depending on the caller's CWD.
    """
    cfg_path = args.config.resolve()
    anchor = cfg_path.parent
    return cfg_path, anchor


def _cmd_run(args: argparse.Namespace) -> int:
    cfg_path, anchor = _resolve_config_path(args)
    cfg = load_audit_config(cfg_path)
    if cfg.app_name != args.app:
        print(
            f"[audit] config app_name={cfg.app_name!r} does not match "
            f"--app={args.app!r}. Refusing to proceed.",
            file=sys.stderr,
        )
        return 2

    now = datetime.now(timezone.utc)
    out_dir = cfg.out_dir_resolved(anchor) / cfg.app_name / now.strftime("%Y-%m-%dT%H-%M-%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[audit] running correctness audit (lookback: {cfg.lookback_days}d)")
    correctness = run_correctness_audit(cfg, now=now, config_anchor=anchor)
    print(
        f"[audit]   {len(correctness.slots)} slot(s) checked, "
        f"{correctness.alarm_count} alarm(s)"
    )
    # Hard structural gate: an incomplete scope (unbaked slot) makes the whole
    # audit meaningless — don't proceed to judge prep, and exit non-zero so
    # automation/cron sees the failure (RCA stress-test CRIT: the CLI used to
    # discard overall_pass and always return 0 → silent green).
    if correctness.incomplete_scope_alarms:
        print(file=sys.stderr)
        print("🛑 [audit] INCOMPLETE SCOPE — refusing to run. Fix before auditing:", file=sys.stderr)
        for a in correctness.incomplete_scope_alarms:
            print(f"   - {a}", file=sys.stderr)
        return 3

    print("[audit] computing effects report")
    effects = compute_effects_report(cfg, now=now)
    print(
        f"[audit]   {effects.total_calls} calls, "
        f"savings = {effects.overall_savings_pct:.1f}%"
    )

    print("[audit] preparing quality batch")
    batch_path = prepare_quality_batch(
        cfg,
        out_dir=out_dir,
        config_path=cfg_path,
        now=now,
    )
    scores_path = out_dir / "judge-scores.json"

    # Write a preliminary report with quality marked pending. The finalize step
    # rewrites this same file with the populated quality section.
    from .quality import QualityReport

    placeholder_quality = QualityReport(
        app_name=cfg.app_name,
        window_start_iso=correctness.window_start_iso,
        window_end_iso=correctness.window_end_iso,
        judge_model=cfg.judge_model,
        warn_threshold_pct=cfg.warn_threshold_pct,
        rebake_threshold_pct=cfg.rebake_threshold_pct,
        slots=[],  # empty until finalize
    )
    report_path = out_dir / "AUDIT-REPORT.md"
    compose_audit_report(
        cfg=cfg,
        correctness=correctness,
        effects=effects,
        quality=placeholder_quality,
        out_path=report_path,
    )
    print(f"[audit] wrote preliminary {report_path}")
    print()
    print("READY FOR JUDGE")
    print(f"  - Read    : {batch_path}")
    print(f"  - Write to: {scores_path}")
    print(
        f"  - Then run: python -m orchestrator.audit finalize "
        f"--app {cfg.app_name} --config {cfg_path}"
    )
    return 0


def _cmd_finalize(args: argparse.Namespace) -> int:
    cfg_path, anchor = _resolve_config_path(args)
    cfg = load_audit_config(cfg_path)
    if cfg.app_name != args.app:
        print(
            f"[audit] config app_name={cfg.app_name!r} does not match "
            f"--app={args.app!r}. Refusing to proceed.",
            file=sys.stderr,
        )
        return 2

    # Find the most recent run dir under out_dir/app_name/.
    base = cfg.out_dir_resolved(anchor) / cfg.app_name
    if not base.exists():
        print(
            f"[audit] no run directories found under {base}. Run `audit run` first.",
            file=sys.stderr,
        )
        return 2
    runs = sorted(p for p in base.iterdir() if p.is_dir())
    if not runs:
        print(
            f"[audit] no run directories under {base}. Run `audit run` first.",
            file=sys.stderr,
        )
        return 2
    out_dir = runs[-1]
    print(f"[audit] finalizing run at {out_dir}")

    now = datetime.now(timezone.utc)
    correctness = run_correctness_audit(cfg, now=now, config_anchor=anchor)
    effects = compute_effects_report(cfg, now=now)
    try:
        quality = finalize_quality(
            cfg,
            out_dir=out_dir,
            config_anchor=anchor,
            now=now,
        )
    except FileNotFoundError as e:
        print(f"[audit] {e}", file=sys.stderr)
        return 2

    report_path = out_dir / "AUDIT-REPORT.md"
    compose_audit_report(
        cfg=cfg,
        correctness=correctness,
        effects=effects,
        quality=quality,
        out_path=report_path,
    )
    print(f"[audit] rewrote {report_path}")

    # --- Verdict exit code (RCA stress-test CRIT) ---------------------------
    # finalize is the machine-readable verdict. Return non-zero when anything
    # actionable is wrong, so cron/CI can't read a silent green:
    #   3 = correctness failed (incomplete scope, unexpected model, no traffic…)
    #   4 = quality drift past the RE-BAKE line
    #   5 = a baked, in-scope slot could not be verified — insufficient samples
    #       to assess drift, OR no quality_pct_of_judge baseline in routing.json
    #       (either way, "unverified" must not read as "healthy")
    rebake = quality.needs_rebake()
    warn = quality.needs_warn()
    insufficient = quality.insufficient_samples()
    no_baseline = quality.missing_baseline()
    if rebake:
        print(f"[audit] RE-BAKE recommended for: {', '.join(rebake)}")
    elif warn:
        print(f"[audit] WARN for: {', '.join(warn)}")
    if insufficient:
        print(
            f"[audit] ⚠️  INSUFFICIENT SAMPLES to assess drift on: "
            f"{', '.join(insufficient)} — these slots are unverified, not healthy.",
            file=sys.stderr,
        )
    if no_baseline:
        print(
            f"[audit] ⚠️  NO BASELINE to assess drift on: {', '.join(no_baseline)} "
            f"— routing.json has no quality_pct_of_judge for these slots; they "
            f"are unverified, not healthy.",
            file=sys.stderr,
        )
    # Effects-accounting flags belong in the verdict summary, not stderr-only.
    if effects.unpriced_models:
        print(
            f"[audit] ⚠️  effects: model(s) missing from the pricing table: "
            f"{', '.join(effects.unpriced_models)} — counted with zero claimed "
            f"displacement."
        )
    if effects.cost_unverified_models:
        print(
            f"[audit] ⚠️  effects: paid model(s) with no logged cost: "
            f"{', '.join(effects.cost_unverified_models)} — savings unverified, "
            f"excluded from the claimed savings."
        )

    if not correctness.overall_pass:
        print("[audit] VERDICT: correctness FAILED — exit 3", file=sys.stderr)
        return 3
    if rebake:
        print("[audit] VERDICT: quality drift past re-bake line — exit 4", file=sys.stderr)
        return 4
    if insufficient or no_baseline:
        print(
            "[audit] VERDICT: unverified slots (insufficient samples / missing "
            "baseline) — exit 5",
            file=sys.stderr,
        )
        return 5
    print("[audit] VERDICT: pass")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Process entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "init":
        return _cmd_init(args)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "finalize":
        return _cmd_finalize(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
