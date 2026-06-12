"""CLI entry point for the orchestration onboarding scaffold.

Single subcommand for now — `init`. Future subcommands (status / audit /
re-bake) are left as TODO since they need real telemetry to be useful.

Invocation::

    python -m orchestrator.onboarding init \\
        --app my-app-name \\
        --workspace /path/to/my-app-workspace

Exit codes::

    0 — plan written
    2 — argument error (workspace doesn't exist, plan already exists w/o --overwrite, etc.)
    3 — template not found (bad --template path, or a broken package install)
"""

from __future__ import annotations

import argparse
import sys
from importlib.resources import files
from pathlib import Path

from .scaffold import render_next_steps, render_plan_from_template
from .scanner import dedupe_patterns, scan_workspace_for_patterns


# Default template — packaged with this module, so it resolves the same way
# for editable installs, wheels, and pip installs into site-packages.
def _default_template_path() -> Path:
    """Locate the packaged orchestration-plan template.

    The template ships inside the package (`onboarding/templates/`), so no
    repo-root walking is needed — `importlib.resources` resolves it wherever
    the package is installed.
    """
    resource = files(__package__).joinpath("templates/orchestration-plan.template.md")
    return Path(str(resource))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orchestrator.onboarding",
        description=(
            "Scaffold an orchestration-plan.md for a production-ready app. "
            "Scans the app's workspace for LLM-call patterns and seeds the "
            "Task inventory table. The human completes the remaining steps "
            "per README.md / AGENTS.md."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=False)

    p_init = sub.add_parser(
        "init",
        help="Create a new orchestration-plan.md from the template + workspace scan.",
        description=(
            "Writes {workspace}/orchestration-plan.md from the canonical "
            "template, pre-filling the Task inventory table with LLM-call "
            "patterns found in the workspace. Refuses to overwrite without "
            "--overwrite."
        ),
    )
    p_init.add_argument(
        "--app",
        required=True,
        help="App name, used in the plan's H1 (e.g., my-app-report-pipeline).",
    )
    p_init.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help=(
            "Path to the app's workspace directory (typically the skill's "
            "parent — containing SKILL.md). The plan is written here too."
        ),
    )
    p_init.add_argument(
        "--template",
        type=Path,
        default=None,
        help=(
            "Override path to orchestration-plan.template.md. Default is "
            "the template bundled with this package (onboarding/templates/)."
        ),
    )
    p_init.add_argument(
        "--onboarded-by",
        default="the user",
        help="Name recorded in the 'Onboarded by' field (default: the user).",
    )
    p_init.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing orchestration-plan.md in the workspace.",
    )
    p_init.add_argument(
        "--no-scan",
        action="store_true",
        help="Skip the workspace scan; emit a plan with the placeholder inventory row.",
    )
    p_init.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Override the output path. Default: {workspace}/orchestration-plan.md. "
            "Useful for dry-runs / testing."
        ),
    )

    return parser


def _cmd_init(args: argparse.Namespace) -> int:
    # 1. Validate workspace
    workspace: Path = args.workspace
    if not workspace.exists():
        print(f"[onboarding] workspace does not exist: {workspace}", file=sys.stderr)
        return 2
    if not workspace.is_dir():
        print(f"[onboarding] workspace is not a directory: {workspace}", file=sys.stderr)
        return 2

    # 2. Locate template
    template_path: Path = args.template or _default_template_path()
    if not template_path.exists():
        print(
            f"[onboarding] template not found: {template_path}\n"
            "Check the --template path, or reinstall the package if the "
            "bundled template is missing.",
            file=sys.stderr,
        )
        return 3

    # 3. Determine output path
    out_path: Path = args.out if args.out is not None else (workspace / "orchestration-plan.md")
    if out_path.exists() and not args.overwrite:
        print(
            f"[onboarding] {out_path} already exists. Pass --overwrite to replace.",
            file=sys.stderr,
        )
        return 2

    # 4. Scan workspace (unless --no-scan)
    if args.no_scan:
        patterns = []
    else:
        raw = scan_workspace_for_patterns(workspace)
        patterns = dedupe_patterns(raw)

    # 5. Render plan
    text = render_plan_from_template(
        template_path=template_path,
        app_name=args.app,
        patterns=patterns,
        onboarded_by=args.onboarded_by,
    )

    # 6. Write
    out_path.write_text(text)
    print(render_next_steps(args.app, out_path, len(patterns)))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "init":
        return _cmd_init(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
