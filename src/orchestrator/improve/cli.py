"""CLI for the self-improving loops.

    python -m orchestrator.improve harvest   --app news-digest         # Loop A
    python -m orchestrator.improve detect-models                        # Loop B
    python -m orchestrator.improve radar-plan                           # Loop C (plan)
    python -m orchestrator.improve gate-check --run-dir <run>           # the hard rule

All commands are SAFE: they harvest / detect / propose / advise and write
staging or proposal files. None of them mutate tasks-v1.yaml or routing.json —
those changes go through the gated bake-off + human confirm (the hard rule).

State lives under the data directory: `./data` relative to the current working
directory by default, overridable with `--data-dir` or the ORCHESTRATOR_DATA_DIR
environment variable. It is never derived from the installed package location.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import loop_a_mistakes as A
from . import loop_b_models as B
from . import loop_c_research as C
from .guard import IntegrityGateError, require_integrity
from .state import StateFileError, load_json_state

_AUDIT_SCORES_FIX = (
    "re-run the audit quality steps (prepare-batch + judge) to regenerate judge-scores.json"
)


def _data_dir(args) -> Path:
    """Resolve the data directory: --data-dir flag > ORCHESTRATOR_DATA_DIR env >
    ./data under the current working directory. Never derived from __file__ —
    an installed package would resolve into site-packages."""
    if getattr(args, "data_dir", None):
        return Path(args.data_dir)
    env = os.environ.get("ORCHESTRATOR_DATA_DIR")
    if env:
        return Path(env)
    return Path.cwd() / "data"


def _load_audit_scores(path: Path) -> dict[str, float]:
    """Audit-quality judge-scores.json (array of {sample_id, mean_quality_score})
    → the {sample_id: score} mapping Loop A's second failure source consumes."""
    data = load_json_state(path, expect=list, how_to_fix=_AUDIT_SCORES_FIX)
    scores: dict[str, float] = {}
    for i, s in enumerate(data):
        if not isinstance(s, dict) or s.get("sample_id") is None:
            continue
        q = s.get("mean_quality_score")
        if q is None:
            continue
        try:
            scores[str(s["sample_id"])] = float(q)
        except (TypeError, ValueError):
            raise StateFileError(
                path, f"entry {i} has a non-numeric mean_quality_score ({q!r})",
                _AUDIT_SCORES_FIX,
            ) from None
    return scores


def _cmd_harvest(args) -> int:
    data = _data_dir(args)
    sample_scores = _load_audit_scores(Path(args.audit_scores)) if args.audit_scores else None
    rep = A.harvest_failures(
        args.app,
        lookback_days=args.lookback_days,
        sample_scores=sample_scores,
        rebake_threshold_pct=args.rebake_threshold,
    )
    staging = data / "improve" / "staged-scenarios.json"
    queue = data / "improve" / "rebake-queue.json"
    A.stage_scenarios(rep, staging)
    A.write_rebake_queue(rep, queue)
    print(f"[loop-a] {len(rep.failures)} failure(s); slots to re-bake: {rep.slots_to_rebake or 'none'}")
    print(f"[loop-a] staged → {staging}")
    print(f"[loop-a] re-bake queue → {queue}")
    return 0


def _cmd_detect_models(args) -> int:
    data = _data_dir(args)
    rep = B.detect_new_models(data / "routing.json", data / "routing-tiered.json")
    print(f"[loop-b] available: {len(rep.available)} | baked: {len(rep.already_baked)} | NEW: {rep.new_models or 'none'}")
    if rep.new_models:
        p = B.propose_rebake(rep, data / "improve" / "rebake-proposal.json")
        print(f"[loop-b] proposal → {p}")
    return 0


def _cmd_radar_plan(args) -> int:
    print(json.dumps(C.radar_plan(), indent=2))
    return 0


def _cmd_gate_check(args) -> int:
    try:
        require_integrity(Path(args.run_dir))
    except IntegrityGateError as e:
        print(f"[gate] FAIL: {e}", file=sys.stderr)
        return 3
    print("[gate] PASS — data is safe for a self-improving loop to act on.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="orchestrator.improve", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--data-dir",
        default=None,
        help="Data directory for state files (default: ./data; env: ORCHESTRATOR_DATA_DIR).",
    )

    p_h = sub.add_parser("harvest", parents=[common],
                         help="Loop A: harvest production failures → staged scenarios + re-bake queue")
    p_h.add_argument("--app", required=True)
    p_h.add_argument("--lookback-days", type=int, default=7)
    p_h.add_argument("--audit-scores", default=None,
                     help="Path to an audit-quality judge-scores.json; enables the second "
                          "failure source (samples the audit judge scored below the re-bake line).")
    p_h.add_argument("--rebake-threshold", type=float, default=80.0,
                     help="Below-this-is-a-failure line (0-100) for --audit-scores.")

    sub.add_parser("detect-models", parents=[common],
                   help="Loop B: detect new unbaked models → re-bake proposal")
    sub.add_parser("radar-plan", help="Loop C: print the research-radar search plan")

    p_g = sub.add_parser("gate-check", help="The hard rule: verify a run dir passed the integrity gate")
    p_g.add_argument("--run-dir", required=True)

    args = p.parse_args(argv)
    handler = {
        "harvest": _cmd_harvest,
        "detect-models": _cmd_detect_models,
        "radar-plan": _cmd_radar_plan,
        "gate-check": _cmd_gate_check,
    }[args.command]
    try:
        return handler(args)
    except StateFileError as e:
        # A corrupt state file must surface as an instruction, not a traceback.
        print(f"[improve] STATE ERROR — {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
