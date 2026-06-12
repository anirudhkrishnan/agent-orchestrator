"""Loop B — learn from NEW models.

A routing table rots: new local models land in Ollama, new frontier tiers ship,
and last quarter's winners get superseded. Loop B detects models that are
AVAILABLE but have never been baked, and PROPOSES a re-bake including them. It
never silently changes routing — a new winner is a proposal for human confirm
(routing decisions affect high-stakes slots).

Detection = (available models) − (already-baked models). "Available" = local
Ollama models (`ollama list`) + a small frontier registry. "Already-baked" =
every model id referenced in routing.json / routing-tiered.json.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .state import load_json_state

# Recovery hint for corrupt routing files read during detection.
_ROUTING_FIX = ("routing files are produced by the bake-off finalize step — restore "
                "from your last run, or delete the corrupt file to treat every "
                "available model as new")

# Frontier models worth tracking as candidates. Update as providers ship.
# Loop C (research radar) is what surfaces "a new model exists" — this list is
# the actionable subset Loop B knows how to bake.
KNOWN_FRONTIER_REGISTRY = (
    "anthropic/claude-opus-4-8",
    "anthropic/claude-opus-4-7",
    "anthropic/claude-sonnet-4-6",
    "anthropic/claude-haiku-4-5",
)


@dataclass
class NewModelReport:
    available: list[str] = field(default_factory=list)
    already_baked: list[str] = field(default_factory=list)
    new_models: list[str] = field(default_factory=list)
    judge_models: list[str] = field(default_factory=list)  # never proposed as candidates


def _ollama_list() -> list[str]:
    """Local Ollama model ids as 'ollama/<name>'. Empty on any failure (Loop B
    must degrade gracefully — a missing Ollama is not a crash)."""
    try:
        out = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10
        )
        if out.returncode != 0:
            return []
        models = []
        for line in out.stdout.splitlines()[1:]:  # skip header
            name = line.split()[0] if line.split() else ""
            if name:
                models.append(f"ollama/{name}")
        return models
    except (FileNotFoundError, subprocess.SubprocessError):
        return []


def known_baked_models(*routing_paths: Path) -> set[str]:
    """Every model id referenced anywhere in the given routing files."""
    baked: set[str] = set()
    for p in routing_paths:
        p = Path(p)
        if not p.exists():
            continue
        data = load_json_state(p, how_to_fix=_ROUTING_FIX)
        _collect_model_strings(data, baked)
    # Drop sentinels — they're not models.
    baked.discard("TO_BE_BAKED")
    baked.discard("queue-for-human")
    return baked


def known_judge_models(*routing_paths: Path) -> set[str]:
    """Model ids referenced as `judge_model` in the given routing files.

    The judge sits OUTSIDE the candidate pool by design, so Loop B must never
    propose a judge as a bake-off candidate."""
    judges: set[str] = set()
    for p in routing_paths:
        p = Path(p)
        if not p.exists():
            continue
        _collect_judge_strings(load_json_state(p, how_to_fix=_ROUTING_FIX), judges)
    return judges


def _collect_judge_strings(obj, out: set[str]) -> None:
    """Walk a routing JSON structure collecting `judge_model` values."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "judge_model" and isinstance(v, str):
                out.add(v)
            else:
                _collect_judge_strings(v, out)
    elif isinstance(obj, list):
        for x in obj:
            _collect_judge_strings(x, out)


def _collect_model_strings(obj, out: set[str]) -> None:
    """Walk a routing JSON structure collecting values of model-ish keys."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("model", "fallback_model", "judge_model") and isinstance(v, str):
                out.add(v)
            elif isinstance(v, str) and ("/" in v) and (v.startswith("ollama/") or v.startswith("anthropic/")):
                out.add(v)
            else:
                _collect_model_strings(v, out)
    elif isinstance(obj, list):
        for x in obj:
            _collect_model_strings(x, out)


def detect_new_models(
    *routing_paths: Path,
    available_models: list[str] | None = None,
) -> NewModelReport:
    """Return models that are available but never baked.

    Args:
        routing_paths: routing.json + routing-tiered.json (the baked sources).
        available_models: override for testing; otherwise = ollama list +
            the frontier registry.
    """
    if available_models is None:
        available_models = _ollama_list() + list(KNOWN_FRONTIER_REGISTRY)
    available = sorted(set(available_models))
    baked = known_baked_models(*routing_paths)
    # A judge model that's baked-as-judge but never baked-as-candidate is still
    # "new" for candidacy — but to keep it simple we treat any referenced id as
    # known. New = available and not referenced anywhere.
    new = sorted(m for m in available if m not in baked)
    return NewModelReport(
        available=available,
        already_baked=sorted(baked),
        new_models=new,
        judge_models=sorted(known_judge_models(*routing_paths)),
    )


def propose_rebake(report: NewModelReport, proposal_path: Path, *, tasks: list[str] | None = None) -> Path | None:
    """Write a re-bake PROPOSAL for the new models. Returns None if nothing new.

    This is a proposal, not an action: it lists the candidates to add and the
    command to run. Routing only changes after the bake-off passes the integrity
    gate AND a human confirms (new winners on high-stakes slots are not
    auto-promoted)."""
    if not report.new_models:
        return None
    proposal_path = Path(proposal_path)
    proposal_path.parent.mkdir(parents=True, exist_ok=True)
    # The suggested command must actually RUN: the eval runner is ollama-only
    # (any other prefix raises ValueError) and the judge sits outside the
    # candidate pool. So --candidates carries ollama/* ids with judges
    # excluded; non-ollama additions go on a comment line for a human to bake
    # through other means.
    judges = set(report.judge_models)
    runnable = [m for m in report.new_models + report.already_baked
                if m.startswith("ollama/") and m not in judges]
    non_ollama_new = [m for m in report.new_models if not m.startswith("ollama/")]
    if runnable:
        command = (
            "python -m orchestrator.evaluation run "
            "--tasks data/evaluation/tasks-v1.yaml "
            f"--candidates {' '.join(runnable)} "
            "--judge claude-opus-4-8-interactive-session "
            "--out-dir data/evaluation/runs --samples-per-cell 5"
        )
        if non_ollama_new:
            command += ("\n# Not in --candidates (eval runner is ollama-only): "
                        + ", ".join(non_ollama_new))
    else:
        command = ("# No runnable candidates (eval runner is ollama-only). "
                   "New non-ollama models: " + ", ".join(non_ollama_new))
    proposal = {
        "_README": "Loop B re-bake PROPOSAL. New models are available but unbaked. "
                   "Review, then run the bake-off command. Routing changes require "
                   "the integrity gate + human confirmation — never auto-applied.",
        "new_models": report.new_models,
        "include_existing": report.already_baked,
        "suggested_command": command,
        "tasks": tasks or "all in tasks-v1.yaml",
        "status": "needs_human_confirm",
    }
    proposal_path.write_text(json.dumps(proposal, indent=2) + "\n")
    return proposal_path
