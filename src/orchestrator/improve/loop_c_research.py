"""Loop C — learn from external research / best practices.

A standing radar that periodically asks "has the field moved?" — new judging
methodologies, routing techniques, eval rigor, model-release notes — and emits
an ADVISORY report. It is deliberately the weakest-authority loop:

  Loop C NEVER changes the methodology or routing. It only surfaces "you might
  consider X, here's the source." Methodology changes are high-risk and
  human-gated; auto-applying a paper's claim is how you regress quietly.

A Python module can't browse the web, so the search itself is an agent-driven
seam: a web-search-capable agent runs the searches in `RADAR_QUERIES`, then
feeds findings (as dicts) into `build_advisory`. This module owns the standing
query plan, the finding schema, the dedup-vs-seen logic, and the advisory
rendering — all deterministic + testable.

Dedup is two-phase: `build_advisory` only READS the seen-claims file; claims
are persisted as seen via `mark_claims_seen` AFTER the advisory has actually
been delivered. Marking before delivery would mean a crash in between silently
drops the claim from every future cycle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .state import load_json_state

# Recovery hint for a corrupt seen-claims file.
_SEEN_FIX = ("delete the file — the next radar pass will simply re-surface "
             "previously seen claims")

# The standing radar. Stable topics; the agent runs these as web searches.
RADAR_QUERIES: tuple[tuple[str, str], ...] = (
    ("judging", "LLM-as-a-judge bias mitigation latest research"),
    ("judging", "pairwise vs pointwise LLM evaluation reliability 2026"),
    ("sampling", "how many samples LLM eval variance best practice"),
    ("routing", "LLM model routing cost quality cascade technique new"),
    ("eval-frameworks", "Inspect AI / promptfoo / Braintrust new features eval"),
    ("models", "new open-weight model release reasoning benchmark"),
    ("models", "Anthropic Claude new model release Opus Sonnet Haiku"),
    ("prompt-opt", "DSPy MIPRO automatic prompt optimization update"),
)


@dataclass
class Finding:
    """One radar hit. `suggested_action` is a PROPOSAL only."""

    topic: str
    claim: str
    source: str            # url or publication
    source_date: str
    relevance: str         # why it matters to THIS primitive
    suggested_action: str  # what a human might consider — never auto-applied


@dataclass
class AdvisoryReport:
    generated_for: str
    findings: list[Finding] = field(default_factory=list)
    note: str = ""


def _claim_key(claim: str) -> str:
    """Dedup key for a claim — case/whitespace-insensitive, length-capped."""
    return claim.strip().lower()[:160]


def _read_seen(seen_claims_path: Path) -> set[str]:
    return set(load_json_state(seen_claims_path, expect=dict, how_to_fix=_SEEN_FIX).get("seen", []))


def build_advisory(
    findings: list[dict],
    *,
    generated_for: str = "agent-orchestrator",
    seen_claims_path: Path | None = None,
) -> AdvisoryReport:
    """Turn agent-collected findings into an advisory, skipping already-seen claims.

    Read-only with respect to the seen-claims file: call `mark_claims_seen`
    AFTER the advisory has been delivered (written / reviewed). Marking here
    would lose claims forever if anything crashed between build and delivery.

    Args:
        findings: list of dicts with keys matching Finding fields. Produced by an
            agent that ran web searches over RADAR_QUERIES.
        seen_claims_path: optional JSON file of previously-surfaced claim keys, so
            the radar doesn't re-report the same thing every cycle.
    """
    seen: set[str] = set()
    if seen_claims_path and Path(seen_claims_path).exists():
        seen = _read_seen(Path(seen_claims_path))

    fresh: list[Finding] = []
    surfaced: set[str] = set()  # within-batch dedup only — never persisted here
    for f in findings:
        key = _claim_key(f.get("claim", ""))
        if not key or key in seen or key in surfaced:
            continue
        surfaced.add(key)
        fresh.append(Finding(
            topic=f.get("topic", "general"),
            claim=f.get("claim", ""),
            source=f.get("source", ""),
            source_date=f.get("source_date", ""),
            relevance=f.get("relevance", ""),
            suggested_action=f.get("suggested_action", ""),
        ))

    note = (
        f"{len(fresh)} new finding(s). ADVISORY ONLY — nothing here changes the "
        f"methodology or routing automatically. Any adopted change goes through "
        f"the normal gated bake-off (integrity gate + human confirm)."
    )
    return AdvisoryReport(generated_for=generated_for, findings=fresh, note=note)


def mark_claims_seen(report: AdvisoryReport, seen_claims_path: Path) -> Path:
    """Persist the claims surfaced in `report` so future radar passes skip them.

    Call this ONLY after the advisory has actually been delivered (written to
    disk / handed to a reviewer). A crash between "marked seen" and "human saw
    it" would otherwise silently drop the claim from every future cycle.
    """
    seen_claims_path = Path(seen_claims_path)
    seen: set[str] = set()
    if seen_claims_path.exists():
        seen = _read_seen(seen_claims_path)
    seen.update(k for k in (_claim_key(f.claim) for f in report.findings) if k)
    seen_claims_path.parent.mkdir(parents=True, exist_ok=True)
    seen_claims_path.write_text(json.dumps({"seen": sorted(seen)}, indent=2) + "\n")
    return seen_claims_path


def render_advisory(report: AdvisoryReport) -> str:
    """Markdown advisory for human review."""
    lines = [
        f"# Loop C — Research Advisory ({report.generated_for})",
        "",
        f"_{report.note}_",
        "",
    ]
    if not report.findings:
        lines.append("No new findings since last radar pass.")
        return "\n".join(lines)
    by_topic: dict[str, list[Finding]] = {}
    for f in report.findings:
        by_topic.setdefault(f.topic, []).append(f)
    for topic, items in sorted(by_topic.items()):
        lines.append(f"## {topic}")
        lines.append("")
        for f in items:
            lines.append(f"- **{f.claim}**")
            lines.append(f"  - Source: {f.source} ({f.source_date})")
            lines.append(f"  - Relevance: {f.relevance}")
            lines.append(f"  - Consider (not auto-applied): {f.suggested_action}")
        lines.append("")
    return "\n".join(lines)


def radar_plan() -> dict:
    """The standing search plan an agent executes. Returned as data so a CLI
    can hand it to a web-search-capable agent."""
    return {
        "_README": "Loop C radar plan. Hand each query to a web-search-capable "
                   "agent, collect findings as dicts (topic, claim, source, "
                   "source_date, relevance, suggested_action), then call "
                   "build_advisory(findings). After the advisory is delivered, "
                   "call mark_claims_seen(report, seen_path).",
        "queries": [{"topic": t, "query": q} for t, q in RADAR_QUERIES],
        "finding_schema": ["topic", "claim", "source", "source_date", "relevance", "suggested_action"],
    }
