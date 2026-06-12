"""Unit tests for scaffold rendering — template + scanner output → plan body."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from orchestrator.onboarding.scaffold import (
    build_inventory_table,
    render_next_steps,
    render_plan_from_template,
)
from orchestrator.onboarding.scanner import LLMCallPattern


# Minimal template fixture matching the real one's load-bearing markers.
_TEMPLATE_FIXTURE = """\
# Orchestration Plan — <APP_NAME>

**Status:** draft
**Last bake-off:** <ISO_TIMESTAMP>
**Routing.json slots used:** <COMMA_SEPARATED_LIST>
**Owner:** <NAME>
**Onboarded by:** <NAME> on <DATE>

## 1. Task inventory

| Task | Input shape | Output shape | Frequency (per app run) | Latency tolerance | Notes |
|---|---|---|---|---|---|
| <task_name> | <input_shape> | <output_shape> | <freq> | <tolerance> | <notes> |

## 2. Slot mapping

| Task | Slot | Rationale |
|---|---|---|
| <task_name> | <routing_slot> | <one_sentence_why> |
"""


def _make_template(tmp_path: Path) -> Path:
    t = tmp_path / "template.md"
    t.write_text(_TEMPLATE_FIXTURE)
    return t


# --- build_inventory_table -------------------------------------------------


def test_build_inventory_table_empty_patterns():
    out = build_inventory_table([])
    assert "TODO" in out
    assert out.count("|") >= 7  # 6 columns → 7 pipes


def test_build_inventory_table_renders_one_row_per_pattern():
    patterns = [
        LLMCallPattern(file="SKILL.md", line=10, kind="skill-narrative", description="D1", excerpt="e1"),
        LLMCallPattern(file="agent.py", line=20, kind="anthropic-sdk", description="D2", excerpt="e2"),
    ]
    out = build_inventory_table(patterns)
    rows = out.strip().split("\n")
    assert len(rows) == 2
    assert "task_01" in rows[0]
    assert "task_02" in rows[1]
    # Each row mentions the file:line
    assert "SKILL.md:10" in rows[0]
    assert "agent.py:20" in rows[1]


def test_build_inventory_table_escapes_pipes_in_excerpts():
    p = LLMCallPattern(
        file="x.md", line=1, kind="skill-narrative",
        description="d", excerpt="excerpt with | pipe inside",
    )
    out = build_inventory_table([p])
    # The literal pipe in the excerpt must be escaped so the table renders.
    assert "\\|" in out


def test_build_inventory_table_truncates_long_excerpts():
    long_excerpt = "x" * 200
    p = LLMCallPattern(file="x.md", line=1, kind="k", description="d", excerpt=long_excerpt)
    out = build_inventory_table([p])
    # Excerpt cell truncated to ≤ 60 chars + "..." marker
    assert "..." in out
    # And the row stays a single line.
    assert "\n" not in out


# --- render_plan_from_template --------------------------------------------


def test_render_plan_substitutes_app_name(tmp_path: Path):
    template = _make_template(tmp_path)
    out = render_plan_from_template(template, "my-app", patterns=[])
    assert "# Orchestration Plan — my-app" in out
    # Both occurrences of <APP_NAME> are replaced
    assert "<APP_NAME>" not in out


def test_render_plan_substitutes_onboarded_by_and_date(tmp_path: Path):
    template = _make_template(tmp_path)
    fixed_date = datetime(2026, 5, 26, tzinfo=timezone.utc)
    out = render_plan_from_template(
        template, "my-app", patterns=[], onboarded_by="example_user", today=fixed_date,
    )
    assert "**Onboarded by:** example_user on 2026-05-26" in out


def test_render_plan_replaces_bake_off_placeholder(tmp_path: Path):
    template = _make_template(tmp_path)
    out = render_plan_from_template(template, "my-app", patterns=[])
    assert "**Last bake-off:** not yet run (draft)" in out
    assert "<ISO_TIMESTAMP>" not in out


def test_render_plan_fills_inventory_table(tmp_path: Path):
    template = _make_template(tmp_path)
    patterns = [
        LLMCallPattern(file="SKILL.md", line=15, kind="skill-narrative", description="d", excerpt="e"),
    ]
    out = render_plan_from_template(template, "my-app", patterns)
    # Placeholder row gone, scanner row in.
    assert "<task_name> | <input_shape>" not in out
    assert "task_01" in out
    assert "SKILL.md:15" in out


def test_render_plan_keeps_other_sections_intact(tmp_path: Path):
    template = _make_template(tmp_path)
    out = render_plan_from_template(template, "my-app", patterns=[])
    # Section 2's placeholder row is NOT touched (different shape).
    assert "<task_name> | <routing_slot> | <one_sentence_why>" in out


def test_render_plan_handles_missing_template(tmp_path: Path):
    missing = tmp_path / "no.md"
    with pytest.raises(FileNotFoundError):
        render_plan_from_template(missing, "my-app", patterns=[])


def test_render_plan_resilient_to_template_drift(tmp_path: Path):
    """If the template's placeholder row was edited away, we don't crash —
    we just leave the table as-is and proceed."""
    template = tmp_path / "drifted.md"
    template.write_text(
        "# Orchestration Plan — <APP_NAME>\n\n"
        "**Last bake-off:** <ISO_TIMESTAMP>\n"
        "**Onboarded by:** <NAME> on <DATE>\n\n"
        "## Task inventory\n\n"
        "Someone deleted the table. No placeholder line here.\n"
    )
    out = render_plan_from_template(template, "my-app", patterns=[
        LLMCallPattern(file="x.md", line=1, kind="k", description="d", excerpt="e"),
    ])
    # No crash. Header still substituted.
    assert "# Orchestration Plan — my-app" in out


# --- render_next_steps ----------------------------------------------------


def test_render_next_steps_includes_path_and_app(tmp_path: Path):
    p = tmp_path / "plan.md"
    txt = render_next_steps("my-app", p, n_patterns_found=3)
    assert str(p) in txt
    assert "3 candidate LLM-call site" in txt
    # 6-step checklist visible
    for marker in ["1.", "2.", "3.", "4.", "5.", "6."]:
        assert marker in txt


def test_render_next_steps_zero_patterns_phrases_softly(tmp_path: Path):
    p = tmp_path / "plan.md"
    txt = render_next_steps("my-app", p, n_patterns_found=0)
    assert "no LLM-call patterns" in txt.lower() or "scanner found no" in txt.lower()


def test_render_next_steps_references_this_repos_artifacts(tmp_path: Path):
    """Regression: the checklist must point at THIS repo's files and CLIs,
    not at paths from some other source tree."""
    txt = render_next_steps("my-app", tmp_path / "plan.md", n_patterns_found=1)
    assert "data/evaluation/tasks-example.yaml" in txt
    assert "orchestrator-eval" in txt
    assert "data/routing.json" in txt
    # No instructions to cd into a sibling checkout or edit foreign packages.
    assert "cd " not in txt
    assert "packages/" not in txt
