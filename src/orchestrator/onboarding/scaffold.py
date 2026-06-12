"""Render the orchestration-plan.md from the template + scanner output.

The packaged template at `onboarding/templates/orchestration-plan.template.md`
is the canonical source-of-truth shape. We do TWO things to it:

1. Substitute the header fields: `<APP_NAME>`, `<NAME>`, `<DATE>` etc.
2. Replace the empty "Task inventory" table row with a pre-filled table from
   the scanner's discoveries.

Everything else stays as placeholder text — the human fills it in. We do not
auto-classify slots, do not guess at quality bars, do not write rationale.
The scaffold removes blank-page friction and nothing more.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .scanner import LLMCallPattern


def build_inventory_table(patterns: list[LLMCallPattern]) -> str:
    """Render scanner output as the markdown table rows for Section 1.

    The template has a 6-column table:

        | Task | Input shape | Output shape | Frequency | Latency tolerance | Notes |

    For each detected pattern, we emit ONE row with:
    - Task: derived from kind + file (the human renames)
    - Input shape: empty (human fills)
    - Output shape: empty (human fills)
    - Frequency: empty
    - Latency tolerance: empty
    - Notes: file:line + excerpt

    If `patterns` is empty, we emit a single TODO row so the human sees the
    section is empty by design (not just truncated).
    """
    if not patterns:
        return (
            "| <task_name> | <input_shape> | <output_shape> | <freq> | <tolerance> | "
            "_TODO: scanner found no LLM-call patterns; inventory by hand._ |"
        )

    rows: list[str] = []
    for i, p in enumerate(patterns, start=1):
        # Derive a placeholder task name from the kind + index. The human
        # will rename to something meaningful (e.g., "constraint_reasoning").
        task_name = f"task_{i:02d}_{p.kind.replace('-', '_')}"
        # Note column: file:line + truncated excerpt + scanner's description.
        site = f"`{p.file}:{p.line}`"
        excerpt_cell = p.excerpt.replace("|", "\\|")  # escape table delim
        if len(excerpt_cell) > 60:
            excerpt_cell = excerpt_cell[:57] + "..."
        note = f"{p.description} Site: {site} — `{excerpt_cell}`"
        # All other columns are placeholders for the human.
        rows.append(
            f"| `{task_name}` | <input_shape> | <output_shape> | <freq> | "
            f"<tolerance> | {note} |"
        )
    return "\n".join(rows)


def _render_header(
    template_text: str,
    app_name: str,
    onboarded_by: str,
    today_iso_date: str,
) -> str:
    """Replace `<APP_NAME>` and `<DATE>` placeholders.

    We do NOT replace `<NAME>` for the Owner field — the human chooses.
    `<ISO_TIMESTAMP>` stays as "not yet run" since the bake-off hasn't fired.
    """
    out = template_text.replace("<APP_NAME>", app_name, 1)  # first occurrence — the H1
    # The first occurrence of <APP_NAME> is in the H1; replace remaining ones too.
    out = out.replace("<APP_NAME>", app_name)
    # Header fields
    out = out.replace(
        "**Last bake-off:** <ISO_TIMESTAMP>",
        "**Last bake-off:** not yet run (draft)",
        1,
    )
    out = out.replace(
        "**Onboarded by:** <NAME> on <DATE>",
        f"**Onboarded by:** {onboarded_by} on {today_iso_date}",
        1,
    )
    return out


def _replace_inventory_table(text: str, inventory_rows: str) -> str:
    """Swap the template's placeholder row with the scanner's rows.

    Template has exactly one placeholder row:

        | <task_name> | <input_shape> | <output_shape> | <freq> | <tolerance> | <notes> |

    We replace that single line with the multi-row block. The header row
    above (`| Task | Input shape | ...`) and separator (`|---|...|`) stay.
    """
    placeholder_line = (
        "| <task_name> | <input_shape> | <output_shape> | <freq> | "
        "<tolerance> | <notes> |"
    )
    if placeholder_line not in text:
        # Template drift — caller error or template was edited. We don't
        # crash; we leave the table alone so the file is still useful.
        return text
    return text.replace(placeholder_line, inventory_rows, 1)


def render_plan_from_template(
    template_path: Path,
    app_name: str,
    patterns: list[LLMCallPattern],
    onboarded_by: str = "the user",
    today: datetime | None = None,
) -> str:
    """Produce the orchestration-plan.md body from the template + scanner output.

    Parameters
    ----------
    template_path : Path to the orchestration-plan template (the packaged
                    default or a --template override)
    app_name      : Name of the app being onboarded (e.g. "my-app-report-pipeline")
    patterns      : Scanner output. Use `[]` if no scanning was done.
    onboarded_by  : Goes into the "Onboarded by" field. Defaults to "the user".
    today         : Override for testing. Production uses UTC now.
    """
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    template_text = template_path.read_text()
    if today is None:
        today = datetime.now(timezone.utc)
    today_iso = today.strftime("%Y-%m-%d")

    text = _render_header(template_text, app_name, onboarded_by, today_iso)
    inventory_rows = build_inventory_table(patterns)
    text = _replace_inventory_table(text, inventory_rows)
    return text


# ─────────────────────────────────────────────────────────────────────
# Human-facing next-steps printer — what `init` prints after writing
# the plan file. The 6-step checklist (README.md / AGENTS.md cover the
# full procedure), with the app's specific paths spliced in.
# ─────────────────────────────────────────────────────────────────────


def render_next_steps(app_name: str, plan_path: Path, n_patterns_found: int) -> str:
    """Plain-text instructions printed to stdout after `init` writes the plan."""
    lines: list[str] = []
    lines.append("")
    lines.append(f"Orchestration plan scaffold written: {plan_path}")
    lines.append("")
    if n_patterns_found:
        lines.append(
            f"Scanner found {n_patterns_found} candidate LLM-call site(s). "
            "Review the 'Task inventory' table in the plan — rename the "
            "placeholder task_NN_* names to meaningful task names and fill "
            "in the input/output/frequency columns."
        )
    else:
        lines.append(
            "Scanner found NO LLM-call patterns in this workspace. Either "
            "(a) the app uses an SDK/CLI pattern not in the scanner, or "
            "(b) the workspace is the wrong directory. Inventory by hand and "
            "update the plan."
        )
    lines.append("")
    lines.append("Next steps (README.md and AGENTS.md cover the full procedure):")
    lines.append("")
    lines.append("  1. Finish the Task inventory table — every LLM call this app makes.")
    lines.append("  2. Map each task to a routing slot (Section 2 of plan). Reuse an")
    lines.append("     existing slot when the task shape matches; new slots need scenarios.")
    lines.append("  3. For NEW slots, author ~5 scenarios drawn from real app traffic.")
    lines.append("     Copy data/evaluation/tasks-example.yaml as your starting point.")
    lines.append("  4. Run the bake-off:")
    lines.append("        orchestrator-eval run \\")
    lines.append("          --tasks data/evaluation/tasks.yaml \\")
    lines.append("          --candidates ollama/qwen3:8b ollama/qwen3.5:9b ollama/gemma4:e4b \\")
    lines.append("          --judge claude-opus-4-7-interactive-session \\")
    lines.append("          --out-dir data/evaluation/runs")
    lines.append("     (Then init-baselines → judge fills → prepare-batch → judge fills → finalize)")
    lines.append("  5. Merge the routing delta from the run's REPORT.md into data/routing.json.")
    lines.append("  6. Wire your runtime router to data/routing.json — resolve each task's")
    lines.append("     slot to its winning model at call time (see plan Section 8).")
    lines.append("")
    lines.append("When done, flip the plan's status from 'draft' to 'production' and commit.")
    return "\n".join(lines)
