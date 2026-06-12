"""Application orchestration onboarding — scaffold tooling for app → router integration.

This module is the supporting infrastructure for the onboarding methodology
documented in README.md and AGENTS.md. It is intentionally SKELETAL — it removes
the blank-page problem (scanning skills, seeding the plan from the template)
but leaves the actual classification + scenario authoring + bake-off run to
the human.

The CLI entry point is `orchestrator.onboarding.init`::

    python -m orchestrator.onboarding init \\
        --app my-app-name \\
        --workspace /path/to/my-app-workspace

Behavior::

    1. Load the packaged plan template (onboarding/templates/)
    2. Scan the app's SKILL.md (+ optional sibling files) for LLM-call patterns
    3. Pre-fill the "Task inventory" table with discovered patterns
    4. Write {workspace}/orchestration-plan.md (refuse to overwrite without --overwrite)
    5. Print the 6-step checklist with app-specific next-step text

Anti-pattern guard: this tool does NOT make routing decisions. It does not
choose slots. It does not run a bake-off. Those steps require the human to
inspect real app traffic. The tool's value is in the scaffolding shape, not
the content.
"""

from .scanner import LLMCallPattern, scan_skill_for_patterns
from .scaffold import build_inventory_table, render_plan_from_template

__all__ = [
    "LLMCallPattern",
    "build_inventory_table",
    "render_plan_from_template",
    "scan_skill_for_patterns",
]
