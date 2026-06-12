"""Heuristic LLM-call detector.

Walks a SKILL.md (and optional sibling source files) and finds places that
likely call a model. Pure regex/keyword matching — there is no AST parsing,
no LSP, no fancy code analysis. The goal is to seed the orchestration plan
with a plausible inventory, NOT to be 100% accurate. The human reviews and
edits.

Patterns we look for:

1. **Skill body LAWS mentioning a model call** — phrases like
   "invoke the model", "ask Claude", "agent call", "send to LLM"
2. **Direct SDK imports / calls** in any sibling .ts / .py / .js / .sh
   - `anthropic.messages.create(`, `claude.messages.create(`
   - `openai.chat.completions.create(`
   - `ollama.chat(`, `ollama.generate(`
   - `oracle ` / `askoracle ` command invocations
3. **Skill invocations within the body** — `Skill(` calls that prompt a model
4. **Subagent dispatch patterns** — "spawn subagent", "Task tool",
   "fanout via subagents"

Output: a list of `LLMCallPattern` records. Each carries enough context for
the human to identify the call site and classify it. We deliberately produce
DUPES (e.g., a LAW description AND a code-pattern match for the same call) —
dedup is the human's job once they understand the actual call site.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────
# Patterns. Each tuple: (regex, kind, short_description).
# Order doesn't matter (we collect all matches), but we keep similar
# patterns grouped for readability.
# ─────────────────────────────────────────────────────────────────────

SKILL_BODY_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    # Skill LAWS / workflow text — natural language hints
    (
        re.compile(r"\b(?:invoke|call|ask|prompt)\s+(?:the\s+)?(?:model|llm|claude|opus|sonnet|haiku|gpt|gemini)\b", re.IGNORECASE),
        "skill-narrative",
        "Skill body mentions an explicit model invocation.",
    ),
    (
        re.compile(r"\b(?:agent\s+call|subagent|fanout|spawn[s]?\s+subagent[s]?|dispatch[es]?\s+(?:to\s+)?subagent[s]?)\b", re.IGNORECASE),
        "subagent-dispatch",
        "Skill body describes a subagent dispatch (each subagent = one LLM call).",
    ),
    (
        re.compile(r"\b(?:send|submit|hand[\s-]?off|forward)\s+(?:to\s+)?(?:the\s+)?(?:llm|model|frontier|judge|oracle)\b", re.IGNORECASE),
        "skill-narrative",
        "Skill body mentions handing off to an LLM/judge.",
    ),
    (
        re.compile(r"\bclaude\s+(?:is\s+)?the\s+classifier\b", re.IGNORECASE),
        "skill-narrative",
        "Skill body declares Claude as classifier (a routed LLM call).",
    ),
    (
        re.compile(r"\bSkill\(\s*[\"\']", re.IGNORECASE),
        "skill-invocation",
        "Skill body invokes another Skill (may itself invoke a model).",
    ),
    # Oracle / askoracle CLI
    (
        re.compile(r"\b(?:askoracle|oracle)\s+[-\w\s\"\']*--models\b", re.IGNORECASE),
        "oracle-cli",
        "Skill body invokes oracle CLI (multi-frontier consultation).",
    ),
]


CODE_FILE_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    # Anthropic SDK
    (
        re.compile(r"\b(?:anthropic|client)\.messages\.create\s*\("),
        "anthropic-sdk",
        "Anthropic SDK call: client.messages.create()",
    ),
    (
        re.compile(r"\bnew\s+Anthropic\s*\("),
        "anthropic-sdk",
        "Anthropic SDK client instantiation (likely followed by .messages.create).",
    ),
    # OpenAI SDK
    (
        re.compile(r"\bopenai\.chat\.completions\.create\s*\("),
        "openai-sdk",
        "OpenAI SDK call: openai.chat.completions.create()",
    ),
    (
        re.compile(r"\b(?:OpenAI|openai)\s*\(\s*\{"),
        "openai-sdk",
        "OpenAI SDK client construction.",
    ),
    # Ollama
    (
        re.compile(r"\bollama\.(?:chat|generate)\s*\("),
        "ollama-sdk",
        "Ollama SDK call: ollama.chat() / ollama.generate()",
    ),
    # Gemini SDK
    (
        re.compile(r"\bgenai\.GenerativeModel\s*\("),
        "google-genai-sdk",
        "Google GenAI SDK model instantiation.",
    ),
    # Generic HTTP completions
    (
        re.compile(r"/v1/chat/completions"),
        "openai-compat-http",
        "OpenAI-compatible HTTP endpoint hit (could be a local gateway, LM Studio, etc.)",
    ),
    # oracle CLI
    (
        re.compile(r"\boracle\s+[-\w]+.*--models\b"),
        "oracle-cli",
        "oracle CLI invocation.",
    ),
]


@dataclass(frozen=True)
class LLMCallPattern:
    """One detected LLM-call site.

    `file` is relative to the workspace root when scanning a workspace dir;
    absolute when scanning an individual file. `line` is 1-indexed.
    `kind` is one of the categories above. `excerpt` is up to ~120 chars
    of context for the human to identify the site.
    """

    file: str
    line: int
    kind: str
    description: str
    excerpt: str


def _scan_text_with_patterns(
    text: str,
    file_label: str,
    patterns: list[tuple[re.Pattern[str], str, str]],
) -> list[LLMCallPattern]:
    """Run all patterns against `text` and emit a record per match.

    Splits into lines so `line` is meaningful for the human. We cap the
    excerpt at 120 chars to keep output readable.
    """
    out: list[LLMCallPattern] = []
    lines = text.splitlines()
    for line_idx, raw_line in enumerate(lines):
        line = raw_line.rstrip()
        for pattern, kind, description in patterns:
            if pattern.search(line):
                excerpt = line.strip()
                if len(excerpt) > 120:
                    excerpt = excerpt[:117] + "..."
                out.append(
                    LLMCallPattern(
                        file=file_label,
                        line=line_idx + 1,
                        kind=kind,
                        description=description,
                        excerpt=excerpt,
                    )
                )
    return out


def scan_skill_for_patterns(skill_md_path: Path) -> list[LLMCallPattern]:
    """Scan a SKILL.md for narrative + invocation patterns.

    Does NOT walk sibling code — use `scan_workspace_for_patterns` for that.
    """
    if not skill_md_path.exists():
        raise FileNotFoundError(f"SKILL.md not found: {skill_md_path}")
    text = skill_md_path.read_text()
    return _scan_text_with_patterns(text, str(skill_md_path), SKILL_BODY_PATTERNS)


# File extensions we consider "code" for the workspace scan. Kept narrow on
# purpose — scanning .json or .lock would produce noise.
_CODE_EXTENSIONS = {".ts", ".tsx", ".js", ".mjs", ".py", ".sh", ".bash"}

# Directories to skip when walking a workspace. Keep this list narrow but
# practical — node_modules/.git/.venv are universally noise.
_SKIP_DIRS = {"node_modules", ".git", ".venv", "venv", "__pycache__", "dist", "build", ".next"}


def scan_workspace_for_patterns(workspace: Path) -> list[LLMCallPattern]:
    """Walk a workspace directory, scan SKILL.md(s) + code files.

    `workspace` is typically a skill's parent directory. Conventionally
    contains: `SKILL.md`, optional `AGENT.md`, and any helper scripts the
    skill invokes via Bash.

    Returns flat list of matches; caller groups / deduplicates as needed.
    """
    if not workspace.exists():
        raise FileNotFoundError(f"Workspace not found: {workspace}")
    if not workspace.is_dir():
        raise NotADirectoryError(f"Workspace is not a directory: {workspace}")

    matches: list[LLMCallPattern] = []

    for path in sorted(workspace.rglob("*")):
        if not path.is_file():
            continue
        # Skip anything inside a skip-dir WITHIN the workspace. Only the
        # path relative to the workspace root counts — an ANCESTOR directory
        # named e.g. "dist" or "build" (think ~/build/my-app) must not blank
        # the whole scan.
        if any(part in _SKIP_DIRS for part in path.relative_to(workspace).parts):
            continue
        # SKILL.md / *.skill.md / SKILL-*.md — apply skill-body patterns.
        if path.name == "SKILL.md" or path.name.endswith(".skill.md"):
            text = path.read_text()
            rel = str(path.relative_to(workspace))
            matches.extend(_scan_text_with_patterns(text, rel, SKILL_BODY_PATTERNS))
            # Skill bodies sometimes embed bash code — also run code patterns.
            matches.extend(_scan_text_with_patterns(text, rel, CODE_FILE_PATTERNS))
            continue
        # Code files — apply code patterns.
        if path.suffix in _CODE_EXTENSIONS:
            try:
                text = path.read_text()
            except UnicodeDecodeError:
                # Binary-ish file with the wrong extension. Skip.
                continue
            rel = str(path.relative_to(workspace))
            matches.extend(_scan_text_with_patterns(text, rel, CODE_FILE_PATTERNS))

    return matches


def dedupe_patterns(patterns: list[LLMCallPattern]) -> list[LLMCallPattern]:
    """Collapse exact duplicates (same file + line + kind).

    Different patterns matching the same line stay separate — they describe
    different aspects of the same site, useful for the human.
    """
    seen: set[tuple[str, int, str]] = set()
    out: list[LLMCallPattern] = []
    for p in patterns:
        key = (p.file, p.line, p.kind)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out
