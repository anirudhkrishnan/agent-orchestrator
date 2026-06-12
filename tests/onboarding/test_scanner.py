"""Unit tests for the LLM-call pattern scanner."""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.onboarding.scanner import (
    LLMCallPattern,
    dedupe_patterns,
    scan_skill_for_patterns,
    scan_workspace_for_patterns,
)


# --- skill body patterns ----------------------------------------------------


def test_scan_skill_finds_invoke_model_phrasing(tmp_path: Path):
    skill = tmp_path / "SKILL.md"
    skill.write_text(
        "# My skill\n"
        "\n"
        "When triggered, invoke the model to produce a summary.\n"
        "Then ask Claude to classify the result.\n"
    )
    patterns = scan_skill_for_patterns(skill)
    kinds = {p.kind for p in patterns}
    assert "skill-narrative" in kinds
    # Two narrative phrases on two lines
    narrative = [p for p in patterns if p.kind == "skill-narrative"]
    assert len(narrative) >= 2


def test_scan_skill_finds_subagent_dispatch(tmp_path: Path):
    skill = tmp_path / "SKILL.md"
    skill.write_text(
        "Spawn parallel subagents (one per document).\n"
        "Then fanout via subagents to draft summaries.\n"
    )
    patterns = scan_skill_for_patterns(skill)
    kinds = [p.kind for p in patterns]
    assert "subagent-dispatch" in kinds


def test_scan_skill_finds_oracle_cli(tmp_path: Path):
    skill = tmp_path / "SKILL.md"
    skill.write_text(
        "Then run: `oracle --models opus,gpt5,gemini -p \"review document\"`\n"
    )
    patterns = scan_skill_for_patterns(skill)
    kinds = [p.kind for p in patterns]
    assert "oracle-cli" in kinds


def test_scan_skill_finds_skill_invocation(tmp_path: Path):
    skill = tmp_path / "SKILL.md"
    skill.write_text(
        'Invoke `Skill("data-fetch")` before the CLI chain.\n'
    )
    patterns = scan_skill_for_patterns(skill)
    kinds = [p.kind for p in patterns]
    assert "skill-invocation" in kinds


def test_scan_skill_finds_claude_classifier_declaration(tmp_path: Path):
    skill = tmp_path / "SKILL.md"
    skill.write_text(
        "For this pipeline, Claude is the classifier.\n"
    )
    patterns = scan_skill_for_patterns(skill)
    assert any(p.kind == "skill-narrative" for p in patterns)


def test_scan_skill_returns_empty_when_no_matches(tmp_path: Path):
    skill = tmp_path / "SKILL.md"
    skill.write_text("# Nothing here\n\nJust prose with no LLM patterns.\n")
    patterns = scan_skill_for_patterns(skill)
    assert patterns == []


def test_scan_skill_records_line_numbers(tmp_path: Path):
    skill = tmp_path / "SKILL.md"
    skill.write_text(
        "Line 1\n"
        "Line 2\n"
        "Line 3: invoke the model here\n"
        "Line 4\n"
    )
    patterns = scan_skill_for_patterns(skill)
    assert len(patterns) == 1
    assert patterns[0].line == 3


def test_scan_skill_truncates_long_excerpts(tmp_path: Path):
    skill = tmp_path / "SKILL.md"
    long_line = "invoke the model " + ("x" * 500)
    skill.write_text(long_line + "\n")
    patterns = scan_skill_for_patterns(skill)
    assert len(patterns) == 1
    assert len(patterns[0].excerpt) <= 120


def test_scan_skill_raises_when_missing(tmp_path: Path):
    missing = tmp_path / "does-not-exist.md"
    with pytest.raises(FileNotFoundError):
        scan_skill_for_patterns(missing)


# --- code file patterns ----------------------------------------------------


def test_scan_workspace_finds_anthropic_sdk_calls(tmp_path: Path):
    (tmp_path / "SKILL.md").write_text("# stub skill\n")
    src = tmp_path / "agent.ts"
    src.write_text(
        "import Anthropic from 'anthropic';\n"
        "const client = new Anthropic({apiKey: process.env.ANTHROPIC_API_KEY});\n"
        "const result = await client.messages.create({model: 'opus-4-7', ...});\n"
    )
    patterns = scan_workspace_for_patterns(tmp_path)
    kinds = {p.kind for p in patterns}
    assert "anthropic-sdk" in kinds


def test_scan_workspace_finds_openai_sdk_calls(tmp_path: Path):
    (tmp_path / "SKILL.md").write_text("# stub\n")
    src = tmp_path / "bot.py"
    src.write_text(
        "from openai import OpenAI\n"
        "client = OpenAI(api_key='...')\n"
        "resp = openai.chat.completions.create(model='gpt-5')\n"
    )
    patterns = scan_workspace_for_patterns(tmp_path)
    kinds = {p.kind for p in patterns}
    assert "openai-sdk" in kinds


def test_scan_workspace_finds_ollama_sdk_calls(tmp_path: Path):
    (tmp_path / "SKILL.md").write_text("# stub\n")
    src = tmp_path / "local.py"
    src.write_text(
        "import ollama\n"
        "resp = ollama.chat(model='qwen3:8b', messages=[...])\n"
    )
    patterns = scan_workspace_for_patterns(tmp_path)
    kinds = {p.kind for p in patterns}
    assert "ollama-sdk" in kinds


def test_scan_workspace_finds_openai_compat_http(tmp_path: Path):
    (tmp_path / "SKILL.md").write_text("# stub\n")
    src = tmp_path / "bridge.sh"
    src.write_text(
        "curl -X POST http://localhost:18789/v1/chat/completions \\\n"
        "    -H 'Authorization: Bearer ...' \\\n"
        "    -d '{\"model\": \"qwen3:8b\"}'\n"
    )
    patterns = scan_workspace_for_patterns(tmp_path)
    kinds = {p.kind for p in patterns}
    assert "openai-compat-http" in kinds


def test_scan_workspace_skips_skip_dirs(tmp_path: Path):
    """node_modules etc. must be ignored, even if they contain code."""
    (tmp_path / "SKILL.md").write_text("# stub\n")
    bad = tmp_path / "node_modules" / "openai" / "dist" / "index.js"
    bad.parent.mkdir(parents=True)
    bad.write_text("openai.chat.completions.create({});\n")
    patterns = scan_workspace_for_patterns(tmp_path)
    # The node_modules match should be absent.
    files = {p.file for p in patterns}
    assert not any("node_modules" in f for f in files)


def test_scan_workspace_under_skip_named_ancestor_dir(tmp_path: Path):
    """Regression: a workspace LIVING UNDER a directory named like a skip-dir
    (e.g. ~/dist/my-app) must still be scanned — only skip-dirs INSIDE the
    workspace count. The old check matched ancestor path components and
    silently scanned nothing."""
    ws = tmp_path / "dist" / "my-app"
    ws.mkdir(parents=True)
    (ws / "SKILL.md").write_text("When triggered, invoke the model to summarize.\n")
    patterns = scan_workspace_for_patterns(ws)
    assert any(p.kind == "skill-narrative" for p in patterns)
    # And skip-dirs inside the workspace are still skipped.
    bad = ws / "node_modules" / "lib" / "index.js"
    bad.parent.mkdir(parents=True)
    bad.write_text("openai.chat.completions.create({});\n")
    patterns = scan_workspace_for_patterns(ws)
    assert not any("node_modules" in p.file for p in patterns)


def test_scan_workspace_skips_binary_files(tmp_path: Path):
    """Binary blobs with code extensions don't blow up the scan."""
    (tmp_path / "SKILL.md").write_text("# stub\n")
    binary = tmp_path / "weird.py"
    # Write actual non-utf8 bytes
    binary.write_bytes(b"\x80\x81\x82 not utf8 \xff\xfe")
    # Should not raise.
    patterns = scan_workspace_for_patterns(tmp_path)
    files = {p.file for p in patterns}
    assert "weird.py" not in files


def test_scan_workspace_handles_skill_with_embedded_bash(tmp_path: Path):
    """SKILL.md often contains bash blocks calling oracle / curl. Scan both ways."""
    (tmp_path / "SKILL.md").write_text(
        "# Skill\n"
        "\n"
        "Run this:\n"
        "```bash\n"
        "curl http://localhost:18789/v1/chat/completions -d '...'\n"
        "```\n"
        "\n"
        "Then invoke the model to summarize.\n"
    )
    patterns = scan_workspace_for_patterns(tmp_path)
    kinds = {p.kind for p in patterns}
    # Should pick up both: the http call from CODE_PATTERNS and the
    # "invoke the model" phrasing from SKILL_BODY_PATTERNS.
    assert "openai-compat-http" in kinds
    assert "skill-narrative" in kinds


def test_scan_workspace_raises_when_workspace_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        scan_workspace_for_patterns(tmp_path / "nope")


def test_scan_workspace_raises_when_workspace_not_dir(tmp_path: Path):
    f = tmp_path / "not-a-dir.txt"
    f.write_text("x")
    with pytest.raises(NotADirectoryError):
        scan_workspace_for_patterns(f)


# --- dedupe ----------------------------------------------------------------


def test_dedupe_collapses_exact_duplicates():
    a = LLMCallPattern(file="x.py", line=1, kind="ollama-sdk", description="d", excerpt="e")
    b = LLMCallPattern(file="x.py", line=1, kind="ollama-sdk", description="d2", excerpt="e2")
    c = LLMCallPattern(file="x.py", line=2, kind="ollama-sdk", description="d", excerpt="e")
    out = dedupe_patterns([a, b, c])
    # a + b collapse (same file/line/kind), c is distinct.
    assert len(out) == 2


def test_dedupe_keeps_different_kinds_on_same_line():
    a = LLMCallPattern(file="x.md", line=1, kind="skill-narrative", description="", excerpt="")
    b = LLMCallPattern(file="x.md", line=1, kind="openai-compat-http", description="", excerpt="")
    out = dedupe_patterns([a, b])
    assert len(out) == 2
