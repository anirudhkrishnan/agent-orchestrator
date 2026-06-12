"""Tests for pure helpers in evaluation.runner.

The HTTP transport itself is integration-tested by running against a real
Ollama instance; here we cover the deterministic helpers (id safety, prefix
stripping, qwen3.5 detection) and the write-to-disk shape.
"""

from datetime import datetime, timezone
from pathlib import Path

from orchestrator.evaluation.runner import (
    CandidateRun,
    _needs_think_false,
    _safe_filename_part,
    _safe_id,
    _strip_provider_prefix,
    _write_run,
)


def test_safe_id_lowercases_and_replaces_specials():
    assert _safe_id("ollama/qwen3:8b") == "ollama-qwen3-8b"
    assert _safe_id("ollama/qwen3.5:9b") == "ollama-qwen3-5-9b"
    assert _safe_id("MyProvider/Big-Model 1.2") == "myprovider-big-model-1-2"


def test_safe_id_strips_edge_dashes():
    assert _safe_id("---weird---") == "weird"


def test_safe_id_falls_back_when_all_special():
    assert _safe_id("///") == "unknown"


def test_strip_provider_prefix_default():
    assert _strip_provider_prefix("ollama/qwen3:8b") == "qwen3:8b"


def test_strip_provider_prefix_no_slash_returns_unchanged():
    assert _strip_provider_prefix("qwen3:8b") == "qwen3:8b"


def test_needs_think_false_matches_qwen3_family():
    """All qwen3.X variants — including qwen3:8b (non-.5) and qwen3.5:9b — exhibit
    thinking-mode-by-default (empirical finding, 2026-05-26 b)."""
    assert _needs_think_false("qwen3:8b") is True
    assert _needs_think_false("qwen3:14b") is True
    assert _needs_think_false("qwen3.5:9b") is True
    assert _needs_think_false("Qwen3.5-mlx") is True


def test_needs_think_false_matches_gemma4_family():
    """gemma4 family also emits thinking tokens on structured-output tasks."""
    assert _needs_think_false("gemma4:e4b") is True
    assert _needs_think_false("Gemma4:27b") is True


def test_needs_think_false_matches_reasoning_models():
    """Explicit reasoning models (DeepSeek-R1, QwQ) always think."""
    assert _needs_think_false("deepseek-r1-distill-qwen-32b") is True
    assert _needs_think_false("qwq-32b-preview") is True


def test_needs_think_false_skips_non_thinking_models():
    """Models without built-in reasoning mode should NOT get think=False —
    passing it to them may error or be silently misinterpreted."""
    assert _needs_think_false("llama3.1:8b") is False
    assert _needs_think_false("gemma3:27b") is False  # gemma3 is pre-thinking
    assert _needs_think_false("mistral:7b") is False


def test_write_run_creates_candidate_subdir_and_correct_filename(tmp_path: Path):
    run = CandidateRun(
        candidate="ollama/qwen3:8b",
        task_id="entity_extraction",
        scenario_id="scn-01",
        output_text='["Acme Corp","Widget Inc"]',
        latency_ms=2_100,
        error=None,
        completed_at=datetime.now(timezone.utc),
    )
    out = _write_run(tmp_path, run)
    expected = tmp_path / "ollama-qwen3-8b" / "entity_extraction_scn-01.json"
    assert out == expected
    assert expected.exists()
    # Round-trip the JSON shape.
    body = expected.read_text()
    reloaded = CandidateRun.model_validate_json(body)
    assert reloaded.output_text == '["Acme Corp","Widget Inc"]'
    assert reloaded.candidate == "ollama/qwen3:8b"


def test_safe_filename_part_passes_through_typical_ids():
    """Typical task/scenario ids must stay byte-identical (stable layout)."""
    assert _safe_filename_part("entity_extraction") == "entity_extraction"
    assert _safe_filename_part("scn-01") == "scn-01"


def test_safe_filename_part_replaces_unsafe_chars():
    assert _safe_filename_part("task/with:specials") == "task-with-specials"
    assert _safe_filename_part("a b\tc") == "a-b-c"
    assert _safe_filename_part("") == "unknown"


def test_write_run_slugifies_filesystem_unsafe_ids(tmp_path: Path):
    """Regression: a '/' or ':' in task/scenario ids must not create nested
    directories or invalid paths — the filename is slugified, the JSON payload
    keeps the original ids."""
    run = CandidateRun(
        candidate="ollama/qwen3:8b",
        task_id="task/with:specials",
        scenario_id="scn 01",
        output_text="out",
        latency_ms=1_000,
        error=None,
        completed_at=datetime.now(timezone.utc),
    )
    out = _write_run(tmp_path, run)
    assert out == tmp_path / "ollama-qwen3-8b" / "task-with-specials_scn-01.json"
    assert out.exists()
    # Original (unsanitized) ids are preserved in the payload.
    reloaded = CandidateRun.model_validate_json(out.read_text())
    assert reloaded.task_id == "task/with:specials"
    assert reloaded.scenario_id == "scn 01"


def test_write_run_overwrites_on_repeat(tmp_path: Path):
    run1 = CandidateRun(
        candidate="ollama/qwen3:8b",
        task_id="tk",
        scenario_id="s1",
        output_text="first",
        latency_ms=1_000,
        error=None,
        completed_at=datetime.now(timezone.utc),
    )
    run2 = CandidateRun(
        candidate="ollama/qwen3:8b",
        task_id="tk",
        scenario_id="s1",
        output_text="second",
        latency_ms=2_000,
        error=None,
        completed_at=datetime.now(timezone.utc),
    )
    _write_run(tmp_path, run1)
    out = _write_run(tmp_path, run2)
    reloaded = CandidateRun.model_validate_json(out.read_text())
    assert reloaded.output_text == "second"
