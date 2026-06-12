"""Tests for the reference judge adapter (`orchestrator.judge_adapter`).

All HTTP traffic is faked by monkeypatching `httpx.post` — no network. Each
fake "endpoint" is a scripted queue of responses/exceptions so the retry
behavior (backoff, temperature-400 fallback, failure tally) is observable
call-by-call.
"""

import json
from pathlib import Path

import httpx
import pytest

from orchestrator import judge_adapter
from orchestrator.judge_adapter import JudgeError, _parse_scores, main

# --- fixtures / helpers ------------------------------------------------------

DIMS = [
    {"name": "accuracy", "description": "factually right", "weight": 0.5},
    {"name": "coverage", "description": "complete", "weight": 0.5},
]


def _ok_response(scores: dict, notes: str = "ok", content: str | None = None) -> httpx.Response:
    """A 200 /chat/completions response whose message content is a scores JSON."""
    if content is None:
        content = json.dumps({"scores": scores, "notes": notes})
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": content}}]},
        request=httpx.Request("POST", "http://judge.test/v1/chat/completions"),
    )


def _error_response(status: int, body: str = "upstream error") -> httpx.Response:
    return httpx.Response(
        status, text=body,
        request=httpx.Request("POST", "http://judge.test/v1/chat/completions"),
    )


class ScriptedPost:
    """Stands in for httpx.post: pops one scripted response (or raises) per call."""

    def __init__(self, script: list):
        self.script = list(script)
        self.calls: list[dict] = []

    def __call__(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "payload": json, "timeout": timeout})
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def _make_run_dir(tmp_path: Path, *, baseline: bool = False, items: list | None = None) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    if items is None:
        item = {
            "item_id": "cand-a::tk1::s1",
            "task_id": "tk1",
            "task_description": "Summarize the input.",
            "quality_dimensions": DIMS,
            "scenario_id": "s1",
            "scenario_input": "some input",
            "candidate": "cand-a",
            "candidate_output": "the candidate answer",
            "latency_ms": 100,
            "error": None,
        }
        if baseline:
            item["baseline_output"] = "the gold answer"
        items = [item]
    (run_dir / "judge-batch.json").write_text(json.dumps({"items": items}))
    return run_dir


@pytest.fixture()
def no_sleep(monkeypatch):
    """Capture backoff sleeps instead of actually sleeping."""
    sleeps: list[float] = []
    monkeypatch.setattr(judge_adapter.time, "sleep", sleeps.append)
    return sleeps


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def _run(run_dir: Path, *extra: str) -> int:
    return main([str(run_dir), "--model", "judge-model", "--base-url", "http://judge.test/v1", *extra])


def _read_scores(run_dir: Path) -> list[dict]:
    return json.loads((run_dir / "judge-scores.json").read_text())


# --- retry behavior (finding A) ----------------------------------------------


def test_transient_failure_then_success_is_retried(tmp_path, monkeypatch, no_sleep):
    """One 500 then a good reply: the item scores normally and exit is 0."""
    post = ScriptedPost([
        _error_response(500),
        _ok_response({"accuracy": 90, "coverage": 80}),
    ])
    monkeypatch.setattr(judge_adapter.httpx, "post", post)
    run_dir = _make_run_dir(tmp_path)
    assert _run(run_dir) == 0
    assert len(post.calls) == 2
    assert no_sleep == [1.0]  # one backoff before the second attempt
    [entry] = _read_scores(run_dir)
    assert "judge_error" not in entry
    assert entry["candidate_scores"]["mean_quality_score"] == 85.0


@pytest.mark.parametrize("step", [
    _error_response(429, "rate limited"),
    httpx.ReadTimeout("timed out", request=httpx.Request("POST", "http://judge.test")),
])
def test_429_and_timeout_are_retryable(tmp_path, monkeypatch, no_sleep, step):
    post = ScriptedPost([step, _ok_response({"accuracy": 70, "coverage": 70})])
    monkeypatch.setattr(judge_adapter.httpx, "post", post)
    run_dir = _make_run_dir(tmp_path)
    assert _run(run_dir) == 0
    assert len(post.calls) == 2


def test_final_failure_tallies_tags_and_exits_nonzero(tmp_path, monkeypatch, no_sleep, capsys):
    """All attempts fail: entry tagged judge_error, FAILED tally printed, exit 1."""
    post = ScriptedPost([_error_response(503)] * 3)
    monkeypatch.setattr(judge_adapter.httpx, "post", post)
    run_dir = _make_run_dir(tmp_path)
    assert _run(run_dir) == 1
    assert len(post.calls) == 3  # 3 attempts total
    assert no_sleep == [1.0, 2.0]  # exponential backoff between attempts
    [entry] = _read_scores(run_dir)
    assert entry["judge_error"] is True
    assert entry["candidate_scores"]["mean_quality_score"] == 0.0
    assert "judge error" in entry["candidate_scores"]["notes"]
    err = capsys.readouterr().err
    assert "1/1 items FAILED" in err
    assert "do NOT finalize" in err


def test_allow_errors_flag_exits_zero_on_failures(tmp_path, monkeypatch, no_sleep):
    post = ScriptedPost([_error_response(503)] * 3)
    monkeypatch.setattr(judge_adapter.httpx, "post", post)
    run_dir = _make_run_dir(tmp_path)
    assert _run(run_dir, "--allow-errors") == 0
    [entry] = _read_scores(run_dir)
    assert entry["judge_error"] is True


def test_non_retryable_status_fails_without_retry(tmp_path, monkeypatch, no_sleep):
    """A 401 is not transient: no retries, item recorded as judge error."""
    post = ScriptedPost([_error_response(401, "bad key")])
    monkeypatch.setattr(judge_adapter.httpx, "post", post)
    run_dir = _make_run_dir(tmp_path)
    assert _run(run_dir) == 1
    assert len(post.calls) == 1


# --- baseline judge-error handling (finding B) --------------------------------


def test_baseline_failure_omits_baseline_scores_and_warns(tmp_path, monkeypatch, no_sleep, capsys):
    """Candidate scores fine; baseline fails all retries → no fabricated 100."""
    post = ScriptedPost([
        _ok_response({"accuracy": 90, "coverage": 80}),  # candidate
        _error_response(500), _error_response(500), _error_response(500),  # baseline
    ])
    monkeypatch.setattr(judge_adapter.httpx, "post", post)
    run_dir = _make_run_dir(tmp_path, baseline=True)
    assert _run(run_dir) == 0  # baseline failure degrades, doesn't fail the run
    [entry] = _read_scores(run_dir)
    assert "baseline_scores" not in entry
    assert "judge_error" not in entry
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "baseline_scores omitted" in err
    assert "100" not in json.dumps(entry)  # nothing fabricated anywhere


def test_baseline_success_is_recorded_and_cached(tmp_path, monkeypatch, no_sleep):
    """Two items in the same cell: baseline judged once, attached to both."""
    base = {
        "task_id": "tk1", "task_description": "t", "quality_dimensions": DIMS,
        "scenario_id": "s1", "scenario_input": "in", "latency_ms": 1, "error": None,
        "baseline_output": "gold",
    }
    items = [
        {**base, "item_id": "cand-a::tk1::s1", "candidate": "cand-a", "candidate_output": "a"},
        {**base, "item_id": "cand-b::tk1::s1", "candidate": "cand-b", "candidate_output": "b"},
    ]
    post = ScriptedPost([
        _ok_response({"accuracy": 80, "coverage": 80}),  # cand-a
        _ok_response({"accuracy": 95, "coverage": 95}),  # baseline (once)
        _ok_response({"accuracy": 60, "coverage": 60}),  # cand-b
    ])
    monkeypatch.setattr(judge_adapter.httpx, "post", post)
    run_dir = _make_run_dir(tmp_path, items=items)
    assert _run(run_dir) == 0
    assert len(post.calls) == 3  # baseline NOT re-judged for the second item
    scores = _read_scores(run_dir)
    assert all(e["baseline_scores"]["mean_quality_score"] == 95.0 for e in scores)


# --- temperature handling (finding C) -----------------------------------------


def test_temperature_400_retried_without_key(tmp_path, monkeypatch, no_sleep):
    """A 400 mentioning 'temperature' drops the key and retries immediately."""
    post = ScriptedPost([
        _error_response(400, '{"error": {"message": "Unsupported parameter: temperature"}}'),
        _ok_response({"accuracy": 90, "coverage": 90}),
    ])
    monkeypatch.setattr(judge_adapter.httpx, "post", post)
    run_dir = _make_run_dir(tmp_path)
    assert _run(run_dir) == 0
    assert "temperature" in post.calls[0]["payload"]
    assert "temperature" not in post.calls[1]["payload"]
    assert no_sleep == []  # immediate retry, no backoff burned


def test_temperature_none_omits_key(tmp_path, monkeypatch, no_sleep):
    post = ScriptedPost([_ok_response({"accuracy": 90, "coverage": 90})])
    monkeypatch.setattr(judge_adapter.httpx, "post", post)
    run_dir = _make_run_dir(tmp_path)
    assert _run(run_dir, "--temperature", "none") == 0
    assert "temperature" not in post.calls[0]["payload"]


def test_temperature_default_and_custom_value_sent(tmp_path, monkeypatch, no_sleep):
    post = ScriptedPost([_ok_response({"accuracy": 90, "coverage": 90})])
    monkeypatch.setattr(judge_adapter.httpx, "post", post)
    run_dir = _make_run_dir(tmp_path)
    assert _run(run_dir, "--temperature", "0.7") == 0
    assert post.calls[0]["payload"]["temperature"] == 0.7


def test_non_temperature_400_is_not_retried(tmp_path, monkeypatch, no_sleep):
    """A 400 about something else is a hard error: no temperature fallback."""
    post = ScriptedPost([_error_response(400, '{"error": {"message": "model not found"}}')])
    monkeypatch.setattr(judge_adapter.httpx, "post", post)
    run_dir = _make_run_dir(tmp_path)
    assert _run(run_dir) == 1
    assert len(post.calls) == 1


# --- reply parsing (finding D) -------------------------------------------------


def test_fenced_json_with_trailing_prose_parses(tmp_path, monkeypatch, no_sleep):
    """The pre-fix failure mode: fence + chatty prose containing a stray brace."""
    content = (
        "Sure! Here are my scores:\n"
        "```json\n"
        '{"scores": {"accuracy": 90, "coverage": 80}, "notes": "solid"}\n'
        "```\n"
        "Let me know if you need more {detail} on any dimension."
    )
    post = ScriptedPost([_ok_response({}, content=content)])
    monkeypatch.setattr(judge_adapter.httpx, "post", post)
    run_dir = _make_run_dir(tmp_path)
    assert _run(run_dir) == 0
    [entry] = _read_scores(run_dir)
    assert entry["candidate_scores"]["scores"] == {"accuracy": 90, "coverage": 80}
    assert entry["candidate_scores"]["mean_quality_score"] == 85.0


def test_unfenced_json_after_prose_brace_parses():
    """Brace-balanced scan skips non-JSON brace groups before the real object."""
    text = 'Notes {not json} then {"scores": {"accuracy": 70, "coverage": 50}, "notes": "x"}'
    parsed = _parse_scores(text, DIMS)
    assert parsed["scores"] == {"accuracy": 70, "coverage": 50}


def test_braces_inside_json_strings_do_not_break_scan():
    text = '{"scores": {"accuracy": 88, "coverage": 88}, "notes": "uses { and } a lot"}'
    parsed = _parse_scores(text, DIMS)
    assert parsed["scores"] == {"accuracy": 88, "coverage": 88}


def test_no_json_in_reply_raises_judge_error():
    with pytest.raises(JudgeError):
        _parse_scores("I cannot score this, sorry.", DIMS)


def test_more_than_half_dimensions_missing_raises_judge_error():
    with pytest.raises(JudgeError, match="missing 2/2"):
        _parse_scores('{"scores": {"unrelated": 50}, "notes": ""}', DIMS)


def test_half_or_fewer_missing_dimensions_scores_zero():
    """Exactly half missing is tolerated (clamped to 0), not a judge error."""
    parsed = _parse_scores('{"scores": {"accuracy": 80}, "notes": ""}', DIMS)
    assert parsed["scores"] == {"accuracy": 80, "coverage": 0}


def test_unparseable_reply_is_retried_then_succeeds(tmp_path, monkeypatch, no_sleep):
    """Extraction failure goes through the retry path, not recorded as a valid 0."""
    post = ScriptedPost([
        _ok_response({}, content="no json here at all"),
        _ok_response({"accuracy": 75, "coverage": 75}),
    ])
    monkeypatch.setattr(judge_adapter.httpx, "post", post)
    run_dir = _make_run_dir(tmp_path)
    assert _run(run_dir) == 0
    assert len(post.calls) == 2
    [entry] = _read_scores(run_dir)
    assert entry["candidate_scores"]["mean_quality_score"] == 75.0


# --- batch shape detection (finding E) ------------------------------------------


def test_audit_shaped_batch_exits_2_with_message(tmp_path, monkeypatch, capsys):
    """Audit batches (sample_id/slot items) are rejected up front, not KeyError'd."""
    post = ScriptedPost([])
    monkeypatch.setattr(judge_adapter.httpx, "post", post)
    items = [{
        "sample_id": "abc123", "slot": "entity_extraction",
        "candidate_model": "m", "input_text": "in", "output_text": "out", "latency_ms": 5,
    }]
    run_dir = _make_run_dir(tmp_path, items=items)
    assert _run(run_dir) == 2
    assert post.calls == []  # rejected before any judge call
    err = capsys.readouterr().err
    assert "audit batch" in err


def test_missing_batch_file_exits_2(tmp_path, capsys):
    run_dir = tmp_path / "empty-run"
    run_dir.mkdir()
    assert _run(run_dir) == 2
    assert "prepare-batch" in capsys.readouterr().err


# --- prompt hardening (finding F) ------------------------------------------------


def test_candidate_output_is_delimited_as_data(tmp_path, monkeypatch, no_sleep):
    """Candidate and gold outputs travel inside data delimiters with an ignore-instructions notice."""
    post = ScriptedPost([
        _ok_response({"accuracy": 50, "coverage": 50}),  # candidate
        _ok_response({"accuracy": 95, "coverage": 95}),  # baseline
    ])
    monkeypatch.setattr(judge_adapter.httpx, "post", post)
    run_dir = _make_run_dir(tmp_path, baseline=True)
    assert _run(run_dir) == 0
    cand_prompt = post.calls[0]["payload"]["messages"][1]["content"]
    assert "<candidate_output>\nthe candidate answer\n</candidate_output>" in cand_prompt
    assert "<gold_reference>\nthe gold answer\n</gold_reference>" in cand_prompt
    assert "NOT instructions" in cand_prompt
    # Baseline is scored as the candidate of its own prompt, without a gold section.
    base_prompt = post.calls[1]["payload"]["messages"][1]["content"]
    assert "<candidate_output>\nthe gold answer\n</candidate_output>" in base_prompt
    assert "GOLD (reference answer" not in base_prompt
