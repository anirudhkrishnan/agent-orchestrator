"""Tests for the three self-improving loops + the hard-rule integrity guard."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from orchestrator.improve import guard
from orchestrator.improve.guard import IntegrityGateError, require_integrity
from orchestrator.improve import loop_a_mistakes as A
from orchestrator.improve import loop_b_models as B
from orchestrator.improve import loop_c_research as C
from orchestrator.telemetry import db as tdb


# ----------------------------- Hard rule (guard) ---------------------------

def _write_run(run_dir: Path, batch_items, scores):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "judge-batch.json").write_text(json.dumps({"items": batch_items}))
    (run_dir / "judge-scores.json").write_text(json.dumps(scores))


def _stamped(run_dir: Path):
    items, scores = [], []
    for k in range(10):
        for j, out in enumerate(["a", "b", "c"]):
            iid = f"ollama/m::summary_synthesis::scn-{k}::sample-{j}"
            items.append({"item_id": iid, "candidate": "ollama/m", "task_id": "summary_synthesis",
                          "scenario_id": f"scn-{k}", "candidate_output": f"{out}-{k}"})
            scores.append({"item_id": iid, "candidate_scores": {"mean_quality_score": 74.5}})
    _write_run(run_dir, items, scores)


def _honest(run_dir: Path):
    items, scores = [], []
    for k in range(10):
        for j, (out, q) in enumerate([("a", 80.0), ("b", 88.0), ("c", 92.0)]):
            iid = f"ollama/m::summary_synthesis::scn-{k}::sample-{j}"
            items.append({"item_id": iid, "candidate": "ollama/m", "task_id": "summary_synthesis",
                          "scenario_id": f"scn-{k}", "candidate_output": f"{out}-{k}"})
            scores.append({"item_id": iid, "candidate_scores": {"mean_quality_score": q}})
    _write_run(run_dir, items, scores)


def test_guard_blocks_stamped_data(tmp_path: Path):
    rd = tmp_path / "run"
    _stamped(rd)
    with pytest.raises(IntegrityGateError):
        require_integrity(rd)


def test_guard_passes_honest_data(tmp_path: Path):
    rd = tmp_path / "run"
    _honest(rd)
    require_integrity(rd)  # no raise


def test_guard_fails_closed_on_missing_files(tmp_path: Path):
    with pytest.raises(IntegrityGateError):
        require_integrity(tmp_path / "nonexistent")


def test_guard_fails_closed_on_empty_batch(tmp_path: Path):
    # A '{}' batch (or one without items) has nothing it can claim to have
    # judged — it must FAIL, not slide through with zero multi-output cells.
    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "judge-batch.json").write_text("{}")
    (rd / "judge-scores.json").write_text(json.dumps(
        [{"item_id": "x", "candidate_scores": {"mean_quality_score": 80.0}}]))
    with pytest.raises(IntegrityGateError, match="no\\s+items"):
        require_integrity(rd)


def test_guard_fails_closed_on_list_batch(tmp_path: Path):
    # A top-level '[]' batch file used to escape as AttributeError; it must be
    # a clean fail-closed gate failure instead.
    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "judge-batch.json").write_text("[]")
    (rd / "judge-scores.json").write_text(json.dumps(
        [{"item_id": "x", "candidate_scores": {"mean_quality_score": 80.0}}]))
    with pytest.raises(IntegrityGateError, match="no\\s+items"):
        require_integrity(rd)


@pytest.mark.parametrize("scores_text", ["[]", "{}"])
def test_guard_fails_closed_on_empty_scores(tmp_path: Path, scores_text: str):
    rd = tmp_path / "run"
    _honest(rd)
    (rd / "judge-scores.json").write_text(scores_text)
    with pytest.raises(IntegrityGateError, match="no\\s+scores"):
        require_integrity(rd)


def test_guard_fails_closed_on_disjoint_item_ids(tmp_path: Path):
    # Scores that match NONE of the batch item_ids belong to a different run —
    # zero overlap used to PASS (no cells formed at all).
    rd = tmp_path / "run"
    _honest(rd)
    scores = [{"item_id": f"other::task::scn-{k}::sample-0",
               "candidate_scores": {"mean_quality_score": 80.0}} for k in range(5)]
    (rd / "judge-scores.json").write_text(json.dumps(scores))
    with pytest.raises(IntegrityGateError, match="match"):
        require_integrity(rd)


def test_guard_fails_closed_on_corrupt_json(tmp_path: Path):
    # Corrupt run files must fail the gate with a pointer at the file, not
    # escape as a raw json traceback.
    rd = tmp_path / "run"
    _honest(rd)
    (rd / "judge-scores.json").write_text("{not json")
    with pytest.raises(IntegrityGateError, match="judge-scores.json"):
        require_integrity(rd)


def test_guard_override_warns_loudly(tmp_path: Path, capsys):
    # allow_override must not be a silent pass-through: the warning has to say
    # the gate FAILED and was overridden.
    rd = tmp_path / "run"
    _stamped(rd)
    require_integrity(rd, allow_override=True)  # no raise
    err = capsys.readouterr().err
    assert "FAILED" in err
    assert "OVERRIDDEN" in err


def test_guard_degenerate_state_is_not_overridable(tmp_path: Path):
    # There is no legitimate "uniform" reading of NO data — empty files must
    # fail even with allow_override=True.
    rd = tmp_path / "run"
    _honest(rd)
    (rd / "judge-scores.json").write_text("[]")
    with pytest.raises(IntegrityGateError):
        require_integrity(rd, allow_override=True)


def test_gated_decorator_blocks_commit(tmp_path: Path):
    rd = tmp_path / "run"
    _stamped(rd)
    committed = []

    @guard.gated(lambda run_dir: run_dir)
    def commit(run_dir):
        committed.append(run_dir)

    with pytest.raises(IntegrityGateError):
        commit(rd)
    assert committed == []  # never ran the body


# ----------------------------- Loop A --------------------------------------

@pytest.fixture
def tmp_db(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_DB_PATH", str(tmp_path / "t.sqlite"))
    tdb.init_db()
    return tmp_path


def test_loop_a_harvests_negative_feedback(tmp_db):
    now = datetime(2026, 5, 28, tzinfo=timezone.utc)
    tdb.log_decision(session_id="s1", app_name="test-app", message_excerpt="summary output looks wrong",
                     classified_slot="summary_synthesis", selected_model="ollama/qwen3:8b",
                     fallback_used=False, latency_ms=100, cost_usd=0.0,
                     user_feedback="thumbs_down", timestamp=now.isoformat())
    tdb.log_decision(session_id="s2", app_name="test-app", message_excerpt="good one",
                     classified_slot="summary_synthesis", selected_model="ollama/qwen3:8b",
                     fallback_used=False, latency_ms=100, cost_usd=0.0,
                     user_feedback="good", timestamp=now.isoformat())
    rep = A.harvest_failures("test-app", now=now.replace(hour=23))
    assert len(rep.failures) == 1
    assert rep.failures[0].reason == "negative_feedback"
    assert rep.slots_to_rebake == ["summary_synthesis"]


def test_loop_a_stages_scenarios_with_provenance(tmp_db, tmp_path: Path):
    now = datetime(2026, 5, 28, tzinfo=timezone.utc)
    tdb.log_decision(session_id="s1", app_name="test-app", message_excerpt="harvest me",
                     classified_slot="relevance_triage", selected_model="ollama/qwen3:8b",
                     fallback_used=False, latency_ms=100, cost_usd=0.0,
                     user_feedback="bad", timestamp=now.isoformat())
    rep = A.harvest_failures("test-app", now=now.replace(hour=23))
    staging = tmp_path / "staging.json"
    A.stage_scenarios(rep, staging)
    data = json.loads(staging.read_text())["candidate_scenarios"]
    assert len(data) == 1
    assert data[0]["status"] == "needs_review"
    assert data[0]["provenance"]["source"] == "routing_decisions"
    # Idempotent: staging again doesn't duplicate.
    A.stage_scenarios(rep, staging)
    assert len(json.loads(staging.read_text())["candidate_scenarios"]) == 1


def test_loop_a_rebake_queue(tmp_db, tmp_path: Path):
    now = datetime(2026, 5, 28, tzinfo=timezone.utc)
    tdb.log_decision(session_id="s1", app_name="test-app", message_excerpt="x",
                     classified_slot="tone_classification", selected_model="ollama/gemma4:e4b",
                     fallback_used=False, latency_ms=1, cost_usd=0.0,
                     user_feedback="wrong", timestamp=now.isoformat())
    rep = A.harvest_failures("test-app", now=now.replace(hour=23))
    q = tmp_path / "queue.json"
    A.write_rebake_queue(rep, q)
    queue = json.loads(q.read_text())["slots"]
    assert "tone_classification" in queue
    assert queue["tone_classification"]["count"] == 1


def test_loop_a_rebake_queue_dedups_across_runs(tmp_db, tmp_path: Path):
    # A daily harvest re-sees the same rows for the whole lookback window; the
    # queue must count each root failure ONCE, not once per run.
    now = datetime(2026, 5, 28, tzinfo=timezone.utc)
    tdb.log_decision(session_id="s1", app_name="test-app", message_excerpt="x",
                     classified_slot="tone_classification", selected_model="ollama/gemma4:e4b",
                     fallback_used=False, latency_ms=1, cost_usd=0.0,
                     user_feedback="wrong", timestamp=now.isoformat())
    q = tmp_path / "queue.json"
    for _ in range(3):  # three "daily" runs over the same window
        rep = A.harvest_failures("test-app", now=now.replace(hour=23))
        A.write_rebake_queue(rep, q)
    queue = json.loads(q.read_text())["slots"]
    assert queue["tone_classification"]["count"] == 1
    # A genuinely NEW failure still increments the count.
    tdb.log_decision(session_id="s2", app_name="test-app", message_excerpt="y",
                     classified_slot="tone_classification", selected_model="ollama/gemma4:e4b",
                     fallback_used=False, latency_ms=1, cost_usd=0.0,
                     user_feedback="bad", timestamp=now.isoformat())
    rep = A.harvest_failures("test-app", now=now.replace(hour=23))
    A.write_rebake_queue(rep, q)
    queue = json.loads(q.read_text())["slots"]
    assert queue["tone_classification"]["count"] == 2


def test_loop_a_harvests_audit_scores_source(tmp_db):
    # Second failure source: a sampled call the audit judge scored below the
    # re-bake line is harvested with a stable sample-backed failure_id.
    now = datetime(2026, 5, 28, tzinfo=timezone.utc)
    tdb.log_sample(sample_id="smp-1", app_name="test-app", slot="summary_synthesis",
                   candidate_model="ollama/qwen3:8b", input_text="summarize this",
                   output_text="meh", latency_ms=50, routed_at=now.isoformat())
    rep = A.harvest_failures("test-app", now=now.replace(hour=23),
                             sample_scores={"smp-1": 42.0})
    assert len(rep.failures) == 1
    f = rep.failures[0]
    assert f.reason == "below_rebake_threshold"
    assert f.source == "routed_call_samples"
    assert f.failure_id == "routed_call_samples:smp-1"
    # At or above the line → not a failure.
    rep = A.harvest_failures("test-app", now=now.replace(hour=23),
                             sample_scores={"smp-1": 80.0})
    assert rep.failures == []


# ----------------------------- Loop B --------------------------------------

def test_loop_b_detects_new_models(tmp_path: Path):
    routing = tmp_path / "routing.json"
    routing.write_text(json.dumps({
        "_README": "x",
        "slot1": {"model": "ollama/qwen3:8b", "fallback_model": "ollama/gemma4:e4b",
                  "judge_model": "claude-opus-4-7", "last_baked_at": "2026-01-01"},
    }))
    rep = B.detect_new_models(
        routing,
        available_models=["ollama/qwen3:8b", "ollama/gemma4:e4b", "ollama/qwen5:14b",
                          "anthropic/claude-opus-4-8"],
    )
    assert "ollama/qwen5:14b" in rep.new_models
    assert "anthropic/claude-opus-4-8" in rep.new_models
    assert "ollama/qwen3:8b" not in rep.new_models  # already baked


def test_loop_b_proposal_none_when_nothing_new(tmp_path: Path):
    routing = tmp_path / "routing.json"
    routing.write_text(json.dumps({"slot1": {"model": "ollama/qwen3:8b", "last_baked_at": "x"}}))
    rep = B.detect_new_models(routing, available_models=["ollama/qwen3:8b"])
    assert rep.new_models == []
    assert B.propose_rebake(rep, tmp_path / "p.json") is None


def test_loop_b_writes_proposal(tmp_path: Path):
    rep = B.NewModelReport(available=["ollama/qwen5:14b"], already_baked=["ollama/qwen3:8b"],
                           new_models=["ollama/qwen5:14b"])
    p = B.propose_rebake(rep, tmp_path / "p.json")
    assert p is not None
    data = json.loads(p.read_text())
    assert data["status"] == "needs_human_confirm"
    assert "ollama/qwen5:14b" in data["suggested_command"]


def test_loop_b_detects_judge_models(tmp_path: Path):
    routing = tmp_path / "routing.json"
    routing.write_text(json.dumps({
        "slot1": {"model": "ollama/qwen3:8b", "judge_model": "ollama/judgey:70b",
                  "last_baked_at": "2026-01-01"},
    }))
    rep = B.detect_new_models(routing, available_models=["ollama/qwen5:14b"])
    assert rep.judge_models == ["ollama/judgey:70b"]


def test_loop_b_command_is_runnable(tmp_path: Path):
    # The eval runner is ollama-only and the judge sits outside the candidate
    # pool — the suggested command must carry neither non-ollama ids nor the
    # judge; non-ollama additions belong on a comment line.
    rep = B.NewModelReport(
        available=["ollama/qwen5:14b", "anthropic/claude-opus-4-8"],
        already_baked=["ollama/qwen3:8b", "ollama/judgey:70b", "claude-opus-4-7"],
        new_models=["ollama/qwen5:14b", "anthropic/claude-opus-4-8"],
        judge_models=["ollama/judgey:70b", "claude-opus-4-7"],
    )
    p = B.propose_rebake(rep, tmp_path / "p.json")
    cmd = json.loads(p.read_text())["suggested_command"]
    run_line, comment_lines = cmd.split("\n")[0], cmd.split("\n")[1:]
    assert "ollama/qwen5:14b" in run_line
    assert "ollama/qwen3:8b" in run_line
    assert "anthropic/" not in run_line
    assert "ollama/judgey:70b" not in run_line
    assert "claude-opus-4-7 " not in run_line  # judge never a candidate
    assert comment_lines and comment_lines[0].startswith("#")
    assert "anthropic/claude-opus-4-8" in comment_lines[0]


def test_loop_b_command_with_no_ollama_candidates(tmp_path: Path):
    # All-new-non-ollama: nothing runnable, so the "command" is a comment that
    # says so instead of an invocation that would raise ValueError.
    rep = B.NewModelReport(available=["anthropic/claude-opus-4-8"], already_baked=[],
                           new_models=["anthropic/claude-opus-4-8"])
    p = B.propose_rebake(rep, tmp_path / "p.json")
    cmd = json.loads(p.read_text())["suggested_command"]
    assert cmd.startswith("#")
    assert "anthropic/claude-opus-4-8" in cmd


# ----------------------------- Loop C --------------------------------------

def test_loop_c_advisory_dedups_seen(tmp_path: Path):
    seen = tmp_path / "seen.json"
    findings = [
        {"topic": "judging", "claim": "Pairwise judging cuts variance 30%",
         "source": "arxiv/x", "source_date": "2026-05", "relevance": "r", "suggested_action": "a"},
    ]
    r1 = C.build_advisory(findings, seen_claims_path=seen)
    assert len(r1.findings) == 1
    C.mark_claims_seen(r1, seen)  # advisory delivered → now persist
    # Second pass with the same claim → deduped to zero.
    r2 = C.build_advisory(findings, seen_claims_path=seen)
    assert len(r2.findings) == 0


def test_loop_c_does_not_mark_seen_before_delivery(tmp_path: Path):
    # build_advisory must be read-only on the seen file: if delivery crashes
    # after building, the claim has to resurface on the next cycle instead of
    # being lost forever.
    seen = tmp_path / "seen.json"
    findings = [
        {"topic": "routing", "claim": "Cascade routing halves cost", "source": "s",
         "source_date": "2026-05", "relevance": "r", "suggested_action": "a"},
    ]
    r1 = C.build_advisory(findings, seen_claims_path=seen)
    assert len(r1.findings) == 1
    assert not seen.exists()  # nothing persisted by the build step
    # Simulated crash before mark_claims_seen → next cycle still surfaces it.
    r2 = C.build_advisory(findings, seen_claims_path=seen)
    assert len(r2.findings) == 1
    # Within one batch, a duplicated claim is still surfaced only once.
    r3 = C.build_advisory(findings * 2, seen_claims_path=seen)
    assert len(r3.findings) == 1


def test_loop_c_advisory_is_advisory_only(tmp_path: Path):
    r = C.build_advisory([{"topic": "models", "claim": "new model", "source": "s",
                           "source_date": "2026", "relevance": "r", "suggested_action": "a"}])
    md = C.render_advisory(r)
    assert "ADVISORY ONLY" in r.note
    assert "not auto-applied" in md.lower()


def test_loop_c_radar_plan_shape():
    plan = C.radar_plan()
    assert plan["queries"]
    assert "finding_schema" in plan
