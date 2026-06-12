"""Tests for the score-integrity gate (M2 mitigation, RCA 2026-05-28).

Guards that the stamped-score signature (distinct outputs, identical scores)
is detected, while genuine judging and deterministic tasks pass.
"""

from __future__ import annotations

from orchestrator.evaluation.integrity import check_score_integrity


def _batch(item_id, candidate, task, scn, output):
    return {"item_id": item_id, "candidate": candidate, "task_id": task,
            "scenario_id": scn, "candidate_output": output}


def _score(item_id, q):
    return {"item_id": item_id, "candidate_scores": {"mean_quality_score": q}}


def _score_v1(item_id, q):
    """v1 flat shape — mean_quality_score at the top level, no nesting."""
    return {"item_id": item_id, "scores": {"quality": q}, "mean_quality_score": q}


def _make_run(cells):
    """cells: list of (candidate, task, scn, [(output, score), ...])."""
    batch, scores = [], []
    for i, (cand, task, scn, pairs) in enumerate(cells):
        for j, (out, q) in enumerate(pairs):
            iid = f"{cand}::{task}::{scn}::sample-{j}-{i}"
            batch.append(_batch(iid, cand, task, scn, out))
            scores.append(_score(iid, q))
    return batch, scores


def test_stamped_run_fails():
    """Distinct outputs, identical scores across many cells → FAIL."""
    cells = []
    for k in range(10):
        # 3 distinct outputs, all scored 74.5 (the stamping signature)
        cells.append(("ollama/qwen3:8b", "summary_synthesis", f"scn-{k}",
                      [(f"out-A-{k}", 74.5), (f"out-B-{k}", 74.5), (f"out-C-{k}", 74.5)]))
    batch, scores = _make_run(cells)
    r = check_score_integrity(batch, scores)
    assert not r.passed
    assert r.multi_output_cells == 10
    assert r.zero_variance_cells == 10
    assert r.zero_variance_fraction == 1.0
    assert len(r.suspicious_cells) == 10


def test_genuine_judging_passes():
    """Distinct outputs with varied scores → PASS (low zero-variance fraction)."""
    cells = []
    for k in range(10):
        cells.append(("ollama/qwen3:8b", "summary_synthesis", f"scn-{k}",
                      [(f"out-A-{k}", 80.0 + k), (f"out-B-{k}", 85.0 + k), (f"out-C-{k}", 90.0)]))
    batch, scores = _make_run(cells)
    r = check_score_integrity(batch, scores)
    assert r.passed
    assert r.zero_variance_fraction < 0.7


def test_deterministic_task_not_flagged():
    """Identical outputs (deterministic) → single-output cells → excluded, PASS."""
    cells = []
    for k in range(10):
        # All 3 samples identical output AND identical score — legit determinism
        cells.append(("ollama/gemma4:e4b", "entity_extraction", f"scn-{k}",
                      [("[\"Acme Corp\"]", 100.0), ("[\"Acme Corp\"]", 100.0), ("[\"Acme Corp\"]", 100.0)]))
    batch, scores = _make_run(cells)
    r = check_score_integrity(batch, scores)
    assert r.passed
    assert r.multi_output_cells == 0  # all single-output → excluded


def test_empty_output_scored_as_real_fails():
    """An empty/whitespace output scored as real quality fails the gate —
    even as a single-output cell (the thinking-mode blind spot)."""
    cells = [
        ("ollama/qwen5:8b", "framework", "scn-0", [("", 88.0), ("", 88.0), ("", 88.0)]),
        ("ollama/qwen5:8b", "framework", "scn-1", [("   ", 90.0)]),
    ]
    batch, scores = _make_run(cells)
    r = check_score_integrity(batch, scores)
    assert not r.passed
    assert len(r.empty_scored_cells) == 2


def test_empty_output_scored_zero_passes():
    """Empty output correctly scored 0 (judge did its job) does NOT fail."""
    cells = [("ollama/qwen5:8b", "framework", "scn-0", [("", 0.0), ("", 0.0)])]
    batch, scores = _make_run(cells)
    r = check_score_integrity(batch, scores)
    assert r.passed
    assert r.empty_scored_cells == []


def test_below_min_signal_passes():
    """Too few multi-output cells → not enough signal → PASS even if uniform."""
    cells = [("ollama/qwen3:8b", "summary_synthesis", "scn-0",
              [("out-A", 74.5), ("out-B", 74.5)])]  # only 1 multi-output cell
    batch, scores = _make_run(cells)
    r = check_score_integrity(batch, scores, min_multi_output_cells=5)
    assert r.passed
    assert "too little signal" in r.note.lower()


def test_v1_flat_score_shape_handled():
    """Regression: v1-flat judge scores (top-level mean_quality_score) must be
    read, not misread as None (which caused false zero-variance positives or a
    TypeError in max())."""
    cells = []
    for k in range(10):
        cells.append(("m", "t", f"scn-{k}",
                      [(f"out-A-{k}", 80.0 + k), (f"out-B-{k}", 90.0)]))
    batch, scores = [], []
    for i, (cand, task, scn, pairs) in enumerate(cells):
        for j, (out, q) in enumerate(pairs):
            iid = f"{cand}::{task}::{scn}::sample-{j}-{i}"
            batch.append(_batch(iid, cand, task, scn, out))
            scores.append(_score_v1(iid, q))
    r = check_score_integrity(batch, scores)
    # Varied scores on distinct outputs → genuine judging → PASS, no crash.
    assert r.passed
    assert r.multi_output_cells == 10
    assert r.zero_variance_cells == 0


def test_v1_flat_stamped_run_still_detected():
    """Stamping detection must work on v1-shape files too."""
    batch, scores = [], []
    for k in range(10):
        for j, out in enumerate([f"out-A-{k}", f"out-B-{k}"]):
            iid = f"m::t::scn-{k}::sample-{j}"
            batch.append(_batch(iid, "m", "t", f"scn-{k}", out))
            scores.append(_score_v1(iid, 74.5))  # identical score everywhere
    r = check_score_integrity(batch, scores)
    assert not r.passed
    assert r.zero_variance_fraction == 1.0


def test_empty_output_detected_per_item_in_mixed_cell():
    """Regression: an empty output scored non-zero must be flagged even when
    the cell ALSO contains real outputs (previously only all-empty cells were
    checked)."""
    cells = [
        ("m", "t", "scn-0", [("real output", 85.0), ("", 90.0)]),
    ]
    batch, scores = _make_run(cells)
    r = check_score_integrity(batch, scores)
    assert not r.passed
    assert len(r.empty_scored_cells) == 1
    assert r.empty_scored_cells[0].score == 90.0


def test_errored_sample_does_not_mask_stamping():
    """Regression: an errored (empty, scored-0) sample must not inject fake
    score variance that hides identical scores on the cell's real outputs."""
    cells = []
    for k in range(10):
        # Two DISTINCT real outputs with identical scores (stamping) plus one
        # errored sample correctly scored 0.
        cells.append(("m", "t", f"scn-{k}",
                      [(f"out-A-{k}", 74.5), (f"out-B-{k}", 74.5), ("", 0.0)]))
    batch, scores = _make_run(cells)
    r = check_score_integrity(batch, scores)
    assert r.multi_output_cells == 10
    assert r.zero_variance_cells == 10
    assert not r.passed


def test_mixed_run_at_threshold():
    """A run that's partly stamped but below the fail fraction passes; above fails."""
    # 10 cells: 6 zero-variance, 4 varied → 60% → under default 0.7 → PASS
    cells = []
    for k in range(6):
        cells.append(("m", "t", f"z-{k}", [(f"a{k}", 70.0), (f"b{k}", 70.0)]))
    for k in range(4):
        cells.append(("m", "t", f"v-{k}", [(f"a{k}", 70.0), (f"b{k}", 88.0)]))
    batch, scores = _make_run(cells)
    r = check_score_integrity(batch, scores)
    assert r.zero_variance_fraction == 0.6
    assert r.passed
    # Tighten the threshold below 0.6 → now fails
    r2 = check_score_integrity(batch, scores, zero_variance_fail_frac=0.5)
    assert not r2.passed
