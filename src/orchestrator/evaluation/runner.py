"""Sequential candidate execution against task scenarios.

Why sequential, not parallel? — On a typical 16-18GB unified-memory machine,
two 9B local Ollama models cannot co-reside; running qwen3:8b and qwen3.5:9b
concurrently would force one to swap out mid-call. The runner enforces a strict
RAM-management contract: each candidate is loaded explicitly with a long
`keep_alive`, all its scenarios run, then it's UNLOADED before the next
candidate is loaded.

Failure isolation: a single (task, scenario, candidate) failure (timeout, model
crash, malformed response) is captured in `CandidateRun.error` and the run
continues. The judge will see the error string and can score appropriately
(usually zero on quality).

Output layout::

    {out_dir}/{ISO-timestamp}/
        {candidate-safe-id}/
            {task_id}_{scenario_id}.json   # one per cell
        manifest.json                       # candidates + tasks + timing

The judge batch is built by a separate module (`batch.py`) walking this tree.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

from .tasks import Scenario, TaskSpec


class CandidateSample(BaseModel):
    """One sample within an N-sample cell.

    Per the framework's N=5 sampling rule (codified 2026-05-26), every
    (candidate, task, scenario) cell runs N times to capture output
    variance. Each sample carries its own output + latency + error so
    downstream stages can dedupe by output_text, score uniquely, and
    aggregate to median/std-dev for the routing decision.
    """

    output_text: str = Field(default="", description="Empty string when error is set.")
    latency_ms: int = Field(..., ge=0)
    error: str | None = Field(default=None)


class CandidateRun(BaseModel):
    """One (candidate, task, scenario) result. Serialized to JSON on disk.

    With N>1 sampling (the post-2026-05-26 standard), this cell contains
    N samples. The `output_text` / `latency_ms` top-level fields point at
    `samples[0]` for backward-compat with old single-sample readers; anything
    computing statistics must read the per-sample data in `samples`.
    """

    candidate: str = Field(..., description="Provider/model id, e.g. 'ollama/qwen3:8b'.")
    task_id: str
    scenario_id: str
    samples: list[CandidateSample] = Field(
        default_factory=list,
        description="N samples for this cell (N=5 default per the framework).",
    )
    # Backward-compat fields — populated from samples[0] when N>=1.
    output_text: str = Field(default="", description="DEPRECATED in favor of samples[0]. Kept for backward compat.")
    latency_ms: int = Field(default=0, ge=0)
    error: str | None = Field(default=None)
    completed_at: datetime


# --- Ollama transport ------------------------------------------------------


def _safe_id(candidate: str) -> str:
    """Convert a provider/model string into a filesystem-safe directory name.

    Rules: lowercase, replace any non-alnum with '-', collapse runs of '-',
    strip leading/trailing '-'. Keeps the result readable so a human scanning
    the run directory can identify which folder is which model.
    """
    s = candidate.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "unknown"


def _safe_filename_part(part: str) -> str:
    """Sanitize a task/scenario id for use in a cell filename.

    Unlike `_safe_id` (which aggressively normalizes model ids), this keeps
    typical ids byte-identical — alphanumerics, '-' and '_' pass through — so
    existing layouts like `entity_extraction_scn-01.json` are unchanged. Any
    filesystem-unsafe character ('/', ':', spaces, ...) becomes '-'. The
    mapping is a pure function of the id, so it is stable across runs; the
    real ids always live inside the JSON payload.
    """
    s = re.sub(r"[^A-Za-z0-9_-]+", "-", part)
    return s or "unknown"


def _strip_provider_prefix(candidate: str) -> str:
    """Strip the 'ollama/' (or other provider) prefix to get the bare model id."""
    return candidate.split("/", 1)[1] if "/" in candidate else candidate


def _needs_think_false(model_name: str) -> bool:
    """Models that emit hidden thinking tokens (consuming the entire response
    budget on reasoning rather than user-facing output) unless we explicitly
    pass `think=False` at the Ollama API level.

    Empirical findings (2026-05-26):
    - (a) qwen3.5 — built-in reasoning mode is on by default; must pass
      think=False to Ollama directly.
    - (b) qwen3 (the non-.5 variant) AND gemma4 ALSO exhibit this behavior on
      structured-output tasks (JSON schema enforcement seems to trigger more
      reasoning, exhausting num_predict). Empirically confirmed via a Phase 1
      eval run on 2026-05-26: 24/35 qwen3:8b outputs and 20/35 gemma4:e4b
      outputs were empty strings until think=False was applied.

    Treat any model in the Qwen3+ family, Gemma4+ family, DeepSeek-R1, and
    QwQ as thinking-by-default. Forward-compat for qwen4 / gemma5 etc.
    """
    name = model_name.lower()
    # Substring match against known reasoning/thinking-by-default families.
    # Broadened 2026-05-28 (stress-test HIGH) to cover more current reasoning
    # models. NOTE: this list will always lag new releases — the real backstop
    # is the empty-output→error guard in _run_one_cell, which catches ANY model
    # that emits only hidden think-tokens regardless of name. Keep both.
    KNOWN_THINKING_PREFIXES = (
        "qwen3", "qwen4", "qwen5",          # Qwen reasoning families (+fwd compat)
        "gemma4", "gemma5",                  # Gemma (+fwd compat)
        "deepseek-r1", "deepseek-r2",        # DeepSeek reasoning
        "qwq",                               # Qwen reasoning preview
        "magistral",                         # Mistral reasoning
        "phi4-reasoning", "phi5",            # Microsoft Phi reasoning
        "glm-4.5", "glm-4.6", "glm-5",       # Zhipu GLM reasoning
        "granite4-reasoning", "granite-4-reasoning",  # IBM Granite reasoning
        "exaone-deep",                       # LG EXAONE reasoning
        "llama4",                            # Llama 4 (reasoning variants)
    )
    return any(prefix in name for prefix in KNOWN_THINKING_PREFIXES)


async def _ollama_chat(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    keep_alive: str,
    timeout_s: float,
) -> tuple[str, int]:
    """Single chat call to Ollama's /api/chat. Returns (text, latency_ms).

    Raises httpx.HTTPError on transport failure; raises ValueError on a 2xx
    response with malformed JSON. The caller (run_evaluation) catches both and
    records the error in the CandidateRun.
    """
    payload: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "keep_alive": keep_alive,
        "stream": False,
        "options": {"num_predict": max_tokens},
    }
    if _needs_think_false(model):
        payload["think"] = False

    start = time.monotonic()
    resp = await client.post(f"{base_url}/api/chat", json=payload, timeout=timeout_s)
    latency_ms = int((time.monotonic() - start) * 1000)
    resp.raise_for_status()
    try:
        data = resp.json()
    except ValueError as e:
        raise ValueError(
            f"Ollama returned non-JSON 200 for model {model!r}: {resp.text[:200]}"
        ) from e
    message = data.get("message") or {}
    text = message.get("content", "") if isinstance(message, dict) else ""
    return text or "", latency_ms


async def _load_model(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    model: str,
    keep_alive_seconds: int,
    timeout_s: float = 600.0,
) -> None:
    """Warm a model into RAM via /api/generate with empty prompt + long keep_alive.

    Ollama treats an empty-prompt call as a load-only request. The keep_alive
    value tells Ollama to keep the weights resident for that many seconds
    after the next call completes. Long timeout because cold-loading a 9GB
    model from disk can take 30-60s on first launch.
    """
    payload = {
        "model": model,
        "prompt": "",
        "keep_alive": f"{keep_alive_seconds}s",
        "stream": False,
    }
    resp = await client.post(f"{base_url}/api/generate", json=payload, timeout=timeout_s)
    resp.raise_for_status()


async def _unload_model(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    model: str,
    timeout_s: float = 60.0,
) -> None:
    """Release a model from RAM via /api/generate with keep_alive=0.

    Best-effort: if the unload call fails (e.g. Ollama already evicted the
    model), we log and continue. The next candidate's _load_model still works
    regardless because Ollama load is idempotent.
    """
    payload = {
        "model": model,
        "prompt": "",
        "keep_alive": 0,
        "stream": False,
    }
    try:
        resp = await client.post(f"{base_url}/api/generate", json=payload, timeout=timeout_s)
        resp.raise_for_status()
    except Exception as e:  # noqa: BLE001 — best-effort, log + continue
        sys.stderr.write(f"[evaluation] WARN: failed to unload {model}: {e}\n")


# --- Output writers --------------------------------------------------------


def _write_run(out_dir: Path, run: CandidateRun) -> Path:
    """Persist a single CandidateRun JSON file. Returns the path written.

    Task/scenario ids are sanitized for the filename only (a '/' or ':' in an
    id must not create subdirectories or invalid paths); the JSON payload
    carries the original ids, and batch preparation reads those, never the
    filename.
    """
    cand_dir = out_dir / _safe_id(run.candidate)
    cand_dir.mkdir(parents=True, exist_ok=True)
    p = cand_dir / f"{_safe_filename_part(run.task_id)}_{_safe_filename_part(run.scenario_id)}.json"
    p.write_text(run.model_dump_json(indent=2) + "\n")
    return p


def _write_manifest(
    run_dir: Path,
    *,
    candidates: list[str],
    tasks: list[TaskSpec],
    started_at: datetime,
    completed_at: datetime,
    ollama_url: str,
    keep_alive_seconds: int,
) -> Path:
    """Write a top-level manifest describing the run's shape and timing.

    Lets `batch.prepare_judge_batch` and `report.generate_report` discover
    candidates + tasks without re-walking the directory tree.
    """
    manifest = {
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "ollama_url": ollama_url,
        "keep_alive_seconds": keep_alive_seconds,
        "candidates": candidates,
        "tasks": [
            {
                "id": t.id,
                "description": t.description,
                "quality_dimensions": [d.model_dump() for d in t.quality_dimensions],
                "scenarios": [s.model_dump() for s in t.scenarios],
                "system_prompt": t.system_prompt,
                "max_response_tokens": t.max_response_tokens,
            }
            for t in tasks
        ],
    }
    p = run_dir / "manifest.json"
    p.write_text(json.dumps(manifest, indent=2) + "\n")
    return p


# --- Public entry point ----------------------------------------------------


async def run_evaluation(
    candidates: list[str],
    tasks: list[TaskSpec],
    out_dir: Path,
    *,
    ollama_url: str = "http://localhost:11434",
    keep_alive_seconds: int = 1800,
    log_progress: bool = True,
    per_call_timeout_s: float = 600.0,
    samples_per_cell: int = 5,
) -> Path:
    """Run every (candidate × task × scenario) cell sequentially.

    Args:
        candidates: Ordered list of provider/model ids. Currently only the
            'ollama/' prefix is supported by this runner; other prefixes raise
            ValueError so future expansion is explicit.
        tasks: Loaded TaskSpec objects. Pass through `load_tasks_yaml`.
        out_dir: Root directory where the timestamped run folder is created.
        ollama_url: Base URL for the Ollama HTTP API.
        keep_alive_seconds: How long Ollama should keep each model resident
            after the last call (the runner unloads explicitly between
            candidates, so this just covers the in-candidate scenarios).
        log_progress: Print per-cell progress lines to stdout.
        per_call_timeout_s: Per-scenario HTTP timeout. 600s default because
            qwen3.5:9b on CPU can take 60-120s on a multi-document synthesis scenario.

    Returns:
        Path to the created `{out_dir}/{ISO-timestamp}/` directory.

    Raises:
        ValueError: if any candidate doesn't start with 'ollama/'.
    """
    for c in candidates:
        if not c.startswith("ollama/"):
            raise ValueError(
                f"Candidate {c!r} unsupported: only 'ollama/' prefix is handled by this "
                f"runner. Wire other providers via a separate transport before adding here."
            )

    started_at = datetime.now(timezone.utc)
    # ISO-with-Z, colons replaced for filesystem safety on case-insensitive FSes.
    run_id = started_at.strftime("%Y-%m-%dT%H-%M-%SZ")
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    total_cells = sum(len(t.scenarios) for t in tasks) * len(candidates)
    total_samples = total_cells * samples_per_cell
    cell_idx = 0
    t0 = time.monotonic()
    if log_progress:
        print(
            f"[evaluation] N={samples_per_cell} samples per cell → "
            f"{total_cells} cells × {samples_per_cell} = {total_samples} total samples",
            flush=True,
        )

    async with httpx.AsyncClient() as client:
        for cand in candidates:
            model_id = _strip_provider_prefix(cand)
            if log_progress:
                print(
                    f"[evaluation] loading {cand} (keep_alive={keep_alive_seconds}s)",
                    flush=True,
                )
            try:
                await _load_model(
                    client,
                    base_url=ollama_url,
                    model=model_id,
                    keep_alive_seconds=keep_alive_seconds,
                )
            except Exception as e:  # noqa: BLE001 — record + skip this candidate
                sys.stderr.write(
                    f"[evaluation] FAIL: could not load {cand}: {e}\n"
                    f"            skipping all scenarios for this candidate.\n"
                )
                # Write per-scenario error records so the judge sees uniform shape.
                for task in tasks:
                    for scn in task.scenarios:
                        cell_idx += 1
                        err_msg = f"model_load_failed: {e}"
                        err_samples = [
                            CandidateSample(output_text="", latency_ms=0, error=err_msg)
                            for _ in range(samples_per_cell)
                        ]
                        run = CandidateRun(
                            candidate=cand,
                            task_id=task.id,
                            scenario_id=scn.id,
                            samples=err_samples,
                            output_text="",
                            latency_ms=0,
                            error=err_msg,
                            completed_at=datetime.now(timezone.utc),
                        )
                        _write_run(run_dir, run)
                continue

            for task in tasks:
                for scn in task.scenarios:
                    cell_idx += 1
                    if log_progress:
                        elapsed = time.monotonic() - t0
                        print(
                            f"[evaluation] {cell_idx}/{total_cells} "
                            f"{cand} :: {task.id}/{scn.id} (elapsed {elapsed:.1f}s)",
                            flush=True,
                        )
                    run = await _run_one_cell(
                        client,
                        candidate=cand,
                        model_id=model_id,
                        task=task,
                        scenario=scn,
                        ollama_url=ollama_url,
                        keep_alive_seconds=keep_alive_seconds,
                        timeout_s=per_call_timeout_s,
                        samples_per_cell=samples_per_cell,
                    )
                    _write_run(run_dir, run)

            if log_progress:
                print(f"[evaluation] unloading {cand}", flush=True)
            await _unload_model(client, base_url=ollama_url, model=model_id)

    completed_at = datetime.now(timezone.utc)
    _write_manifest(
        run_dir,
        candidates=candidates,
        tasks=tasks,
        started_at=started_at,
        completed_at=completed_at,
        ollama_url=ollama_url,
        keep_alive_seconds=keep_alive_seconds,
    )
    if log_progress:
        total_elapsed = (completed_at - started_at).total_seconds()
        print(
            f"[evaluation] complete in {total_elapsed:.1f}s — wrote {run_dir}",
            flush=True,
        )
    return run_dir


async def _run_one_cell(
    client: httpx.AsyncClient,
    *,
    candidate: str,
    model_id: str,
    task: TaskSpec,
    scenario: Scenario,
    ollama_url: str,
    keep_alive_seconds: int,
    timeout_s: float,
    samples_per_cell: int = 5,
) -> CandidateRun:
    """Run a single (candidate, task, scenario) N=samples_per_cell times.

    Per the framework's N=5 sampling rule (2026-05-26): each cell runs N
    times to capture output variance. Local models are non-deterministic
    at default temperature; without resampling, a routing decision near
    a tier boundary could be a tail-sample artifact rather than a real
    score difference.

    Returns a single CandidateRun whose `samples` field holds N entries.
    If a sample errors, the error is captured in that sample's `error`
    field but subsequent samples still run — partial sampling is allowed
    so transient failures don't poison the whole cell.
    """
    samples: list[CandidateSample] = []
    for _ in range(samples_per_cell):
        try:
            text, latency_ms = await _ollama_chat(
                client,
                base_url=ollama_url,
                model=model_id,
                system_prompt=task.system_prompt,
                user_prompt=scenario.input,
                max_tokens=task.max_response_tokens,
                keep_alive=f"{keep_alive_seconds}s",
                timeout_s=timeout_s,
            )
            # Empty/whitespace output is a FAILURE, not a valid sample — the
            # thinking-mode bug (model emitted only hidden <think> tokens, or a
            # reasoning model we didn't pass think=False to). Record it as an
            # error so it scores 0 downstream and the integrity gate sees it,
            # rather than being silently judged as real quality. This is the
            # transport-level backstop for any reasoning model that slips
            # _needs_think_false (RCA stress-test HIGH).
            if not text or not text.strip():
                samples.append(
                    CandidateSample(output_text="", latency_ms=latency_ms, error="empty_output")
                )
            else:
                samples.append(CandidateSample(output_text=text, latency_ms=latency_ms, error=None))
        except Exception as e:  # noqa: BLE001 — record + continue per spec
            samples.append(
                CandidateSample(output_text="", latency_ms=0, error=f"{type(e).__name__}: {e}")
            )

    # Populate backward-compat fields from samples[0].
    first = samples[0]
    return CandidateRun(
        candidate=candidate,
        task_id=task.id,
        scenario_id=scenario.id,
        samples=samples,
        output_text=first.output_text,
        latency_ms=first.latency_ms,
        error=first.error,
        completed_at=datetime.now(timezone.utc),
    )


# --- Sync wrapper for CLI convenience -------------------------------------


def run_evaluation_sync(
    candidates: list[str],
    tasks: list[TaskSpec],
    out_dir: Path,
    **kwargs,
) -> Path:
    """Synchronous wrapper around `run_evaluation`.

    Lets the CLI stay synchronous (and easier to test) while the I/O path
    remains async for transport efficiency.
    """
    return asyncio.run(run_evaluation(candidates, tasks, out_dir, **kwargs))
