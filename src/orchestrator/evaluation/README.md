# `orchestrator.evaluation`

An evaluation framework for per-application model bake-offs: run a pool of
candidate models against your application's real tasks, have a frontier judge
score the outputs from OUTSIDE the candidate pool, and turn the results into
a routing recommendation.

## Why the judge sits outside the pool

A common shortcut is to pull one of the candidate models out of the pool to
serve as the judge. When the candidates and the judge come from the same
pool, two things go wrong:

1. **Self-judging bias.** A model grading its own outputs (even via a
   different prompt) can't be trusted to penalize its own hallucinations or
   weak formatting.
2. **Free-tier ceilings.** When the judge is a frontier API model, every
   grading call eats quota. A 30-scenario × 5-candidate bake-off = 150 judge
   calls minimum; on a 20/day free tier that's a `RESOURCE_EXHAUSTED`
   mid-run (observed 2026-05-26).

This module avoids both by putting the frontier judge **outside** the
candidate pool, in an interactive session. The runner writes candidate
outputs to disk; your judge model reads the batch, scores each item, and
writes a `judge-scores.json` array back to disk. No API tokens spent on
judging, no self-judging.

## What changed in v2

Two upgrades over the v1 release of this module:

1. **0-100 scoring scale.** Quality and speed are both scored on 0-100 (was
   1-5). The earlier scale clustered most candidates in the 3.5-4.5 band;
   0-100 gives enough granularity to see real degradation between, say, a
   92% candidate and a 78% candidate on the same task.

2. **Judge baselines.** The judge produces a gold-standard answer for each
   (task, scenario) BEFORE seeing any candidate output. When scoring, the
   judge grades both its own baseline and each candidate on the same rubric.
   The framework computes `quality_pct_of_baseline = candidate / baseline ×
   100` — the load-bearing signal for delegation decisions ("can this local
   model reproduce 90% of what the judge would have written?"). Mechanical
   threshold buckets on that signal feed the per-task callouts and the
   end-of-report delegation matrix.

The baseline pattern is the durable insight. Re-running with a new judge or
new candidate set keeps the scaffolding intact; only the answers under each
scenario change.

## Pipeline

```
+--------------+      +-----------+      +------------------+
|  tasks.yaml  | -->  |  runner   | -->  | init-baselines   |
+--------------+      +-----------+      +------------------+
                            |                     |
                            v                     v
                  per-cell JSON files    baselines.json skeleton
                            |                     |
                            |          (judge fills in answers)
                            |                     |
                            |                     v
                            |          +------------------+
                            +--------->|  prepare-batch   |
                                       +------------------+
                                                 |
                                                 v
                                       judge-batch.json
                                                 |
                                       (judge scores both
                                       baseline + candidate)
                                                 |
                                                 v
                                       judge-scores.json
                                                 |
                                                 v
                                       +------------------+
                                       |     finalize     |
                                       +------------------+
                                                 |
                                                 v
                                          REPORT.md
                                       + routing.json suggestion
                                       + delegation matrix
```

## Four-step CLI

```bash
# Phase 1 — run candidates
python -m orchestrator.evaluation run \
  --tasks data/evaluation/tasks-example.yaml \
  --candidates ollama/qwen3:8b ollama/qwen3.5:9b ollama/gemma4:e4b \
  --judge interactive-judge-session \
  --out-dir data/evaluation/runs

# Phase 1.5 — create the baselines.json skeleton
python -m orchestrator.evaluation init-baselines data/evaluation/runs/<timestamp>

# Phase 2 — your judge model opens baselines.json and writes a
#           gold-standard answer in each (task_id, scenario_id) slot

# Phase 2.5 — bundle baselines + candidate outputs into judge-batch.json
python -m orchestrator.evaluation prepare-batch data/evaluation/runs/<timestamp>

# Phase 3 — judge reads judge-batch.json and writes judge-scores.json
#           scoring BOTH the baseline and the candidate per item

# Phase 4 — produce REPORT.md (leaderboards + routing.json suggestion + matrix)
python -m orchestrator.evaluation finalize data/evaluation/runs/<timestamp>
```

If you skip the baseline steps (1.5 → 2 → 2.5), `prepare-batch` still works
without `baselines.json` — it emits the no-baseline variant of the judge
instructions, the judge scores candidates only, and the resulting REPORT.md
omits the `% of Judge` column and the delegation matrix. Everything else
(per-task leaderboards, winners, routing.json suggestion) still renders.

### Design rationale: why baselines

The "% of judge" signal answers the question that actually matters for
delegation: **how much quality do I give up by routing this task to a
small local model instead of the frontier?**

* A combined score of 75 means little on its own — is that "good enough" or
  "embarrassingly weak"?
* A `% of judge` of 92% means "the local model is 92% as good as my judge on
  this rubric, in this scenario." That's a number that maps directly to a
  routing decision.

Honest baseline scoring matters. The judge should score its own baseline
imperfectly when imperfect (a 92/100, not a faked 100). Otherwise every
candidate's `% of judge` is inflated downward. The judge instructions
explicitly call this out.

## Authoring a new task

Tasks live in YAML, not code. The runner is generic — it doesn't know what
`entity_extraction` means; it just sends scenarios to each candidate and
captures outputs. To add a task, append to the YAML file:

```yaml
- id: my_new_task
  description: One-line summary for routing.json.
  system_prompt: |
    You are an assistant that does <X>. Be concise.
  max_response_tokens: 256
  quality_dimensions:
    - name: accuracy
      description: Did the candidate get the right answer?
      weight: 0.6
    - name: format
      description: Did it respect the requested output shape?
      weight: 0.4
  scenarios:
    - id: scn-01
      input: |
        <the actual prompt to send to the candidate>
      notes: This scenario probes <Y>; correct answer should be <Z>.
      expected_output_shape: JSON array of strings.
```

Two hard rules enforced at load time:
- `quality_dimensions` weights MUST sum to 1.0 (within 1e-6 tolerance).
- Scenario ids must be unique within a task; task ids must be unique within
  the file.

## Scenario authoring rules (framework-level)

Codified 2026-05-26 after the second eval cycle. These rules apply to
EVERY task added to the bake-off; violating them makes downstream routing
decisions misleading. They're listed here because the eval framework is
the place new scenarios get authored.

### Rule 1 — Scenarios must resemble actual application need

Bake-off scenarios are the ground truth that decides which model gets
routed. If the scenarios don't reflect the workflow's real input
distribution, the winner doesn't generalize.

**Concrete:** every scenario MUST be authored from one of:
- A real production output (e.g., a recent output file from the pipeline
  for your workflow stage).
- A real document snippet from the data source the workflow consumes.
- A real schema-extraction input the workflow consumes (an API response,
  a data file snippet, a structured document fragment).

**Bad scenarios:** synthetic prose generated by the model itself, fabricated
entities, made-up field values, hand-waved "kinda looks like what the user
might paste in" content. These pass the bake-off and fail in production.

The `notes:` field of each scenario MUST cite the source — file path + date,
or X-handle + post URL, or production-output filename. Source-less
scenarios are an authoring smell; require them as part of code review.

### Rule 2 — Completeness: never leave a slot unbaked

Every application that onboards via the orchestration workflow (see the
repo-level `README.md` and `AGENTS.md`) inventories the tasks its workflow
makes. Every one of those tasks MUST be baked off before that app can
produce an audit verdict.

The `audit` module's `check_scope_completeness()` enforces this at runtime:
audits with unbaked slots in scope short-circuit with `INCOMPLETE_SCOPE`
alarms (see `audit/correctness.py`).

If a slot's bake-off shows no candidate hits the quality bar, route the
slot to `queue-for-human` BUT keep `last_baked_at` populated — the
slot is measured, the measurement just produced the verdict "no OSS pick".
That counts as baked.

### Rule 2b — N=5 sampling per cell (default)

Local models aren't deterministic at default temperature. Single-shot
scoring (N=1) makes routing decisions on noisy point estimates: a 67%
score might really be 52% or 82%, and we can't tell without resampling.

**The framework's default is N=5 samples per (candidate, task, scenario)
cell.** Each cell runs 5 times; the aggregator takes the median for the
routing decision and reports std-dev so high-variance slots are flagged.

#### Why N=5

N=5 is a pragmatic rule of thumb, not a statistically derived constant:
it is the point where, in our runs, per-cell medians stopped jumping
around between re-runs while the marginal cost of another sample
(5 model calls per cell instead of 1, before dedupe) stayed tolerable.
More samples always tighten the estimate; they also multiply runtime.

Two honest caveats:

- The **routing decision uses the per-cell MEDIAN**, which is robust to a
  single tail sample — that, more than the exact N, is what protects the
  decision.
- The **std-dev over 5 samples is a coarse signal**. It is good for "this
  cell is unstable, treat the median with caution" and not much more; do
  not read precision into one decimal place of a 5-sample std-dev.

#### Aggregation

Per-cell aggregation (what `finalize` computes and renders):
- **Median** is used for routing decisions (robust to tail samples). It is
  the `Quality` column of the REPORT.md leaderboard and the `p50_quality`
  key of the routing.json suggestion.
- **Std-dev** is the `Stdev` column of the REPORT.md leaderboard, with a
  ⚡ flag appended when std-dev > 10 points (the median is a coarse
  signal there). It is also emitted as `stdev_quality` in the
  routing.json suggestion block.
- **Min/Max** bracket the observed range — available as the
  `min_quality` / `max_quality` keys of the aggregation dict
  (`aggregate_per_task_candidate`).

`prepare-batch` dedupes identical outputs within a cell — if all 5
samples are byte-identical, only 1 judge-batch item is emitted with
`sample_count: 5`. The judge scores it once; aggregation expands by
`sample_count` to compute median + std-dev correctly. This makes
deterministic tasks (e.g. entity_extraction) cheap to judge without
losing variance signal.

#### Override

Lower with `--samples-per-cell 1` for quick smoke tests. Don't lower
below 3 for routing decisions — below that, a single tail sample can
flip the median.

### Rule 3 — When new scenarios surface mid-eval, re-run the loop

It is impossible to capture every scenario upfront. The eval and audit
machinery is designed to surface gaps continuously. When a gap is
identified (e.g., during workflow testing, audit telemetry sampling, or
a real production failure):

1. **LIST the gap explicitly** — name the scenario, link to the
   production case it came from.
2. **Add to your tasks YAML** in the appropriate task — or author a new
   task if it doesn't fit any existing one.
3. **Re-run the bake-off** with the new scenarios:
   `python -m orchestrator.evaluation run --tasks data/evaluation/tasks-example.yaml ...`
4. **Update the orchestration plan** for any app that consumes the
   affected slots.
5. **Re-run any pending production audit**: `orchestrator audit run --config ...`

This loop is the framework's scalability story. New applications and new
workflows will surface scenarios that didn't exist when the bake-off was
authored. The pipeline is fast enough (5-30 minutes end-to-end depending
on task count) that re-running on every gap is cheap.

**Trigger words for self-catch:** if you think "we can add that scenario
later" or "this gap doesn't really need a new bake-off run" — STOP. That's
the regression. The completeness gate exists precisely because the cost
of a stale routing decision (a document being processed by an under-spec'd
model, a structured output being half-baked) is much higher than the 20-minute
cost of a re-run.

## Changing the judge

The default judge is `interactive-judge-session` — a label, not a model
handle. The framework never CALLS the judge programmatically; it just
records which judge produced which scores. To swap judges:

```bash
python -m orchestrator.evaluation run --judge <new-label> ...
python -m orchestrator.evaluation init-baselines --judge <new-label> <run-dir>
python -m orchestrator.evaluation prepare-batch --judge <new-label> <run-dir>
```

This module is built around the interactive disk-mediated judge protocol;
an API-driven judge would need its own transport, written against the same
`judge-batch.json` / `judge-scores.json` contract.

## Changing candidates

Pass `--candidates` as a space-separated list of `ollama/<model:tag>` strings.
Currently only Ollama models are supported as candidates by the runner; other
providers raise `ValueError` so future expansion is explicit (not silent).

To add a non-Ollama candidate provider, edit `runner._run_one_cell` to
dispatch by provider prefix, and update the `for c in candidates:` validation
block at the top of `run_evaluation`.

## Delegation matrix thresholds

The final section of REPORT.md classifies each task's winner into one of
three buckets, derived from the `% of judge` value:

| Threshold | Label                  |
|-----------|------------------------|
| ≥ 80%     | Delegate freely        |
| 60–80%    | Delegate with monitor  |
| < 60%     | Keep on judge          |

Constants live in `scoring.py` (`DELEGATE_FREELY_BASELINE_PCT`,
`DELEGATE_WITH_MONITOR_BASELINE_PCT`). Tune there if your risk
tolerance shifts.

## Re-running after model updates

The framework is intentionally stateless — each run produces a fresh
timestamped directory under `--out-dir`. To re-validate routing after pulling
a new model version:

```bash
ollama pull qwen3:8b   # or whatever model updated
python -m orchestrator.evaluation run --tasks ... --candidates ... --out-dir ...
python -m orchestrator.evaluation init-baselines <run-dir>
# ... judge fills baselines.json + judge-scores.json ...
python -m orchestrator.evaluation prepare-batch <run-dir>
# ... judge writes judge-scores.json ...
python -m orchestrator.evaluation finalize <run-dir>
# Compare new REPORT.md against the previous run's REPORT.md
```

If the judge hasn't changed, baselines from a prior run can be copy-pasted
into the new run's `baselines.json` to save the gold-answer-authoring time.

There's no destructive update to `routing.json`; the report emits a paste-ready
JSON block that the user merges manually.

## Sequential execution + RAM rationale

On a typical 16–18GB unified-memory machine, two 9B models cannot
co-reside. A single `qwen3.5:9b` resident uses ~9GB; two candidates resident
at once would force swapping mid-call (or OOM the Ollama process).

The runner enforces explicit RAM management via Ollama's `keep_alive`
protocol:

1. **Load** — `POST /api/generate` with empty prompt + `keep_alive: <30m>`
   warms the candidate into RAM (cold load can take 30-60s for a 9GB model).
2. **Run all scenarios for this candidate**, with each chat call passing the
   same `keep_alive` so weights stay resident.
3. **Unload** — `POST /api/generate` with empty prompt + `keep_alive: 0`
   evicts the model before the next candidate loads.

This is slower than parallel candidate execution (which is impossible here
anyway) but guarantees fair latency measurements: no candidate's call is
poisoned by another's weights being swapped in.

If you ever run this on a machine with more RAM, parallelism is still a
non-goal — sequential keeps the latency comparisons clean.

## Reasoning-model `think: false` handling

Empirical finding (2026-05-26): reasoning-by-default model families (qwen3 /
qwen3.5, gemma4, DeepSeek-R1, QwQ, and friends) bury all output in hidden
reasoning tokens unless `think: false` is passed at the Ollama API level —
the response budget is consumed by thinking and the user-facing output comes
back empty. The rule lives in `runner._needs_think_false`, a case-insensitive
substring match on the bare model name (after stripping the `ollama/` prefix)
against a list of known thinking-by-default families.

That list will always lag new releases, so the runner also has a transport-
level backstop: an empty/whitespace output is recorded as an `empty_output`
error rather than silently judged as real quality, which catches any
reasoning model that slips the name match.

## Output layout

```
<out-dir>/
  2026-05-26T18-30-00Z/                # ISO timestamp (colons replaced)
    manifest.json                       # candidates + tasks + timing
    ollama-qwen3-8b/                    # safe id of candidate
      entity_extraction_scn-01.json     # one CandidateRun per cell
      entity_extraction_scn-02.json
      summary_synthesis_scn-01.json
    ollama-qwen3-5-9b/
      ...
    baselines.json                      # written by init-baselines (skeleton),
                                        # filled in by the judge
    judge-batch.json                    # written by prepare-batch
    judge-scores.json                   # written by the JUDGE
    REPORT.md                           # written by finalize
```

A run directory is fully self-contained — you can tar it up, ship it
elsewhere, and re-run `finalize` against it.

## Files in this module

| File          | Purpose                                                              |
|---------------|----------------------------------------------------------------------|
| `__init__.py` | Module exports                                                       |
| `__main__.py` | `python -m orchestrator.evaluation` entry point                      |
| `tasks.py`    | Pydantic models + YAML loader                                        |
| `runner.py`   | Sequential RAM-aware candidate execution against Ollama              |
| `batch.py`    | Build `judge-batch.json` + write baselines.json skeleton             |
| `scoring.py`  | Pure functions: speed bin, combined, baseline pct, delegation tiers  |
| `report.py`   | Markdown report generation (with delegation matrix when baselined)   |
| `cli.py`      | Four-phase CLI: `run`, `init-baselines`, `prepare-batch`, `finalize` |

## Tests

```bash
uv run pytest tests/evaluation/ -v
```

All scoring + aggregation logic is unit-tested with synthetic data. The
Ollama HTTP transport is exercised by running against a real Ollama instance
(no mock for that — the contract IS that we talk to the local server).
