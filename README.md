# Agent Orchestrator

> **Pick the right model for each task in your app — and *prove* it stays right.**
> A per-application bake-off → tiered routing → audit → self-improvement loop, with
> integrity gates that refuse to lie to you about whether it's working.

[![CI](https://github.com/anirudhkrishnan/agent-orchestrator/actions/workflows/ci.yml/badge.svg)](https://github.com/anirudhkrishnan/agent-orchestrator/actions/workflows/ci.yml)
&nbsp;Python 3.11+ &nbsp;·&nbsp; MIT &nbsp;·&nbsp; zero required cloud deps (runs against local models out of the box)

---

## What is this?

Most apps that call an LLM use **one big model for everything** — and overpay, because
most individual tasks (extract these fields, classify this, triage that) don't need a
frontier model. The hard part isn't *knowing* that; it's **deciding which cheaper model
is safe for which task, and noticing when that stops being true.**

Agent Orchestrator does exactly that, per application:

1. **Bake-off** — run your app's real tasks against a pool of candidate models, scored
   by a strong "judge" model *held outside the candidate pool* (no candidate grades its
   own outputs).
2. **Route** — write the winner per task into a routing table. Tiered: prefer the
   cheapest model that clears your quality bar (local OSS → mid frontier → top frontier).
3. **Audit** — sample real production calls, re-judge them, and alarm when a model drifts
   below your line. The audit **refuses to report "healthy" when it can't actually tell.**
4. **Self-improve** — three optional loops feed production failures, new models, and new
   research back into the bake-off — each gated so the system can't quietly fool itself.

## Why is it different?

The individual pieces aren't new. What's unusual here is the **composition + the paranoia**:

- **Your eval set is your app's real task distribution**, not a generic benchmark or a
  learned classifier. The routing decision generalizes because it was measured on the
  thing you actually do.
- **Integrity is a first-class feature.** The framework has gates that *fail the build*
  when results look fabricated, when scope is incomplete, or when a metric would mislead.
  See [Safety](#safety-the-framework-distrusts-itself).
- **Personal / subscription / local-first.** Built for an individual or small team running
  local models + whatever frontier access they have — not a platform serving millions. No
  vendor lock, no required pay-per-token API.
- **It learns.** Optional self-improvement loops keep the routing current as your traffic,
  the model landscape, and best practices evolve.

It is **model-agnostic** — ships working against local [Ollama](https://ollama.com) models
and any model you can call; bring whatever you like.

## Table of contents

- [Quickstart](#quickstart)
- [The pipeline](#the-pipeline)
- [Tiered routing + cost-weighted savings](#tiered-routing--cost-weighted-savings)
- [What the output looks like](#what-the-output-looks-like)
- [Why N=5 sampling](#why-n5-sampling)
- [Bring your own judge](#bring-your-own-judge)
- [Models we developed against (use any)](#models-we-developed-against-use-any)
- [Results from internal use](#results-from-internal-use)
- [Safety: the framework distrusts itself](#safety-the-framework-distrusts-itself)
- [Telemetry & privacy](#telemetry--privacy)
- [Self-improving loops + the hard rule](#self-improving-loops--the-hard-rule)
- [Repository layout](#repository-layout)
- [For agents](#for-agents)
- [A note on terms of service](#a-note-on-terms-of-service)
- [License](#license)

## Quickstart

```bash
# 0. Get the code
git clone https://github.com/anirudhkrishnan/agent-orchestrator.git
cd agent-orchestrator

# 1. Install (pulls only httpx + pydantic + pyyaml)
pip install -e ".[dev]"

# 2. Run the test suite (no models or network needed — everything is unit-tested)
pytest -q

# 3. See the whole thing work on shipped example data (no models needed):
orchestrator-tiered dry-run examples/example-run-oss examples/example-run-frontier \
  --workflow ingest_pipeline
```

To run a real bake-off you'll want [Ollama](https://ollama.com) with a couple of models
pulled (e.g. `ollama pull qwen3:8b`). The four-phase flow:

```bash
# Phase 1 — run candidates against your tasks (N=5 samples per cell by default)
orchestrator-eval run --tasks data/evaluation/tasks-example.yaml \
  --candidates ollama/qwen3:8b ollama/gemma3:4b \
  --judge my-judge --out-dir data/evaluation/runs --samples-per-cell 5

# Phase 2 — the judge writes gold baselines, then scores candidates.
#            Use the bundled reference judge (see "Bring your own judge"), or your own.
orchestrator-eval init-baselines  <run-dir>
#   → edit <run-dir>/baselines.json: fill in each empty gold answer before prepare-batch
orchestrator-eval prepare-batch   <run-dir>
python -m orchestrator.judge_adapter <run-dir> --model <judge-model>   # or bring your own

# Phase 3 — finalize: REPORT.md + a paste-ready routing table.
#            Runs the score-integrity gate; REFUSES to finalize stamped/empty-scored runs.
orchestrator-eval finalize <run-dir>

# Tiered routing table + per-workflow savings:
orchestrator-tiered build-table <run-dir-oss> <run-dir-frontier>

# Audit a deployment later (non-zero exit on drift / incomplete scope / unverified):
orchestrator-audit run --app example-app --config data/audit.example.yaml
orchestrator-audit finalize --app example-app --config data/audit.example.yaml
```

## The pipeline

```
   your real tasks                  judge model (OUTSIDE the candidate pool)
        │                                       │
        ▼                                       ▼
   ┌──────────┐  N=5    ┌──────────┐  gold + scores   ┌────────────┐
   │ BAKE-OFF │ ───────▶│  JUDGE   │ ────────────────▶│  ROUTING   │
   └──────────┘         └──────────┘                  │   TABLE    │
                                                       └─────┬──────┘
   ┌──────────┐  re-judge sampled calls                     │ runtime picks a model per task
   │  AUDIT   │◀──────────────────────────────────────  production
   └────┬─────┘  alarms on drift / silent failure / incomplete scope
        │
        ▼  feeds failures, new models, new research back in (each gated)
   ┌────────────────────┐
   │ SELF-IMPROVE A/B/C  │
   └────────────────────┘
```

## Tiered routing + cost-weighted savings

The router doesn't just pick "best" or "cheapest." It picks the **cheapest tier that still
clears your quality bar**, per task:

```
        cost ($/token)         when it's used
  ┌────────────────────────────────────────────────────────────────┐
  │ Tier 0  top frontier      1.0×    tasks nothing cheaper does safely
  │ Tier 1  mid / low frontier ~0.2× / ~0.07×   most genuinely hard tasks
  │ Tier 2  local OSS          0×     mechanical / well-scoped tasks (usually most)
  └────────────────────────────────────────────────────────────────┘
   gate: a task drops to a cheaper tier ONLY if its WORST-scenario quality
         still clears your threshold (downside-risk, not the flattering average)
```

`orchestrator-tiered dry-run` compares three modes on your workflow's exact call sequence —
**tiered** (balance), **frontier-only** (max quality), **oss-only** (max savings) — and
reports cost-weighted savings + delegated-call quality + worst-slot quality for each.

## What the output looks like

**`finalize` delegation matrix** (per-task winner vs the judge) — illustrative:

```
| Task                     | Winner            | % of judge | Action               |
|--------------------------|-------------------|-----------:|----------------------|
| entity_extraction        | ollama/qwen3:8b   |       98%  | Delegate freely      |
| schema_extraction        | ollama/gemma3:4b  |       96%  | Delegate freely      |
| document_qa              | ollama/qwen3:8b   |       90%  | Delegate w/ monitor  |
| summary_synthesis        | anthropic/<mid>   |       95%  | Delegate (Tier 1)    |
| sentiment_classification | ollama/gemma3:4b  |       84%  | Delegate w/ monitor  |
```

**`orchestrator-tiered dry-run`** headline — this is the *actual* output of the quickstart
command above on the shipped example data (reproduce it yourself):

```
ingest_pipeline (11 calls):
| Mode                    | Cost-weighted savings | Delegated quality | Worst slot | Tiers 0/1/2 |
|-------------------------|----------------------:|------------------:|-----------:|-------------|
| Tiered (balance)        |                 69.0% |     96.9% (n=8)   |    96.0%   |   3 / 3 / 5 |
| Frontier-only (quality) |                 65.9% |     97.5% (n=8)   |    96.0%   |   3 / 8 / 0 |
| OSS-only (max savings)  |                 46.2% |     97.2% (n=5)   |    96.0%   |   6 / 0 / 5 |
```

Note the **mix** in tiered mode (`3 / 3 / 5`): 5 calls safely drop to free local models, 3 to
a mid-frontier model, and 3 genuinely-hard tasks correctly *stay* on the top frontier — the
tool finds the right tier per task rather than forcing everything cheap.

**Audit verdict** — the machine-readable signal automation reads:

```
[audit] VERDICT: pass                              → exit 0
[audit] RE-BAKE recommended for: document_qa       → exit 4   (drifted past your line)
[audit] INCOMPLETE SCOPE — refusing to run         → exit 3   (a slot was never measured)
[audit] VERDICT: unverified                        → exit 5   (couldn't actually verify)
```

(Exit `2` means setup is incomplete — e.g. the judge step hasn't produced its scores
yet, or the `--app` name doesn't match the config — with a message saying what to fix.)

## Why N=5 sampling

Local models aren't deterministic. Score a task **once** and a tier-boundary decision can
be a tail-sample artifact, not a real difference. So every (model, task, scenario) **cell
runs N times** and we route on the **median** (with std-dev surfaced as a stability signal —
the leaderboard carries a Stdev column and flags ⚡ any cell with stdev > 10).

We default to **N=5** as a rule of thumb. The field hasn't converged on a magic number, but
repeated-sampling configs in common eval tooling typically sit in the 3–10 range. With
LLM-judge score noise on the order of σ≈8–12 on a 0–100 scale, the standard-error math
lands roughly like this:

| N | ~95% CI on the mean | verdict |
|---|---|---|
| 1 | no variance signal | a single sample; tier-boundary calls unreliable |
| 3 | ≈ ±12 | spread visible, mean still noisy |
| **5** | **≈ ±9** | **distinguishes most boundary cases — the default** |
| 10 | ≈ ±6 | tighter, ~2× the cost |

These figures are a back-of-envelope guide, not a published result. It's a flag:
`--samples-per-cell N`. Raise it for high-stakes slots, drop to 1 for a quick smoke test.

## Bring your own judge

The "judge" is a strong model, kept **out of the candidate pool**, that writes gold-standard
answers and scores candidates. Two options:

1. **Reference adapter (bundled):** `python -m orchestrator.judge_adapter <run-dir> --model <id>`
   talks to any **OpenAI-compatible** chat endpoint — set `OPENAI_BASE_URL` + `OPENAI_API_KEY`.
   That covers OpenAI, many hosted providers, *and* a local Ollama model as judge
   (`OPENAI_BASE_URL=http://localhost:11434/v1`). No extra install needed — it uses the
   bundled `httpx` dependency.
2. **Your own / an agent:** the protocol is just files. After `prepare-batch`, a judge reads
   `judge-batch.json` and writes `judge-scores.json` (one score object per item). Any agent or
   script that follows that contract works; `src/orchestrator/evaluation/README.md` documents
   the exact shapes.

Either way, `finalize` runs the **score-integrity gate** on the result, so a lazy or broken
judge can't slip stamped/empty scores through.

One honest limitation: the judge sits outside the *candidate* pool, but the gold baselines
are judge-authored **and** judge-scored — they are the denominator of `% of judge`. Every
percentage is therefore anchored to the judge's own idea of a good answer: a weak or biased
judge shifts the whole scale rather than one candidate. Use the strongest judge you can,
and re-run the bake-off when you upgrade it.

## Models we developed against (use any)

During development we ran bake-offs across:

- **Local (zero marginal cost):** `qwen3:8b`, `qwen3.5:9b`, `gemma`-class models via Ollama
- **Frontier tiers:** a top model (judge + Tier-0 reference) and mid/low frontier models as
  Tier-1 candidates

**None of this is required.** A "candidate" is just an id the runner knows how to call. Point
it at whatever you have — other local models, other providers, future releases. The framework
re-benchmarks cleanly when you swap the judge or add a model (we re-ran the entire pipeline
against a newer judge model in a single pass; quality held or rose on nearly every task and
cost savings *increased*, because the sharper judge certified more slots as safe to delegate).

## Results from internal use

Measured on **two real internal workflows** (kept anonymous; ~19 model calls each per run),
using tiered routing at a 95%-of-frontier quality bar:

| Workflow | Avg quality vs frontier | Cost-weighted savings (tiered) |
|---|---|---|
| A — high-frequency ingest/triage | ~98% | **~74%** |
| B — multi-step analysis/report | ~98% | **~69%** |

"Cost-weighted savings" = the fraction of the frontier-model token cost avoided by routing
cheaper where it's safe (cost-weighted because a binary "displaced or not" hides the
difference between a mid-tier and a free local model). The headline: **most tasks in a real
workflow don't need your most expensive model, and this tells you precisely which ones —
with receipts.**

Your numbers will differ — they depend entirely on your tasks and your quality bar. That's
the point: you measure it, you don't guess.

## Safety: the framework distrusts itself

A model-selection system that's quietly wrong is worse than none — it routes real work to an
unfit model behind a green checkmark. So the framework is designed to **distrust its own
outputs**, and was hardened by an adversarial self-review against one failure class in
particular: **silent wrongness at the seams.** What's built in:

- **Score-integrity gate** — `finalize` *refuses* to produce a report if distinct candidate
  outputs receive suspiciously identical scores (the signature of recycled/"stamped" judging)
  or if an empty model output was scored as real quality. Stamping becomes un-shippable.
- **Completeness gate** — an audit/verdict cannot pass while any in-scope task is unmeasured.
  No "100% on the subset we happened to measure."
- **Honest exit codes** — the audit returns non-zero on drift, incomplete scope, *or*
  "couldn't actually verify"; the bake-off runner reports per-sample error counts and exits
  non-zero when every call failed; the reference judge retries, tallies its failures, and
  exits non-zero on any — so automation can't read a silent pass.
- **Sample-weighted median, not a mislabeled mean**; **worst-case** quality reported next to
  the average so a single bad slot can't hide; unknown/unpriced models flagged rather than
  silently counted as savings.
- **Fail-closed** everywhere: missing data is never treated as a pass.

The design rationale behind these gates is written up in
[`docs/DESIGN-PRINCIPLES.md`](docs/DESIGN-PRINCIPLES.md).

## Telemetry & privacy

Routed calls are logged to a **local** SQLite telemetry DB — default
`~/.agent-orchestrator/telemetry.sqlite`, overridable — so the audit can sample real
production calls. It stores **full prompt and response content**. Nothing leaves your
machine, but treat the file as sensitive data: it contains whatever your app sent to its
models. Don't commit it or sync it anywhere you wouldn't put the underlying data.

## Self-improving loops + the hard rule

Three optional loops keep the routing from going stale (in `orchestrator.improve`):

- **Loop A — learn from mistakes:** harvest thumbs-down / below-threshold production calls,
  stage them as *review-gated* new scenarios (carrying the real call as provenance), and
  queue the slot for re-bake. Active-learning for your eval set.
- **Loop B — learn from new models:** detect models that are available but never baked →
  propose a re-bake including them (routing changes stay human-confirmed).
- **Loop C — learn from research:** a web-search radar over eval/judging/routing best
  practices that emits an **advisory** — it never changes your methodology automatically.

> **The hard rule (non-negotiable):** no loop may *commit* a change derived from eval data
> that hasn't passed the integrity gate. Self-improvement without verification is just
> *scaled* self-deception. Enforced by `improve/guard.py` (`require_integrity` / `@gated`),
> which fails closed.

## Repository layout

```
src/orchestrator/
  evaluation/   bake-off runner, judge-batch prep, scoring, report, integrity gate
  tiered/       3-mode dry-run + routing-tiered table generator (cost-weighted savings)
  audit/        correctness + effects + quality drift, completeness gate, honest exit codes
  improve/      self-improving loops A/B/C + the integrity guard (the hard rule)
  onboarding/   template + scanner to onboard a new app's tasks
  telemetry/    local SQLite for routing decisions + sampled calls
  judge_adapter.py   optional reference judge (OpenAI-compatible endpoints)
data/
  evaluation/tasks-example.yaml   generic example task set (replace with yours)
  routing.json                    example routing table (a bake-off's output)
  audit.example.yaml              example audit config
examples/                         a runnable illustrative bake-off run (no models needed)
schemas/routing.schema.json       the routing-table contract
tests/                            full unit suite (run with `pytest -q`)
```

## For agents

Most humans will hand this README to an agent and say "set this up for my app." There is a
dedicated, extensive agent guide: **[`AGENTS.md`](AGENTS.md)** — it explains, in operational
detail, what the framework is for, when to reach for it, and the exact end-to-end procedure
(including the safety rules an agent must honor). If you are an AI agent reading this: open
`AGENTS.md` and follow it.

## A note on terms of service

This framework calls models you are entitled to call: **local models** (e.g. via Ollama) and
**any provider/model through that provider's legitimate API or your own authenticated
session.** It does **not** circumvent any provider's terms, scrape, or share credentials. The
judge and frontier tiers can be any model *you* are licensed to use. Keep it that way.

## License

[MIT](LICENSE) © anirudhkrishnan.
