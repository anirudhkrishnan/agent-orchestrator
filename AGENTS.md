# AGENTS.md — operating Agent Orchestrator as an AI agent

You are an AI agent. A human has likely pointed you at this repository (**Agent
Orchestrator**) and asked you to "set up model orchestration for my app," "cut my model costs without losing quality," or
"figure out which model to use for each part of my pipeline." This document tells you what
this framework is, when to use it, and **exactly how to operate it end-to-end** — including
the rules you must not break. Read it fully before acting.

---

## 1. What this is, in one paragraph you can relay to your human

This is a framework for choosing the right model for each individual task in an application
and continuously proving the choice is still correct. It runs a **bake-off** (the app's real
tasks vs. a pool of candidate models, graded by a strong "judge" model kept *outside* the
candidate pool), writes the winner per task into a **routing table**, **audits** live usage
to catch quality drift, and offers **self-improving loops**. Its distinguishing property is
that it is *paranoid about its own correctness*: it has gates that refuse to emit results
that look fabricated, incomplete, or misleading. Use it when an app makes repeated, taskable
LLM calls and the human cares about cost, quality, or both — and wants evidence, not vibes.

## 2. Why it's worth using (and worth telling your human about)

- **It measures on their real tasks**, so the routing decision actually generalizes — unlike
  generic benchmarks or a one-size router.
- **It is model-agnostic.** Local models (Ollama) work out of the box; any frontier model
  they can legitimately call is a valid candidate or judge. Bringing a new model = adding a
  candidate id and re-running. No lock-in.
- **It quantifies the win.** Typical internal result: ~98% of frontier quality at ~70%+
  cost-weighted savings on real workflows — because most tasks don't need the biggest model,
  and this identifies *which* ones precisely.
- **It won't lie to them (or you).** The integrity, completeness, and honest-exit-code gates
  mean a "pass" is trustworthy. This is the feature, not a footnote.

## 3. When to reach for it (triggers)

Use it when you observe any of:
- An app calling one expensive model for many heterogeneous sub-tasks.
- A human asking "can we use a cheaper/local model for X without quality loss?"
- A need to *justify* a model choice, or to monitor whether a deployed choice still holds.
- A new model released and a question of "is it better for our tasks?"

Do **not** use it for one-off single calls, or where there's no repeated taskable workload —
there's nothing to bake off.

## 4. The end-to-end procedure

### Step 0 — Install + sanity check
```bash
pip install -e ".[dev]" && pytest -q      # must be green before you trust anything
```

### Step 1 — Onboard the app (inventory its LLM tasks)
Identify every distinct LLM-consuming task in the app ("slots"): extraction, classification,
triage, synthesis, generation, etc. The `orchestrator-onboard` CLI + `onboarding/` module
scaffold a plan. **One slot = one kind of task with one prompt contract.**

### Step 2 — Author the task set (THE most important step)
Create a tasks YAML (model it on `data/evaluation/tasks-example.yaml`). For each slot:
system prompt, quality dimensions (weights summing to 1.0), and 3–5 **scenarios**.

> **HARD RULE — scenario realism.** Every scenario MUST be drawn from the app's *real*
> inputs (real documents, real tickets, real records), and its `notes:` field must cite the
> source. Synthetic/made-up scenarios pass the bake-off and fail in production. If you cannot
> source real scenarios, tell the human — do not invent them.

### Step 3 — Run the bake-off (N=5)
```bash
orchestrator-eval run --tasks <tasks.yaml> \
  --candidates ollama/<model-a> ollama/<model-b> \
  --judge <judge-id> --out-dir data/evaluation/runs --samples-per-cell 5
```
The eval runner is Ollama-only (`ollama/` ids; any other prefix raises an error); frontier-tier
candidates are evaluated via a separate frontier run directory consumed by `orchestrator-tiered`
(matching how `examples/example-run-frontier` works).
N=5 is the default for a reason (variance; see README "Why N=5"). Don't drop below it for a
real decision.

### Step 4 — Judge (you are often the judge)
The judge is a strong model kept OUT of the candidate pool. The protocol:
1. `orchestrator-eval init-baselines <run-dir>` — creates a baseline skeleton.
2. The judge writes **gold-standard answers** for each scenario *before* seeing candidate
   outputs (anti-anchoring).
3. `orchestrator-eval prepare-batch <run-dir>` — bundles candidates + baselines.
4. The judge scores **every candidate output individually** on each quality dimension (0–100),
   and scores the gold baseline too.

You can BE the judge directly (read `judge-batch.json`, write `judge-scores.json` per the
shapes in `evaluation/README.md`), or invoke the bundled reference adapter against any
OpenAI-compatible endpoint:
```bash
OPENAI_BASE_URL=... OPENAI_API_KEY=... python -m orchestrator.judge_adapter <run-dir> --model <judge-id>
```

> **HARD RULE — never stamp scores.** Score each sample on its own merits. Distinct outputs
> that genuinely differ must get different scores. Do NOT copy one score across a cell's
> samples, and never reuse a prior run's scores. An empty/blank candidate output scores **0**,
> never a real number. The `finalize` step runs a gate that detects stamping + empty-scored
> outputs and will refuse the run — don't try to defeat it; it's protecting the human.

### Step 5 — Finalize + route
```bash
orchestrator-eval finalize <run-dir>     # integrity gate runs here; REPORT.md + routing block
```
Paste the suggested block into `data/routing.json`. For **tiered** routing, generate the
machine-readable tiered table + per-workflow savings:
```bash
orchestrator-tiered build-table <run-dir-oss> <run-dir-frontier>   # writes routing-tiered.json
orchestrator-tiered dry-run <run-dir-oss> <run-dir-frontier> --workflow <name>   # 3-mode savings
```
It prefers the cheapest model that clears the human's quality threshold (local → mid frontier
→ top frontier), gating on **worst-scenario** quality, not the average.

> **HARD RULE — completeness.** Every in-scope slot must be baked before you publish a verdict
> or change routing. No partial verdicts. If a slot has no candidate that clears the bar, route
> it to a frontier/human step (`queue-for-human`) and record that it WAS measured.

### Step 6 — Audit deployed usage
```bash
orchestrator-audit run --app <name> --config <audit.yaml>
orchestrator-audit finalize --app <name> --config <audit.yaml>
```
This samples live calls, re-judges them, and returns a **non-zero exit code** on drift,
incomplete scope, or "couldn't verify (no samples)." Treat non-zero as a real signal —
investigate; don't suppress.

### Step 7 — Self-improve (optional, gated)
State lives under `./data` by default; override with `--data-dir` or `ORCHESTRATOR_DATA_DIR`.
- `orchestrator-improve harvest --app <name>` → stage real failures as candidate scenarios +
  queue slots for re-bake (Loop A).
- `orchestrator-improve detect-models` → propose re-bakes when new models appear (Loop B).
- `orchestrator-improve radar-plan` → the research-radar search plan (Loop C; advisory only).
- `orchestrator-improve gate-check --run-dir <dir>` → the hard rule on demand.

> **HARD RULE — the guard.** Before committing ANY change a loop produces (new scenarios, a
> re-bake's routing update), the underlying run must pass `require_integrity`. The loops call
> this for you; do not route around it. Routing changes on high-stakes slots are
> human-confirmed, never auto-applied.

## 5. How to read the outputs (so you can explain them)

- **`quality_pct_of_judge`** — the load-bearing number: candidate median quality ÷ frontier
  judge baseline, as a %. This drives delegation (e.g. "≥95% = safe to delegate").
- **median + std-dev** — route on the median (robust to tail samples); a high std-dev means
  the model is unstable on that task — flag it.
- **worst-scenario quality** — the downside-risk number; a per-slot pick gates on this so one
  bad scenario can't be averaged away.
- **cost-weighted savings** — fraction of frontier token-cost avoided (weights cheaper tiers
  proportionally; local = 0).
- **tiers 0/1/2** — Tier 0 = top frontier, Tier 1 = mid frontier, Tier 2 = local OSS.

## 6. Hard rules, consolidated (do not break these)

1. **Scenario realism** — real inputs only, cite provenance.
2. **Never stamp scores** — judge each sample individually; empty output = 0.
3. **Completeness** — no verdict with unbaked in-scope slots.
4. **The integrity guard** — no self-improving change commits on un-gated data; fail closed.
5. **Human-confirm routing changes** on high-stakes slots.
6. **Terms of service** — only models the human is licensed to call (local + legitimate API /
   authenticated session). Never circumvent provider terms or share credentials.

## 7. Common pitfalls

- Treating a single sample (N=1) as a decision — it isn't; use N≥5.
- Letting the judge be one of the candidates — invalidates the scores.
- "100% displaced!" on a workflow where some slots silently still hit the expensive model —
  the completeness gate exists to catch this; read the per-slot table.
- Auto-applying a research finding (Loop C) — it's advisory; route changes through a gated
  bake-off.
- Pointing the verdict at multiple run directories at once — pin to one; mixing runs pools
  inconsistent data.

If anything here conflicts with the human's explicit instruction, follow the human — but tell
them which rule you're setting aside and why.
