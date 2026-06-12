# audit — orchestration audit engine

Part 3 of the orchestration primitive. Parts 1 and 2 are:

1. **The bake-off framework** (`orchestrator.evaluation`) — runs candidate
   models against task scenarios; produces a `routing.json` with per-slot
   winners + `quality_pct_of_judge` baselines.
2. **The router plugin** — an integration in the host agent framework that
   reads `routing.json` at model-resolve time and writes a row to
   `routing_decisions` per call.

This module closes the loop: it consumes the telemetry the plugin writes
and answers three questions worth asking every week.

## The three questions

1. **Is the orchestration happening as expected?**
   _Correctness audit_ — for each slot in scope, verifies the model the
   plugin actually picked matches `routing.json`. Flags:
   - **Incomplete scope** — any in-scope slot that is unbaked or routed to
     `queue-for-human` without a measured fallback. This is the
     completeness gate (codified 2026-05-26); audits short-circuit until
     every in-scope slot is baked. See "The completeness rule" below.
   - Unexpected model selections (config drift, plugin bug)
   - High fallback rate (primary timing out)
   - High error rate (provider unhealthy)
   - Silent slots (no traffic = telemetry not wired up)

2. **What are the effects?**
   _Effects report_ — the headline number. Computes USD spend on the
   actual routed models vs. the counterfactual ("had we used the
   frontier model for every call"). Reports:
   - Frontier displacement % (the X in "reduced Opus by X%")
   - USD savings $ per app per window
   - Per-slot p50/p95 latency
   - Per-slot success rate

3. **Is quality holding?**
   _Quality drift_ — takes the most recent ~`sample_rate` fraction of
   routed calls per slot (default 5%, at least 1, capped at
   `max_samples_per_slot`), hands them to the frontier judge in an
   interactive session, compares the resulting mean quality to the
   bake-off baseline in `routing.json`, and classifies each slot as
   `ok` / `warn` / `rebake` / `unknown` / `no_samples` per the per-app
   thresholds. `unknown` and `no_samples` are treated as *unverified* —
   they produce a non-zero `finalize` exit, never a silent pass.

## The completeness rule

Codified 2026-05-26 after the first end-to-end workflow test. The rule:
every eval and audit must end with full and complete results — never
with missing slots.

### What it gates

Before the audit reads any telemetry, it inspects `slots_in_scope` against
`routing.json`. A slot is INCOMPLETE if any of:

- Not present in `routing.json` at all (scope drift).
- `model == "queue-for-human"` AND `last_baked_at is None` (never
  measured — there's no fallback to fall back to).
- `last_baked_at is None` (slot exists but was never baked).

If any in-scope slot fails this check, the audit short-circuits with
`INCOMPLETE_SCOPE` alarms and `overall_pass = False`. The per-slot
correctness table is empty in this case — by construction, you can't
produce a meaningful audit when slots are missing, because the
"this audit looks healthy" verdict on the measured subset hides the
slots still flowing to the frontier model.

### Why this matters

Without the gate, the verdict for a workflow that mixes measured + unbaked
slots reports something like "100% displacement on the 7 measured slots"
— and silently excludes the 4 unbaked slots that still hit the frontier
model every call. A casual reader takes that as "we displaced 100%" — false.

The gate forces you to either bake the slot off or remove it from
scope before the audit can produce numbers. No silent partial audits.

### Unblock procedure

When an audit fails on `INCOMPLETE_SCOPE`:

1. Read the alarms — they list the offending slots.
2. For each: author scenarios in your eval tasks YAML (start from
   `data/evaluation/tasks-example.yaml`) that resemble the real workflow
   inputs (see eval README's "Scenario authoring rules" for what counts
   as a valid scenario).
3. Run the bake-off:
   ```
   python -m orchestrator.evaluation run --tasks data/evaluation/tasks-example.yaml \
     --candidates ollama/qwen3:8b ollama/qwen3.5:9b ollama/gemma4:e4b \
     --judge claude-opus-4-7-interactive-session \
     --out-dir data/evaluation/runs
   ```
4. Update `routing.json` with the verdict — either pick a winner OR set
   `queue-for-human` with a populated `last_baked_at` (the bake-off
   ran; the verdict is documented).
5. Re-run the audit.

### Related rule — scenario-realism

The completeness gate is paired with the eval framework's scenario-realism
rule (see `evaluation/README.md` → "Scenario authoring rules"). A slot
"baked" against synthetic scenarios isn't really baked — the verdict is
just an artifact of made-up inputs. Together the two rules ensure: every
in-scope slot has been measured against real workflow inputs.

## The closed loop

```
bake-off → routing.json → router plugin runs → telemetry DB → audit → re-bake-off
   ↑                                                              │
   └──────────────── triggered when audit recommends ─────────────┘
```

The audit doesn't author baselines — that's the bake-off's job. It compares
current behavior against the baseline + the configured thresholds, and
surfaces "this slot's drift is past your re-bake line" so a human decides
to re-bake.

## Per-app config; per-app thresholds

The guiding invariant: the acceptable quality % is a call made per build
based on the audit result — there is no universal threshold that can be
defined ahead of time for all builds.

The audit honors this with per-app YAML configs under
`data/audit/{app_name}.yaml`. Each config carries:

- `slots_in_scope` — only these slots are audited (other slots ignored)
- `warn_threshold_pct` — quality % below this → WARN
- `rebake_threshold_pct` — quality % below this → RE-BAKE recommendation
- `sample_rate` — fraction of calls to sample for re-evaluation
- `lookback_days` — audit window
- `pricing` — USD/M-token rates per model (drives counterfactual)
- `max_fallback_rate_pct` / `max_error_rate_pct` — correctness alarms

Defaults match the reference defaults (95% warn, 80% re-bake, 5% sample
rate, 7-day lookback). Per-app YAML always wins.

## Lightweight by design

The audit is deliberately lightweight (don't add heavy cost) — the engine
does **sampling, not 100% re-evaluation**. At a 20-sample cap per slot per
audit, a typical week's audit costs at most:

- 5 slots × 20 samples = 100 items in the judge batch
- One interactive judge session, no API spend (the judge IS the
  interactive session, no API)
- ~10 minutes of human review on the batch + scores write-back

The non-quality portions (correctness + effects) are fully autonomous
and can run nightly via a scheduler (e.g., launchd or cron).

## Three-phase workflow

The interactive-judge protocol is identical in shape to the bake-off:

```bash
# Phase 0 — scaffold a config for a new app (once per app)
python -m orchestrator.audit init --app news-digest \
    --out data/audit/news-digest.yaml

# ... edit slots_in_scope + thresholds ...

# Phase 1 — autonomous: correctness + effects + prepare quality batch
python -m orchestrator.audit run --app news-digest \
    --config data/audit/news-digest.yaml
# Prints "READY FOR JUDGE", writes a preliminary AUDIT-REPORT.md

# Phase 2 — JUDGE STEP (interactive judge session reads + writes)
# (judge reads judge-batch.json, writes judge-scores.json)

# Phase 3 — finalize: compute drift, rewrite AUDIT-REPORT.md
python -m orchestrator.audit finalize --app news-digest \
    --config data/audit/news-digest.yaml
```

Phase 1 can run unattended on a schedule; phase 2 is a short interactive
judge session whenever convenient; phase 3 is one command.

## The headline

Every audit report opens with one sentence:

> **Reduced `anthropic/claude-opus-4-7` usage by 96.8%** while staying at
> **89.3%** of `anthropic/claude-opus-4-7` quality across 105 routed
> call(s) in the window.

If you skim nothing else, this is the number. Everything else in
the report exists to defend / refine that single claim.

## Adding the audit to a new app

```bash
# 1. Scaffold a config
python -m orchestrator.audit init --app newsletter-digest \
    --out data/audit/newsletter-digest.yaml

# 2. Edit slots_in_scope to match the slots this app actually uses
$EDITOR data/audit/newsletter-digest.yaml

# 3. Run the audit (or wire it into your scheduler of choice)
python -m orchestrator.audit run --app newsletter-digest \
    --config data/audit/newsletter-digest.yaml

# 4. Wait for the judge step, then:
python -m orchestrator.audit finalize --app newsletter-digest \
    --config data/audit/newsletter-digest.yaml
```

## TODOs for downstream work (out of scope here)

- **Router plugin sample writes.** The schema for `routed_call_samples`
  is in place, and the audit consumes it, but the router plugin
  doesn't yet write to it. A future PR should add a sampling hook on the
  plugin side that writes (input, output, model, latency, tokens) to
  `routed_call_samples` per routed call. The audit then keeps the most
  recent ~`sample_rate` fraction per slot when building the judge batch.
  Until that lands, the audit's quality section reports `no_samples`
  for every slot — correctness + effects are unaffected.
- **Per-call token columns.** The current effects estimate back-computes
  tokens from `cost_usd` using a 1:3 input:output blend. Adding explicit
  `input_tokens` / `output_tokens` columns to `routing_decisions` lets
  the counterfactual use exact token counts. Schema change + plugin
  write update.
- **Multi-app aggregation.** Today's report is per-app. A future
  cross-app rollup ("total Opus tokens displaced across the stack
  this week") could chain multiple AUDIT-REPORT.md files into a single
  weekly summary.

## Module layout

```
src/orchestrator/audit/
  __init__.py         Public surface
  config.py           AuditConfig + PricingTable + load_audit_config
  correctness.py      run_correctness_audit → CorrectnessReport
  effects.py          compute_effects_report → EffectsReport (the X)
  quality.py          prepare_quality_batch + finalize_quality (the Y)
  report.py           compose_audit_report — the markdown writer
  cli.py              argparse: init, run, finalize subcommands
  __main__.py         python -m orchestrator.audit
  README.md           you are here

tests/audit/
  conftest.py         tmp_db + audit_cfg + seed helpers
  test_config.py      Pydantic + YAML + skeleton roundtrip
  test_correctness.py expected vs actual model + alarm thresholds
  test_effects.py     counterfactual math + percentiles + savings
  test_quality.py     classify_alert + drift + batch sampling/capping
  test_report.py      end-to-end markdown composition
  test_cli.py         all three subcommands + error paths
```
