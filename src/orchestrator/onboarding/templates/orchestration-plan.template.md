# Orchestration Plan — <APP_NAME>

**Status:** draft
**Last bake-off:** <ISO_TIMESTAMP>
**Routing.json slots used:** <COMMA_SEPARATED_LIST>
**Owner:** <NAME>
**Onboarded by:** <NAME> on <DATE>

> One plan per application. This document is the contract between the app and
> the orchestrator: which LLM tasks the app runs, which routing slot serves
> each one, what quality bar each must clear, and how the winning models are
> wired into the app's runtime. Replace every `<placeholder>`; delete the
> guidance blockquotes once a section is filled in.

## 1. Task inventory

> Every distinct LLM call the app makes — one row per task. `init` pre-fills
> rows from the workspace scan; rename the `task_NN_*` placeholders to
> meaningful names and complete the remaining columns by hand.

| Task | Input shape | Output shape | Frequency (per app run) | Latency tolerance | Notes |
|---|---|---|---|---|---|
| <task_name> | <input_shape> | <output_shape> | <freq> | <tolerance> | <notes> |

## 2. Slot mapping

> Map each task to a routing slot. Reuse an existing slot when the task shape
> matches (same input/output contract, same quality dimensions). Create a new
> slot only when nothing fits — every new slot needs scenarios in Section 3.

| Task | Slot | Rationale |
|---|---|---|
| <task_name> | <routing_slot> | <one_sentence_why> |

## 3. Scenarios

> ~5 scenarios per slot, drawn from REAL app traffic — not invented examples.
> Author them in your tasks YAML (copy `data/evaluation/tasks-example.yaml`
> for the shape). Record the slot → scenario-id mapping here for traceability.

| Slot | Scenario ids | Source (where the real inputs came from) |
|---|---|---|
| <routing_slot> | <scn_ids> | <source> |

## 4. Quality bars

> Per slot: the minimum acceptable score (as % of the judge's gold-standard
> baseline) and the dimensions that matter most. The bake-off and the audit
> both enforce these numbers.

| Slot | Quality bar (% of baseline) | Critical dimensions | Why this bar |
|---|---|---|---|
| <routing_slot> | <pct> | <dimensions> | <one_sentence_why> |

## 5. Candidate pool

> Models eligible to serve this app's slots, cheapest first. The judge model
> must stay OUTSIDE this pool — no model grades itself.

- Candidates: <model_ids>
- Judge: <judge_model_id>

## 6. Bake-off results

> Filled in after running the bake-off (see README.md for the phase-by-phase
> procedure). Record the run directory and the per-slot winners from REPORT.md.

| Slot | Winner | Score (% of baseline) | Run dir |
|---|---|---|---|
| <routing_slot> | <model_id> | <pct> | <run_dir> |

## 7. Audit plan

> How drift gets caught: sample rate for live calls, lookback window, and
> what reacts to an alarm. See `data/audit.example.yaml` for the config shape.

- Sample rate: <rate>
- Lookback window: <days> days
- Alarm response: <what_happens_when_quality_drops>

## 8. Runtime wiring

> How the app consumes the routing decision: point your runtime router at
> `data/routing.json` and resolve each task's slot to its winning model at
> call time. Note here exactly where the integration lives in the app's code.

- Routing table: `data/routing.json`
- Integration point(s): <file_or_module>
- Fallback model (when a slot is missing or the router is unreachable): <model_id>
