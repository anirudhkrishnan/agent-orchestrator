"""Audit engine — verifies routing is happening as expected at runtime.

Part 3 of the orchestration primitive (Parts 1+2 = bake-off framework and the
router plugin). The audit engine consumes the telemetry that the
plugin writes per call and answers three questions:

  1. **Is the orchestration happening as expected?** (correctness audit)
     Verifies the actual model used per (app, slot) matches what routing.json
     says it should be; flags fallback overuse, error spikes, missing slots.

  2. **What are the effects?** (savings/quality report)
     Computes frontier-model token displacement — the headline number
     ("reduced Opus usage by X% staying at Y% quality") — plus per-slot
     latency, success rate, and a USD savings estimate vs the counterfactual
     "had we used frontier for everything."

  3. **Is quality holding?** (quality drift via sampling)
     Samples N% of routed calls per slot, hands them to the frontier judge
     (interactive session — same protocol as the bake-off), compares
     the resulting quality to the baseline in routing.json, and alerts when
     quality crosses the configured per-app thresholds.

Per-app configurable; thresholds are NOT globally defined — they are a
per-build call based on the audit result. The module ships defaults
(warn=95%, rebake=80%) but the YAML config can override.

Public surface
--------------
- `AuditConfig` / `load_audit_config(path)`         — Pydantic + YAML loader
- `CorrectnessReport` / `run_correctness_audit(...)` — pure read against telemetry
- `EffectsReport` / `compute_effects_report(...)`    — frontier counterfactual
- `QualityReport` / `prepare_quality_batch(...)`     — interactive-judge batch
- `finalize_quality(...)`                            — read judge scores → drift
- `compose_audit_report(...)`                        — write AUDIT-REPORT.md

CLI (see `cli.py`)::

    python -m orchestrator.audit init     --app <name>
    python -m orchestrator.audit run      --app <name> --config <path>
    python -m orchestrator.audit finalize --app <name> --config <path>
"""

from .config import (
    AuditConfig,
    PricingEntry,
    PricingTable,
    init_audit_config_skeleton,
    load_audit_config,
)
from .correctness import CorrectnessReport, SlotCorrectness, run_correctness_audit
from .effects import EffectsReport, SlotEffects, compute_effects_report
from .quality import (
    QualityReport,
    SlotQuality,
    finalize_quality,
    prepare_quality_batch,
)
from .report import compose_audit_report

__all__ = [
    "AuditConfig",
    "CorrectnessReport",
    "EffectsReport",
    "PricingEntry",
    "PricingTable",
    "QualityReport",
    "SlotCorrectness",
    "SlotEffects",
    "SlotQuality",
    "compose_audit_report",
    "compute_effects_report",
    "finalize_quality",
    "init_audit_config_skeleton",
    "load_audit_config",
    "prepare_quality_batch",
    "run_correctness_audit",
]
