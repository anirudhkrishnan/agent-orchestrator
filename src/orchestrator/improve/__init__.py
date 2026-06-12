"""Self-improving loops for the orchestration primitive.

Three loops, one hard rule (codified 2026-05-28):

  Loop A (loop_a_mistakes) — learn from the system's own mistakes: harvest
    below-threshold / thumbs-down production calls → stage as review-gated
    scenarios + queue the slot for re-bake.
  Loop B (loop_b_models)   — learn from new models: detect available-but-unbaked
    models → propose a re-bake including them (human-confirmed routing).
  Loop C (loop_c_research) — learn from research: a web-search radar that emits
    an ADVISORY (never auto-applies methodology/routing changes).

  THE HARD RULE (guard) — no loop may COMMIT a change derived from eval data
    that has not passed the score-integrity gate (+ completeness where routing
    is touched). Self-improvement without verification is scaled self-deception
    (RCA 2026-05-28). `guard.require_integrity` / `guard.gated` enforce it.
"""

from .guard import IntegrityGateError, gated, require_integrity

__all__ = ["IntegrityGateError", "gated", "require_integrity"]
