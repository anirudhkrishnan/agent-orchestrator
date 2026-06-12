"""Routing-decision + bake-off-result telemetry.

SQLite-backed. Single DB at ~/.agent-orchestrator/telemetry.sqlite,
overridable via the ORCHESTRATOR_DB_PATH env var. Note that
`init_db(path=...)` SETS that env var for the rest of the process — every
later telemetry call resolves to the same path.

Caveats worth knowing:

- The DB persists raw prompt/response content (message excerpts, full
  sampled input/output text). Relocate it via ORCHESTRATOR_DB_PATH if your
  home directory is synced or shared.
- `bakeoff_results` slots are one global namespace across every app writing
  to the DB — see `slot_winner` for the isolation options.
"""

from .db import (
    DEFAULT_DB_PATH,
    db_path,
    init_db,
    log_bakeoff_result,
    log_decision,
    log_sample,
    routing_decisions_for_audit,
    samples_for_audit,
    slot_winner,
)

__all__ = [
    "DEFAULT_DB_PATH",
    "db_path",
    "init_db",
    "log_bakeoff_result",
    "log_decision",
    "log_sample",
    "routing_decisions_for_audit",
    "samples_for_audit",
    "slot_winner",
]
