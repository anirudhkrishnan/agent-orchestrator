-- Routing-decision telemetry + bake-off results.
-- Apply via `init_db()` in db.py. Idempotent (`IF NOT EXISTS`).

CREATE TABLE IF NOT EXISTS routing_decisions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp TEXT NOT NULL,
  session_id TEXT NOT NULL,
  app_name TEXT,  -- owning app; NULL only on rows written before this column existed
  message_excerpt TEXT NOT NULL,
  classified_slot TEXT,
  selected_model TEXT NOT NULL,
  fallback_used INTEGER DEFAULT 0,
  latency_ms INTEGER,
  cost_usd REAL,
  user_feedback TEXT  -- 'good', 'bad', NULL (set by user later)
);
CREATE INDEX IF NOT EXISTS idx_routing_slot ON routing_decisions(classified_slot);
CREATE INDEX IF NOT EXISTS idx_routing_model ON routing_decisions(selected_model);
CREATE INDEX IF NOT EXISTS idx_routing_timestamp ON routing_decisions(timestamp);
CREATE INDEX IF NOT EXISTS idx_routing_app ON routing_decisions(app_name);

CREATE TABLE IF NOT EXISTS bakeoff_results (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  slot TEXT NOT NULL,
  candidate_model TEXT NOT NULL,
  example_id TEXT NOT NULL,
  output TEXT NOT NULL,
  judge_model TEXT NOT NULL,
  rubric_scores TEXT NOT NULL,  -- JSON
  mean_score REAL NOT NULL,
  latency_ms INTEGER,
  cost_usd REAL,
  baked_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bakeoff_slot ON bakeoff_results(slot);

-- Per-call samples kept for periodic quality re-evaluation by the audit engine.
-- Populated by the runtime router (an external process may write to the same
-- DB) when a coin-flip against AuditConfig.sample_rate succeeds. The audit
-- engine reads from here to build a judge batch; the router only writes.
-- Until a runtime router is wired up, the table is written only by audit
-- tests + manual fixture-seeding.
CREATE TABLE IF NOT EXISTS routed_call_samples (
    sample_id        TEXT PRIMARY KEY,
    routed_at        TEXT NOT NULL,
    app_name         TEXT NOT NULL,
    slot             TEXT NOT NULL,
    candidate_model  TEXT NOT NULL,
    input_text       TEXT NOT NULL,
    output_text      TEXT NOT NULL,
    latency_ms       INTEGER NOT NULL,
    input_tokens     INTEGER,
    output_tokens    INTEGER,
    sampled_for      TEXT  -- 'audit', 'manual', etc.
);
CREATE INDEX IF NOT EXISTS idx_samples_app_slot ON routed_call_samples(app_name, slot);
CREATE INDEX IF NOT EXISTS idx_samples_routed_at ON routed_call_samples(routed_at);
