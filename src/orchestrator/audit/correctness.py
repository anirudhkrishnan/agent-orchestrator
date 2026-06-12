"""Correctness audit — is the orchestration happening as expected?

Reads the telemetry DB for the audit window and verifies::

  * COMPLETENESS (per the completeness rule, codified 2026-05-26): every slot
    listed in ``slots_in_scope`` is present in ``routing.json`` with a
    non-null ``last_baked_at``. A slot routed to ``queue-for-human`` is
    acceptable ONLY if it was measured (``last_baked_at`` set) — the bake-off
    ran and documented "no candidate hit the bar". Slots that don't pass this
    gate are surfaced as ``INCOMPLETE_SCOPE`` alarms and the audit
    short-circuits to ``overall_pass=False``. Partial audits are misleading
    by construction — they always pass on the measured subset and silently
    exclude the un-measured slots.
  * The model actually selected per (app, slot) matches what ``routing.json``
    declares the slot's primary model to be.
  * Fallback usage is within ``max_fallback_rate_pct`` — high fallback rate is
    an early-warning that the primary is timing out / 429ing.
  * Error rate per slot is within ``max_error_rate_pct`` — surfaces transport
    failures the audit shouldn't sweep under the rug.
  * Every slot listed in ``slots_in_scope`` actually received traffic — silent
    drops (telemetry not written, plugin disabled, app regressed) show up as
    "no calls" rather than "looks healthy".

The module is pure: no I/O except DB reads. Output is a
``CorrectnessReport`` dataclass that ``report.py`` formats into markdown.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from orchestrator.telemetry import db as tdb

from .config import AuditConfig


@dataclass
class SlotCorrectness:
    """Per-slot correctness check result."""

    slot: str
    expected_primary_model: str | None
    expected_fallback_model: str | None
    n_calls: int
    n_with_expected_primary: int
    n_with_fallback: int
    n_with_unexpected_model: int
    n_errors: int
    fallback_rate_pct: float
    error_rate_pct: float
    unexpected_models: list[str] = field(default_factory=list)
    alarms: list[str] = field(default_factory=list)
    """Human-readable strings; empty list means slot is healthy."""


@dataclass
class CorrectnessReport:
    """Full correctness audit across all slots_in_scope.

    ``overall_pass`` is True iff:
      * ``incomplete_scope_alarms`` is empty (every in-scope slot is baked), AND
      * every slot's ``alarms`` list is empty, AND
      * every slot received at least one call (silent-drop catch).
    """

    app_name: str
    window_start_iso: str
    window_end_iso: str
    routing_json_path: Path
    slots: list[SlotCorrectness]
    overall_pass: bool
    incomplete_scope_alarms: list[str] = field(default_factory=list)
    """Top-level alarms for slots that shouldn't be in scope yet — unbaked,
    or queue-for-human without a measured verdict. Per the completeness rule
    (codified 2026-05-26): every workflow verdict must cover EVERY in-scope
    slot. These alarms force you to either bake the slot off OR remove it
    from scope before the audit can pass."""

    @property
    def alarm_count(self) -> int:
        """Total alarms across all slots + incomplete-scope alarms."""
        return sum(len(s.alarms) for s in self.slots) + len(self.incomplete_scope_alarms)


def _load_routing_json(path: Path) -> dict:
    """Read routing.json off disk.

    Tolerates the "_README" sentinel key the file uses for in-band docs
    (skipped by callers iterating over slot keys).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"routing.json not found at {p}. The audit needs to know the "
            f"expected primary model per slot."
        )
    return json.loads(p.read_text())


def _window_start_iso(now: datetime, lookback_days: int) -> str:
    """Compute the ISO timestamp for `now - lookback_days`."""
    return (now - timedelta(days=lookback_days)).isoformat()


def check_scope_completeness(
    slots_in_scope: list[str],
    routing: dict,
) -> list[str]:
    """Return a list of INCOMPLETE_SCOPE alarms for unbaked/unroutable slots.

    Per the completeness rule (codified 2026-05-26):
    every workflow verdict must cover EVERY in-scope slot. A slot is
    considered "incomplete" if any of:

      * It is not present in routing.json at all.
      * It is routed to ``queue-for-human`` AND has null last_baked_at
        (i.e. there is no measured fallback to fall back to).
      * Its ``last_baked_at`` is null (slot exists but was never baked).

    Returns an empty list if all in-scope slots are fully baked. Otherwise
    returns one human-readable alarm string per missing slot. Callers should
    use ``len(alarms) > 0`` as a hard-fail signal — partial audits are
    misleading by construction.
    """
    alarms: list[str] = []
    for slot in slots_in_scope:
        slot_cfg = routing.get(slot)
        if slot_cfg is None:
            alarms.append(
                f"INCOMPLETE_SCOPE: slot {slot!r} not present in routing.json. "
                f"Either author scenarios in the eval tasks YAML + bake-off + add "
                f"to routing.json, OR drop it from audit scope."
            )
            continue

        primary_model = slot_cfg.get("model")
        last_baked = slot_cfg.get("last_baked_at")

        # Routed to queue-for-human AND never baked = fully un-measured.
        if primary_model == "queue-for-human" and last_baked is None:
            alarms.append(
                f"INCOMPLETE_SCOPE: slot {slot!r} is queue-for-human "
                f"with no measured fallback (last_baked_at is null). The "
                f"completeness rule requires every in-scope slot "
                f"to be baked. Author scenarios for this slot in the eval tasks "
                f"YAML and re-run the bake-off, OR drop it from audit scope."
            )
            continue

        # Exists in routing but never baked (legitimate or stale).
        if last_baked is None:
            alarms.append(
                f"INCOMPLETE_SCOPE: slot {slot!r} has last_baked_at=null "
                f"(never measured). Run the bake-off for this slot before "
                f"including it in audit scope."
            )

    return alarms


def run_correctness_audit(
    cfg: AuditConfig,
    *,
    now: datetime | None = None,
    config_anchor: Path | None = None,
) -> CorrectnessReport:
    """Run the correctness audit.

    Args:
        cfg: Loaded AuditConfig.
        now: Override for "now" (defaults to UTC now). Tests pass an explicit
            value to make windows deterministic.
        config_anchor: Directory the AuditConfig was loaded from — used to
            resolve relative routing_json_path. None means the path is
            already absolute or relative to CWD.

    Returns:
        CorrectnessReport with per-slot detail + overall pass/fail.

    Raises:
        FileNotFoundError: if routing.json is missing.
    """
    now = now or datetime.now(timezone.utc)
    window_start = _window_start_iso(now, cfg.lookback_days)

    routing_path = cfg.routing_json_path
    if not routing_path.is_absolute() and config_anchor is not None:
        routing_path = (config_anchor / routing_path).resolve()
    routing = _load_routing_json(routing_path)

    # ── Completeness gate (codified 2026-05-26) ───────────────────────
    # Before measuring anything, refuse to audit if any in-scope slot is
    # unbaked. A partial audit that excludes the un-measured slots would
    # always report ~100% on the measured subset — misleading by construction.
    incomplete_scope_alarms = check_scope_completeness(cfg.slots_in_scope, routing)

    if incomplete_scope_alarms:
        # Short-circuit: don't bother reading telemetry. Return a hard-fail
        # report so the CLI exits non-zero and the gap gets fixed before
        # producing a verdict.
        return CorrectnessReport(
            app_name=cfg.app_name,
            window_start_iso=window_start,
            window_end_iso=now.isoformat(),
            routing_json_path=routing_path,
            slots=[],
            overall_pass=False,
            incomplete_scope_alarms=incomplete_scope_alarms,
        )

    rows = tdb.routing_decisions_for_audit(
        app_name=cfg.app_name,
        slots=cfg.slots_in_scope,
        since_iso=window_start,
    )

    # Bucket telemetry rows by slot for per-slot processing.
    rows_by_slot: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        slot = r.get("classified_slot")
        if slot is None:
            continue
        rows_by_slot[slot].append(r)

    slot_reports: list[SlotCorrectness] = []
    for slot in cfg.slots_in_scope:
        slot_cfg = routing.get(slot) or {}
        expected_primary = slot_cfg.get("model")
        expected_fallback = slot_cfg.get("fallback_model")
        slot_rows = rows_by_slot.get(slot, [])

        n_calls = len(slot_rows)
        n_errors = 0
        n_with_primary = 0
        n_with_fallback = 0
        n_with_unexpected = 0
        unexpected_seen: set[str] = set()

        for row in slot_rows:
            selected = row.get("selected_model") or ""
            fallback_used = bool(row.get("fallback_used"))
            user_fb = row.get("user_feedback")
            # `user_feedback == 'error'` is the convention we ask the plugin
            # to use for transport failures (429, timeout, 5xx). Surfacing
            # those here avoids treating an errored call as "successful with
            # the right model".
            if user_fb == "error":
                n_errors += 1

            if fallback_used:
                n_with_fallback += 1
                # A fallback hit isn't "unexpected" — it's the configured
                # backup. Bucket it separately so the report can show both
                # rates without double-counting.
                if expected_fallback and selected != expected_fallback:
                    n_with_unexpected += 1
                    unexpected_seen.add(selected)
                continue

            if expected_primary is None or selected == expected_primary:
                # When routing.json has no expectation yet (e.g. NOT YET BAKED),
                # we can't call any model "unexpected"; count it as primary.
                n_with_primary += 1
            else:
                n_with_unexpected += 1
                unexpected_seen.add(selected)

        fallback_rate = (n_with_fallback / n_calls * 100.0) if n_calls else 0.0
        error_rate = (n_errors / n_calls * 100.0) if n_calls else 0.0

        alarms: list[str] = []
        # A slot present + baked in routing.json but MISSING its `model` key
        # means expected_primary is None, which makes the "unexpected model"
        # check below count EVERY selected model as correct — UNEXPECTED can
        # never fire (stress-test HIGH). Surface the malformed entry instead of
        # silently trusting it.
        if expected_primary is None and slot in routing and slot_cfg.get("last_baked_at") is not None:
            alarms.append(
                f"MALFORMED ROUTING: slot {slot!r} is baked (last_baked_at set) but "
                f"has no `model` key — the unexpected-model check is disabled for it. "
                f"Fix routing.json."
            )
        if n_calls == 0:
            alarms.append(
                f"NO TRAFFIC: slot {slot!r} received 0 calls in the last "
                f"{cfg.lookback_days} day(s). Either the app is dormant or "
                f"the router-plugin isn't writing telemetry for it."
            )
        if n_with_unexpected > 0:
            sample = ", ".join(sorted(unexpected_seen)[:3])
            alarms.append(
                f"UNEXPECTED MODEL: {n_with_unexpected}/{n_calls} calls used "
                f"a model other than {expected_primary!r} / {expected_fallback!r}. "
                f"Examples: {sample}. Check that routing.json is in sync with "
                f"the live router-plugin config."
            )
        if fallback_rate > cfg.max_fallback_rate_pct:
            alarms.append(
                f"HIGH FALLBACK RATE: {fallback_rate:.1f}% of calls hit the "
                f"fallback model (threshold: {cfg.max_fallback_rate_pct}%). "
                f"Primary {expected_primary!r} may be timing out or rate-limited."
            )
        if error_rate > cfg.max_error_rate_pct:
            alarms.append(
                f"HIGH ERROR RATE: {error_rate:.1f}% of calls errored "
                f"(threshold: {cfg.max_error_rate_pct}%). Check provider "
                f"health + plugin retry config."
            )

        slot_reports.append(
            SlotCorrectness(
                slot=slot,
                expected_primary_model=expected_primary,
                expected_fallback_model=expected_fallback,
                n_calls=n_calls,
                n_with_expected_primary=n_with_primary,
                n_with_fallback=n_with_fallback,
                n_with_unexpected_model=n_with_unexpected,
                n_errors=n_errors,
                fallback_rate_pct=fallback_rate,
                error_rate_pct=error_rate,
                unexpected_models=sorted(unexpected_seen),
                alarms=alarms,
            )
        )

    overall_pass = all(not s.alarms for s in slot_reports)
    return CorrectnessReport(
        app_name=cfg.app_name,
        window_start_iso=window_start,
        window_end_iso=now.isoformat(),
        routing_json_path=routing_path,
        slots=slot_reports,
        overall_pass=overall_pass,
        incomplete_scope_alarms=[],
    )
