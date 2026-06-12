"""Effects audit — what is the orchestration actually saving us?

Computes the headline metric::

    "Reduced frontier usage by X% staying at Y% of frontier quality."

Where X = frontier-token displacement and Y comes from quality.py. This
module owns X; report.py glues them together.

Counterfactual model
--------------------
For each routed call in the window, we have:

  * `selected_model`  — what the router chose
  * `cost_usd`        — what that call cost (logged by the plugin)
  * `latency_ms`      — what that call took

We don't (yet) have per-call token counts in `routing_decisions` — the
existing schema carries cost_usd but not tokens. Per call, the
counterfactual ("had we used the frontier model instead") is computed as:

  * Frontier-model calls — counterfactual = actual cost (no displacement).
  * Models ABSENT from the pricing table — counterfactual = actual cost,
    zero claimed displacement; the model is flagged in the report
    (`unpriced_models`) until a pricing entry is added.
  * KNOWN PAID models whose calls carry no logged cost — savings are
    unverifiable; counterfactual = actual cost (zero claimed displacement)
    and the model is flagged (`cost_unverified_models`).
  * KNOWN FREE/local models (both rates 0.0) — a nominal 500-input /
    1500-output call is priced at frontier rates (footnoted in the report).
  * Otherwise — tokens are back-computed from cost via the pricing table
    assuming a 1:3 input:output blend, then re-priced at frontier rates.
    Crude but conservative: the actual frontier output ratio is usually
    much higher than the candidate's, so the blend undercounts savings.

Savings are SIGNED, per slot and in aggregate — a router that costs MORE
than the frontier counterfactual shows negative savings, never a 0 floor.

For the longer-horizon roadmap there's a TODO to extend
`routing_decisions` with explicit input_tokens / output_tokens columns —
then this module switches to exact counterfactual without the blend.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from orchestrator.telemetry import db as tdb

from .config import AuditConfig, PricingTable


@dataclass
class SlotEffects:
    """Per-slot effects breakdown."""

    slot: str
    n_calls: int
    actual_cost_usd: float
    counterfactual_cost_usd: float
    savings_usd: float
    savings_pct: float
    """100 * (1 - actual/counterfactual). Pinned to 0 when counterfactual is 0."""
    p50_latency_ms: float
    p95_latency_ms: float
    success_rate_pct: float
    """100 - error_rate (errors counted as user_feedback == 'error')."""


@dataclass
class EffectsReport:
    """Top-level effects report — aggregated + per-slot."""

    app_name: str
    window_start_iso: str
    window_end_iso: str
    frontier_model: str
    total_calls: int
    total_actual_cost_usd: float
    total_counterfactual_cost_usd: float
    total_savings_usd: float
    """SIGNED: negative when routing cost more than the frontier counterfactual."""
    overall_savings_pct: float
    """Headline X value — token (cost-proxy) displacement vs. frontier."""
    slots: list[SlotEffects] = field(default_factory=list)
    unpriced_models: list[str] = field(default_factory=list)
    """Models absent from the pricing table — counted at actual cost with zero
    claimed displacement. Surfaced in the report so the table gets fixed."""
    cost_unverified_models: list[str] = field(default_factory=list)
    """Known PAID models whose calls carried no logged cost — their savings are
    unverifiable and excluded from the claimed savings (zero displacement)."""


def _estimate_tokens_from_cost(
    cost_usd: float | None,
    pricing: PricingTable,
    model: str,
) -> tuple[float, float]:
    """Estimate (input_tokens, output_tokens) for a row from cost + rates.

    Coarse: assumes a 1:3 input:output ratio (typical for chat-style calls),
    then back-solves from the blended USD/M-token rate. If the row's cost is
    None / 0 (local model), returns (0, 0).

    Returns:
        Tuple of (input_tokens, output_tokens) as floats — fractional is fine
        since these are estimates, not authoritative counts.
    """
    if not cost_usd or cost_usd <= 0:
        return (0.0, 0.0)
    entry = pricing.lookup(model)
    if entry is None or (entry.input_usd_per_1m == 0 and entry.output_usd_per_1m == 0):
        return (0.0, 0.0)
    # Solve cost = (in_tokens/1M) * in_rate + (out_tokens/1M) * out_rate
    # with the assumption out_tokens = 3 * in_tokens.
    # => cost = in_tokens/1M * (in_rate + 3*out_rate)
    blended_rate = entry.input_usd_per_1m + 3.0 * entry.output_usd_per_1m
    if blended_rate <= 0:
        return (0.0, 0.0)
    in_tokens = cost_usd / blended_rate * 1_000_000.0
    return (in_tokens, in_tokens * 3.0)


def _frontier_cost_for(
    in_tokens: float,
    out_tokens: float,
    pricing: PricingTable,
) -> float:
    """USD cost the same call would have incurred on the frontier model.

    Multiplies token-count estimates by the frontier rates.
    """
    fe = pricing.frontier_entry()
    return (
        in_tokens / 1_000_000.0 * fe.input_usd_per_1m
        + out_tokens / 1_000_000.0 * fe.output_usd_per_1m
    )


def _percentile(values: list[int], pct: float) -> float:
    """Compute a percentile from a list of ints. Linear interpolation.

    Avoids depending on numpy; the stdlib `statistics.quantiles` works
    but is awkward for arbitrary percentiles. Pure-python is fine at our
    sample sizes (< thousands per slot per week).
    """
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def compute_effects_report(
    cfg: AuditConfig,
    *,
    now: datetime | None = None,
) -> EffectsReport:
    """Compute the effects report.

    Args:
        cfg: Loaded AuditConfig.
        now: Override for "now" (defaults to UTC now).

    Returns:
        EffectsReport with the headline savings_pct + per-slot detail.
    """
    now = now or datetime.now(timezone.utc)
    window_start = (now - timedelta(days=cfg.lookback_days)).isoformat()

    rows = tdb.routing_decisions_for_audit(
        app_name=cfg.app_name,
        slots=cfg.slots_in_scope,
        since_iso=window_start,
    )

    # Bucket rows by slot for per-slot rollup; aggregate totals on the side.
    rows_by_slot: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        slot = r.get("classified_slot")
        if slot is None:
            continue
        rows_by_slot[slot].append(r)

    slot_effects: list[SlotEffects] = []
    total_actual = 0.0
    total_counter = 0.0
    total_calls = 0
    unknown_models: set[str] = set()  # models absent from the pricing table
    cost_unverified: set[str] = set()  # paid models whose calls had no logged cost

    for slot in cfg.slots_in_scope:
        slot_rows = rows_by_slot.get(slot, [])
        n = len(slot_rows)
        total_calls += n

        actual_cost = 0.0
        counter_cost = 0.0
        latencies: list[int] = []
        n_errors = 0

        for row in slot_rows:
            model = row.get("selected_model") or ""
            cost = row.get("cost_usd")
            lat = row.get("latency_ms")
            if isinstance(lat, int):
                latencies.append(lat)
            if row.get("user_feedback") == "error":
                n_errors += 1

            row_actual = float(cost) if cost is not None else 0.0
            actual_cost += row_actual

            # Counterfactual: had we used frontier for this call instead.
            entry = cfg.pricing.lookup(model)
            if model == cfg.pricing.frontier_model:
                # Already frontier — no displacement.
                counter_cost += row_actual
            elif entry is None:
                # UNKNOWN model (not in the pricing table). We cannot compute a
                # trustworthy counterfactual — so DON'T fabricate displacement
                # (the old code fell into the nominal-500/1500 branch and counted
                # a renamed/mispriced model that genuinely cost money as a WIN —
                # stress-test HIGH). Count its real cost, claim zero displacement,
                # and flag it.
                counter_cost += row_actual
                unknown_models.add(model)
            elif (
                entry.input_usd_per_1m > 0 or entry.output_usd_per_1m > 0
            ) and (cost is None or cost <= 0):
                # KNOWN PAID model whose call carries no logged cost. The old
                # code dropped this into the nominal-500/1500 branch and claimed
                # 100% savings for a call that genuinely cost money. We can't
                # verify the savings — claim zero displacement and flag it.
                counter_cost += row_actual
                cost_unverified.add(model)
            else:
                in_tok, out_tok = _estimate_tokens_from_cost(cost, cfg.pricing, model)
                if in_tok == 0 and out_tok == 0:
                    # KNOWN free/local model (priced at 0) — a nominal call keeps
                    # the counterfactual from being a flat zero. Anchor on a
                    # typical 500-input / 1500-output call. Estimate, footnoted.
                    in_tok, out_tok = 500.0, 1500.0
                counter_cost += _frontier_cost_for(in_tok, out_tok, cfg.pricing)

        # SIGNED savings (no max(...,0) floor): if the router actually cost MORE
        # than frontier would have, that must show as negative, not be hidden as
        # 0 (stress-test MED).
        savings = counter_cost - actual_cost
        savings_pct = (
            100.0 * savings / counter_cost if counter_cost > 0 else 0.0
        )
        p50 = statistics.median(latencies) if latencies else 0.0
        p95 = _percentile(latencies, 95.0)
        success_rate = 100.0 - (n_errors / n * 100.0 if n else 0.0)

        total_actual += actual_cost
        total_counter += counter_cost

        slot_effects.append(
            SlotEffects(
                slot=slot,
                n_calls=n,
                actual_cost_usd=actual_cost,
                counterfactual_cost_usd=counter_cost,
                savings_usd=savings,
                savings_pct=savings_pct,
                p50_latency_ms=float(p50),
                p95_latency_ms=p95,
                success_rate_pct=success_rate,
            )
        )

    # Both flags also travel on the EffectsReport so they land in the report
    # markdown + verdict summary — stderr alone is too easy to miss.
    if unknown_models:
        import sys as _sys
        _sys.stderr.write(
            f"⚠️  [effects] {len(unknown_models)} model(s) absent from the pricing "
            f"table {sorted(unknown_models)} — their calls are counted at actual cost "
            f"with ZERO claimed displacement (not fabricated as savings). Add them to "
            f"the pricing table for accurate effects accounting.\n"
        )
    if cost_unverified:
        import sys as _sys
        _sys.stderr.write(
            f"⚠️  [effects] {len(cost_unverified)} PAID model(s) had calls with no "
            f"logged cost {sorted(cost_unverified)} — their savings are unverifiable "
            f"and excluded from the claimed savings. Fix the plugin's cost logging.\n"
        )

    overall_savings_pct = (
        100.0 * (total_counter - total_actual) / total_counter
        if total_counter > 0
        else 0.0
    )

    return EffectsReport(
        app_name=cfg.app_name,
        window_start_iso=window_start,
        window_end_iso=now.isoformat(),
        frontier_model=cfg.pricing.frontier_model,
        total_calls=total_calls,
        total_actual_cost_usd=total_actual,
        total_counterfactual_cost_usd=total_counter,
        # SIGNED, like the per-slot figure: a $0 floor here would hide an
        # aggregate that cost MORE than the frontier counterfactual.
        total_savings_usd=total_counter - total_actual,
        overall_savings_pct=overall_savings_pct,
        slots=slot_effects,
        unpriced_models=sorted(unknown_models),
        cost_unverified_models=sorted(cost_unverified),
    )
