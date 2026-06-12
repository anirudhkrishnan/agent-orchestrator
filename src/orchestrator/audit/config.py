"""Per-app audit configuration — YAML on disk, Pydantic in memory.

One YAML file per app under `data/audit/{app_name}.yaml`. The file is
hand-authored (or scaffolded via `init_audit_config_skeleton`) and read by
every audit subcommand.

Design notes
------------
* **No global thresholds.** What quality % is acceptable is a judgment call
  made per build based on the audit result — there is no universal threshold
  that can be defined ahead of time. The defaults (95% warn / 80% rebake)
  are illustrative reference numbers; the per-app YAML always wins.

* **Pricing table is per-config, not per-module.** Frontier USD/M-token rates
  drift quarterly; sticking them in the audit module would force a code
  release every time a provider changes prices. Sticking them in the YAML
  lets the config evolve at the cadence of the world.

* **`slots_in_scope` is explicit.** An audit deliberately ignores slots not
  listed — the report should reflect the surface area actively monitored
  for this app, not every slot that happened to receive traffic.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class PricingEntry(BaseModel):
    """One row in the pricing table — USD per 1M tokens for a model.

    Both input and output rates are required. Local models are priced at 0.0
    (local compute). The frontier model entry is used by `effects.py` for the
    counterfactual calculation.
    """

    model: str = Field(..., description="Provider/model id, e.g. 'ollama/qwen3:8b'.")
    input_usd_per_1m: float = Field(..., ge=0.0, description="USD per 1M input tokens.")
    output_usd_per_1m: float = Field(..., ge=0.0, description="USD per 1M output tokens.")


class PricingTable(BaseModel):
    """Pricing table embedded in AuditConfig.

    `frontier_model` names the model whose pricing is used for the
    counterfactual. Must be a key in `entries` — validated at load time.
    """

    frontier_model: str = Field(
        ...,
        description=(
            "Identifier of the frontier model used for the cost-counterfactual "
            "(usually 'anthropic/claude-opus-4-7'). Must appear in `entries`."
        ),
    )
    entries: list[PricingEntry] = Field(default_factory=list)

    def lookup(self, model: str) -> PricingEntry | None:
        """Return the PricingEntry for `model`, or None if unknown.

        Local models default to free compute when missing; callers that need
        a fail-hard semantic should check the None explicitly.
        """
        for e in self.entries:
            if e.model == model:
                return e
        return None

    def frontier_entry(self) -> PricingEntry:
        """Return the pricing row for the configured frontier model.

        Raises ValueError if `frontier_model` isn't in the table — `validate`
        normally catches this at load time but the runtime check guards
        against programmatic mutations after construction.
        """
        e = self.lookup(self.frontier_model)
        if e is None:
            raise ValueError(
                f"PricingTable.frontier_model={self.frontier_model!r} is not present "
                f"in entries. Add a PricingEntry for it before computing effects."
            )
        return e

    @field_validator("entries")
    @classmethod
    def _at_least_one(cls, v: list[PricingEntry]) -> list[PricingEntry]:
        if not v:
            raise ValueError("PricingTable.entries must contain at least one row.")
        return v


# Sensible defaults — used by the skeleton initializer. Numbers reflect public
# Anthropic + Ollama pricing as of 2026-05-26. Update via YAML, not code.
#
# The three Anthropic entries are the tier ladder used by the orchestration
# primitive's cost-weighted audit:
#   - Opus  4.7: $15/$75 per 1M tokens (tier 0 = reference)
#   - Sonnet 4.6: $3/$15 per 1M tokens (tier 1 = 5x cheaper than Opus)
#   - Haiku 4.5: $1/$5 per 1M tokens   (tier 1 = 15x cheaper than Opus)
# Local Ollama entries are zero (electricity-only) — tier 2.
_DEFAULT_PRICING_ENTRIES = [
    PricingEntry(model="anthropic/claude-opus-4-7", input_usd_per_1m=15.0, output_usd_per_1m=75.0),
    PricingEntry(model="anthropic/claude-sonnet-4-6", input_usd_per_1m=3.0, output_usd_per_1m=15.0),
    PricingEntry(model="anthropic/claude-haiku-4-5", input_usd_per_1m=1.0, output_usd_per_1m=5.0),
    PricingEntry(model="ollama/qwen3:8b", input_usd_per_1m=0.0, output_usd_per_1m=0.0),
    PricingEntry(model="ollama/qwen3.5:9b", input_usd_per_1m=0.0, output_usd_per_1m=0.0),
    PricingEntry(model="ollama/gemma4:e4b", input_usd_per_1m=0.0, output_usd_per_1m=0.0),
]


class AuditConfig(BaseModel):
    """Per-app audit configuration.

    Authored as YAML; the loader round-trips through this Pydantic class for
    validation. The fields collectively define BOTH what the audit checks
    (which slots, what window, what samples) and HOW it reports (warn/rebake
    thresholds, pricing for cost counterfactual).
    """

    app_name: str = Field(..., min_length=1, description="Stable identifier, e.g. 'news-digest'.")
    slots_in_scope: list[str] = Field(
        ...,
        description="Slots this audit covers. Slots outside this list are ignored.",
    )
    routing_json_path: Path = Field(
        default=Path("data/routing.json"),
        description=(
            "Path to the routing.json used at runtime. Read to verify the actual "
            "model selected per slot matches expectation, and to fetch the "
            "`quality_pct_of_judge` baseline for drift computation."
        ),
    )
    warn_threshold_pct: float = Field(
        default=95.0,
        ge=0.0,
        le=200.0,
        description=(
            "Quality % below this triggers a WARN in the report. Matches "
            "the reference defaults; per-app overridable."
        ),
    )
    rebake_threshold_pct: float = Field(
        default=80.0,
        ge=0.0,
        le=200.0,
        description=(
            "Quality % below this triggers a RE-BAKE recommendation. Used to "
            "decide whether the routing winner is still valid."
        ),
    )
    sample_rate: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Fraction of routed calls to sample for re-evaluation (default 5%).",
    )
    lookback_days: int = Field(
        default=7,
        ge=1,
        description="How many days back to consider for both correctness + effects + quality.",
    )
    max_samples_per_slot: int = Field(
        default=20,
        ge=1,
        description=(
            "Hard cap on samples per slot in the judge batch — keeps the batch "
            "manageable for the interactive judge even on busy slots."
        ),
    )
    judge_model: str = Field(
        default="claude-opus-4-7-interactive-session",
        description="Identifier of the judge — recorded in the AUDIT-REPORT.md front matter.",
    )
    pricing: PricingTable = Field(
        ...,
        description="USD/M-token rates for cost counterfactual + USD savings claim.",
    )
    max_fallback_rate_pct: float = Field(
        default=5.0,
        ge=0.0,
        le=100.0,
        description=(
            "Correctness alarm threshold — if more than this % of routing "
            "decisions for a slot used the fallback model, flag the slot. "
            "Useful early-warning that the primary model is timing out."
        ),
    )
    max_error_rate_pct: float = Field(
        default=2.0,
        ge=0.0,
        le=100.0,
        description=(
            "Correctness alarm threshold — % of routed calls that errored "
            "(429/timeout/HTTP-5xx). Above this triggers a slot-level flag."
        ),
    )
    out_dir: Path = Field(
        default=Path("data/audit/reports"),
        description="Where AUDIT-REPORT.md + intermediate JSON files land.",
    )

    @model_validator(mode="after")
    def _rebake_below_warn(self) -> "AuditConfig":
        # STRICTLY below, not <=: when rebake == warn the WARN tier is
        # unreachable (a slot below the line classifies straight to rebake,
        # never warn), silently collapsing a configured band (stress-test MED).
        # A model_validator (not a field_validator) so the invariant also holds
        # when rebake_threshold_pct comes from its default — field validators
        # don't run on defaulted fields, which let warn=70 + rebake-omitted
        # (default 80) slip through.
        if self.rebake_threshold_pct >= self.warn_threshold_pct:
            raise ValueError(
                f"rebake_threshold_pct ({self.rebake_threshold_pct}) must be "
                f"STRICTLY < warn_threshold_pct ({self.warn_threshold_pct}). Equal "
                f"thresholds collapse the WARN tier (it becomes unreachable); "
                f"rebake is the deeper concern and must sit below warn."
            )
        return self

    def out_dir_resolved(self, anchor: Path) -> Path:
        """Return out_dir resolved against `anchor` if it's relative.

        Lets the config-file author write relative paths without depending on
        whoever invokes the CLI happening to chdir to the right place.
        """
        if self.out_dir.is_absolute():
            return self.out_dir
        return (anchor / self.out_dir).resolve()


def load_audit_config(path: Path) -> AuditConfig:
    """Load an audit config from YAML.

    Args:
        path: Path to a YAML file matching the AuditConfig shape.

    Returns:
        Parsed AuditConfig.

    Raises:
        FileNotFoundError: if `path` doesn't exist.
        ValueError: on YAML parse failure or Pydantic validation error
            (the original ValidationError is chained).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"audit config not found at {p}")
    try:
        raw: dict[str, Any] = yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"failed to parse YAML at {p}: {e}") from e
    return AuditConfig.model_validate(raw)


def init_audit_config_skeleton(
    *,
    app_name: str,
    out_path: Path,
    slots: list[str] | None = None,
    frontier_model: str = "anthropic/claude-opus-4-7",
    overwrite: bool = False,
) -> Path:
    """Write a starter audit-config YAML at `out_path`.

    Pre-fills sensible defaults so the author only needs to edit
    `slots_in_scope` and any per-app threshold overrides. Idempotent unless
    `overwrite=True`.

    Args:
        app_name: Stable identifier, becomes `app_name` in the file.
        out_path: Where to write the YAML.
        slots: Slot names to seed; defaults to the canonical 5 if None.
        frontier_model: Recorded in pricing.frontier_model.
        overwrite: If False (default) and the file exists, raises FileExistsError.

    Returns:
        Path to the written file.
    """
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"{out_path} already exists. Pass overwrite=True to replace "
            f"(this will discard any local edits)."
        )
    if slots is None:
        slots = [
            "entity_extraction",
            "relevance_triage",
            "summary_synthesis",
            "document_qa",
            "schema_extraction",
        ]
    cfg = AuditConfig(
        app_name=app_name,
        slots_in_scope=slots,
        pricing=PricingTable(
            frontier_model=frontier_model,
            entries=list(_DEFAULT_PRICING_ENTRIES),
        ),
    )
    yaml_text = _dump_skeleton_yaml(cfg, app_name=app_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml_text)
    return out_path


def _dump_skeleton_yaml(cfg: AuditConfig, *, app_name: str) -> str:
    """Render AuditConfig to YAML with inline comments.

    yaml.safe_dump doesn't preserve comments; emit them by hand for the
    skeleton so the human author sees the same explanations the Pydantic
    docstrings carry.
    """
    now = datetime.now(timezone.utc).isoformat()
    body = yaml.safe_dump(
        cfg.model_dump(mode="json"),
        sort_keys=False,
        default_flow_style=False,
    )
    header = (
        f"# Audit config for app: {app_name}\n"
        f"# Generated: {now}\n"
        f"# Edit `slots_in_scope` to match the slots this app actually uses.\n"
        f"# `warn_threshold_pct` / `rebake_threshold_pct` are a per-build call —\n"
        f"# no universal default. Defaults below match the reference defaults\n"
        f"# (95% warn, 80% re-bake).\n"
        f"#\n"
        f"# Pricing entries: USD per 1M tokens. Update when provider rates change.\n"
        f"# Local models are priced at 0.0 (local compute).\n\n"
    )
    return header + body
