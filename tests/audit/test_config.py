"""Tests for orchestrator.audit.config — Pydantic model + YAML loader.

Covers:
  * Skeleton initializer produces a YAML the loader can read back.
  * rebake_threshold_pct cannot exceed warn_threshold_pct.
  * Pricing table requires the frontier model to be in entries.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from orchestrator.audit import (
    AuditConfig,
    PricingEntry,
    PricingTable,
    init_audit_config_skeleton,
    load_audit_config,
)


def test_skeleton_roundtrips_through_loader(tmp_path: Path):
    """The YAML the skeleton emits is loadable by load_audit_config."""
    out = tmp_path / "skel.yaml"
    init_audit_config_skeleton(app_name="probe", out_path=out)
    cfg = load_audit_config(out)
    assert cfg.app_name == "probe"
    # Skeleton seeds the canonical 5 slots if no override provided.
    assert "entity_extraction" in cfg.slots_in_scope
    assert cfg.pricing.frontier_model == "anthropic/claude-opus-4-7"
    assert cfg.pricing.lookup("anthropic/claude-opus-4-7") is not None


def test_skeleton_refuses_overwrite_by_default(tmp_path: Path):
    out = tmp_path / "skel.yaml"
    init_audit_config_skeleton(app_name="probe", out_path=out)
    with pytest.raises(FileExistsError):
        init_audit_config_skeleton(app_name="probe", out_path=out)


def test_skeleton_overwrite_replaces(tmp_path: Path):
    out = tmp_path / "skel.yaml"
    init_audit_config_skeleton(app_name="probe", out_path=out)
    # Different app_name on the second pass — should win.
    init_audit_config_skeleton(app_name="probe-v2", out_path=out, overwrite=True)
    cfg = load_audit_config(out)
    assert cfg.app_name == "probe-v2"


def test_skeleton_custom_slots(tmp_path: Path):
    out = tmp_path / "skel.yaml"
    init_audit_config_skeleton(
        app_name="probe",
        out_path=out,
        slots=["custom_slot_a", "custom_slot_b"],
    )
    cfg = load_audit_config(out)
    assert cfg.slots_in_scope == ["custom_slot_a", "custom_slot_b"]


def test_rebake_below_warn_validation():
    """rebake_threshold_pct > warn_threshold_pct must raise."""
    with pytest.raises(ValueError, match="rebake_threshold_pct"):
        AuditConfig(
            app_name="x",
            slots_in_scope=["s1"],
            warn_threshold_pct=90.0,
            rebake_threshold_pct=95.0,
            pricing=PricingTable(
                frontier_model="anthropic/claude-opus-4-7",
                entries=[
                    PricingEntry(
                        model="anthropic/claude-opus-4-7",
                        input_usd_per_1m=15.0,
                        output_usd_per_1m=75.0,
                    )
                ],
            ),
        )


def test_rebake_default_must_sit_below_explicit_warn(tmp_path: Path):
    """warn=70 with rebake OMITTED (default 80) must raise.

    Regression: a field_validator doesn't run on defaulted fields, so the
    warn>rebake invariant silently inverted whenever rebake_threshold_pct came
    from its default. The model_validator must enforce it always.
    """
    p = tmp_path / "warn-only.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "app_name": "x",
                "slots_in_scope": ["s1"],
                "warn_threshold_pct": 70.0,
                # rebake_threshold_pct omitted → default 80.0 >= warn → invalid
                "pricing": {
                    "frontier_model": "anthropic/claude-opus-4-7",
                    "entries": [
                        {
                            "model": "anthropic/claude-opus-4-7",
                            "input_usd_per_1m": 15.0,
                            "output_usd_per_1m": 75.0,
                        }
                    ],
                },
            }
        )
    )
    with pytest.raises(ValueError, match="rebake_threshold_pct"):
        load_audit_config(p)


def test_pricing_frontier_must_be_in_entries():
    """Looking up the frontier rate when it isn't in the table raises."""
    pt = PricingTable(
        frontier_model="anthropic/claude-opus-4-7",
        entries=[
            PricingEntry(model="ollama/qwen3:8b", input_usd_per_1m=0.0, output_usd_per_1m=0.0),
        ],
    )
    with pytest.raises(ValueError, match="frontier_model"):
        pt.frontier_entry()


def test_pricing_lookup_returns_none_for_unknown():
    pt = PricingTable(
        frontier_model="anthropic/claude-opus-4-7",
        entries=[
            PricingEntry(
                model="anthropic/claude-opus-4-7",
                input_usd_per_1m=15.0,
                output_usd_per_1m=75.0,
            )
        ],
    )
    assert pt.lookup("does-not-exist") is None


def test_load_audit_config_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_audit_config(tmp_path / "nope.yaml")


def test_load_audit_config_malformed_yaml(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("not: valid: yaml: [")
    with pytest.raises(ValueError, match="parse"):
        load_audit_config(p)


def test_load_audit_config_missing_required_field(tmp_path: Path):
    p = tmp_path / "missing.yaml"
    # Missing pricing.
    p.write_text(yaml.safe_dump({"app_name": "x", "slots_in_scope": ["s1"]}))
    with pytest.raises(Exception):  # ValidationError from pydantic
        load_audit_config(p)


def test_out_dir_resolved_against_anchor(tmp_path: Path):
    """Relative out_dir is resolved against the supplied anchor."""
    pt = PricingTable(
        frontier_model="anthropic/claude-opus-4-7",
        entries=[
            PricingEntry(
                model="anthropic/claude-opus-4-7",
                input_usd_per_1m=15.0,
                output_usd_per_1m=75.0,
            )
        ],
    )
    cfg = AuditConfig(
        app_name="x",
        slots_in_scope=["s1"],
        pricing=pt,
        out_dir=Path("relative/path"),
    )
    resolved = cfg.out_dir_resolved(tmp_path)
    assert resolved.is_absolute()
    assert "relative/path" in str(resolved)


def test_out_dir_resolved_absolute_passthrough(tmp_path: Path):
    pt = PricingTable(
        frontier_model="anthropic/claude-opus-4-7",
        entries=[
            PricingEntry(
                model="anthropic/claude-opus-4-7",
                input_usd_per_1m=15.0,
                output_usd_per_1m=75.0,
            )
        ],
    )
    abs_path = tmp_path / "abs"
    cfg = AuditConfig(
        app_name="x",
        slots_in_scope=["s1"],
        pricing=pt,
        out_dir=abs_path,
    )
    assert cfg.out_dir_resolved(tmp_path / "anchor") == abs_path
