"""Schema-validation test for data/routing.json.

Uses jsonschema (already a transitive dep via inspect-ai) to ensure routing.json
conforms to schemas/routing.schema.json. Run as part of pytest so an editing
mistake to routing.json by hand fails CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = ROOT / "schemas" / "routing.schema.json"
ROUTING = ROOT / "data" / "routing.json"


def test_routing_schema_file_exists():
    assert SCHEMA.exists(), f"Missing schema at {SCHEMA}"


def _slots(data: dict) -> dict:
    """Routing entries only — drop the _README in-band-docs sentinel."""
    return {k: v for k, v in data.items() if not k.startswith("_")}


def test_routing_json_exists_and_has_core_slots():
    """routing.json must contain (at least) the core orchestration slots.

    Superset check, not exact-set: the table grows as apps onboard new slots
    (updated 2026-05-28 — was a stale exact-5 assertion that had been red for
    weeks once the slots were baked + expanded).
    """
    assert ROUTING.exists(), f"Missing routing.json at {ROUTING}"
    data = _slots(json.loads(ROUTING.read_text()))
    core = {"entity_extraction", "summary_synthesis", "relevance_triage"}
    missing = core - set(data.keys())
    assert not missing, f"routing.json missing core slots: {missing}"


def test_routing_json_validates_against_schema():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA.read_text())
    data = json.loads(ROUTING.read_text())
    # Validates the _README sentinel + every slot entry against the v2 schema.
    jsonschema.validate(data, schema)


def test_every_slot_has_model_and_last_baked_at_key():
    """Structural invariant the plugin relies on: each slot has a `model` string
    and a `last_baked_at` key (value may be null for unbaked slots)."""
    data = _slots(json.loads(ROUTING.read_text()))
    assert data, "routing.json has no slots"
    for slot, entry in data.items():
        assert isinstance(entry.get("model"), str), f"{slot} missing string model"
        assert "last_baked_at" in entry, f"{slot} missing last_baked_at key"


def test_baked_slots_carry_quality_pct():
    """Completeness invariant: a slot with a non-null last_baked_at (i.e. it was
    measured) must carry a quality_pct_of_judge — even queue-for-human
    slots, whose bake-off ran and produced a 'no OSS pick' verdict."""
    data = _slots(json.loads(ROUTING.read_text()))
    for slot, entry in data.items():
        if entry.get("last_baked_at") is not None:
            assert entry.get("quality_pct_of_judge") is not None, (
                f"{slot} is baked (last_baked_at set) but has no quality_pct_of_judge"
            )
