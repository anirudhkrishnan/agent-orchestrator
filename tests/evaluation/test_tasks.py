"""Tests for evaluation.tasks loading + validation."""

from pathlib import Path

import pytest
import yaml

from orchestrator.evaluation.tasks import (
    QualityDimension,
    Scenario,
    TaskSpec,
    load_tasks_yaml,
)


def _minimal_task_dict() -> dict:
    """Smallest valid task dict — used as a starting point for failure tests."""
    return {
        "id": "entity_extraction",
        "description": "Extract named entities from a document.",
        "system_prompt": "You extract named entities.",
        "max_response_tokens": 256,
        "quality_dimensions": [
            {"name": "recall", "description": "catches real entities", "weight": 0.5},
            {"name": "precision", "description": "avoids false positives", "weight": 0.5},
        ],
        "scenarios": [
            {
                "id": "scn-01",
                "input": "Nvidia and AMD beat estimates.",
                "notes": "two real entities",
                "expected_output_shape": "JSON array of uppercased strings",
            }
        ],
    }


# --- Pydantic-level validation -------------------------------------------


def test_taskspec_round_trips_clean_dict():
    spec = TaskSpec.model_validate(_minimal_task_dict())
    assert spec.id == "entity_extraction"
    assert len(spec.scenarios) == 1
    assert spec.max_response_tokens == 256


def test_taskspec_defaults_max_tokens_to_512():
    d = _minimal_task_dict()
    del d["max_response_tokens"]
    spec = TaskSpec.model_validate(d)
    assert spec.max_response_tokens == 512


def test_taskspec_rejects_weights_not_summing_to_one():
    d = _minimal_task_dict()
    d["quality_dimensions"][0]["weight"] = 0.4
    # 0.4 + 0.5 = 0.9 → invalid
    with pytest.raises(ValueError, match="sum to"):
        TaskSpec.model_validate(d)


def test_taskspec_accepts_weights_within_float_tolerance():
    d = _minimal_task_dict()
    # Three dimensions of 1/3 each — float rep won't sum exactly to 1.0.
    d["quality_dimensions"] = [
        {"name": "a", "description": "a", "weight": 1.0 / 3.0},
        {"name": "b", "description": "b", "weight": 1.0 / 3.0},
        {"name": "c", "description": "c", "weight": 1.0 / 3.0},
    ]
    spec = TaskSpec.model_validate(d)
    assert len(spec.quality_dimensions) == 3


def test_taskspec_rejects_duplicate_scenario_ids():
    d = _minimal_task_dict()
    d["scenarios"].append(dict(d["scenarios"][0]))  # same id as scn-01
    with pytest.raises(ValueError, match="Duplicate scenario ids"):
        TaskSpec.model_validate(d)


def test_taskspec_rejects_duplicate_dimension_names():
    d = _minimal_task_dict()
    d["quality_dimensions"] = [
        {"name": "x", "description": "x", "weight": 0.5},
        {"name": "x", "description": "x2", "weight": 0.5},
    ]
    with pytest.raises(ValueError, match="Duplicate quality_dimension names"):
        TaskSpec.model_validate(d)


def test_quality_dimension_weight_bounds():
    with pytest.raises(ValueError):
        QualityDimension(name="x", description="x", weight=-0.1)
    with pytest.raises(ValueError):
        QualityDimension(name="x", description="x", weight=1.5)


def test_scenario_requires_nonempty_input():
    with pytest.raises(ValueError):
        Scenario(id="s", input="")


# --- load_tasks_yaml -----------------------------------------------------


def test_load_tasks_yaml_round_trips(tmp_path: Path):
    p = tmp_path / "tasks.yaml"
    p.write_text(yaml.safe_dump([_minimal_task_dict()]))
    tasks = load_tasks_yaml(p)
    assert len(tasks) == 1
    assert tasks[0].id == "entity_extraction"
    assert tasks[0].scenarios[0].id == "scn-01"


def test_load_tasks_yaml_rejects_non_list_root(tmp_path: Path):
    p = tmp_path / "tasks.yaml"
    p.write_text(yaml.safe_dump(_minimal_task_dict()))  # dict at root
    with pytest.raises(ValueError, match="must be a list"):
        load_tasks_yaml(p)


def test_load_tasks_yaml_rejects_empty_list(tmp_path: Path):
    p = tmp_path / "tasks.yaml"
    p.write_text("[]\n")
    with pytest.raises(ValueError, match="zero tasks"):
        load_tasks_yaml(p)


def test_load_tasks_yaml_rejects_duplicate_task_ids(tmp_path: Path):
    p = tmp_path / "tasks.yaml"
    p.write_text(yaml.safe_dump([_minimal_task_dict(), _minimal_task_dict()]))
    with pytest.raises(ValueError, match="Duplicate task ids"):
        load_tasks_yaml(p)


def test_load_tasks_yaml_raises_on_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_tasks_yaml(tmp_path / "missing.yaml")
