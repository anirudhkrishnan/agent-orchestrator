"""Task taxonomy + scenario schema for the evaluation framework.

A `TaskSpec` declares a class of work (e.g. entity_extraction) with:
    - a shared system prompt used across all scenarios
    - a set of quality dimensions the judge scores on
    - a list of scenarios (concrete inputs to send the candidate)

Tasks are authored in YAML by the user — the loader round-trips that YAML into
the Pydantic models below. Code never hard-codes task content; the YAML is the
single source of truth so tasks evolve without touching Python.

YAML structure mirrors the Pydantic models 1:1::

    - id: entity_extraction
      description: Pull named entities out of a document snippet.
      system_prompt: |
        You are a precise entity-extraction assistant.
      max_response_tokens: 256
      quality_dimensions:
        - name: recall
          description: Did the model catch every real entity?
          weight: 0.5
        - name: precision
          description: Did the model avoid false positives?
          weight: 0.5
      scenarios:
        - id: scn-01
          input: "Acme Corp and Widget Inc both announced record results; IT spend was strong."
          notes: "IT is a false-positive trap."
          expected_output_shape: "JSON array of uppercased entity names"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class QualityDimension(BaseModel):
    """One scoring axis the judge evaluates.

    Weights across dimensions in a single task MUST sum to 1.0 (within 1e-6).
    The runner enforces this at load time so a malformed YAML doesn't silently
    skew the weighted mean.
    """

    name: str = Field(..., min_length=1, description="Axis identifier, e.g. 'accuracy'.")
    description: str = Field(..., min_length=1, description="What the judge looks for on this axis.")
    weight: float = Field(..., ge=0.0, le=1.0, description="Contribution to mean quality (0.0-1.0).")


class Scenario(BaseModel):
    """One concrete input sent to a candidate model.

    The judge sees `notes` and `expected_output_shape` as context — they
    explain WHY this scenario is interesting and what shape of response would
    be considered well-formed. They are NOT sent to the candidate; the candidate
    only sees the task system_prompt + this scenario's `input`.
    """

    id: str = Field(..., min_length=1, description="Stable identifier, e.g. 'scn-01'.")
    input: str = Field(..., min_length=1, description="The actual prompt body sent to the candidate.")
    notes: str | None = Field(default=None, description="Optional context for the judge.")
    expected_output_shape: str | None = Field(
        default=None,
        description="Free-text description, e.g. 'JSON array of strings' or '1-2 sentences max'.",
    )


class TaskSpec(BaseModel):
    """A single class of work to evaluate across all candidates."""

    id: str = Field(..., min_length=1, description="Task identifier, e.g. 'entity_extraction'.")
    description: str = Field(..., min_length=1, description="One-line summary suitable for routing.json.")
    system_prompt: str = Field(..., min_length=1, description="Shared system prompt for all scenarios.")
    quality_dimensions: list[QualityDimension] = Field(..., min_length=1)
    scenarios: list[Scenario] = Field(..., min_length=1)
    max_response_tokens: int = Field(default=512, ge=1, description="Cap on tokens per candidate call.")

    @field_validator("scenarios")
    @classmethod
    def _scenario_ids_unique(cls, scenarios: list[Scenario]) -> list[Scenario]:
        ids = [s.id for s in scenarios]
        if len(ids) != len(set(ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"Duplicate scenario ids: {dupes}")
        return scenarios

    @field_validator("quality_dimensions")
    @classmethod
    def _dimension_names_unique(cls, dims: list[QualityDimension]) -> list[QualityDimension]:
        names = [d.name for d in dims]
        if len(names) != len(set(names)):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"Duplicate quality_dimension names: {dupes}")
        return dims

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> "TaskSpec":
        total = sum(d.weight for d in self.quality_dimensions)
        # 1e-6 tolerance — YAML float rounding can drift the sum slightly.
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Task {self.id!r}: quality_dimension weights sum to {total!r}, expected 1.0. "
                f"Adjust weights in the YAML so they total exactly 1.0."
            )
        return self


def load_tasks_yaml(path: Path | str) -> list[TaskSpec]:
    """Load a list of TaskSpecs from a YAML file.

    The YAML's top-level structure must be a list of task objects (see module
    docstring). Accepts a Path or a str path. Raises:
        FileNotFoundError if `path` doesn't exist.
        ValueError if the YAML root isn't a list, or any task fails validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Task YAML not found: {path}")
    with path.open() as f:
        raw: Any = yaml.safe_load(f)
    if not isinstance(raw, list):
        raise ValueError(
            f"Task YAML root must be a list of tasks; got {type(raw).__name__} in {path}."
        )
    tasks = [TaskSpec.model_validate(item) for item in raw]
    if not tasks:
        raise ValueError(f"Task YAML at {path} contained zero tasks.")
    # Cross-task uniqueness check — duplicate task ids would break the runner's
    # output-path scheme (`{task_id}_{scenario_id}.json`).
    task_ids = [t.id for t in tasks]
    if len(task_ids) != len(set(task_ids)):
        dupes = sorted({i for i in task_ids if task_ids.count(i) > 1})
        raise ValueError(f"Duplicate task ids in {path}: {dupes}")
    return tasks
