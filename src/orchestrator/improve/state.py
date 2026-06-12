"""Shared state-file loading for the self-improving loops.

Loop state lives in small JSON files (staged scenarios, the re-bake queue,
seen research claims, routing files read for detection). A corrupt or
hand-mangled file must surface as a clear, actionable error — which file,
what's wrong, how to recover — never as a raw json traceback. The CLI catches
StateFileError and exits non-zero with the message.
"""

from __future__ import annotations

import json
from pathlib import Path


class StateFileError(RuntimeError):
    """A JSON state/input file is unreadable or has the wrong shape.

    Carries which file, what is wrong, and how to recover, so the CLI can
    print an actionable message (and exit non-zero) instead of a traceback.
    """

    def __init__(self, path: Path | str, problem: str, how_to_fix: str):
        self.path = Path(path)
        self.problem = problem
        self.how_to_fix = how_to_fix
        super().__init__(f"{self.path}: {problem}. How to fix: {how_to_fix}")


def load_json_state(path: Path | str, *, expect: type | None = None, how_to_fix: str):
    """Read + parse a JSON state file, converting failure into StateFileError.

    Args:
        path: the file to read. Existence checks stay at call sites — a missing
            state file usually legitimately means "first run".
        expect: optional required top-level type (dict or list).
        how_to_fix: recovery instruction, surfaced to the user verbatim.

    Raises:
        StateFileError: unreadable file, invalid JSON, or wrong top-level type.
    """
    path = Path(path)
    try:
        raw = path.read_text()
    except OSError as e:
        raise StateFileError(path, f"cannot read file ({e.strerror or e})", how_to_fix) from e
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise StateFileError(
            path, f"invalid JSON at line {e.lineno} column {e.colno} ({e.msg})", how_to_fix
        ) from e
    if expect is not None and not isinstance(data, expect):
        raise StateFileError(
            path,
            f"expected top-level JSON {expect.__name__}, found {type(data).__name__}",
            how_to_fix,
        )
    return data
