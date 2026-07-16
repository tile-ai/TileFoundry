from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Protocol

from .constraints import ConstraintList
from .cost import CostTable
from .graph import ScheduleGraph
from .solution import ScheduleSolution
from .space import ScheduleSpace


@dataclass(frozen=True, slots=True)
class SolveOptions:
    deterministic: bool = True


class ScheduleSolver(Protocol):
    def solve(self, problem: "SolveProblem", options: SolveOptions) -> ScheduleSolution:
        ...


def _stable(value):
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("fingerprints cannot contain non-finite floats")
        return value
    if isinstance(value, Enum):
        return {"enum": type(value).__qualname__, "value": _stable(value.value)}
    if isinstance(value, (tuple, list)):
        return [_stable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _stable(item) for key, item in sorted(value.items(), key=lambda p: str(p[0]))}
    if is_dataclass(value):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {field.name: _stable(getattr(value, field.name)) for field in fields(value)},
        }
    if isinstance(value, type):
        return {"type": f"{value.__module__}.{value.__qualname__}"}
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}", "repr": repr(value)}


def problem_fingerprint(
    graph: ScheduleGraph,
    space: ScheduleSpace,
    costs: CostTable,
    constraints: ConstraintList,
) -> str:
    payload = _stable((graph, space, costs, constraints))
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class SolveProblem:
    graph: ScheduleGraph
    space: ScheduleSpace
    costs: CostTable
    constraints: ConstraintList
    problem_fingerprint: str

    def __post_init__(self) -> None:
        if not self.problem_fingerprint:
            raise ValueError("SolveProblem requires a problem fingerprint")


__all__ = [
    "ScheduleSolver",
    "SolveOptions",
    "SolveProblem",
    "problem_fingerprint",
]
