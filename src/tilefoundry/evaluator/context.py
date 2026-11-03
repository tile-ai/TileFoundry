"""Evaluation context shared by recursive walks and registered Op handlers."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class EvaluateContext:
    """Operands, Op-local values, and runtime state for one evaluation."""

    op: Any = None
    args: tuple[Any, ...] = ()
    arg_types: tuple[Any, ...] = ()
    """The operands' global types; ``per_unit`` keeps them while it localizes ``args``."""
    result_type: Any = None
    loaded_module: Any | None = None
    device: str | None = None
    dim_bindings: Mapping[str, int] = field(default_factory=dict)

    def for_op(self, op: Any, args: tuple[Any, ...], result_type: Any) -> EvaluateContext:
        """Add one Call's evaluated operands while preserving runtime state."""
        types = tuple(getattr(arg, "type", None) for arg in args)
        return replace(self, op=op, args=args, arg_types=types, result_type=result_type)
