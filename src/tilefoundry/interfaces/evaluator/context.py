"""Evaluation context shared by recursive walks and registered Op handlers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class EvaluateContext:
    """Operands, Op-local values, and runtime state for one evaluation.

    *module* declares the running function and resolves the references it
    calls; *loaded_module* is the reading its instance calls take weights from.
    """

    op: Any = None
    args: tuple[Any, ...] = ()
    result_type: Any = None
    module: Any | None = None
    loaded_module: Any | None = None
    device: str | None = None
    dim_bindings: Mapping[str, int] = field(default_factory=dict)

    def for_op(self, op: Any, args: tuple[Any, ...], result_type: Any) -> EvaluateContext:
        """Add one Call's evaluated operands while preserving runtime state."""
        return replace(self, op=op, args=args, result_type=result_type)


__all__ = ["EvaluateContext"]
