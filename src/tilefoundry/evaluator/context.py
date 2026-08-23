"""Eval context handed to a ``@register_eval`` handler."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tilefoundry.visitor_registry.contexts import CallFeed


@dataclass(frozen=True)
class EvalContext:
    """Operands + op + result type for one Op evaluation."""

    op: Any
    args: tuple[Any, ...]
    result_type: Any
    device: str = "cpu"
    dim_bindings: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.dim_bindings is None:
            object.__setattr__(self, "dim_bindings", {})


@dataclass(frozen=True)
class FunctionEvalContext:
    """Runtime state for one recursive function invocation."""

    feed: CallFeed[Any]
    loaded_module: Any | None = None
    device: str = "cpu"
    dim_bindings: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.dim_bindings is None:
            object.__setattr__(self, "dim_bindings", {})
