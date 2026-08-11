"""Eval context handed to a ``@register_eval`` handler."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class EvalContext:
    """Operands + op + result type for one Op evaluation."""

    op: Any
    args: tuple[Any, ...]
    result_type: Any
    device: str = "cpu"
    eval_expr: Callable[[Any], Any] | None = None
