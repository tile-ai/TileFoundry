"""HIR reference interpreter — a codegen-independent value oracle."""
from __future__ import annotations

from typing import Any

from tilefoundry.evaluator.context import EvalContext
from tilefoundry.evaluator.registry import eval_registry, register_eval
from tilefoundry.evaluator.value import (
    EvalError,
    TensorValue,
    TupleValue,
    Value,
    as_layout_view,
    from_layout_view,
    to_torch_dtype,
)

__all__ = [
    "evaluate",
    "register_eval",
    "eval_registry",
    "Value",
    "TensorValue",
    "TupleValue",
    "EvalContext",
    "EvalError",
    "to_torch_dtype",
    "as_layout_view",
    "from_layout_view",
]

# `evaluate` lives in `interpreter.py`, which pulls in the IR visitor stack.
# Import lazily so op modules can `from tilefoundry.evaluator.registry import
# register_eval` while the IR package is still loading.


def __getattr__(name: str) -> Any:
    if name == "evaluate":
        import importlib  # noqa: PLC0415 — lazy to avoid an IR import cycle

        return importlib.import_module("tilefoundry.evaluator.interpreter").evaluate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
