from __future__ import annotations

from dataclasses import dataclass, field

from tilefoundry.ir.core import Var
from tilefoundry.ir.tir.stmt import Stmt
from tilefoundry.ir.tir.stmts import Sequential
from tilefoundry.target.base import Target, target_instance


def _default_target():
    """Construct the compiler-owned default after the IR import graph is ready."""
    from tilefoundry.target import default_target  # noqa: PLC0415

    return default_target()


@dataclass(frozen=True)
class PrimFunction(Stmt):
    """tir function container. No return value (@prim_func is effect-only).

    Inherits ``Stmt`` per [tir §2](docs/spec/tir.md#2-tir-expr-and-callable-constructs) — PrimFunction sits inside the tir
    stmt tree rather than outside it. Body is a ``Sequential`` wrapper.

    ``output_count`` records the number of trailing output parameters (set
    by the lowering pass, consumed by codegen → CallableType).
    """
    name: str
    params: tuple[Var, ...]
    body: Sequential
    output_count: int = 1
    target: Target = field(default_factory=_default_target)

    def __post_init__(self) -> None:
        target_instance(self.target)


__all__ = ["PrimFunction"]
