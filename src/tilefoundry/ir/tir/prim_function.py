from __future__ import annotations

from dataclasses import dataclass, field

from tilefoundry.ir.core import Var
from tilefoundry.ir.tir.stmt import Stmt
from tilefoundry.ir.tir.stmts import Sequential
from tilefoundry.target import CudaTarget, Target


@dataclass(frozen=True)
class PrimFunction(Stmt):
    """tir function container. No return value (@prim_func is effect-only).

    Inherits ``Stmt`` per tir.md §2 — PrimFunction sits inside the tir
    stmt tree rather than outside it. Body is a ``Sequential`` wrapper.

    ``output_count`` records the number of trailing output parameters (set
    by the lowering pass, consumed by codegen → CallableType).
    """
    name: str
    params: tuple[Var, ...]
    body: Sequential
    output_count: int = 1
    target: Target = field(default_factory=CudaTarget)


__all__ = ["PrimFunction"]
