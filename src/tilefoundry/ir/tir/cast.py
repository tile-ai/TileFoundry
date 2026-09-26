"""Effect-ful TIR dtype conversion operation."""

from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import (
    DistinctConstraint,
    SameConstraint,
    TensorPattern,
    any_threads,
)
from tilefoundry.ir.types import StorageKind, UnitType
from tilefoundry.visitor_registry import register_typeinfer

_REGISTER = TensorPattern(storage=StorageKind.RMEM)


@register_op(dialect="T", category="arith", name="cast")
class Cast(Op):
    """Convert a register tile to another dtype without changing its shape."""

    src = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_REGISTER)
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=_REGISTER)
    between = (
        SameConstraint("shape", "src", "dst"),
        DistinctConstraint("dtype", "src", "dst"),
    )
    scope = any_threads()


@register_typeinfer(Cast)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


__all__ = ["Cast"]
