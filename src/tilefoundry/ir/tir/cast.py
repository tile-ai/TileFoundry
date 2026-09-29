"""Effect-ful TIR dtype conversion operation."""

from __future__ import annotations

from tilefoundry.ir.core import Op, OpCapability
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import (
    DistinctConstraint,
    SameConstraint,
    utils,
)
from tilefoundry.ir.types import DType, StorageKind, UnitType
from tilefoundry.visitor_registry import register_typeinfer, register_verify_stmt
from tilefoundry.visitor_registry.access_relation import (
    identity_relations,
    register_access_relation,
)

_EXECUTION_MESH = utils.thread_execution_mesh()
_IN_RMEM = utils.tensor_in(StorageKind.RMEM, _EXECUTION_MESH)


@register_op(dialect="T", category="arith", name="cast")
class Cast(Op):
    """Convert a register tile to another dtype without changing its shape."""

    capability = OpCapability(None)

    src = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_IN_RMEM)
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=_IN_RMEM)
    dtype = ParamDef(kind="attribute", annotation=DType)
    between = (
        SameConstraint("shape", "src", "dst"),
        DistinctConstraint("dtype", "src", "dst"),
    )
    execution_mesh = _EXECUTION_MESH

    def __init__(self, **attrs) -> None:
        dtype = attrs.get("dtype")
        if isinstance(dtype, str):
            attrs["dtype"] = DType.from_name(dtype)
        super().__init__(**attrs)


@register_typeinfer(Cast)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


register_access_relation(Cast)(identity_relations(2))


@register_verify_stmt(Cast)
def _(call: "Call", ctx: "VerifyContext") -> None:
    op = call.target
    src = ctx.type_of(call.args[0])
    dst = ctx.type_of(call.args[1])
    if src.shape != dst.shape:
        ctx.error(call, f"Cast shape mismatch: {src.shape} vs {dst.shape}")
    if src.dtype == dst.dtype:
        ctx.error(call, f"Cast requires distinct dtypes, both are {src.dtype}")
    if op.dtype != dst.dtype:
        ctx.error(call, f"Cast dtype {op.dtype} does not match dst dtype {dst.dtype}")


__all__ = ["Cast"]
