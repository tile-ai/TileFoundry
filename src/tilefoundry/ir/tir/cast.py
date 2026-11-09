"""Effect-ful TIR dtype conversion operation."""

from __future__ import annotations

from dataclasses import replace

from tilefoundry.interfaces import access, typeinfer
from tilefoundry.ir import expr, pattern, types

EXECUTION_MESH = pattern.thread_execution_mesh()
IN_RMEM = pattern.tensor_in(expr.StorageKind.RMEM, EXECUTION_MESH)


@expr.register_op(dialect="T", category="arith", name="cast")
class Cast(expr.Op):
    """Convert a register tile to another dtype without changing its shape."""

    capability = expr.OpCapability(None)

    src = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ, pattern=IN_RMEM)
    dst = expr.ParamDef(
        kind="input",
        effect=expr.MemoryEffect.WRITE,
        pattern=replace(IN_RMEM, dtype=pattern.Wildcard("cast_dtype")),
    )
    dtype = expr.ParamDef(kind="attribute", annotation=expr.DType, pattern=pattern.Wildcard("cast_dtype"))
    between = (
        pattern.SameConstraint("shape", "src", "dst"),
        pattern.DistinctConstraint("dtype", "src", "dst"),
    )
    execution_mesh = EXECUTION_MESH

    def __init__(self, **attrs) -> None:
        dtype = attrs.get("dtype")
        if isinstance(dtype, str):
            attrs["dtype"] = expr.DType.from_name(dtype)
        super().__init__(**attrs)


@typeinfer.register(Cast)
def _(call: "Call", ctx: "TypeInferContext") -> types.UnitType:
    return types.UnitType()


access.register_relation(Cast)(access.identity_relations)


__all__ = ["Cast"]
