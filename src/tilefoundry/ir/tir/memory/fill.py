"""Effect-ful TIR Op ``tir.memory.Fill``.

Fills ``tensor`` with the rank-0 scalar ``value`` (in-place memory
write). Wrapped by ``Evaluate(Fill, ...)`` in Stmt position.
"""

from __future__ import annotations

from tilefoundry.interfaces import access, typeinfer, verify
from tilefoundry.ir import expr, pattern, types


@expr.register_op(dialect='T', category='memory')
class Fill(expr.Op):
    """Fills ``tensor`` element-wise with ``value`` (rank-0 scalar)."""

    execution_mesh = pattern.memory_execution_mesh()

    tensor = expr.ParamDef(
        kind="input",
        effect=expr.MemoryEffect.WRITE,
        pattern=pattern.operand_tile(0, execution_mesh=execution_mesh),
    )
    value = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ, pattern=pattern.is_scalar_tensor())


@typeinfer.register(Fill)
def _(call: "Call", ctx: "TypeInferContext") -> types.UnitType:
    return types.UnitType()


access.register_relation(Fill)(access.identity_relations)


@verify.register(Fill)
def _(call: "Call", ctx: "verify.Context") -> None:
    t_ty = ctx.type_of(call.args[0])
    v_ty = ctx.type_of(call.args[1])
    if v_ty.dtype != t_ty.dtype and not (
        isinstance(call.args[1], expr.Constant) and call.args[1].value == 0
    ):
        ctx.error(call, f"Fill dtype mismatch: {v_ty.dtype} vs {t_ty.dtype}")


__all__ = ["Fill"]
