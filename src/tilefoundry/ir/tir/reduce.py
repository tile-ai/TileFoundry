"""Effect-form TIR Op ``tir.tensor.Reduce`` — axis reduction dispatched by ``ReduceKind`` tag."""

from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.kinds import ReduceKind
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import UnitType
from tilefoundry.visitor_registry import register_typeinfer, register_verify_stmt

__all__ = ["ReduceKind", "Reduce"]

@register_op(dialect="T", category="tensor")
class Reduce(Op):
    """Generic axis reduction; dispatched by the ``kind`` tag."""
    src = ParamDef(kind="input", pattern=Tensor)
    dst = ParamDef(kind="input", pattern=Tensor)
    workspace = ParamDef(
        kind="input", pattern=Tensor, optional=True, default=None
    )
    axes = ParamDef(kind="attribute", annotation=tuple)
    kind = ParamDef(kind="attribute", annotation=ReduceKind)

@register_typeinfer(Reduce)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()

@register_verify_stmt(Reduce)
def _(call: "Call", ctx: "VerifyContext") -> None:
    op = call.target
    if not isinstance(op.kind, ReduceKind):
        ctx.error(call, f"Reduce: kind must be ReduceKind enum, got {type(op.kind)}")
    src_ty = ctx.type_of(call.args[0])  # noqa: F841
    dst_ty = ctx.type_of(call.args[1])  # noqa: F841
    # Per-shard reshard lowering may produce rank-N (e.g.
    # ``(1, 1, 1, 8)``) src tensors. The runtime template
    # (``tilefoundry::ops::reduce<Op, Axes>``) iterates via
    # ``cute::size(src)`` so rank is no longer relevant at the verifier level —
    # the old rank<=2 guard predates the sharded reduce path.
