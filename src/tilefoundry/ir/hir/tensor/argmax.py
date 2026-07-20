"""ArgMax HIR primitive.

SGLang baseline kernel H3 (greedy sampling). Returns int64 indices along the
reduction axis; ``keepdim=False``.

"""
from __future__ import annotations

import isl

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir._shard_checks import reject_partials
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelations,
    register_access_relation,
)


@register_op
class ArgMax(Op):
    x = ParamDef(kind="input", pattern=Tensor)
    axis = ParamDef(kind="attribute", annotation=int, default=-1)
@register_typeinfer(ArgMax)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    if not x_ty.shape:
        ctx.error(call, "x must be at least rank-1")
    rank = len(x_ty.shape)
    axis = call.target.axis
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        ctx.error(call, f"axis {call.target.axis} out of range for rank {rank}")
    # The winning index is not recoverable from a partial (per-shard) reduction.
    reject_partials(ctx, call, "x", x_ty.layout)
    out_shape = tuple(d for i, d in enumerate(x_ty.shape) if i != axis)
    return TensorType(
        shape=out_shape,
        dtype=DType.i64,
        layout=x_ty.layout,
        storage=x_ty.storage,
    )

@register_access_relation(ArgMax)
def _argmax_access_relation(call: "Call", ctx: "TypeInferContext") -> AccessRelations:
    """GLOBAL: input scanned over the reduction axis (isl.map). Output is
    identity over the leading dims (axis collapsed away)."""
    x_ty = ctx.type_of(call.args[0])
    rank = len(x_ty.shape)
    axis = call.target.axis
    if axis < 0:
        axis += rank
    in_dims = ", ".join(f"i{i}" for i in range(rank))
    leading = [f"i{i}" for i in range(rank) if i != axis]
    out_dims = ", ".join(leading) if leading else ""
    if out_dims:
        in_rel = isl.map(f"{{ [{out_dims}] -> [{in_dims}] }}")
        out_id = isl.multi_aff(f"{{ [{out_dims}] -> [{out_dims}] }}")
    else:
        # rank-1 input → scalar output (rank-0 not constructible; ArgMax of
        # a rank-1 vector returns a scalar i64; degenerate ISL form).
        in_rel = isl.map(f"{{ [] -> [{in_dims}] }}")
        out_id = isl.multi_aff("{ [] -> [] }")
    return AccessRelations(inputs=(in_rel,), outputs=(out_id,))

__all__ = ["ArgMax"]
