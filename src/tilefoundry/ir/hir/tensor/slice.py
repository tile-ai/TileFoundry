from __future__ import annotations

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Expr, Op
from tilefoundry.ir.core.expr import Call, Constant
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.dim import DimAdd, DimFloorDiv, DimSub, simplify_dim
from tilefoundry.ir.types.shape_helpers import i64_const
from tilefoundry.visitor_registry import register_typeinfer


@register_op
class Slice(Op):
    x = ParamDef(kind="input", pattern=Tensor)
    begin = ParamDef(kind="attribute", annotation=tuple)
    end = ParamDef(kind="attribute", annotation=tuple)
    strides = ParamDef(kind="attribute", annotation=tuple)
    def __init__(self, **attrs):
        # Lift Python ints in begin/end/strides to i64 Constants so DSL
        # callers can pass plain int tuples (e.g. ``slice(x, begin=(0, 0),
        # end=(1, 4096), strides=(1, 1))``) without manually wrapping each
        # bound. Symbolic Expr / Constant values pass through unchanged.
        for key in ("begin", "end", "strides"):
            if key in attrs and isinstance(attrs[key], tuple):
                attrs[key] = tuple(
                    _i64(v) if isinstance(v, int) and not isinstance(v, bool) else v
                    for v in attrs[key]
                )
        super().__init__(**attrs)

def _i64(value: int) -> Constant:
    return i64_const(value)

def _slice_dim(begin: Expr, end: Expr, stride: Expr) -> Expr:
    """`(end - begin + stride - 1) // stride` — ceil-div — to correctly handle
    the final partial window when `(end - begin) % stride != 0`.

    Construction-time folding via ``simplify_dim`` collapses
    all-Constant chains to a single ``Constant`` bottom-up. The
    explicit non-positive-stride guard remains because stride <= 0
    is a domain-level edge case (not algebraic folding) — without
    the guard ``simplify_dim`` would either preserve the Call
    (stride == 0) or floor-div with a negative stride, neither of
    which matches the slice semantics ``max(0, ceil((e - b) / s))``.
    """
    if (
        isinstance(begin, Constant)
        and isinstance(end, Constant)
        and isinstance(stride, Constant)
    ):
        b, e, s = int(begin.value), int(end.value), int(stride.value)
        if s <= 0:
            return _i64(0)
        n = max(0, (e - b + s - 1) // s)
        return _i64(n)
    diff = simplify_dim(DimSub, (end, begin))
    # (diff + stride - 1) // stride
    bump = simplify_dim(
        DimAdd,
        (diff, simplify_dim(DimSub, (stride, _i64(1)))),
    )
    return simplify_dim(DimFloorDiv, (bump, stride))

@register_typeinfer(Slice)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    op = call.target
    rank = len(x_ty.shape)
    if not (len(op.begin) == len(op.end) == len(op.strides) == rank):
        ctx.error(call, f"Slice begin/end/strides rank must match input rank {rank}")
    shape = []
    for b, e, s in zip(op.begin, op.end, op.strides):
        b_e = b if isinstance(b, Expr) else _i64(int(b))
        e_e = e if isinstance(e, Expr) else _i64(int(e))
        s_e = s if isinstance(s, Expr) else _i64(int(s))
        shape.append(_slice_dim(b_e, e_e, s_e))
    # A sliced sharded axis changes its per-shard extent, which cannot in
    # general be re-expressed; drop to an unsharded output rather than carry the
    # input layout onto the smaller shape. Re-expressing a slice of a sharded
    # axis is left to a follow-up.
    return TensorType(
        shape=tuple(shape), dtype=x_ty.dtype, layout=None, storage=x_ty.storage
    )


@register_eval(Slice)
def _eval_slice(ctx):
    op = ctx.op
    key = tuple(
        slice(int(b.value), int(e.value), int(s.value))
        for b, e, s in zip(op.begin, op.end, op.strides)
    )
    return TensorValue(data=ctx.args[0].data[key], type=ctx.result_type)
