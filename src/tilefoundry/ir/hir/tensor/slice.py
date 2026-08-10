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
from tilefoundry.ir.types.shard import (
    ComposedLayout,
    Layout,
    ShardLayout,
)
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    identity_relations,
    register_access_relation,
)


@register_op
class Slice(Op):
    x = ParamDef(kind="input", pattern=Tensor)
    begin = ParamDef(kind="attribute", annotation=tuple)
    end = ParamDef(kind="attribute", annotation=tuple)
    strides = ParamDef(kind="attribute", annotation=tuple)

    def __init__(self, **attrs):

        for key in ("begin", "end", "strides"):
            if key in attrs and isinstance(attrs[key], tuple):
                attrs[key] = tuple(
                    _i64(v) if isinstance(v, int) and not isinstance(v, bool) else v
                    for v in attrs[key]
                )
        super().__init__(**attrs)


register_access_relation(Slice)(identity_relations(1))


def _i64(value: int) -> Constant:
    return i64_const(value)


def _slice_dim(begin: Expr, end: Expr, stride: Expr) -> Expr:
    """Return ``max(0, ceil((end - begin) / stride))`` as a dimension Expr.

    Constant chains fold through ``simplify_dim``. A non-positive constant
    stride returns zero explicitly because generic arithmetic folding does not
    encode slice-domain semantics.
    """
    if isinstance(begin, Constant) and isinstance(end, Constant) and isinstance(stride, Constant):
        b, e, s = int(begin.value), int(end.value), int(stride.value)
        if s <= 0:
            return _i64(0)
        n = max(0, (e - b + s - 1) // s)
        return _i64(n)
    diff = simplify_dim(DimSub, (end, begin))

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
    layout_shape = tuple(int(dim.value) if isinstance(dim, Constant) else dim for dim in shape)
    source = x_ty.layout
    inherited_offset = 0
    if isinstance(source, ShardLayout):
        source = None
    elif (
        isinstance(source, ComposedLayout)
        and source.inner is None
        and isinstance(source.outer, Layout)
    ):
        inherited_offset = source.offset
        source = source.outer

    new_layout = None
    if isinstance(source, Layout) and source.strides is not None:
        starts = []
        steps = []
        for begin, end, stride in zip(op.begin, op.end, op.strides):
            if not (
                isinstance(begin, Constant)
                and isinstance(begin.value, int)
                and isinstance(end, Constant)
                and isinstance(end.value, int)
                and isinstance(stride, Constant)
                and isinstance(stride.value, int)
            ):
                break
            starts.append(int(begin.value))
            steps.append(int(stride.value))
        else:
            new_layout = ComposedLayout(
                inner=None,
                offset=inherited_offset
                + sum(start * stride for start, stride in zip(starts, source.strides)),
                outer=Layout(
                    shape=layout_shape,
                    strides=tuple(stride * step for stride, step in zip(source.strides, steps)),
                ),
            )
    return TensorType(shape=tuple(shape), dtype=x_ty.dtype, layout=new_layout, storage=x_ty.storage)


@register_eval(Slice)
def _eval_slice(ctx):
    op = ctx.op
    key = tuple(
        slice(int(b.value), int(e.value), int(s.value))
        for b, e, s in zip(op.begin, op.end, op.strides)
    )
    return TensorValue(data=ctx.args[0].data[key], type=ctx.result_type)
