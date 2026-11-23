"""dst = a reduction of src over feature-map axes on the processing engine."""

from enum import Enum
from types import MappingProxyType

from tilefoundry import utils
from tilefoundry.interfaces import access, evaluator, schedule, typeinfer, verify
from tilefoundry.ir import expr, pattern, types
from tilefoundry.ir.hir import tensor
from tilefoundry_apple.ane.ir.tir import evaluation, layout


@expr.register_type(dialect="T", namespace="ane")
class Post(Enum):
    """What the engine applies to a reduction result: identity, 1/x, sqrt(x) or 1/sqrt(x)."""

    NONE = "none"
    INVERSE = "inverse"
    SQRT = "sqrt"
    RSQRT = "rsqrt"


AXES = ((2, 3), (3,), (1, 2, 3), (1, 3))
KINDS = (expr.ReduceKind.SUM, expr.ReduceKind.MAX, expr.ReduceKind.MIN)
REDUCTIONS = MappingProxyType(
    {
        expr.ReduceKind.SUM: lambda value, axes, keepdim: value.sum(axes, keepdim),
        expr.ReduceKind.MAX: lambda value, axes, keepdim: value.amax(axes, keepdim),
        expr.ReduceKind.MIN: lambda value, axes, keepdim: value.amin(axes, keepdim),
    }
)
POSTS = MappingProxyType(
    {
        Post.NONE: lambda value: value,
        Post.INVERSE: lambda value: value.reciprocal(),
        Post.SQRT: lambda value: value.sqrt(),
        Post.RSQRT: lambda value: value.rsqrt(),
    }
)
KEEPDIM = pattern.Wildcard(
    "keepdim",
    predicates=(
        pattern.In(pattern.Wildcard("keepdim"), (True,))
        | pattern.In(pattern.Wildcard("axes"), ((2, 3),)),
    ),
)
RESULT = layout.feature("dst")
BOUND = MappingProxyType({name: pattern.Wildcard(name) for name in ("axes", "keepdim", "kind")})


@expr.register_op(dialect="T", category="pe", namespace="ane", name="reduce")
class Reduce(expr.Op):
    """dst = post(scale * reduce(square(src))) over axes, each step rounded to fp16.

    keepdim=False drops the axes (2, 3); the other axes forms keep them as 1.
    """

    capability = expr.OpCapability("pe.reduce")
    execution_mesh = layout.ENGINE
    hir_matches = (
        schedule.HirMatch(tensor.Reduce, when=BOUND, operands={"dst": "result", "src": "x"}),
    )

    axes = expr.ParamDef(kind="attribute", annotation=tuple, pattern=layout.one_of("axes", AXES))
    keepdim = expr.ParamDef(kind="attribute", annotation=bool, pattern=KEEPDIM)
    kind = expr.ParamDef(
        kind="attribute", annotation=expr.ReduceKind, pattern=layout.one_of("kind", KINDS)
    )
    square = expr.ParamDef(
        kind="attribute", annotation=bool, pattern=pattern.Or(False, True), default=False
    )
    scale = expr.ParamDef(kind="attribute", annotation=float, default=1.0)
    post = expr.ParamDef(kind="attribute", annotation=Post, default=Post.NONE)
    dst = expr.ParamDef(
        kind="input",
        effect=expr.MemoryEffect.WRITE,
        pattern=pattern.Switch(
            "keepdim",
            {
                True: RESULT,
                False: RESULT.replace(
                    shape=(1, layout.extent("dst.shape[1]", layout.MAX_CHANNELS)), layout=None
                ),
            },
        ),
    )
    tile_height = expr.ParamDef(kind="attribute", annotation=int, optional=True, default=None)
    tile_width = expr.ParamDef(kind="attribute", annotation=int, optional=True, default=None)
    patch = expr.ParamDef(kind="attribute", annotation=tuple, optional=True, default=None)
    width_code = expr.ParamDef(kind="attribute", annotation=int, optional=True, default=None)
    src = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ, pattern=layout.feature("src"))


@typeinfer.register(Reduce)
def _(call, ctx) -> types.TupleType:
    return types.TupleType.Void


@access.register_relation(Reduce)
def _(call, ctx):
    op = call.target
    hir = tensor.Reduce(axes=op.axes, keepdim=op.keepdim, kind=op.kind)
    read, written = access.relations_of(
        expr.Call(hir, call.args[1:], type=types.TupleType.Void), ctx
    )
    return written, read, written


def reduced_shape(op, shape) -> tuple:
    """The shape of *shape* reduced over axes; keepdim leaves the reduced axes as 1."""
    return tuple(
        1 if axis in op.axes else extent
        for axis, extent in enumerate(shape)
        if op.keepdim or axis not in op.axes
    )


@verify.register(Reduce)
def _(call, ctx) -> None:
    dst, src = (ctx.type_of(arg) for arg in call.args)
    expected = reduced_shape(call.target, src.shape)
    if tuple(dst.shape) != expected:
        ctx.error(call, f"dst shape {tuple(dst.shape)} differs from the reduced shape {expected}")


@evaluator.register(Reduce)
def _(ctx):
    op = ctx.op
    value = ctx.args[1].data.to(utils.torch.float32)
    if op.square:
        value = evaluation.f16(value * value)
    reduced = evaluation.f16(REDUCTIONS[op.kind](value, op.axes, op.keepdim))
    return evaluation.write(ctx, evaluation.f16(POSTS[op.post](evaluation.f16(op.scale * reduced))))


__all__ = ["Post", "Reduce"]
