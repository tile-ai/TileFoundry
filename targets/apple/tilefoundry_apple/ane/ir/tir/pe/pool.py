"""dst = max or average pooling of src on the processing engine."""

from math import prod
from types import MappingProxyType

from tilefoundry import utils
from tilefoundry.interfaces import access, evaluator, schedule, typeinfer, verify
from tilefoundry.ir import expr, pattern, types
from tilefoundry.ir.hir import nn
from tilefoundry_apple.ane.ir.tir import evaluation, layout

KINDS = (expr.PoolKind.MAX, expr.PoolKind.AVG)
KERNEL = pattern.Sequence(pattern.Wildcard("kernel_h"), pattern.Wildcard("kernel_w"))
PADDING = pattern.Sequence(
    pattern.Wildcard("padding_h", predicates=(pattern.Range(lo=0),)),
    pattern.Wildcard("padding_w", predicates=(pattern.Range(lo=0),)),
    predicates=(
        pattern.Wildcard("padding_h") < pattern.Wildcard("kernel_h"),
        pattern.Wildcard("padding_w") < pattern.Wildcard("kernel_w"),
    ),
)
SAME_AVERAGE = (
    pattern.In(pattern.Wildcard("kind"), (expr.PoolKind.MAX,))
    | pattern.In(pattern.Wildcard("ceil_mode"), (False,))
    | pattern.In(pattern.Wildcard("count_include_pad"), (False,))
)
BOUND = MappingProxyType(
    {
        **{
            name: pattern.Wildcard(name)
            for name in ("kernel_size", "stride", "padding", "ceil_mode", "kind")
        },
        "count_include_pad": pattern.Wildcard("count_include_pad", predicates=(SAME_AVERAGE,)),
    }
)


@expr.register_op(dialect="T", category="pe", namespace="ane", name="pool")
class Pool(expr.Op):
    """dst = pooling of src over kernel_size windows; padding is zero for the average.

    An average divides by the window size, or by the elements the window holds
    that are not padding, as a multiplication by the fp16 reciprocal. A ceil_mode
    window past the padded input still divides by the window size, so an HIR average
    with ceil_mode and count_include_pad is not paired.
    """

    capability = expr.OpCapability("pe.pool")
    execution_mesh = layout.ENGINE
    hir_matches = (
        schedule.HirMatch(nn.Pool2D, when=BOUND, operands={"dst": "result", "src": "input"}),
    )

    kernel_size = expr.ParamDef(kind="attribute", annotation=tuple, pattern=KERNEL)
    stride = expr.ParamDef(
        kind="attribute", annotation=tuple, pattern=layout.one_of("stride", layout.STRIDES)
    )
    padding = expr.ParamDef(kind="attribute", annotation=tuple, pattern=PADDING, default=(0, 0))
    ceil_mode = expr.ParamDef(
        kind="attribute", annotation=bool, pattern=pattern.Or(False, True), default=False
    )
    count_include_pad = expr.ParamDef(
        kind="attribute", annotation=bool, pattern=pattern.Or(False, True), default=True
    )
    kind = expr.ParamDef(
        kind="attribute", annotation=expr.PoolKind, pattern=layout.one_of("kind", KINDS)
    )
    tile_height = expr.ParamDef(kind="attribute", annotation=int, optional=True, default=None)
    dst = expr.ParamDef(kind="input", effect=expr.MemoryEffect.WRITE, pattern=layout.feature("dst"))
    src = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ, pattern=layout.feature("src"))


def windows(size: int, kernel: int, stride: int, padding: int, ceil_mode: bool) -> int:
    """The windows along one axis; ceil_mode drops a last window that starts past the input."""
    count = (size + 2 * padding - kernel + (stride - 1 if ceil_mode else 0)) // stride + 1
    return count - 1 if ceil_mode and (count - 1) * stride >= size + padding else count


@typeinfer.register(Pool)
def _(call, ctx) -> types.TupleType:
    return types.TupleType.Void


@access.register_relation(Pool)
def _(call, ctx):
    op = call.target
    hir = nn.Pool2D(
        kernel_size=op.kernel_size,
        stride=op.stride,
        padding=op.padding,
        ceil_mode=op.ceil_mode,
        count_include_pad=op.count_include_pad,
        kind=op.kind,
    )
    reached, written = access.relations_of(
        expr.Call(hir, call.args[1:], type=types.TupleType.Void), ctx
    )
    return written, reached, written


@verify.register(Pool)
def _(call, ctx) -> None:
    op = call.target
    dst, src = (ctx.type_of(arg) for arg in call.args)
    expected = (
        *src.shape[:2],
        *(
            windows(size, kernel, stride, padding, op.ceil_mode)
            for size, kernel, stride, padding in zip(
                src.shape[2:], op.kernel_size, op.stride, op.padding
            )
        ),
    )
    if tuple(dst.shape) != expected:
        ctx.error(call, f"dst shape {tuple(dst.shape)} differs from the pooled shape {expected}")


@evaluator.register(Pool)
def _(ctx):
    op = ctx.op
    src = ctx.args[1].data.to(utils.torch.float32)
    window = (op.kernel_size, op.stride, op.padding)
    if op.kind is expr.PoolKind.MAX:
        return evaluation.write(ctx, utils.F.max_pool2d(src, *window, ceil_mode=op.ceil_mode))
    sums = evaluation.f16(
        utils.F.avg_pool2d(src, *window, ceil_mode=op.ceil_mode, divisor_override=1)
    )
    count = utils.torch.tensor(float(prod(op.kernel_size)))
    if not op.count_include_pad:
        ones = utils.torch.ones_like(src[:1, :1])
        count = utils.F.avg_pool2d(ones, *window, ceil_mode=op.ceil_mode, divisor_override=1)
    return evaluation.write(ctx, evaluation.f16(sums * evaluation.f16(1 / count)))


__all__ = ["Pool"]
