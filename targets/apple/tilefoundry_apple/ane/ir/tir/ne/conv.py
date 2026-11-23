"""dst = nonlinear(conv2d(src, weight) + bias) as one neural-engine convolution task."""

from types import MappingProxyType

from tilefoundry import utils
from tilefoundry.interfaces import access, evaluator, schedule, typeinfer, verify
from tilefoundry.ir import expr, pattern, types
from tilefoundry.ir.hir import nn
from tilefoundry_apple.ane.ir.tir import evaluation, layout

from .nonlinear import Nonlinear, apply

DILATION = (1, 1)
MATRIX_AXES = (-2, -1, -1, -2)
BOUND = MappingProxyType(
    {name: pattern.Wildcard(name) for name in ("stride", "padding", "dilation", "groups")}
)
PADDING = pattern.Sequence(
    *(pattern.Wildcard(f"padding_{axis}", predicates=(pattern.Range(lo=0),)) for axis in "hw")
)
GROUPS = pattern.Wildcard("groups", predicates=(pattern.Range(lo=1),))
SRC = (layout.feature_layout("src"), layout.matrix_layout("src", layout.ACTIVATION_LIMITS))
DST = (layout.feature_layout("dst"), layout.matrix_layout("dst", layout.ACTIVATION_LIMITS))
WEIGHT = (layout.kernel_layout("weight"), layout.matrix_layout("weight", layout.WEIGHT_LIMITS))


@expr.register_op(dialect="T", category="ne", namespace="ane", name="conv")
class Conv(expr.Op):
    """dst = nonlinear(conv2d(src, weight) + bias), the result rounded once to fp16.

    The feature form convolves [1, C, H, W] with [O, C / groups, KH, KW] weights. The
    matrix form multiplies the channel-outer [M, K] src (K channels, M wide) by the
    row-major [K, N] weight, with stride (1, 1), no padding and one group.
    """

    capability = expr.OpCapability("ne.conv")
    execution_mesh = layout.ENGINE
    hir_matches = (
        schedule.HirMatch(
            nn.Conv2D,
            when=BOUND,
            operands={"dst": "result", "src": "input", "weight": "weight", "bias": "bias"},
        ),
        schedule.HirMatch(
            nn.MatMul,
            when={"a_layout": "MK", "b_layout": "KN"},
            operands={"dst": "result", "src": "lhs", "weight": "rhs"},
        ),
    )

    stride = expr.ParamDef(
        kind="attribute",
        annotation=tuple,
        pattern=layout.one_of("stride", layout.STRIDES),
        default=(1, 1),
    )
    padding = expr.ParamDef(kind="attribute", annotation=tuple, pattern=PADDING, default=(0, 0))
    dilation = expr.ParamDef(kind="attribute", annotation=tuple, pattern=DILATION, default=DILATION)
    groups = expr.ParamDef(kind="attribute", annotation=int, pattern=GROUPS, default=1)
    nonlinear = expr.ParamDef(
        kind="attribute",
        annotation=Nonlinear,
        pattern=pattern.Wildcard("nonlinear"),
        default=Nonlinear.NONE,
    )
    tile_height = expr.ParamDef(kind="attribute", annotation=int, optional=True, default=None)
    dst = expr.ParamDef(
        kind="input", effect=expr.MemoryEffect.WRITE, pattern=layout.forms("dst", DST)
    )
    src = expr.ParamDef(
        kind="input", effect=expr.MemoryEffect.READ, pattern=layout.forms("src", SRC)
    )
    weight = expr.ParamDef(
        kind="input",
        effect=expr.MemoryEffect.READ,
        pattern=layout.forms("weight", WEIGHT, layout.GMEM_ONLY),
    )
    bias = expr.ParamDef(
        kind="input",
        effect=expr.MemoryEffect.READ,
        optional=True,
        default=None,
        pattern=pattern.Tensor(
            shape=(layout.extent("bias.shape[0]", layout.MAX_CHANNELS),),
            dtype=layout.one_of("dtype", layout.FLOATS),
            storage=layout.stored("bias", layout.GMEM_ONLY),
        ),
    )


@typeinfer.register(Conv)
def _(call, ctx) -> types.TupleType:
    return types.TupleType.Void


@access.register_relation(Conv)
def _(call, ctx):
    op = call.target
    src, weight = (ctx.type_of(arg) for arg in call.args[1:3])
    if len(src.shape) == 2:
        lhs, rhs, written = access.matmul_relations(src.shape, weight.shape, MATRIX_AXES)
        rows, columns = src.shape[0], weight.shape[1]
        reads = (lhs, rhs, written.compose(access.broadcast((rows, columns), (columns,))))
    else:
        hir = nn.Conv2D(
            stride=op.stride, padding=op.padding, dilation=op.dilation, groups=op.groups
        )
        *reads, written = access.relations_of(
            expr.Call(hir, call.args[1:], type=types.TupleType.Void), ctx
        )
    return written, *reads[: len(call.args) - 1], written


def spatial(size: int, kernel: int, stride: int, padding: int) -> int:
    """The windows of *kernel* that fit *size* extended by *padding* on both sides."""
    return (size + 2 * padding - kernel) // stride + 1


def check_feature(call, ctx, dst, src, weight, bias) -> None:
    """Refuse a feature-form call whose channels, groups, padding or dst shape do not agree."""
    op = call.target
    channels, per_group, *kernel = weight.shape
    if src.shape[1] % op.groups or channels % op.groups:
        ctx.error(
            call,
            f"groups {op.groups} must divide the input channels {src.shape[1]} "
            f"and the output channels {channels}",
        )
    if per_group * op.groups != src.shape[1]:
        ctx.error(
            call,
            f"weight shape {tuple(weight.shape)} takes {per_group} channels per group, "
            f"but src has {src.shape[1]} over {op.groups} groups",
        )
    if bias is not None and tuple(bias.shape) != (channels,):
        ctx.error(
            call, f"bias shape {tuple(bias.shape)} differs from the output channels ({channels},)"
        )
    if any(padding >= extent for padding, extent in zip(op.padding, kernel)):
        ctx.error(call, f"padding {tuple(op.padding)} must be less than the kernel {tuple(kernel)}")
    expected = (
        1,
        channels,
        *(
            spatial(size, extent, stride, padding)
            for size, extent, stride, padding in zip(src.shape[2:], kernel, op.stride, op.padding)
        ),
    )
    if tuple(dst.shape) != expected:
        ctx.error(call, f"dst shape {tuple(dst.shape)} differs from the convolved shape {expected}")


def check_matrix(call, ctx, dst, src, weight, bias) -> None:
    """Refuse a matrix-form call that is not a plain product with agreeing shapes."""
    op = call.target
    for name, value, plain in (
        ("stride", op.stride, (1, 1)),
        ("padding", op.padding, (0, 0)),
        ("groups", op.groups, 1),
    ):
        if value != plain:
            ctx.error(call, f"{name} {value} must be {plain} for a matrix product")
    (rows, depth), (inner, columns) = src.shape, weight.shape
    layout.require_strides(call, ctx, "src", src, (1, rows))
    layout.require_strides(call, ctx, "weight", weight, (columns, 1))
    layout.require_strides(call, ctx, "dst", dst, (1, rows), (columns, 1))
    if depth != inner:
        ctx.error(
            call, f"src shape {tuple(src.shape)} and weight shape {tuple(weight.shape)} differ in K"
        )
    if bias is not None and tuple(bias.shape) != (columns,):
        ctx.error(call, f"bias shape {tuple(bias.shape)} differs from the columns ({columns},)")
    if tuple(dst.shape) != (rows, columns):
        ctx.error(
            call, f"dst shape {tuple(dst.shape)} differs from the product shape {(rows, columns)}"
        )


@verify.register(Conv)
def _(call, ctx) -> None:
    dst, src, weight, *held = (ctx.type_of(arg) for arg in call.args)
    ranks = tuple(len(type_.shape) for type_ in (dst, src, weight))
    if len(set(ranks)) != 1:
        ctx.error(
            call,
            f"dst, src and weight shapes must all be feature maps or all matrices, got ranks {ranks}",
        )
    check = check_matrix if ranks[0] == 2 else check_feature
    check(call, ctx, dst, src, weight, held[0] if held else None)


@evaluator.register(Conv)
def _(ctx):
    op = ctx.op
    src, weight, *held = (arg.data.to(utils.torch.float32) for arg in ctx.args[1:])
    bias = held[0] if held else None
    if src.dim() == 4:
        value = utils.F.conv2d(
            src, weight, bias, stride=op.stride, padding=op.padding, groups=op.groups
        )
    else:
        value = src @ weight if bias is None else src @ weight + bias
    return evaluation.write(ctx, apply(op.nonlinear, evaluation.f16(value)))


__all__ = ["Conv"]
