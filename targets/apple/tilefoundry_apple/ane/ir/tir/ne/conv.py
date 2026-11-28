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
SRC = layout.activation("src", layout.SMEM_ONLY)
DST = layout.activation("dst", layout.SMEM_ONLY)
WEIGHT = (layout.kernel_layout("weight"), layout.matrix_layout("weight", layout.WEIGHT_LIMITS))
ACCUMULATOR_EXPONENT = -16
ACCUMULATOR_STEP = 2.0**ACCUMULATOR_EXPONENT
FP16_MIN_EXPONENT = -14
ACCUMULATOR_LIMIT = 2.0**15


@expr.register_op(dialect="T", category="ne", namespace="ane", name="conv")
class Conv(expr.Op):
    """dst = nonlinear(conv2d(src, weight) + bias), the result rounded once to fp16.

    The sum is held in signed 16.16 fixed point: each product is rounded to 2**-16, a
    product whose operands' exponents sum below -16 is dropped, and a partial sum
    outside [-2**15, 2**15) makes the result infinite.

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
            operands={"src": "input", "weight": "weight", "dst": "result", "bias": "bias"},
        ),
        schedule.HirMatch(
            nn.MatMul,
            when={"a_layout": "MK", "b_layout": "KN"},
            operands={"src": "lhs", "weight": "rhs", "dst": "result"},
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
    patch = expr.ParamDef(kind="attribute", annotation=tuple, optional=True, default=None)
    width_code = expr.ParamDef(kind="attribute", annotation=int, optional=True, default=None)
    ocg_size = expr.ParamDef(kind="attribute", annotation=int, optional=True, default=None)
    omit_ocg = expr.ParamDef(kind="attribute", annotation=bool, optional=True, default=False)
    src = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ, pattern=SRC)
    weight = expr.ParamDef(
        kind="input",
        effect=expr.MemoryEffect.READ,
        pattern=layout.forms("weight", WEIGHT, layout.GMEM_ONLY),
    )
    dst = expr.ParamDef(kind="input", effect=expr.MemoryEffect.WRITE, pattern=DST)
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
    src, weight = (ctx.type_of(arg) for arg in call.args[:2])
    if len(src.shape) == 2:
        lhs, rhs, written = access.matmul_relations(src.shape, weight.shape, MATRIX_AXES)
        rows, columns = src.shape[0], weight.shape[1]
        reads = (lhs, rhs, written.compose(access.broadcast((rows, columns), (columns,))))
    else:
        hir = nn.Conv2D(
            stride=op.stride, padding=op.padding, dilation=op.dilation, groups=op.groups
        )
        *reads, written = access.relations_of(
            expr.Call(hir, (*call.args[:2], *call.args[3:]), type=types.TupleType.Void), ctx
        )
    return *reads[:2], written, *reads[2 : len(call.args) - 1], written


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
    layout.require_innermost_contiguous(call, ctx, "weight", weight)
    layout.require_strides_or_dense(call, ctx, "dst", dst, (1, rows), (columns, 1))
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
    src, weight, dst, *held = (ctx.type_of(arg) for arg in call.args)
    ranks = tuple(len(type_.shape) for type_ in (dst, src, weight))
    if len(set(ranks)) != 1:
        ctx.error(
            call,
            f"dst, src and weight shapes must all be feature maps or all matrices, got ranks {ranks}",
        )
    check = check_matrix if ranks[0] == 2 else check_feature
    check(call, ctx, dst, src, weight, held[0] if held else None)


def exponent(value):
    """The binary exponent an fp16 operand carries; subnormals share the smallest normal's."""
    return utils.torch.frexp(value)[1].sub(1).clamp_min(FP16_MIN_EXPONENT)


def accumulate(inputs, weights, axis: int):
    """Sum inputs * weights along *axis* as the engine's signed 16.16 fixed-point accumulator does.

    Each product is rounded to a multiple of 2**-16, a tie away from zero, and the
    rounded products add exactly in index order. A product whose operands' exponents
    sum below -16 contributes nothing, whatever its mantissas. The first partial sum
    outside [-2**15, 2**15) makes the result infinite with that sum's sign.
    """
    torch = utils.torch
    products = inputs * weights
    if (products.abs() >= ACCUMULATOR_LIMIT).any():
        raise evaluator.EvalError("T.ane.conv: a product of magnitude 2**15 or more is not decoded yet")
    steps = torch.copysign(torch.floor(products.abs() / ACCUMULATOR_STEP + 0.5), products)
    kept = exponent(inputs) + exponent(weights) >= ACCUMULATOR_EXPONENT
    partial = torch.where(kept, steps, 0.0).cumsum(axis) * ACCUMULATOR_STEP
    outside = (partial >= ACCUMULATOR_LIMIT) | (partial < -ACCUMULATOR_LIMIT)
    first = outside.to(torch.int8).argmax(axis, keepdim=True)
    overflow = torch.copysign(torch.full_like(partial, torch.inf), partial).gather(axis, first)
    total = partial.select(axis, -1)
    return torch.where(outside.any(axis), overflow.squeeze(axis), total)


@evaluator.register(Conv)
def _(ctx):
    op = ctx.op
    torch = utils.torch
    src, weight, dst, *held = ctx.args
    src, weight, *held = (arg.data.to(torch.float64) for arg in (src, weight, *held))
    if src.dim() == 4:
        columns = utils.F.unfold(src, weight.shape[2:], padding=op.padding, stride=op.stride)
        columns = columns.unflatten(1, (op.groups, -1))
        kernels = weight.reshape(op.groups, weight.shape[0] // op.groups, -1)
        value = torch.stack(
            [
                accumulate(columns[:, group], kernel[:, None], 1)
                for group in range(op.groups)
                for kernel in kernels[group]
            ],
            1,
        )
        height = (src.shape[2] + 2 * op.padding[0] - weight.shape[2]) // op.stride[0] + 1
        value = value.unflatten(2, (height, -1))
        bias = held[0].reshape(1, -1, 1, 1) if held else 0.0
    else:
        value = accumulate(src[:, :, None], weight, 1)
        bias = held[0] if held else 0.0
    return evaluation.write(dst, apply(op.nonlinear, evaluation.f16((value + bias).to(torch.float32))))


__all__ = ["Conv"]
