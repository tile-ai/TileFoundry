"""Lower ANE TIR statements to intrinsic sequences.

An emitter turns one ``T.ane.*`` call into the hardware primitives of a single
engine pass: read the operands, run the arithmetic, write the result.  Nothing
here knows about bit positions or about descriptor framing; both live in the ISA
layer and in :mod:`.serialize`.
"""

from collections.abc import Callable, Mapping

from tilefoundry import dispatch, target
from tilefoundry.ir import core, expr, types
from tilefoundry_apple.ane import isa
from tilefoundry_apple.ane import target as ane_target
from tilefoundry_apple.ane.ir import tir
from tilefoundry_apple.ane.ir.tir.ne import nonlinear as tir_nonlinear
from tilefoundry_apple.ane.isa import intrinsic

from . import serialize

#: The activation a neural engine pass selects.
NONLINEAR = {
    tir_nonlinear.Nonlinear.NONE: isa.NENonlinear.NONE,
    tir_nonlinear.Nonlinear.RELU: isa.NENonlinear.RELU,
}

registry = dispatch.Registry[type, Callable[[expr.Call], list[Mapping[int, int]]]](
    "ANE intrinsic emitters"
)
EMITTERS = registry.entries
emitter = registry.register_for


def feature_map(type_: types.TensorType, name: str) -> intrinsic.Surface:
    """Read one supported static fp16 feature map in global memory."""
    if not isinstance(type_, types.TensorType) or type_.dtype is not expr.DType.f16:
        raise ValueError(f"{name} must be an fp16 tensor")
    shape = tuple(expr.static_dim_value(extent) for extent in type_.shape)
    if len(shape) != 4 or shape[0] != 1 or any(extent is None for extent in shape):
        raise ValueError(f"{name} must have static shape [1, C, H, W]")
    _, channels, height, width = shape
    for axis, extent, limit in (
        ("C", channels, tir.layout.MAX_CHANNELS),
        ("H", height, tir.layout.MAX_HEIGHT),
        ("W", width, tir.layout.MAX_WIDTH),
    ):
        if not 1 <= extent <= limit:
            raise ValueError(f"{name} {axis} extent {extent} outside [1, {limit}]")
    if type_.storage is not expr.StorageKind.GMEM:
        raise ValueError(f"{name} must be in gmem for a single descriptor")
    if width > 1 and expr.static_dim_value(type_.layout.strides[3]) != 1:
        raise ValueError(f"{name} W stride must be one")
    surface = intrinsic.surface(channels, height, width)
    try:
        row, channel = tir.layout.feature_dma_strides(type_)
    except ValueError as error:
        raise ValueError(f"{name}: {error}") from error
    declared = (row or surface.row, channel or surface.channel)
    if declared != (surface.row, surface.channel):
        raise ValueError(
            f"{name} declares a {declared} byte row/channel stride, but a "
            f"{channels}x{height}x{width} fp16 surface needs "
            f"{(surface.row, surface.channel)}; declare strides "
            f"{surface.element_strides} instead"
        )
    return surface


#: The planar engine patch pair: four columns by four rows for a plain pass, and
#: five by four or five by three once a pooling window sweeps with a stride.
#: This is a rasterization candidate the L2 configuration layer picks, not a
#: derived quantity; the pair below reproduces the sampled descriptors and is
#: expected to be replaced when that layer is modelled.
PATCH_EDGE = 4
PATCH_SWEEP = 5


def patch_for(kernel: tuple[int, int], stride: tuple[int, int]) -> tuple[int, int]:
    """The patch pair a pooling pass declares."""
    width = PATCH_EDGE + (1 if kernel[1] > 1 else 0)
    height = PATCH_SWEEP - stride[0]
    return width, height


#: The planar engine op and second-source select of each binary kind.
BINARY = {
    expr.BinaryKind.ADD: (isa.PEOp.ADD, 2),
    expr.BinaryKind.SUB: (isa.PEOp.ADD, 3),
    expr.BinaryKind.MUL: (isa.PEOp.MUL, 2),
    expr.BinaryKind.MIN: (isa.PEOp.MIN, 2),
    expr.BinaryKind.MAX: (isa.PEOp.MAX, 2),
}


#: The L2 compare code and pooling mode of each reduction kind.
REDUCE = {
    expr.ReduceKind.SUM: (None, isa.PoolMode.NONE_OR_SUM),
    expr.ReduceKind.MAX: (1, isa.PoolMode.MAX),
    expr.ReduceKind.MIN: (3, isa.PoolMode.MIN),
}


#: The pooling mode of each pool kind.
POOL = {
    expr.PoolKind.MAX: isa.PoolMode.MAX,
    expr.PoolKind.AVG: isa.PoolMode.AVG,
}


@emitter(tir.Pool)
def pool(call: expr.Call) -> list[Mapping[int, int]]:
    """Stage the operand, run one planar engine pooling pass, write the result."""
    op = call.target
    destination, source = call.args
    src = feature_map(source.type, "src")
    dst = feature_map(destination.type, "dst")
    if op.ceil_mode:
        raise ValueError("T.ane.pool: ceil_mode is not emitted yet")
    kernel, stride = tuple(op.kernel_size), tuple(op.stride)
    if len(kernel) != 2 or len(stride) != 2:
        raise ValueError(f"T.ane.pool: kernel {kernel} and stride {stride} must be 2D")
    for axis, value in zip(("kernel", "stride"), (*kernel, *stride)):
        if not 1 <= value <= 2:
            raise ValueError(f"T.ane.pool: {axis} {value} is outside [1, 2]")
    padding = tuple(op.padding) if op.padding else (0, 0)
    if any(padding):
        raise ValueError("T.ane.pool: padding is not emitted yet")
    try:
        pool_mode = POOL[op.kind]
    except KeyError:
        raise ValueError(f"T.ane.pool: {op.kind} has no planar engine encoding") from None
    return [
        intrinsic.load(src),
        intrinsic.compute_pe(
            isa.TaskType.PE_POOL_UNARY,
            dst,
            pool_mode=pool_mode,
            compare=1 if pool_mode is isa.PoolMode.MAX else None,
            declared_in=src,
            declared_out=dst,
            window=(kernel[1], kernel[0], stride[1], stride[0]),
            padding=padding,
            patch=patch_for(kernel, stride),
        ),
        intrinsic.store(dst, staging=src, declare=False),
    ]


@emitter(tir.Reduce)
def reduce(call: expr.Call) -> list[Mapping[int, int]]:
    """Stage the operand, run one planar engine reduction, write the result.

    A reduction keeps the staged geometry as its declared window and places the
    result one staged image further along the buffer.
    """
    op = call.target
    destination, source = call.args
    src = feature_map(source.type, "src")
    dst = feature_map(destination.type, "dst")
    if tuple(op.axes) != (2, 3) or not op.keepdim:
        raise ValueError(f"T.ane.reduce: axes {tuple(op.axes)} are not emitted yet")
    expected = (src.channels, 1, 1)
    if (dst.channels, dst.height, dst.width) != expected:
        raise ValueError(
            f"T.ane.reduce: expected a {expected[0]}x1x1 result, got "
            f"{dst.channels}x{dst.height}x{dst.width}"
        )
    if op.square or op.scale != 1.0 or op.post is not tir.Post.NONE:
        raise ValueError("T.ane.reduce: square, scale and post are not emitted yet")
    try:
        compare, pool_mode = REDUCE[op.kind]
    except KeyError:
        raise ValueError(f"T.ane.reduce: {op.kind} has no planar engine encoding") from None
    parts = [
        intrinsic.load(src),
        intrinsic.compute_pe(
            isa.TaskType.PE_EW_REDUCE_UNARY,
            dst,
            declared_in=src,
            declared_out=src,
            tile_height=src.height,
            pool_mode=pool_mode,
            compare=compare,
            patch=(4, 4),
        ),
        intrinsic.store(dst, staging=src, declare=False),
    ]
    if compare is None:
        parts.append(intrinsic.end_marker())
    return parts


@emitter(tir.Binary)
def binary(call: expr.Call) -> list[Mapping[int, int]]:
    """Stage both operands, run one planar engine binary pass, write the result."""
    op = call.target
    destination, lhs, rhs = call.args
    dst = feature_map(destination.type, "dst")
    left = feature_map(lhs.type, "lhs")
    right = feature_map(rhs.type, "rhs")
    if (left.channels, left.height, left.width) != (dst.channels, dst.height, dst.width):
        raise ValueError("T.ane.binary: lhs shape must equal dst shape")
    if (right.channels, right.height, right.width) != (dst.channels, dst.height, dst.width):
        raise ValueError("T.ane.binary: rhs shape must equal dst shape")
    try:
        engine_op, second_source = BINARY[op.kind]
    except KeyError:
        raise ValueError(f"T.ane.binary: {op.kind} has no planar engine encoding") from None
    return [
        intrinsic.load(left),
        intrinsic.load_second(right),
        intrinsic.compute_pe(
            isa.TaskType.PE_EW_BINARY,
            dst,
            op=engine_op,
            second_source=second_source,
            source_routing=3,
            patch=(4, 4),
        ),
        intrinsic.end_marker(),
        intrinsic.store(dst, staged=2),
    ]


@emitter(tir.Bypass)
def bypass(call: expr.Call) -> list[Mapping[int, int]]:
    """Stage the operand, run one neural engine pass-through, write the result."""
    op = call.target
    destination, source = call.args
    src = feature_map(source.type, "src")
    dst = feature_map(destination.type, "dst")
    if (src.channels, src.height, src.width) != (dst.channels, dst.height, dst.width):
        raise ValueError("T.ane.bypass: dst shape must equal src shape")
    if tuple(op.perm) != (0, 1, 2):
        raise ValueError(f"T.ane.bypass: perm {tuple(op.perm)} is not emitted yet")
    return [
        intrinsic.load(src, isa.Engine.NEURAL),
        intrinsic.compute_ne(
            isa.NEOp.BYPASS,
            dst,
            nonlinear=NONLINEAR[op.nonlinear],
            kernel_mode=1,
        ),
        intrinsic.store(dst, isa.Engine.NEURAL),
    ]


@emitter(tir.Unary)
def unary(call: expr.Call) -> list[Mapping[int, int]]:
    """Read the operand, run the planar engine pass, write the result."""
    op = call.target
    destination, source = call.args
    src = feature_map(source.type, "src")
    dst = feature_map(destination.type, "dst")
    if (src.channels, src.height, src.width) != (dst.channels, dst.height, dst.width):
        raise ValueError("T.ane.unary: dst shape must equal src shape")
    return [
        intrinsic.load(src),
        intrinsic.compute_pe(
            isa.TaskType.PE_EW_UNARY,
            dst,
            pre_scale=op.pre_scale,
            scale=op.scale,
            bias=op.bias,
            absolute=op.abs,
            square=op.square,
            clamp=op.clamp,
            patch=(4, 4),
        ),
        intrinsic.store(dst),
    ]


class Builder(core.StmtVisitor[None]):
    """Visit statements, keeping each Evaluate's operands and intrinsic sequence."""

    def __init__(self) -> None:
        self.parts: list[Mapping[int, int]] = []
        self.operands: list[tuple[expr.Expr, ...]] = []

    def generic_visit(self, stmt, ctx) -> None:
        raise ValueError(f"{ctx}: cannot emit {type(stmt).__name__} as an ANE task descriptor")

    def visit_Sequential(self, stmt: core.Sequential, ctx: str) -> None:
        super().generic_visit(stmt, ctx)

    def visit_MeshScope(self, stmt: core.MeshScope, ctx: str) -> None:
        if tuple((item.name, item.size) for item in stmt.mesh.topologies) != (("engine", 1),):
            raise ValueError(f"{ctx}: expected one ANE engine in the mesh")
        super().generic_visit(stmt, ctx)

    def visit_Evaluate(self, stmt: core.Evaluate, ctx: str) -> None:
        op = stmt.callable
        emit = EMITTERS.get(type(op))
        if emit is None:
            try:
                name = expr.get_registered_name(type(op))
            except ValueError:
                name = type(op).__name__
            raise dispatch.MissingHandlerError(f"{ctx}: no intrinsic emitter for {name}")
        self.parts.extend(emit(expr.Call(op, stmt.args, type=types.TupleType.Void)))
        self.operands.append(stmt.args)


def build_program(fn: core.PrimFunction) -> isa.Program:
    """Build one task descriptor stream from a lowered H14 PrimFunction."""
    label = f"{target.summary(fn.target)} {fn.name}"
    if not isinstance(fn.target, ane_target.Target) or fn.target.architecture.name != "apple_h14":
        raise ValueError(f"{label}: expected an H14 ANE target")
    if fn.specializations or fn.variants:
        raise ValueError(f"{label}: descriptor emission does not support specialization")
    builder = Builder()
    builder.visit(fn.body, label)
    (args,) = builder.operands
    if fn.output_count != 1 or len(fn.params) < 2:
        raise ValueError(f"{label}: expected one output and at least one input parameter")
    if args[0] is not fn.params[-1]:
        raise ValueError(f"{label}: the schedule must write the output parameter first")
    if any(arg not in fn.params[:-1] for arg in args[1:]):
        raise ValueError(f"{label}: the schedule must read the function's input parameters")
    return serialize.to_program(builder.parts)


__all__ = [
    "EMITTERS",
    "build_program",
    "binary",
    "bypass",
    "emitter",
    "feature_map",
    "reduce",
    "unary",
]
