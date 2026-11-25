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
from tilefoundry_apple.ane.codegen import image
from tilefoundry_apple.ane.codegen import image as image_mod
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


def feature_map(type_: types.TensorType, name: str, *, strided: bool = False) -> intrinsic.Surface:
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
    if strided:
        return intrinsic.Surface(channels, height, width, declared[0], declared[1])
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


def tile_rows(op, surface) -> int:
    """Return how many rows one descriptor stages for this call.

    The tile height is a scheduling decision supplied by the caller; when it is
    absent the descriptor stages the whole surface.  A supplied height must be a
    positive row count that fits within the surface it stages.
    """
    rows = getattr(op, "tile_height", None)
    if rows is None:
        return surface.height
    if rows < 1:
        raise ValueError(f"tile_height must be a positive row count, got {rows}")
    if rows > surface.height:
        raise ValueError(
            f"tile_height {rows} exceeds the {surface.height}-row surface it stages"
        )
    return rows


def pool_slots(rows: int, surface) -> int:
    """Return how many staged slots a pooled result starts after.

    A pooled descriptor that stages a whole plane writes its result into the
    slot right behind the source; a single-staged-row tile keeps the source and
    the result in separate staged images.
    """
    return POOL_RESULT_SLOTS if rows == surface.height else 2


def op_patch(op, rows: int, surface) -> tuple[int, int]:
    """Return the rasterization patch: the caller's choice, else the local rule."""
    declared = getattr(op, "patch", None)
    return raster_patch(rows, surface) if declared is None else tuple(declared)


def raster_patch(rows: int, surface) -> tuple[int, int]:
    """Return the rasterization patch for a staged tile.

    A tile that stages a whole plane is rasterized in square patches; a single
    staged row cannot be patched vertically, so the vertical field stays clear
    and the horizontal field widens by one.
    """
    return (4, 4) if rows == surface.height else (5, 0)


def patch_for(kernel: tuple[int, int], stride: tuple[int, int]) -> tuple[int, int]:
    """The patch pair a pooling pass declares."""
    width = PATCH_EDGE + (1 if kernel[1] > 1 else 0)
    height = PATCH_SWEEP - stride[0]
    return width, height


#: The planar engine op and second-source select of each binary kind.
#: Staged-image slots the result buffer starts after, per operation family.
#: The canonical layouts all place the result one staged image along; the tiled
#: streams observed so far place pool at two and reduce at thirty-two, so the
#: slot count is a property of the staging plan rather than of the operation.
POOL_RESULT_SLOTS = 1
REDUCE_RESULT_SLOTS = 1

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
def pool(
    call: expr.Call, address: Mapping[expr.Expr, int] | None = None, fresh: bool = True
) -> list[Mapping[int, int]]:
    """Stage the operand, run one planar engine pooling pass, write the result."""
    op = call.target
    destination, source = call.args
    src = feature_map(source.type, "src").staged(getattr(op, "tile_width", None))
    dst = feature_map(destination.type, "dst").staged(getattr(op, "tile_width", None))
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
        (
            intrinsic.load_chained(src, rows=tile_rows(op, src))
            if chain_source(op)
            else intrinsic.load(src, address=_address(address, source), fresh=fresh)
        ),
        intrinsic.compute_pe(
            isa.TaskType.PE_POOL_UNARY,
            dst,
            pool_mode=pool_mode,
            compare=1 if pool_mode is isa.PoolMode.MAX else None,
            declared_in=src,
            declared_out=dst,
            window=(kernel[1], kernel[0], stride[1], stride[0]),
            padding=padding,
            tile_height=tile_rows(op, dst),
            patch=(
                patch_for(kernel, stride) if getattr(op, "patch", None) is None else tuple(op.patch)
            ),
        ),
        intrinsic.store(
            dst,
            staging=src,
            rows=tile_rows(op, src),
            staged=pool_slots(tile_rows(op, dst), dst),
            address=_address(address, destination),
            separated=False,
            fresh=fresh,
            declare=False,
        ),
    ]


@emitter(tir.Reduce)
def reduce(
    call: expr.Call, address: Mapping[expr.Expr, int] | None = None, fresh: bool = True
) -> list[Mapping[int, int]]:
    """Stage the operand, run one planar engine reduction, write the result.

    A reduction keeps the staged geometry as its declared window and places the
    result one staged image further along the buffer.
    """
    op = call.target
    destination, source = call.args
    declared = feature_map(source.type, "src")
    src = declared.staged(getattr(op, "tile_width", None))
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
        intrinsic.load(
            src,
            address=_address(address, source),
            fresh=fresh,
            rows=tile_rows(op, src),
            planar_result=True,
        ),
        intrinsic.compute_pe(
            isa.TaskType.PE_EW_REDUCE_UNARY,
            dst,
            declared_in=src,
            declared_out=src,
            tile_height=tile_rows(op, src),
            pool_mode=pool_mode,
            compare=compare,
            patch=op_patch(op, tile_rows(op, dst), dst),
        ),
        intrinsic.store(
            dst,
            staging=src,
            rows=tile_rows(op, src),
            staged=REDUCE_RESULT_SLOTS,
            address=_address(address, destination),
            fresh=fresh,
            declare=False,
        ),
    ]
    return parts


@emitter(tir.Binary)
def binary(
    call: expr.Call, address: Mapping[expr.Expr, int] | None = None, fresh: bool = True
) -> list[Mapping[int, int]]:
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
        (
            intrinsic.load_chained(left, rows=tile_rows(op, dst))
            if chain_source(op)
            else intrinsic.load(left, address=_address(address, lhs), fresh=fresh)
        ),
        intrinsic.load_second(
            right, rows=tile_rows(op, dst), address=_address(address, rhs), fresh=fresh
        ),
        intrinsic.compute_pe(
            isa.TaskType.PE_EW_BINARY,
            dst,
            op=engine_op,
            second_source=second_source,
            source_routing=3,
            tile_height=tile_rows(op, dst),
            patch=op_patch(op, tile_rows(op, dst), dst),
        ),
        intrinsic.store(
            dst,
            staged=2,
            rows=tile_rows(op, dst),
            address=_address(address, destination),
            fresh=fresh,
            separated=True,
        ),
    ]


@emitter(tir.Bypass)
def bypass(
    call: expr.Call, address: Mapping[expr.Expr, int] | None = None, fresh: bool = True
) -> list[Mapping[int, int]]:
    """Stage the operand, run one neural engine pass-through, write the result.

    A pass-through that permutes the feature map writes a destination whose DMA
    strides follow the transposed binding, so the declared strides are used
    verbatim instead of the canonical surface.
    """
    op = call.target
    destination, source = call.args
    permuted = tuple(op.perm) != (0, 1, 2)
    src = feature_map(source.type, "src")
    dst = feature_map(destination.type, "dst", strided=permuted)
    if not permuted and (src.channels, src.height, src.width) != (
        dst.channels,
        dst.height,
        dst.width,
    ):
        raise ValueError("T.ane.bypass: dst shape must equal src shape")
    return [
        (
            intrinsic.load_chained(src, isa.Engine.NEURAL, rows=tile_rows(op, dst))
            if chain_source(op)
            else intrinsic.load(
                src,
                isa.Engine.NEURAL,
                address=_address(address, source),
                fresh=fresh,
                rows=tile_rows(op, dst),
            )
        ),
        intrinsic.compute_ne(
            isa.NEOp.BYPASS,
            dst,
            nonlinear=NONLINEAR[op.nonlinear],
            kernel_mode=1,
            tile_height=tile_rows(op, dst),
            width_code=getattr(op, "width_code", None),
            declared=src if permuted else None,
            transpose=permuted,
            unicast_channels=(getattr(op, "unicast_channels", None) or (2 if permuted else 1)),
            ocg_size=(
                None if getattr(op, "omit_ocg", False) else (getattr(op, "ocg_size", None) or 0)
            ),
            address=_address(address, destination),
        ),
        intrinsic.store(
            dst,
            isa.Engine.NEURAL,
            rows=tile_rows(op, dst),
            address=_address(address, destination),
            fresh=fresh,
            declared=src if permuted else None,
        ),
    ]


@emitter(tir.Unary)
def unary(
    call: expr.Call, address: Mapping[expr.Expr, int] | None = None, fresh: bool = True
) -> list[Mapping[int, int]]:
    """Read the operand, run the planar engine pass, write the result."""
    op = call.target
    destination, source = call.args
    src = feature_map(source.type, "src").staged(getattr(op, "tile_width", None))
    dst = feature_map(destination.type, "dst").staged(getattr(op, "tile_width", None))
    if (src.channels, src.height, src.width) != (dst.channels, dst.height, dst.width):
        raise ValueError("T.ane.unary: dst shape must equal src shape")
    return [
        (
            intrinsic.load_chained(src, rows=tile_rows(op, src))
            if chain_source(op)
            else intrinsic.load(src, address=_address(address, source), fresh=fresh)
        ),
        intrinsic.compute_pe(
            isa.TaskType.PE_EW_UNARY,
            dst,
            pre_scale=op.pre_scale,
            scale=op.scale,
            bias=op.bias,
            absolute=op.abs,
            square=op.square,
            clamp=op.clamp,
            tile_height=tile_rows(op, dst),
            patch=op_patch(op, tile_rows(op, dst), dst),
        ),
        intrinsic.store(
            dst, rows=tile_rows(op, dst), address=_address(address, destination), fresh=fresh
        ),
    ]


@emitter(tir.Conv)
def conv(
    call: expr.Call, address: Mapping[expr.Expr, int] | None = None, fresh: bool = True
) -> list[Mapping[int, int]]:
    """Read the operand, run one convolution pass over it, write the result.

    A standalone convolution reads its feature map from global memory in the same
    descriptor as the convolution itself; the coefficient stream is a separate
    base address the loader binds.
    """
    op = call.target
    destination, source = call.args[0], call.args[1]
    src = feature_map(source.type, "src").staged(getattr(op, "tile_width", None))
    dst = feature_map(destination.type, "dst").staged(getattr(op, "tile_width", None))
    if tuple(getattr(op, "dilation", (1, 1))) != (1, 1) or getattr(op, "groups", 1) != 1:
        raise ValueError("T.ane.conv: dilated and grouped convolutions are not emitted yet")
    ocg = getattr(op, "ocg_size", None)
    if ocg is None:
        ocg = channel_groups(dst.channels)
    rows = tile_rows(op, dst)
    weight_type = call.args[2].type
    weight_shape = tuple(expr.static_dim_value(n) for n in weight_type.shape)
    _lanes, _per_lane, coeff_stride = image.coefficient_layout(weight_shape)
    kernel_width = weight_shape[3]
    return [
        intrinsic.load(
            src, isa.Engine.NEURAL, address=_address(address, source), fresh=fresh, rows=rows
        ),
        intrinsic.convolution(
            src,
            dst,
            index=0,
            ocg_size=ocg,
            coeff_stride=coeff_stride,
            kernel_width=kernel_width,
            result_base=getattr(op, "l2_result_base", None),
            activate=op.nonlinear is not tir_nonlinear.Nonlinear.NONE,
            staged_source=False,
            to_dram=True,
            address=_address(address, destination),
        ),
    ]


def channel_groups(channels: int) -> int:
    """Encode how many 16-channel engine groups a pass covers, as a log2.

    A pass states the field only while the output fits four groups; wider outputs
    leave it at its reset value.
    """
    group = intrinsic.NE_CHANNEL_GROUP
    groups = -(-channels // group)
    return 0 if groups <= 1 else (groups - 1).bit_length()


def chain_source(op) -> bool:
    """Whether this pass reads the source its predecessor left staged."""
    return bool(getattr(op, "chained_source", False))


def _address(addresses: Mapping[expr.Expr, int] | None, operand: expr.Expr) -> int:
    """Return the byte address of one operand, defaulting to the tensor base."""
    return 0 if addresses is None else addresses.get(operand, 0)


def view_address(value: expr.Expr) -> int:
    """Return the byte address a bound view starts at, relative to its tensor.

    A view over a slice of a tensor names that slice's first element, so the DMA
    programs the window's byte displacement rather than the tensor base.
    """
    if not isinstance(value, expr.Call):
        return 0
    pointer = value.args[0] if value.args else None
    if not isinstance(pointer, expr.Call) or len(pointer.args) != 1:
        return 0
    window = pointer.args[0]
    if not isinstance(window, expr.Call) or len(window.args) != 2:
        return 0
    tensor, starts = window.args
    parent = tensor.type if hasattr(tensor, "type") else None
    layout = getattr(parent, "layout", None)
    if layout is None:
        return 0
    try:
        stride = tuple(layout.strides)
    except AttributeError:
        return 0
    total = 0
    for begin, step in zip(getattr(starts, "elements", ()), stride, strict=False):
        index = begin.value if isinstance(begin, expr.Constant) else 0
        if not isinstance(index, int):
            return 0
        total += index * step
    return total * 2


class Builder(core.StmtVisitor[None]):
    """Visit statements, keeping each Evaluate's operands and intrinsic sequence."""

    def __init__(self) -> None:
        self.passes: list[list[Mapping[int, int]]] = []
        self.operands: list[tuple[expr.Expr, ...]] = []
        self.bound: set[expr.Expr] = set()
        self.address: dict[expr.Expr, int] = {}

    def generic_visit(self, stmt, ctx) -> None:
        raise ValueError(f"{ctx}: cannot emit {type(stmt).__name__} as an ANE task descriptor")

    def visit_Sequential(self, stmt: core.Sequential, ctx: str) -> None:
        super().generic_visit(stmt, ctx)

    def visit_LetStmt(self, stmt, ctx) -> None:
        """A binding names a slice; record where that slice starts."""
        self.bound.add(stmt.var)
        self.address[stmt.var] = view_address(stmt.value)

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
        call = expr.Call(op, stmt.args, type=types.TupleType.Void)
        self.passes.append(emit(call, self.address, fresh=not self.passes))
        self.operands.append(stmt.args)


def matmul_stream(cin: int, cout: int, *, ocg_size: int) -> isa.Program:
    """Build the two-descriptor stream of a matmul lowered to one convolution.

    The first descriptor copies the operand into the L2 buffer with the channels
    on the fast axis; the second runs the convolution with the coefficient stream
    and writes the result to DRAM.  ``ocg_size`` is the output channel group
    count the scheduler chooses.
    """
    source = isa.surface(cin, 1, 1)
    destination = isa.surface(cout, 1, 1)
    return serialize.to_program(
        [
            [isa.staged_copy(source)],
            [isa.convolution(source, destination, index=1, ocg_size=ocg_size)],
        ]
    )


def coefficient_relocs(program: isa.Program) -> tuple[int, ...]:
    """Return the text word offsets of the flagged coefficient packets.

    Each descriptor that fetches coefficients states their base as a flagged
    two-word packet; those words are what the relocation records patch.
    """
    offsets: list[int] = []
    start = len(isa.PREAMBLE) // 4
    for descriptor in program.tds:
        cursor = start + descriptor.header_size_words()
        for packet in descriptor.packets:
            if len(packet.writes) == 2 and all(addr == 0x1908 + 4 * i for i, (addr, _) in enumerate(packet.writes)):
                offsets.append(cursor + 1)
            cursor += 1 + len(packet.writes)
        start += descriptor.size_words()
    return tuple(offsets)


def coefficient_span(cin: int, inter: int, *, full: bool) -> dict[str, tuple[int, int]]:
    """Return each coefficient tensor's (section offset, packed length).

    The gate tensor carries the activation lookup table behind its weights, so
    its packed length is the one that also holds the table.
    """
    tensors = {"gate": (inter, cin, 1, 1, intrinsic.NE_LUT_BYTES),
               "up": (inter, cin, 1, 1, 0)}
    if full:
        tensors["down"] = (cin, inter, 1, 1, 0)
    out: dict[str, tuple[int, int]] = {}
    base = 0
    for name, (out_channels, in_channels, kh, kw, lut) in tensors.items():
        lanes, per_lane, _stride = image_mod.coefficient_layout(
            (out_channels, in_channels, kh, kw), lut_bytes=lut
        )
        length = lanes * _stride
        out[name] = (base, length)
        base += length
    return out


def fused_image(
    weights: Mapping[str, tuple[tuple[int, int, int, int], bytes]],
    *,
    slots: Mapping[str, int],
    full: bool,
) -> image.Image:
    """Assemble one fused block: descriptor stream plus its coefficient section.

    ``weights`` names the block's coefficient tensors (``gate``/``up`` and, in the
    full feed-forward block, ``down``); each carries its shape and bytes.  Each
    tensor's section offset is the operand of the pass that consumes it.
    """
    cin, inter = slots["cin"], slots["inter"]
    program = ffn_stream(cin, inter, slots) if full else glu_stream(cin, inter, slots)
    order = ("gate", "up", "down") if full else ("gate", "up")
    coefficients = tuple(
        (weights[name][0], weights[name][1], intrinsic.NE_LUT_BYTES if name == "gate" else 0)
        for name in order
    )
    return image.Image(
        program.to_bytes(),
        b"",
        (),
        (),
        coefficients=coefficients,
        kernel_reloc=coefficient_relocs(program),
    )


def build_program(fn: core.PrimFunction) -> isa.Program:
    """Build one task descriptor stream from a lowered H14 PrimFunction."""
    label = f"{target.summary(fn.target)} {fn.name}"
    if not isinstance(fn.target, ane_target.Target) or fn.target.architecture.name != "apple_h14":
        raise ValueError(f"{label}: expected an H14 ANE target")
    if fn.specializations or fn.variants:
        raise ValueError(f"{label}: descriptor emission does not support specialization")
    builder = Builder()
    builder.visit(fn.body, label)
    if fn.output_count != 1 or len(fn.params) < 2:
        raise ValueError(f"{label}: expected one output and at least one input parameter")
    sources = set(fn.params[:-1]) | builder.bound
    for args in builder.operands:
        if len(args) < 2:
            raise ValueError(f"{label}: every pass must name an output and a source")
        if args[0] not in (fn.params[-1], *builder.bound):
            raise ValueError(f"{label}: the schedule must write the output parameter first")
        if any(arg not in sources for arg in args[1:]):
            raise ValueError(f"{label}: the schedule must read the function's inputs")
    return serialize.to_program(builder.passes)


__all__ = [
    "EMITTERS",
    "build_program",
    "binary",
    "bypass",
    "conv",
    "emitter",
    "feature_map",
    "reduce",
    "unary",
]


def glu_stream(cin: int, inter: int, slots: Mapping[str, int]) -> isa.Program:
    """Build the fused gate/up/multiply stream of one GLU block.

    Every L2 slot, the rasterization patch and the DRAM ownership come from the
    memory allocator, so ``slots`` names them; only the register encoding is
    derived here.
    """
    operand = isa.surface(cin, 1, 1)
    hidden = isa.surface(inter, 1, 1)
    return serialize.to_program(
        [
            [
                isa.staged_copy(
                    operand,
                    source_base=slots["staged_source"],
                    result_base=slots["staged_result"],
                )
            ],
            [
                isa.convolution(
                    operand,
                    hidden,
                    index=1,
                    ocg_size=slots["gate_ocg"],
                    coeff_base=coefficient_span(cin, inter, full=True)["gate"][0],
                    activate=True,
                    armed=False,
                    staged_source=True,
                    to_dram=False,
                    source_base=slots["hidden_input"],
                )
            ],
            [
                isa.convolution(
                    operand,
                    hidden,
                    index=2,
                    ocg_size=slots["up_ocg"],
                    coeff_base=coefficient_span(cin, inter, full=True)["up"][0],
                    armed=False,
                    staged_source=True,
                    to_dram=False,
                    source_base=slots["hidden_input"],
                    result_base=slots["gate_result"],
                    read_config=False,
                )
            ],
            [
                isa.multiply(
                    *slots["multiply_shape"],
                    second_base=slots["gate_result"],
                    result_base=slots["block_result"],
                    patch=slots["multiply_patch"],
                    transpose=slots["multiply_transpose"],
                    to_dram=True,
                    result_config=1,
                    lut_armed=True,
                    index=3,
                )
            ],
        ]
    )


def ffn_stream(cin: int, inter: int, slots: Mapping[str, int]) -> isa.Program:
    """Build the fused gate/up/multiply/down stream of one SwiGLU FFN block.

    The GLU passes come from the same emitter as a standalone block; the down
    projection adds the pass that writes the block result to DRAM.
    """
    operand = isa.surface(cin, 1, 1)
    hidden = isa.surface(inter, 1, 1)
    passes = [
        [
            isa.staged_copy(
                operand,
                source_base=slots["staged_source"],
                result_base=slots["staged_result"],
            )
        ],
        [
            isa.convolution(
                operand,
                hidden,
                index=1,
                ocg_size=slots["gate_ocg"],
                coeff_base=coefficient_span(cin, inter, full=True)["gate"][0],
                activate=True,
                armed=False,
                staged_source=True,
                to_dram=False,
                source_base=slots["hidden_input"],
            )
        ],
        [
            isa.convolution(
                operand,
                hidden,
                index=2,
                ocg_size=slots["up_ocg"],
                coeff_base=coefficient_span(cin, inter, full=True)["up"][0],
                armed=False,
                staged_source=True,
                to_dram=False,
                source_base=slots["hidden_input"],
                result_base=slots["gate_result"],
                read_config=False,
            )
        ],
        [
            isa.multiply(
                *slots["multiply_shape"],
                second_base=slots["gate_result"],
                result_base=slots["block_result"],
                patch=slots["multiply_patch"],
                transpose=slots["multiply_transpose"],
                to_dram=False,
                result_config=0,
                result_stride=slots["multiply_stride"],
            )
        ],
        [
            isa.convolution(
                hidden,
                operand,
                index=4,
                ocg_size=slots["down_ocg"],
                coeff_base=coefficient_span(cin, inter, full=True)["down"][0],
                armed=True,
                staged_source=True,
                to_dram=True,
                source_base=slots["hidden_input"],
                result_base=slots["ffn_result"],
            )
        ],
    ]
    return serialize.to_program(passes)
