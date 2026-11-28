"""What one ANE entry accumulates while its statements are emitted.

An emitter turns one ``T.ane.*`` call into the hardware primitives of a single
task: read the sources, run the arithmetic, keep the result in L2 or write it
out. A ``T.ane.store`` has no task of its own; the task that produces its source
writes it out. The context keeps the tasks in schedule order and answers what an
emitter cannot read from its own call: where a bound view starts, which engine
produced an L2 source and whether a result is written out.
"""

from collections.abc import Mapping

from tilefoundry import codegen, dispatch
from tilefoundry.ir import expr, types
from tilefoundry_apple.ane import isa, target
from tilefoundry_apple.ane.ir import tir

from . import hwx

FULL_PLANE_PATCH = (4, 4)
PATCH_COLUMNS = 16
NEURAL = (tir.Conv, tir.Bypass)
SOURCE = isa.TileDMASrc.tile_dma_src_base_addr_lo.addr
RESULT = isa.TileDMADst.tile_dma_dst_base_addr_lo.addr


class Context:
    """The tasks of one entry, the GMEM tensors their DMA moves and its bound view addresses."""

    def __init__(self) -> None:
        self.passes: list[list[Mapping[int, int]]] = []
        self.transfers: list[Mapping[int, expr.Expr]] = []
        self.addresses: dict[expr.Expr, int] = {}
        self.symbols: dict[expr.Expr, expr.Expr] = {}
        self.coefficients: dict[expr.Expr, hwx.CoefficientLayout] = {}
        self.biases: dict[expr.Expr, expr.Expr] = {}
        self.weight_uses: list[tuple[expr.Expr, int, hwx.CoefficientLayout]] = []
        self.producers: dict[expr.Expr, expr.Call] = {}
        self.stores: dict[expr.Expr, expr.Expr] = {}

    @property
    def fresh(self) -> bool:
        """Whether no pass ran yet, so the next one programs every register it reads."""
        return not self.passes

    def coefficient_use(
        self, weight: expr.Expr, lut: bytes = b"", bias: expr.Expr | None = None
    ) -> hwx.CoefficientLayout:
        """Plan a constant's first use, its lanes led by *lut* and *bias*, and retain its passes."""
        for constant in (weight,) if bias is None else (weight, bias):
            source = self.symbols.get(constant, constant)
            if source is not constant or not getattr(source, "is_const", False):
                raise ValueError("convolution weights and biases must be direct constant parameters")
        layout = self.coefficients.get(weight)
        if layout is None:
            shape = tuple(expr.static_dim_value(n) for n in weight.type.shape)
            base = sum(item.size for item in self.coefficients.values())
            layout = hwx.CoefficientLayout(shape, base, lut, bias is not None)
            _, inputs, kh, kw = layout.shape
            expected = (inputs * kh * kw, kh * kw, kw, 1)
            if not tir.layout.strided(weight.type, expected):
                raise ValueError("convolution coefficients require contiguous OIHW weights")
            if bias is not None:
                if any(other is bias for other in self.biases.values()):
                    raise ValueError("a bias consumed with different weights is not decoded yet")
                if not tir.layout.strided(bias.type, (1,)):
                    raise ValueError("convolution biases require contiguous vectors")
                self.biases[weight] = bias
            self.coefficients[weight] = layout
        elif layout.lut != lut or self.biases.get(weight) is not bias:
            raise ValueError(
                "a constant consumed under different lookup tables or biases is not decoded yet"
            )
        self.weight_uses.append((weight, len(self.passes), layout))
        return layout

    def bind(self, var: expr.Expr, source: expr.Expr, address: int) -> None:
        """Record the root tensor and accumulate the parent view's byte displacement.

        An SMEM allocation keeps the byte offset the framework assigned it as its L2 base.
        """
        if var.type.storage is expr.StorageKind.SMEM and isinstance(source, expr.Call):
            desc = tensor_desc_of(var.type, var.name)
            if address < 0 or address % isa.intrinsic.L2_UNIT:
                raise ValueError(f"{var.name}: SMEM offset {address} must be a nonnegative "
                                 f"multiple of {isa.intrinsic.L2_UNIT}")
            if desc.storage_bytes != desc.l2_extent:
                raise ValueError(f"{var.name}: SMEM storage {desc.storage_bytes} bytes must equal "
                                 f"its L2 extent {desc.l2_extent}")
            self.addresses[var] = address
            self.symbols[var] = var
            return
        if var.type.storage is expr.StorageKind.SMEM and address:
            raise ValueError("offset SMEM view continuation is not decoded yet")
        self.addresses[var] = address + self.address_of(source)
        self.symbols[var] = self.symbols.get(source, source)

    def plan(self, calls: list[expr.Call]) -> None:
        """Record each SMEM tensor's producer, the store that writes it out and each task's DMA.

        A load moves a GMEM source into its destination; any other task that produces a
        stored tensor writes it out to the store's GMEM destination, so that tensor has
        no other reader.
        """
        read = set()
        for call in calls:
            sources, destination = operands(call)
            roots = tuple(self.symbols.get(operand, operand) for operand in sources)
            for operand, root in zip(sources, roots):
                if root not in self.producers:
                    raise ValueError("L2 source requires a preceding producer")
                if root.type != operand.type:
                    raise ValueError("L2 source must match producer geometry and layout")
            if isinstance(call.target, tir.Store):
                (root,) = roots
                if isinstance(self.producers[root].target, tir.Load):
                    raise ValueError("T.ane.store: its source must be the result of a computation; "
                                     "write T.ane.unary() between them")
                if root in self.stores:
                    raise ValueError("an L2 tensor is written out once")
                self.stores[root] = destination
                continue
            read.update(roots)
            root = self.symbols.get(destination, destination)
            if root in self.producers:
                raise ValueError("L2 tensor must have exactly one producer")
            self.producers[root] = call
        if read & self.stores.keys():
            raise ValueError("T.ane.store: its source is read again after being written out; "
                             "store a copy made by T.ane.unary()")
        for root, call in self.producers.items():
            loaded = isinstance(call.target, tir.Load)
            moved = call.args[0] if loaded else self.stores.get(root)
            register = SOURCE if loaded else RESULT
            self.transfers.append({} if moved is None else {register: self.symbols.get(moved, moved)})

    def source(self, operand: expr.Expr) -> tuple[int, isa.Engine]:
        """The L2 base of an SMEM source and the engine that produced it."""
        producer = self.producers[self.symbols.get(operand, operand)]
        return self.address_of(operand), (
            isa.Engine.NEURAL if isinstance(producer.target, NEURAL) else isa.Engine.PLANAR
        )

    def result(self, operand: expr.Expr) -> tuple[int, expr.Expr | None]:
        """The L2 base of an SMEM result and the GMEM tensor its store writes, if any."""
        return self.address_of(operand), self.stores.get(self.symbols.get(operand, operand))

    def address_of(self, operand: expr.Expr) -> int:
        """Return the allocated L2 byte base or a view's DRAM-relative byte offset."""
        return self.addresses.get(operand, 0)

    def binding_desc(self, param: expr.Var) -> isa.TensorDesc:
        """Describe an entry buffer and validate any lowered GMEM scratch views."""
        if len(param.type.shape) != 1:
            return tensor_desc_of(param.type, param.name, strided=True)
        count = expr.static_dim_value(param.type.shape[0])
        desc = isa.TensorDesc.from_shape(param.type.dtype.name, (1, 1, 1, count))
        for operand, source in self.symbols.items():
            if source is param:
                view = tensor_desc_of(operand.type, operand.name)
                if self.address_of(operand) + view.storage_bytes > count * desc.element_bytes:
                    raise ValueError("scratch view leaves its entry buffer")
        return desc

    def emit(self, *parts: Mapping[int, int]) -> None:
        """Append one pass made of the register maps of its primitives."""
        self.passes.append(list(parts))

    def emit_call(self, call: expr.Call, label: str) -> None:
        """Emit the pass of one ``T.ane.*`` call through its registered handler."""
        key = (target.Target, codegen.Role.EMIT, type(call.target))
        handler = codegen.registry.get(key, None)
        if handler is None:
            name = expr.get_registered_name(type(call.target))
            raise dispatch.MissingHandlerError(f"{label}: no ANE emitter for {name}")
        handler(call, self)


def tensor_desc_of(
    type_: types.TensorType, name: str, *, strided: bool = False
) -> isa.TensorDesc:
    """Read the memory layout of one static fp16 feature map.

    *strided* keeps the strides the type declares; otherwise the type must
    declare the canonical feature map of its shape.
    """
    if not isinstance(type_, types.TensorType):
        raise ValueError(f"{name} must be an fp16 tensor")
    shape = tuple(expr.static_dim_value(extent) for extent in type_.shape)
    if len(shape) != 4 or any(extent is None for extent in shape):
        raise ValueError(f"{name} must have static shape [1, C, H, W]")
    _, channels, height, width = shape
    for axis, extent, limit in (
        ("C", channels, tir.layout.MAX_CHANNELS),
        ("H", height, tir.layout.MAX_HEIGHT),
        ("W", width, tir.layout.MAX_WIDTH),
    ):
        if not 1 <= extent <= limit:
            raise ValueError(f"{name} {axis} extent {extent} outside [1, {limit}]")
    if type_.storage not in (expr.StorageKind.GMEM, expr.StorageKind.SMEM):
        raise ValueError(f"{name} must be in gmem or smem")
    if width > 1 and expr.static_dim_value(type_.layout.strides[3]) != 1:
        raise ValueError(f"{name} W stride must be one")
    if not isinstance(type_.layout, types.Layout):
        raise ValueError(f"{name}: ANE I/O needs a flat tensor layout")
    strides = tuple(expr.static_dim_value(step) for step in type_.layout.strides)
    desc = isa.TensorDesc(type_.dtype.name, shape, strides)
    try:
        desc.validate_feature()
        desc.validate("declared" if strided or type_.storage is expr.StorageKind.SMEM else "canonical")
        tir.layout.feature_dma_strides(type_)
    except ValueError as error:
        raise ValueError(f"{name}: {error}") from error
    return desc


def operands(call: expr.Call) -> tuple[tuple[expr.Expr, ...], expr.Expr]:
    """The SMEM tensors *call* reads and the tensor it writes."""
    params = (param for param in expr.signature_of(type(call.target)) if param.kind == "input")
    bound = tuple(zip(params, call.args))
    (written,) = (arg for param, arg in bound if param.effect & expr.MemoryEffect.WRITE)
    read = tuple(
        arg for param, arg in bound
        if not param.effect & expr.MemoryEffect.WRITE and arg.type.storage is expr.StorageKind.SMEM
    )
    return read, written


def written(desc: isa.TensorDesc, stored: expr.Expr | None) -> isa.TensorDesc:
    """The feature map a task writes: *desc* in L2, or the GMEM tensor a store writes it to."""
    return desc if stored is None else tensor_desc_of(stored.type, "dst", strided=True)


def raster_patch(desc: isa.TensorDesc, patch: tuple | None = None) -> tuple[int, int]:
    """The rasterization patch of a whole feature map; a stated *patch* must equal it.

    A feature map wider than the 16 columns of the base patch needs the horizontal
    field ``ceil(log2(width))``.
    """
    expected = (isa.rules.ceil_log2(max(desc.width, PATCH_COLUMNS)), FULL_PLANE_PATCH[1])
    if patch is not None:
        patch = tuple(patch)
        if len(patch) != 2 or any(type(n) is not int or not 0 <= n <= 15 for n in patch):
            raise ValueError(f"patch {patch!r} must contain two integers within [0, 15]")
        if patch != expected:
            raise ValueError(f"patch {patch} is not decoded yet")
    return expected


__all__ = ["RESULT", "SOURCE", "Context", "operands", "raster_patch", "tensor_desc_of", "written"]
