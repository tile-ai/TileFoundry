"""What one ANE entry accumulates while its statements are emitted.

An emitter turns one ``T.ane.*`` call into the hardware primitives of a single
engine pass: read the operands, run the arithmetic, write the result. The context
keeps the passes in schedule order and answers what an emitter cannot read from
its own call: where a bound view starts and whether an earlier pass ran.
"""

from collections.abc import Mapping

from tilefoundry import codegen, dispatch
from tilefoundry.ir import expr, types
from tilefoundry.ir.tir import memory
from tilefoundry_apple.ane import isa, target
from tilefoundry_apple.ane.ir import tir

from . import hwx
from .allocation import Interval, allocate

FULL_PLANE_PATCH = (4, 4)
SINGLE_ROW_PATCH = (5, 0)


class Context:
    """The passes of one entry, their operands and the addresses of its bound views."""

    def __init__(self) -> None:
        self.passes: list[list[Mapping[int, int]]] = []
        self.operands: list[tuple[expr.Expr, ...]] = []
        self.addresses: dict[expr.Expr, int] = {}
        self.symbols: dict[expr.Expr, expr.Expr] = {}
        self.coefficients: dict[expr.Expr, hwx.CoefficientLayout] = {}
        self.weight_uses: list[tuple[expr.Expr, int, hwx.CoefficientLayout]] = []
        self.staging: dict[tuple[int, int], int] = {}

    @property
    def fresh(self) -> bool:
        """Whether no pass ran yet, so the next one programs every register it reads."""
        return not self.passes

    def coefficient_use(self, weight: expr.Expr, lut: bytes = b"") -> hwx.CoefficientLayout:
        """Plan a constant's first use, its lanes led by *lut*, and retain every consuming pass."""
        source = self.symbols.get(weight, weight)
        if source is not weight or not getattr(source, "is_const", False):
            raise ValueError("convolution weights must be direct constant parameters")
        layout = self.coefficients.get(weight)
        if layout is None:
            shape = tuple(expr.static_dim_value(n) for n in weight.type.shape)
            base = sum(item.size for item in self.coefficients.values())
            layout = hwx.CoefficientLayout(shape, base, lut)
            _, inputs, kh, kw = layout.shape
            expected = (inputs * kh * kw, kh * kw, kw, 1)
            if not tir.layout.strided(weight.type, expected):
                raise ValueError("convolution coefficients require contiguous OIHW weights")
            self.coefficients[weight] = layout
        elif layout.lut != lut:
            raise ValueError("a constant consumed under different lookup tables is not decoded yet")
        self.weight_uses.append((weight, len(self.passes), layout))
        return layout

    def bind(self, var: expr.Expr, source: expr.Expr, address: int) -> None:
        """Record the root tensor and accumulate the parent view's byte displacement."""
        address += self.address_of(source)
        if var.type.storage is expr.StorageKind.SMEM and address:
            raise ValueError("offset SMEM view continuation is not decoded yet")
        if var.type.storage is expr.StorageKind.SMEM and isinstance(source, expr.Call):
            if (not isinstance(source.target, memory.TensorView) or not source.args
                    or not isinstance(source.args[0], expr.Constant) or source.args[0].value != 0):
                raise ValueError("SMEM view allocation mapping is not decoded yet")
            self.addresses[var] = 0
            self.symbols[var] = var
        else:
            self.addresses[var] = address
            self.symbols[var] = self.symbols.get(source, source)

    def plan(self, calls: list[expr.Call]) -> None:
        """Allocate SMEM lifetimes and each pass's transient DMA staging together."""
        resident = any(arg.type.storage is expr.StorageKind.SMEM for call in calls for arg in call.args)
        if not resident and not any(isinstance(call.target, tir.Conv) for call in calls):
            return
        lifetimes = {}
        intervals = []
        for index, call in enumerate(calls):
            if not isinstance(call.target, (tir.Unary, tir.Conv)):
                if resident:
                    raise ValueError("L2 continuation for this operation is not decoded yet")
                continue
            engine = isa.Engine.NEURAL if isinstance(call.target, tir.Conv) else isa.Engine.PLANAR
            for position in (1, 0):
                operand = call.args[position]
                desc = tensor_desc_of(operand.type, "L2 operand")
                rows = tile_rows(call.target, desc)
                if resident and (rows != desc.height or call.target.tile_width is not None):
                    raise ValueError("partial L2 continuation is not decoded yet")
                size = desc.staging_extent(engine, rows, call.target.tile_width)
                if operand.type.storage is expr.StorageKind.SMEM:
                    root = self.symbols.get(operand, operand)
                    if position == 0:
                        if root in lifetimes:
                            raise ValueError("L2 tensor must have exactly one producer")
                        lifetimes[root] = Interval(root, index, index, size)
                    else:
                        previous = lifetimes.get(root)
                        if previous is None or previous.first == index:
                            raise ValueError("L2 source requires a preceding producer")
                        if root.type != operand.type:
                            raise ValueError("L2 source must match producer geometry and layout")
                        if previous.size != size:
                            raise ValueError("cross-engine L2 staging pitch conversion is not decoded yet")
                        lifetimes[root] = Interval(root, previous.first, index, size)
                else:
                    intervals.append(Interval((index, position), index, index, size))
        addresses = allocate((*lifetimes.values(), *intervals))
        for operand in tuple(self.addresses):
            root = self.symbols.get(operand, operand)
            if root in lifetimes:
                self.addresses[operand] = addresses[root]
        for root in lifetimes:
            self.addresses[root] = addresses[root]
        for index, call in enumerate(calls):
            for position, operand in enumerate(call.args):
                if operand.type.storage is expr.StorageKind.SMEM:
                    self.staging[index, position] = self.address_of(operand)
                elif (index, position) in addresses:
                    self.staging[index, position] = addresses[index, position]

    def staging_address(self, position: int) -> int | None:
        """Return the current pass's allocated slot, or its legacy implicit placement."""
        return self.staging.get((len(self.passes), position))

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
        self.operands.append(call.args)


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


def tile_rows(op: expr.Op, desc: isa.TensorDesc) -> int:
    """Return the scheduled tile height, or the full feature-map height."""
    rows = op.tile_height
    if rows is None:
        return desc.height
    if type(rows) is not int or not 1 <= rows <= desc.height:
        raise ValueError(f"tile_height {rows} is outside [1, {desc.height}]")
    return rows


def raster_patch(op: expr.Op, rows: int, desc: isa.TensorDesc) -> tuple[int, int]:
    """The rasterization patch: the schedule's ``patch``, else the one the tile height implies.

    A single staged row cannot be patched vertically, so its vertical field stays
    clear and its horizontal field widens by one.
    """
    expected = FULL_PLANE_PATCH if rows == desc.height else SINGLE_ROW_PATCH
    if op.patch is not None:
        patch = tuple(op.patch)
        if len(patch) != 2 or any(type(n) is not int or not 0 <= n <= 15 for n in patch):
            raise ValueError(f"patch {patch!r} must contain two integers within [0, 15]")
        if patch != expected:
            raise ValueError(f"patch {patch} for tile_height {rows} is not decoded yet")
    return expected


__all__ = ["Context", "raster_patch", "tensor_desc_of", "tile_rows"]
