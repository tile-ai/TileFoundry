"""What one ANE entry accumulates while its statements are emitted.

An emitter turns one ``T.ane.*`` call into the hardware primitives of a single
engine pass: read the operands, run the arithmetic, write the result. The context
keeps the passes in schedule order and answers what an emitter cannot read from
its own call: where a bound view starts and whether an earlier pass ran.
"""

from collections.abc import Mapping

from tilefoundry import codegen, dispatch
from tilefoundry.ir import expr, types
from tilefoundry_apple.ane import isa, target
from tilefoundry_apple.ane.ir import tir

from . import hwx

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

    @property
    def fresh(self) -> bool:
        """Whether no pass ran yet, so the next one programs every register it reads."""
        return not self.passes

    def coefficient_use(self, weight: expr.Expr) -> hwx.CoefficientLayout:
        """Plan a constant's first consumption and retain every consuming pass."""
        source = self.symbols.get(weight, weight)
        if source is not weight or not getattr(source, "is_const", False):
            raise ValueError("convolution weights must be direct constant parameters")
        layout = self.coefficients.get(weight)
        if layout is None:
            shape = tuple(expr.static_dim_value(n) for n in weight.type.shape)
            base = sum(item.size for item in self.coefficients.values())
            layout = hwx.CoefficientLayout(shape, base)
            _, inputs, kh, kw = layout.shape
            expected = (inputs * kh * kw, kh * kw, kw, 1)
            if not tir.layout.strided(weight.type, expected):
                raise ValueError("convolution coefficients require contiguous OIHW weights")
            self.coefficients[weight] = layout
        self.weight_uses.append((weight, len(self.passes), layout))
        return layout

    def bind(self, var: expr.Expr, source: expr.Expr, address: int) -> None:
        """Record the source tensor and byte address of a bound view."""
        self.addresses[var] = address
        self.symbols[var] = self.symbols.get(source, source)

    def address_of(self, operand: expr.Expr) -> int:
        """The byte address of *operand* inside its tensor; a parameter starts at zero."""
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
    """Read the memory layout of one static fp16 feature map in global memory.

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
    if type_.storage is not expr.StorageKind.GMEM:
        raise ValueError(f"{name} must be in gmem for a single descriptor")
    if width > 1 and expr.static_dim_value(type_.layout.strides[3]) != 1:
        raise ValueError(f"{name} W stride must be one")
    if not isinstance(type_.layout, types.Layout):
        raise ValueError(f"{name}: ANE I/O needs a flat tensor layout")
    strides = tuple(expr.static_dim_value(step) for step in type_.layout.strides)
    desc = isa.TensorDesc(type_.dtype.name, shape, strides)
    try:
        desc.validate_feature()
        desc.validate("declared" if strided else "canonical")
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
