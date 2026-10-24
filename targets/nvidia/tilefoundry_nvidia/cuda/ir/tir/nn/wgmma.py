"""The parameterized SM90 warpgroup MMA declaration."""

from __future__ import annotations

from dataclasses import replace
from enum import Enum

from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_type
from tilefoundry.ir.pattern import (
    ComposedLayoutPattern,
    LayoutPattern,
    OrPattern,
    RangePattern,
    ShardLayoutPattern,
    SwitchPattern,
    SwizzlePattern,
    TensorPattern,
    WildcardPattern,
)
from tilefoundry.ir.pattern import (
    predicates as P,
)
from tilefoundry.ir.types import (
    Broadcast,
    ComposedLayout,
    DType,
    Layout,
    Mesh,
    ShardLayout,
    Split,
    Swizzle,
    TensorType,
    Topology,
)
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.layout_algebra import logical_divide
from tilefoundry.ir.types.storage import StorageKind as S

from .mma_atom import MmaAtom, execution_mesh_pattern

WARPGROUP = Mesh(
    (Topology("thread", 128),),
    Layout((4, 8, 4), (32, 4, 1)),
    ("warp", "lane8", "lane4"),
)


@register_type(dialect="T", namespace="cuda.sm90")
class Major(Enum):
    """Which of a shared descriptor's axes runs contiguously."""

    MN = "MN-major"
    K = "K-major"


@register_type(dialect="T", namespace="cuda.sm90")
class Form(Enum):
    """Whether WGMMA reads A from shared memory or registers."""

    SS = "SS"
    RS = "RS"


_ = WildcardPattern()
n, b, W, p, r, l, s, c = (
    WildcardPattern(name) for name in ("n", "a_swizzle", "W", "k0", "r", "l", "s", "c")
)
n_extent = n
bb, Wb, rb, lb, sb, pb = (
    WildcardPattern(name) for name in ("b_swizzle", "Wb", "rb", "lb", "sb", "k0b")
)
SWIZZLE_BYTES = (16, 32, 64, 128)
DTYPES = (DType.bf16, DType.f16, DType.fp8e4m3)
BROADCAST = (Broadcast(), Broadcast(), Broadcast())
PER_THREAD = (Split(0), Split(2), Split(4))


def _swizzle_row_elements(swizzle: WildcardPattern, element_bytes: int):
    """Elements in one swizzled row for an operand of *element_bytes* bytes."""
    return P.Table(tuple(width // element_bytes for width in SWIZZLE_BYTES))[swizzle]


def _mma_k_elements(element_bytes: int) -> int:
    """One issue reads 32 bytes of K."""
    return 32 // element_bytes


def canonical_smem_layout(layout, element_bytes: int, major: Major, *, b_operand=False):
    """Return CuTe make_gmma_desc's mode tree, in element units, for matching.

    Its (inner, repeat) coordinates are not logical-axis row-major digits;
    the view must not flow into address analysis or emission.
    """
    if isinstance(layout, ShardLayout):
        return replace(
            layout,
            layout=canonical_smem_layout(layout.layout, element_bytes, major, b_operand=b_operand),
        )
    if isinstance(layout, Layout):
        outer, swizzle = layout, 0
    elif (
        isinstance(layout, ComposedLayout)
        and isinstance(layout.outer, Layout)
        and (layout.inner is None or isinstance(layout.inner, Swizzle))
    ):
        outer, swizzle = layout.outer, 0 if layout.inner is None else layout.inner.bits
    else:
        return layout
    if not 0 <= swizzle < 4:
        return layout
    if b_operand:
        outer = Layout(outer.shape[::-1], outer.strides[::-1])
    width = SWIZZLE_BYTES[swizzle] // element_bytes
    tiles = (width, 8) if major is Major.MN else (8, 16 // element_bytes)
    outer = logical_divide(outer, tuple(Layout((tile,), (1,)) for tile in tiles), major="row")
    if b_operand:
        outer = Layout(outer.shape[::-1], outer.strides[::-1])
    return replace(layout, outer=outer) if isinstance(layout, ComposedLayout) else outer


def _smem_pattern(element_bytes: int, major: Major, *, b_operand=False):
    sw, w, rows, leading, stride, offset, extent = (
        (bb, Wb, rb, lb, sb, pb, n_extent) if b_operand else (b, W, r, l, s, p, 64)
    )
    vector = 16 // element_bytes
    prefix = "b" if b_operand else "a"
    stride_00, stride_10 = (
        WildcardPattern(f"{prefix}_stride_{position}") for position in ("00", "10")
    )
    if major is Major.MN:
        shape = ((w, rows), (8, 2))
        strides = (
            (stride_00, SwitchPattern(sw.name, {0: stride, 1: leading, 2: leading, 3: leading})),
            (stride_10, SwitchPattern(sw.name, {0: leading, 1: stride, 2: stride, 3: stride})),
        )
        origin = 0
        predicates = (
            rows * w == extent,
            stride_00 == 1,
            stride_10 == w,
            leading % vector == 0,
            stride % vector == 0,
        )
    else:
        leading = WildcardPattern(f"{prefix}_stride_11")
        shape = ((8, rows), (vector, 2))
        strides = ((stride_00, stride), (stride_10, leading))
        origin = offset
        predicates = (
            8 * rows == extent,
            stride_00 == _swizzle_row_elements(sw, element_bytes),
            stride_10 == 1,
            (sw == 0) | (leading == vector),
            leading % vector == 0,
            stride % vector == 0,
            offset % _mma_k_elements(element_bytes) == 0,
            0 <= offset,
            offset < _swizzle_row_elements(sw, element_bytes),
        )
    return ComposedLayoutPattern(
        SwizzlePattern(sw, 4, 3),
        origin,
        LayoutPattern(shape[::-1], strides[::-1]) if b_operand else LayoutPattern(shape, strides),
        predicates=((w == _swizzle_row_elements(sw, element_bytes),) if major is Major.MN else ())
        + predicates,
    )


def _a_register_fragment_pattern(element_bytes: int) -> LayoutPattern:
    """A's register fragment; each thread holds 4 bytes of K per pair of rows."""
    per_register = 4 // element_bytes
    return LayoutPattern(
        (4, 2, 8, 2, 4, per_register), (16, 8, 1, 256 * per_register, 64 * per_register, 64)
    )


fragment = LayoutPattern((4, 2, 8, c, 4, 2), (16, 8, 1, 512, 128, 64))


N = DimVar("n", 8, 256)


def _by_major(element_bytes: int, *, b_operand=False) -> dict:
    """Descriptor modes by major; only 16-bit operands may be MN-major."""
    return {
        major: _smem_pattern(element_bytes, major, b_operand=b_operand)
        for major in Major
        if major is Major.K or element_bytes == 2
    }


def _a_layout(dtype: DType) -> SwitchPattern:
    element_bytes = dtype.bit_width // 8
    return SwitchPattern(
        "form",
        {
            Form.SS: ShardLayoutPattern(
                SwitchPattern(
                    "a_major",
                    _by_major(element_bytes),
                ),
                BROADCAST,
                execution_mesh_pattern(WARPGROUP),
            ),
            Form.RS: ShardLayoutPattern(
                _a_register_fragment_pattern(element_bytes),
                PER_THREAD,
                execution_mesh_pattern(WARPGROUP),
            ),
        },
    )


def _b_layout(dtype: DType) -> ShardLayoutPattern:
    return ShardLayoutPattern(
        SwitchPattern(
            "b_major",
            _by_major(dtype.bit_width // 8, b_operand=True),
        ),
        BROADCAST,
        execution_mesh_pattern(WARPGROUP),
    )


_A_MAJOR_16 = SwitchPattern("form", {Form.SS: OrPattern(*tuple(Major)), Form.RS: Major.K})


_OPERAND_DTYPE = SwitchPattern("dtype", {dtype: dtype for dtype in DTYPES})
_K_ELEMENTS = SwitchPattern(
    "dtype", {dtype: _mma_k_elements(dtype.bit_width // 8) for dtype in DTYPES}
)


class Wgmma(MmaAtom):
    """A warpgroup MMA, 64 x n x k, accumulating in F32; k is 32 bytes of the operand dtype."""

    namespace = "T.cuda.sm90"
    execution_mesh = WARPGROUP
    capability = "wgmma.mma_async"
    resource = "wgmma_engine"

    def match_view(self, param_name: str, type_: TensorType) -> TensorType:
        """Use descriptor modes only for shared-operand pattern matching."""
        if type_.storage is not S.SMEM:
            return type_
        b_operand = param_name == "rhs"
        return replace(
            type_,
            layout=canonical_smem_layout(
                type_.layout,
                type_.dtype.bit_width // 8,
                self.bindings["b_major" if b_operand else "a_major"],
                b_operand=b_operand,
            ),
        )

    n = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern(
            "n",
            predicates=(
                RangePattern(lo=N.lo, hi=N.hi),
                WildcardPattern("n") % 8 == 0,
            ),
        ),
    )
    dtype = ParamDef(
        kind="attribute",
        annotation=DType,
        pattern=WildcardPattern("dtype", predicates=(OrPattern(*DTYPES),)),
    )
    form = ParamDef(
        kind="attribute",
        annotation=Form,
        pattern=WildcardPattern(
            "form",
            predicates=(OrPattern(*tuple(Form)),),
        ),
    )
    a_major = ParamDef(
        kind="attribute",
        annotation=Major,
        pattern=SwitchPattern(
            "dtype",
            {
                DType.bf16: _A_MAJOR_16,
                DType.f16: _A_MAJOR_16,
                DType.fp8e4m3: Major.K,
            },
        ),
        default=Major.MN,
    )
    b_major = ParamDef(
        kind="attribute",
        annotation=Major,
        pattern=SwitchPattern(
            "dtype",
            {
                DType.bf16: OrPattern(*tuple(Major)),
                DType.f16: OrPattern(*tuple(Major)),
                DType.fp8e4m3: Major.K,
            },
        ),
        default=Major.MN,
    )

    C = TensorPattern(
        shape=(64, N),
        dtype=DType.f32,
        storage=S.RMEM,
        layout=ShardLayoutPattern(
            fragment,
            PER_THREAD,
            execution_mesh_pattern(WARPGROUP),
        ),
        predicates=(8 * c == n_extent,),
    )
    A = TensorPattern(
        dtype=_OPERAND_DTYPE,
        storage=SwitchPattern("form", {Form.SS: S.SMEM, Form.RS: S.RMEM}),
        shape=(
            64,
            _K_ELEMENTS,
        ),
        layout=SwitchPattern("dtype", {dtype: _a_layout(dtype) for dtype in DTYPES}),
    )
    B = TensorPattern(
        dtype=_OPERAND_DTYPE,
        storage=S.SMEM,
        shape=(
            _K_ELEMENTS,
            N,
        ),
        layout=SwitchPattern("dtype", {dtype: _b_layout(dtype) for dtype in DTYPES}),
    )


__all__ = [
    "BROADCAST",
    "Form",
    "Major",
    "N",
    "PER_THREAD",
    "WARPGROUP",
    "Wgmma",
    "fragment",
]
