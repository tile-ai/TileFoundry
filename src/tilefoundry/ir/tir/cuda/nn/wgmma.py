"""The parameterized SM90 warpgroup MMA declaration."""

from __future__ import annotations

from enum import Enum

from tilefoundry.ir.core.param_def import ParamDef
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
from tilefoundry.ir.types import Broadcast, DType, Layout, Mesh, Split, Topology
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.storage import StorageKind as S

from .mma_atom import MmaAtom, execution_mesh_pattern

WARPGROUP = Mesh(
    (Topology("thread", 128),),
    Layout((4, 8, 4), (32, 4, 1)),
    ("warp", "lane8", "lane4"),
)


class Major(Enum):
    """Which of a shared descriptor's axes runs contiguously."""

    MN = "MN-major"
    K = "K-major"


class Form(Enum):
    """Whether WGMMA reads A from shared memory or registers."""

    SS = "SS"
    RS = "RS"


_ = WildcardPattern()
n, b, W, p, r, l, s, c = (
    WildcardPattern(name) for name in ("n", "a_swizzle", "W", "k0", "r", "l", "s", "c")
)
n_extent = n
bb, Wb, rb, lb, sb, pb, gb = (
    WildcardPattern(name) for name in ("b_swizzle", "Wb", "rb", "lb", "sb", "k0b", "gb")
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


def _smem_fields(b_operand):
    return (bb, Wb, rb, lb, sb, pb, gb, n_extent) if b_operand else (b, W, r, l, s, p, 8, 64)


def _smem_layout(shape, strides, b_operand):
    return LayoutPattern(shape[::-1], strides[::-1]) if b_operand else LayoutPattern(shape, strides)


def _a_mn_major_smem_pattern(element_bytes: int, b_operand=False) -> ComposedLayoutPattern:
    sw, w, rows, leading, stride, _, _, extent = _smem_fields(b_operand)
    return ComposedLayoutPattern(
        SwizzlePattern(sw, 4, 3),
        0,
        _smem_layout(((rows, w), (2, 8)), ((leading, 1), (stride, w)), b_operand),
        predicates=(
            w == _swizzle_row_elements(sw, element_bytes),
            rows * w == extent,
            leading % (8 * w) == 0,
            stride % (8 * w) == 0,
        ),
    )


def _a_k_major_whole_smem_pattern(element_bytes: int, b_operand=False) -> ComposedLayoutPattern:
    """SW32 has rows == 1: its fixed LBO lies inside the contiguous w mode."""
    sw, w, rows, leading, stride, _, groups, extent = _smem_fields(b_operand)
    return ComposedLayoutPattern(
        SwizzlePattern(sw, 4, 3),
        0,
        _smem_layout(((groups, 8), (rows, w)), ((leading, w), (stride, 1)), b_operand),
        predicates=(
            w == _swizzle_row_elements(sw, element_bytes),
            sw <= 1,
            rows * w == _mma_k_elements(element_bytes),
            *((8 * groups == extent,) if b_operand else ()),
            leading % (8 * w) == 0,
            stride % (8 * w) == 0,
        ),
    )


def _a_k_major_sliced_smem_pattern(element_bytes: int, b_operand=False) -> ComposedLayoutPattern:
    sw, w, _, leading, _, offset, groups, extent = _smem_fields(b_operand)
    k_elements = _mma_k_elements(element_bytes)
    return ComposedLayoutPattern(
        SwizzlePattern(sw, 4, 3),
        offset,
        _smem_layout(((groups, 8), k_elements), ((leading, w), 1), b_operand),
        predicates=(
            w == _swizzle_row_elements(sw, element_bytes),
            sw >= 2,
            *((8 * groups == extent,) if b_operand else ()),
            leading % (8 * w) == 0,
            offset % k_elements == 0,
            0 <= offset,
            offset < w,
        ),
        issues_per_row=lambda captures: (int(not b_operand), captures[w.name] // k_elements),
    )


def _a_register_fragment_pattern(element_bytes: int) -> LayoutPattern:
    """A's register fragment; each thread holds 4 bytes of K per pair of rows."""
    per_register = 4 // element_bytes
    return LayoutPattern(
        (4, 2, 8, 2, 4, per_register), (16, 8, 1, 256 * per_register, 64 * per_register, 64)
    )


def _b_mn_major_smem_pattern(element_bytes: int) -> ComposedLayoutPattern:
    return _a_mn_major_smem_pattern(element_bytes, True)


def _b_k_major_whole_smem_pattern(element_bytes: int) -> ComposedLayoutPattern:
    return _a_k_major_whole_smem_pattern(element_bytes, True)


def _b_k_major_sliced_smem_pattern(element_bytes: int) -> ComposedLayoutPattern:
    return _a_k_major_sliced_smem_pattern(element_bytes, True)


fragment = LayoutPattern((4, 2, 8, c, 4, 2), (16, 8, 1, 512, 128, 64))


N = DimVar("n", 8, 256)


def _by_major(mn_major_pattern, k_major_patterns, element_bytes: int) -> dict:
    """Shared-memory arrangements by major; only 16-bit operands may be MN-major."""
    arrangements = {Major.MN: mn_major_pattern(element_bytes)} if element_bytes == 2 else {}
    arrangements[Major.K] = OrPattern(*(pattern(element_bytes) for pattern in k_major_patterns))
    return arrangements


def _a_layout(dtype: DType) -> SwitchPattern:
    element_bytes = dtype.bit_width // 8
    return SwitchPattern(
        "form",
        {
            Form.SS: ShardLayoutPattern(
                SwitchPattern(
                    "a_major",
                    _by_major(
                        _a_mn_major_smem_pattern,
                        (_a_k_major_whole_smem_pattern, _a_k_major_sliced_smem_pattern),
                        element_bytes,
                    ),
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
            _by_major(
                _b_mn_major_smem_pattern,
                (_b_k_major_whole_smem_pattern, _b_k_major_sliced_smem_pattern),
                dtype.bit_width // 8,
            ),
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
