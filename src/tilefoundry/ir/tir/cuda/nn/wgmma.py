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
PER_THREAD = (Split(2), Split(0), Split(4))


def run(swizzle: WildcardPattern, e: int):
    """Elements in one swizzled row for an operand of *e* bytes."""
    return P.Table(tuple(width // e for width in SWIZZLE_BYTES))[swizzle]


def k_extent(e: int) -> int:
    """One issue reads 32 bytes of K."""
    return 32 // e


def a_mn(e: int) -> ComposedLayoutPattern:
    return ComposedLayoutPattern(
        SwizzlePattern(b, 4, 3),
        0,
        LayoutPattern(((r, W), (2, 8)), ((l, 1), (s, W))),
        predicates=(W == run(b, e), r * W == 64, l == 16 * W, s == 8 * W),
    )


def a_k_whole(e: int) -> ComposedLayoutPattern:
    return ComposedLayoutPattern(
        SwizzlePattern(b, 4, 3),
        0,
        LayoutPattern(((8, 8), (r, W)), ((l, W), (s, 1))),
        predicates=(W == run(b, e), b <= 1, r * W == k_extent(e), l == 16 * W, s == 8 * W),
    )


def a_k_sliced(e: int) -> ComposedLayoutPattern:
    k = k_extent(e)
    return ComposedLayoutPattern(
        SwizzlePattern(b, 4, 3),
        p,
        LayoutPattern(((8, 8), k), ((l, W), 1)),
        predicates=(W == run(b, e), b >= 2, l == 8 * W, p % k == 0, 0 <= p, p < W),
        issues_per_row=lambda captures: (1, captures["W"] // k),
    )


def rs_fragment(e: int) -> LayoutPattern:
    """A's register fragment; each thread holds 4 bytes of K per pair of rows."""
    v = 4 // e
    return LayoutPattern((8, 2, 4, v, 4, 2), (1, 8, 16, 64, 64 * v, 256 * v))


def b_mn(e: int) -> ComposedLayoutPattern:
    return ComposedLayoutPattern(
        SwizzlePattern(bb, 4, 3),
        0,
        LayoutPattern(((2, 8), (rb, Wb)), ((sb, Wb), (lb, 1))),
        predicates=(
            Wb == run(bb, e),
            rb * Wb == n_extent,
            lb == 16 * Wb,
            sb == 8 * Wb,
        ),
    )


def b_k_whole(e: int) -> ComposedLayoutPattern:
    """``a_k_whole`` on B's ``(K, N)`` axes, with ``n / 8`` core-matrix groups along N."""
    return ComposedLayoutPattern(
        SwizzlePattern(bb, 4, 3),
        0,
        LayoutPattern(((rb, Wb), (gb, 8)), ((sb, 1), (lb, Wb))),
        predicates=(
            Wb == run(bb, e),
            bb <= 1,
            rb * Wb == k_extent(e),
            8 * gb == n_extent,
            lb == 16 * Wb,
            sb == 8 * Wb,
        ),
    )


def b_k_sliced(e: int) -> ComposedLayoutPattern:
    """``a_k_sliced`` on B's ``(K, N)`` axes, with ``n / 8`` core-matrix groups along N."""
    k = k_extent(e)
    return ComposedLayoutPattern(
        SwizzlePattern(bb, 4, 3),
        pb,
        LayoutPattern((k, (gb, 8)), (1, (lb, Wb))),
        predicates=(
            Wb == run(bb, e),
            bb >= 2,
            8 * gb == n_extent,
            lb == 8 * Wb,
            pb % k == 0,
            0 <= pb,
            pb < Wb,
        ),
        issues_per_row=lambda captures: (0, captures["Wb"] // k),
    )


fragment = LayoutPattern((8, 2, 4, 2, 4, c), (1, 8, 16, 64, 128, 512))


N = DimVar("n", 8, 256)


def _by_major(mn, k_major, e: int) -> dict:
    """Shared-memory arrangements by major; only 16-bit operands may be MN-major."""
    arrangements = {Major.MN: mn(e)} if e == 2 else {}
    arrangements[Major.K] = OrPattern(*(factory(e) for factory in k_major))
    return arrangements


def _a_role(dtype: DType) -> SwitchPattern:
    e = dtype.bit_width // 8
    return SwitchPattern(
        "form",
        {
            Form.SS: TensorPattern(
                shape=(64, k_extent(e)),
                dtype=dtype,
                storage=S.SMEM,
                layout=ShardLayoutPattern(
                    SwitchPattern("a_major", _by_major(a_mn, (a_k_whole, a_k_sliced), e)),
                    BROADCAST,
                    execution_mesh_pattern(WARPGROUP),
                ),
            ),
            Form.RS: TensorPattern(
                shape=(64, k_extent(e)),
                dtype=dtype,
                storage=S.RMEM,
                layout=ShardLayoutPattern(
                    rs_fragment(e),
                    PER_THREAD,
                    execution_mesh_pattern(WARPGROUP),
                ),
            ),
        },
    )


def _b_role(dtype: DType) -> TensorPattern:
    e = dtype.bit_width // 8
    return TensorPattern(
        shape=(k_extent(e), N),
        dtype=dtype,
        storage=S.SMEM,
        layout=ShardLayoutPattern(
            SwitchPattern("b_major", _by_major(b_mn, (b_k_whole, b_k_sliced), e)),
            BROADCAST,
            execution_mesh_pattern(WARPGROUP),
        ),
    )


_A_MAJOR_16 = SwitchPattern("form", {Form.SS: OrPattern(*tuple(Major)), Form.RS: Major.K})


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
    A = SwitchPattern("dtype", {dtype: _a_role(dtype) for dtype in DTYPES})
    B = SwitchPattern("dtype", {dtype: _b_role(dtype) for dtype in DTYPES})


__all__ = [
    "BROADCAST",
    "Form",
    "Major",
    "N",
    "PER_THREAD",
    "WARPGROUP",
    "Wgmma",
    "a_k_sliced",
    "a_k_whole",
    "a_mn",
    "fragment",
]
