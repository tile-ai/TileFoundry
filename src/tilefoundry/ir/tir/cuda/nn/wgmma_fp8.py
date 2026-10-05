"""The parameterized SM90 FP8 warpgroup MMA declaration."""

from __future__ import annotations

from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.pattern import (
    ComposedLayoutPattern,
    LayoutPattern,
    OrPattern,
    RangePattern,
    ShardLayoutPattern,
    SwizzlePattern,
    TensorPattern,
    WildcardPattern,
)
from tilefoundry.ir.pattern import (
    predicates as P,
)
from tilefoundry.ir.types import DType
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.storage import StorageKind as S

from .mma_atom import MmaAtom, execution_mesh_pattern
from .wgmma import BROADCAST, PER_THREAD, WARPGROUP, fragment

K_EXTENT = 32

n_extent = WildcardPattern("n")
b, W, p, r, l, s, c = (
    WildcardPattern(name) for name in ("a_swizzle", "W", "k0", "r", "l", "s", "c")
)
bb, Wb, pb, rb, lb, sb, nb = (
    WildcardPattern(name) for name in ("b_swizzle", "Wb", "kb0", "rb", "lb", "sb", "nb")
)
RUN = P.Table((16, 32, 64, 128))[b]
RUN_B = P.Table((16, 32, 64, 128))[bb]

a_k_whole = ComposedLayoutPattern(
    SwizzlePattern(b, 4, 3),
    0,
    LayoutPattern(((8, 8), (r, W)), ((l, W), (s, 1))),
    predicates=(W == RUN, b <= 1, r * W == K_EXTENT, l == 16 * W, s == 8 * W),
)
a_k_sliced = ComposedLayoutPattern(
    SwizzlePattern(b, 4, 3),
    p,
    LayoutPattern(((8, 8), K_EXTENT), ((l, W), 1)),
    predicates=(W == RUN, b >= 2, l == 8 * W, p % K_EXTENT == 0, 0 <= p, p < W),
    issues_per_row=lambda captures: (1, captures["W"] // K_EXTENT),
)
b_k_whole = ComposedLayoutPattern(
    SwizzlePattern(bb, 4, 3),
    0,
    LayoutPattern(((rb, Wb), (nb, 8)), ((sb, 1), (lb, Wb))),
    predicates=(
        Wb == RUN_B,
        bb <= 1,
        rb * Wb == K_EXTENT,
        lb == 16 * Wb,
        sb == 8 * Wb,
        8 * nb == n_extent,
    ),
)
b_k_sliced = ComposedLayoutPattern(
    SwizzlePattern(bb, 4, 3),
    pb,
    LayoutPattern((K_EXTENT, (nb, 8)), (1, (lb, Wb))),
    predicates=(
        Wb == RUN_B,
        bb >= 2,
        lb == 8 * Wb,
        8 * nb == n_extent,
        pb % K_EXTENT == 0,
        0 <= pb,
        pb < Wb,
    ),
    issues_per_row=lambda captures: (0, captures["Wb"] // K_EXTENT),
)


N = DimVar("n", 8, 256)


class WgmmaFp8(MmaAtom):
    """An FP8 (e4m3) warpgroup MMA, 64 x n x 32, accumulating in F32.

    SM90 reads FP8 A and B from shared memory K-major only, so B's (K, n) tile
    runs contiguously along K. One 128-byte swizzle row holds four K atoms.
    """

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
        shape=(64, K_EXTENT),
        dtype=DType.fp8e4m3,
        storage=S.SMEM,
        layout=ShardLayoutPattern(
            OrPattern(a_k_whole, a_k_sliced),
            BROADCAST,
            execution_mesh_pattern(WARPGROUP),
        ),
    )
    B = TensorPattern(
        shape=(K_EXTENT, N),
        dtype=DType.fp8e4m3,
        storage=S.SMEM,
        layout=ShardLayoutPattern(
            OrPattern(b_k_whole, b_k_sliced),
            BROADCAST,
            execution_mesh_pattern(WARPGROUP),
        ),
    )


__all__ = ["K_EXTENT", "N", "WgmmaFp8", "a_k_sliced", "a_k_whole", "b_k_sliced", "b_k_whole"]
