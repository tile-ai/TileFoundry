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
bb, Wb, rb, lb, sb = (WildcardPattern(name) for name in ("b_swizzle", "Wb", "rb", "lb", "sb"))
RUN = P.Table((8, 16, 32, 64))[b]
BROADCAST = (Broadcast(), Broadcast(), Broadcast())
PER_THREAD = (Split(2), Split(0), Split(4))

a_mn = ComposedLayoutPattern(
    SwizzlePattern(b, 4, 3),
    0,
    LayoutPattern(((r, W), (2, 8)), ((l, 1), (s, W))),
    predicates=(W == RUN, r * W == 64, l == 16 * W, s == 8 * W),
)
a_k_whole = ComposedLayoutPattern(
    SwizzlePattern(b, 4, 3),
    0,
    LayoutPattern(((8, 8), (r, W)), ((l, W), (s, 1))),
    predicates=(W == RUN, b <= 1, r * W == 16, l == 16 * W, s == 8 * W),
)
a_k_sliced = ComposedLayoutPattern(
    SwizzlePattern(b, 4, 3),
    p,
    LayoutPattern(((8, 8), 16), ((l, W), 1)),
    predicates=(W == RUN, b >= 2, l == 8 * W, p % 16 == 0, 0 <= p, p < W),
    issues_per_row=lambda captures: (1, captures["W"] // 16),
)
fragment = LayoutPattern((8, 2, 4, 2, 4, c), (1, 8, 16, 64, 128, 512))


N = DimVar("n", 8, 256)


class Wgmma(MmaAtom):
    """A BF16 warpgroup MMA, 64 x n x 16, accumulating in F32."""

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
            "form",
            {
                Form.SS: OrPattern(*tuple(Major)),
                Form.RS: Major.K,
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
    A = SwitchPattern(
        "form",
        {
            Form.SS: TensorPattern(
                shape=(64, 16),
                dtype=DType.bf16,
                storage=S.SMEM,
                layout=ShardLayoutPattern(
                    SwitchPattern(
                        "a_major",
                        {
                            Major.MN: a_mn,
                            Major.K: OrPattern(a_k_whole, a_k_sliced),
                        },
                    ),
                    BROADCAST,
                    execution_mesh_pattern(WARPGROUP),
                ),
            ),
            Form.RS: TensorPattern(
                shape=(64, 16),
                dtype=DType.bf16,
                storage=S.RMEM,
                layout=ShardLayoutPattern(
                    LayoutPattern(
                        (8, 2, 4, 2, 4, 2),
                        (1, 8, 16, 64, 128, 512),
                    ),
                    PER_THREAD,
                    execution_mesh_pattern(WARPGROUP),
                ),
            ),
        },
    )
    B = TensorPattern(
        shape=(16, N),
        dtype=DType.bf16,
        storage=S.SMEM,
        layout=ShardLayoutPattern(
            ComposedLayoutPattern(
                SwizzlePattern(bb, 4, 3),
                0,
                LayoutPattern(((2, 8), (rb, Wb)), ((sb, Wb), (lb, 1))),
                predicates=(
                    Wb == P.Table((8, 16, 32, 64))[bb],
                    rb * Wb == n_extent,
                    lb == 16 * Wb,
                    sb == 8 * Wb,
                ),
            ),
            BROADCAST,
            execution_mesh_pattern(WARPGROUP),
        ),
    )


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
