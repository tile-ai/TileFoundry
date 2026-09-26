"""The parameterized SM90 warpgroup MMA declaration."""

from __future__ import annotations

from enum import Enum

from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.pattern import (
    AndPattern,
    CapturePattern,
    ComposedLayoutPattern,
    GuardPattern,
    LayoutPattern,
    MeshPattern,
    MultipleOfPattern,
    OneOfPattern,
    OrPattern,
    RangePattern,
    ShardLayoutPattern,
    SwitchPattern,
    SwizzlePattern,
    TensorPattern,
    WildcardPattern,
)
from tilefoundry.ir.pattern.match import is_symbolic
from tilefoundry.ir.types import Broadcast, DType, Layout, Mesh, Split, Topology
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.storage import StorageKind as S

from .mma_atom import MmaAtom

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


class Swizzled(Enum):
    """The SM90 descriptor's shared-memory swizzle width."""

    INTERLEAVE = "INTERLEAVE"
    SW32 = "SW32"
    SW64 = "SW64"
    SW128 = "SW128"

    @property
    def bits(self) -> int:
        return ("INTERLEAVE", "SW32", "SW64", "SW128").index(self.value)

    @property
    def run(self) -> int:
        return DESCRIPTOR_UNIT << self.bits


DESCRIPTOR_UNIT_BYTES = 16
DESCRIPTOR_UNIT = DESCRIPTOR_UNIT_BYTES * 8 // DType.bf16.bit_width
START = "k0"


def Descriptor(
    rows,
    cols: int,
    major,
    *,
    selects: str,
    k_first: bool = False,
) -> SwitchPattern:
    """Every shared BF16 descriptor arrangement for one logical tile."""
    if isinstance(major, str):
        return SwitchPattern(
            major,
            {
                read: Descriptor(
                    rows,
                    cols,
                    read,
                    selects=selects,
                    k_first=k_first,
                )
                for read in Major
            },
        )
    unit = DESCRIPTOR_UNIT
    branches = {}
    for mode in Swizzled:
        width = mode.run
        along = rows if major is Major.MN else cols
        leading, stride = cols * width, width * unit
        guarded, sliced = False, False
        if is_symbolic(along):
            if major is not Major.MN:
                raise ValueError("a K-major descriptor reads a K it states")
            guarded = True
        elif along % width:
            if major is Major.MN or width % along:
                continue
            sliced = True
        if sliced:
            shape, strides = ((rows // unit, unit), cols), ((unit * width, width), 1)
        elif major is Major.MN:
            shape = ((rows // width, width), (cols // unit, unit))
            strides = ((leading, 1), (stride, width))
        else:
            shape = ((rows // unit, unit), (cols // width, width))
            strides = ((leading, width), (stride, 1))
        if k_first:
            shape, strides = shape[::-1], strides[::-1]
        held = LayoutPattern(shape, strides)
        if mode.bits:
            start = (
                CapturePattern(START, OneOfPattern(tuple(range(0, width, along))))
                if sliced
                else 0
            )
            held = ComposedLayoutPattern(
                SwizzlePattern(mode.bits, 4, 3),
                start,
                held,
            )
        branches[mode] = (
            GuardPattern(along, MultipleOfPattern(width), held) if guarded else held
        )
    return SwitchPattern(selects, branches)


def Fragment(rows: int, cols) -> LayoutPattern:
    """CuTe ``CLayout_64xN``; at N=16, also the RS A fragment."""
    if rows != 64:
        raise ValueError("the only register fragment this back end reads is 64 rows deep")
    return LayoutPattern((8, 2, 4, 2, 4, cols // 8), (1, 8, 16, 64, 128, 512))


SHARED_BY_ALL = (Broadcast(), Broadcast(), Broadcast())
HELD_PER_THREAD = (Split(2), Split(0), Split(4))

_WARPGROUP_LAYOUT = LayoutPattern.from_layout(WARPGROUP.layout, per_mode=True)
_WARPGROUP_PATTERN = MeshPattern(
    ("thread",),
    OrPattern(
        ComposedLayoutPattern(offset=WildcardPattern(), outer=_WARPGROUP_LAYOUT),
        _WARPGROUP_LAYOUT,
    ),
)


def shared(arrangement) -> ShardLayoutPattern:
    return ShardLayoutPattern(arrangement, SHARED_BY_ALL, _WARPGROUP_PATTERN)


def held(arrangement) -> ShardLayoutPattern:
    return ShardLayoutPattern(arrangement, HELD_PER_THREAD, _WARPGROUP_PATTERN)


N = DimVar("n", 8, 257)


class Wgmma(MmaAtom):
    """A BF16 warpgroup MMA, 64 x n x 16, accumulating in F32."""

    namespace = "T.cuda.sm90"
    scope = WARPGROUP
    capability = "wgmma"

    n = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=AndPattern((MultipleOfPattern(8), RangePattern(lo=N.lo, hi=N.hi - 1))),
    )
    form = ParamDef(
        kind="attribute",
        annotation=Form,
        pattern=OneOfPattern(tuple(Form)),
    )
    a_major = ParamDef(
        kind="attribute",
        annotation=Major,
        pattern=SwitchPattern(
            "form",
            {
                Form.SS: OneOfPattern(tuple(Major)),
                Form.RS: Major.K,
            },
        ),
        default=Major.MN,
    )

    C = TensorPattern(
        shape=(64, N),
        dtype=DType.f32,
        storage=S.RMEM,
        layout=held(Fragment(64, N)),
    )
    A = SwitchPattern(
        "form",
        {
            Form.SS: TensorPattern(
                shape=(64, 16),
                dtype=DType.bf16,
                storage=S.SMEM,
                layout=shared(Descriptor(64, 16, "a_major", selects="a_swizzle")),
            ),
            Form.RS: TensorPattern(
                shape=(64, 16),
                dtype=DType.bf16,
                storage=S.RMEM,
                layout=held(Fragment(64, 16)),
            ),
        },
    )
    B = TensorPattern(
        shape=(16, N),
        dtype=DType.bf16,
        storage=S.SMEM,
        layout=shared(
            Descriptor(
                N,
                16,
                Major.MN,
                selects="b_swizzle",
                k_first=True,
            )
        ),
    )


__all__ = [
    "DESCRIPTOR_UNIT",
    "DESCRIPTOR_UNIT_BYTES",
    "Descriptor",
    "Form",
    "Fragment",
    "HELD_PER_THREAD",
    "Major",
    "N",
    "SHARED_BY_ALL",
    "START",
    "Swizzled",
    "WARPGROUP",
    "Wgmma",
    "held",
    "shared",
]
