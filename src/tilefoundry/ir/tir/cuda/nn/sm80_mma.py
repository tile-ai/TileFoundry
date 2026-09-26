"""The fixed SM80 warp MMA declaration."""

from __future__ import annotations

from tilefoundry.ir.pattern import (
    ComposedLayoutPattern,
    LayoutPattern,
    MeshPattern,
    OrPattern,
    ShardLayoutPattern,
    TensorPattern,
    WildcardPattern,
)
from tilefoundry.ir.types import DType, Layout, Mesh, ShardLayout, Split, Topology
from tilefoundry.ir.types.storage import StorageKind as S

from .mma_atom import MmaAtom

WARP = Mesh(
    topologies=(Topology("thread", 32),),
    layout=Layout(shape=(4, 8), strides=(1, 4)),
    names=("warp", "lane"),
)

_A_FRAGMENT = ShardLayout(
    layout=Layout(shape=(2, 4, 2, 8, 2), strides=(1, 2, 8, 16, 128)),
    attrs=(Split(1), Split(3)),
    mesh=WARP,
)
_B_FRAGMENT = ShardLayout(
    layout=Layout(shape=(8, 2, 4, 2), strides=(1, 8, 16, 64)),
    attrs=(Split(2), Split(0)),
    mesh=WARP,
)
_C_FRAGMENT = ShardLayout(
    layout=Layout(shape=(2, 4, 8, 2), strides=(1, 2, 8, 64)),
    attrs=(Split(1), Split(2)),
    mesh=WARP,
)

_WARP_LAYOUT = LayoutPattern.from_layout(WARP.layout, per_mode=True)
_WARP_PATTERN = MeshPattern(
    ("thread",),
    OrPattern(
        ComposedLayoutPattern(offset=WildcardPattern(), outer=_WARP_LAYOUT),
        _WARP_LAYOUT,
    ),
)


def _fragment(shape: tuple, dtype, held: ShardLayout) -> TensorPattern:
    return TensorPattern(
        shape=shape,
        dtype=dtype,
        storage=S.RMEM,
        layout=ShardLayoutPattern(
            layout=LayoutPattern.from_layout(held.layout),
            attrs=held.attrs,
            mesh=_WARP_PATTERN,
        ),
    )


class Mma(MmaAtom):
    """A BF16 warp MMA, 16 x 8 x 16, accumulating in F32."""

    namespace = "T.cuda.sm80"
    scope = WARP
    capability = "tensor_core"

    C = _fragment((16, 8), DType.f32, _C_FRAGMENT)
    A = _fragment((16, 16), DType.bf16, _A_FRAGMENT)
    B = _fragment((16, 8), DType.bf16, _B_FRAGMENT)


__all__ = ["Mma", "WARP"]
