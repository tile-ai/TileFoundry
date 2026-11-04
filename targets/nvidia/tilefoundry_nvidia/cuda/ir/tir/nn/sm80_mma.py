"""The fixed SM80 warp MMA declaration."""

from __future__ import annotations

from tilefoundry.ir.types import DType, Layout, Mesh, ShardLayout, Split, Topology

from .mma_atom import MmaAtom, rmem_fragment

WARP = Mesh(
    topologies=(Topology("thread", 32),),
    layout=Layout(shape=(4, 8), strides=(1, 4)),
    names=("warp", "lane"),
)

_A_FRAGMENT = ShardLayout(
    layout=Layout(shape=(2, 8, 2, 4, 2), strides=(128, 16, 8, 2, 1)),
    attrs=(Split(3), Split(1)),
    mesh=WARP,
)
_B_FRAGMENT = ShardLayout(
    layout=Layout(shape=(2, 4, 2, 8), strides=(64, 16, 8, 1)),
    attrs=(Split(1), Split(3)),
    mesh=WARP,
)
_C_FRAGMENT = ShardLayout(
    layout=Layout(shape=(2, 8, 4, 2), strides=(64, 8, 2, 1)),
    attrs=(Split(2), Split(1)),
    mesh=WARP,
)


class Mma(MmaAtom):
    """A BF16 warp MMA, 16 x 8 x 16, accumulating in F32."""

    namespace = "T.cuda.sm80"
    execution_mesh = WARP
    capability = "mma.sync"
    resource = "tensor_core_engine"

    C = rmem_fragment((16, 8), DType.f32, _C_FRAGMENT)
    A = rmem_fragment((16, 16), DType.bf16, _A_FRAGMENT)
    B = rmem_fragment((16, 8), DType.bf16, _B_FRAGMENT)


__all__ = ["Mma", "WARP"]
