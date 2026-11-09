"""The fixed SM80 warp MMA declaration."""

from __future__ import annotations

from tilefoundry.ir import expr, types

from .mma_atom import MmaAtom, rmem_fragment

WARP = types.Mesh(
    topologies=(types.Topology("thread", 32),),
    layout=types.Layout(shape=(4, 8), strides=(1, 4)),
    names=("warp", "lane"),
)

A_FRAGMENT = types.ShardLayout(
    layout=types.Layout(shape=(2, 8, 2, 4, 2), strides=(128, 16, 8, 2, 1)),
    attrs=(types.Split(3), types.Split(1)),
    mesh=WARP,
)
B_FRAGMENT = types.ShardLayout(
    layout=types.Layout(shape=(2, 4, 2, 8), strides=(64, 16, 8, 1)),
    attrs=(types.Split(1), types.Split(3)),
    mesh=WARP,
)
C_FRAGMENT = types.ShardLayout(
    layout=types.Layout(shape=(2, 8, 4, 2), strides=(64, 8, 2, 1)),
    attrs=(types.Split(2), types.Split(1)),
    mesh=WARP,
)


class Mma(MmaAtom):
    """A BF16 warp MMA, 16 x 8 x 16, accumulating in F32."""

    namespace = "T.cuda.sm80"
    execution_mesh = WARP
    capability = "mma.sync"
    resource = "tensor_core_engine"

    C = rmem_fragment((16, 8), expr.DType.f32, C_FRAGMENT)
    A = rmem_fragment((16, 16), expr.DType.bf16, A_FRAGMENT)
    B = rmem_fragment((16, 8), expr.DType.bf16, B_FRAGMENT)


__all__ = ["Mma", "WARP"]
