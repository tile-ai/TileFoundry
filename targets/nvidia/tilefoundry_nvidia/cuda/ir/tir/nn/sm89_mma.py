"""The fixed SM89 FP8 warp MMA declaration."""

from __future__ import annotations

from tilefoundry.ir import expr, types

from . import sm80_mma
from .mma_atom import MmaAtom, rmem_fragment
from .sm80_mma import WARP

A_FRAGMENT = types.ShardLayout(
    layout=types.Layout(shape=(2, 8, 2, 4, 4), strides=(256, 32, 16, 4, 1)),
    attrs=(types.Split(3), types.Split(1)),
    mesh=WARP,
)
B_FRAGMENT = types.ShardLayout(
    layout=types.Layout(shape=(2, 4, 4, 8), strides=(128, 32, 8, 1)),
    attrs=(types.Split(1), types.Split(3)),
    mesh=WARP,
)


class Mma(MmaAtom):
    """An E4M3 warp MMA, 16 x 8 x 32, accumulating in F32."""

    namespace = "T.cuda.sm89"
    execution_mesh = WARP
    capability = "mma.sync"
    resource = "tensor_core_engine"

    C = sm80_mma.Mma.C
    A = rmem_fragment((16, 32), expr.DType.fp8e4m3, A_FRAGMENT)
    B = rmem_fragment((32, 8), expr.DType.fp8e4m3, B_FRAGMENT)


__all__ = ["Mma"]
