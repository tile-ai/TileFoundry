"""The fixed SM89 FP8 warp MMA declaration."""

from __future__ import annotations

from tilefoundry.ir.types import DType, Layout, ShardLayout, Split

from .mma_atom import MmaAtom, rmem_fragment
from .sm80_mma import WARP
from .sm80_mma import Mma as Sm80Mma

_A_FRAGMENT = ShardLayout(
    layout=Layout(shape=(2, 8, 2, 4, 4), strides=(256, 32, 16, 4, 1)),
    attrs=(Split(3), Split(1)),
    mesh=WARP,
)
_B_FRAGMENT = ShardLayout(
    layout=Layout(shape=(2, 4, 4, 8), strides=(128, 32, 8, 1)),
    attrs=(Split(1), Split(3)),
    mesh=WARP,
)


class Mma(MmaAtom):
    """An E4M3 warp MMA, 16 x 8 x 32, accumulating in F32."""

    namespace = "T.cuda.sm89"
    execution_mesh = WARP
    capability = "mma.sync"
    resource = "tensor_core_engine"

    C = Sm80Mma.C
    A = rmem_fragment((16, 32), DType.fp8e4m3, _A_FRAGMENT)
    B = rmem_fragment((32, 8), DType.fp8e4m3, _B_FRAGMENT)


__all__ = ["Mma"]
