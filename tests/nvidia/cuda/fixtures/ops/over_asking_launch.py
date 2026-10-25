"""A launch that asks for more dynamic shared memory than any CUDA device grants."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget
from tilefoundry_nvidia import CudaTarget

_CUDA = CudaTarget("nvidia.h200_sxm")


@module(entry="over_asking_host", target=_CUDA, topologies=(Topology("thread", 128),))
class OverAskingLaunch:
    """One kernel launched with 1 MiB of dynamic shared memory.

    No CUDA device grants a block that much, so every device refuses the launch.
    """

    @prim_func(target=_CUDA)
    def over_asking_device(a: Tensor[(4, 32), "f32"]):
        with Mesh((Topology("thread", 128),), layout=(4, 32), names=("w", "t")) as m:
            view = T.tensor_view(T.ptr_of(a), layout=((4 @ m.w, 32 @ m.t), (32, 1)))
            reg = T.alloc_tensor(
                tensor_type=Tensor[(4, 32), "f32", ((4 @ m.w, 32 @ m.t), (32, 1)), "rmem"]
            )
            T.copy(view, reg)
            T.copy(reg, view)

    @prim_func(target=CpuTarget())
    def over_asking_host(a: Tensor[(4, 32), "f32"]):
        launch(over_asking_device, a, grid=(1, 1, 1), block=(128, 1, 1), dynamic_smem=1048576)
