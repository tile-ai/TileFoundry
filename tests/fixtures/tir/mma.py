from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget
from tilefoundry_nvidia import CudaTarget


@module(entry="mm_host", target=CudaTarget("nvidia.h200_sxm"))
class MmHandwritten:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def mm_device(
        a: Tensor[(16, 16), "bf16"], b: Tensor[(16, 8), "bf16"], c: Tensor[(16, 8), "f32"]
    ):
        with Mesh(
            (Topology("thread", 32),), layout=((4, 8), (1, 4)), names=("warp", "lane")
        ) as _warp:
            a_view = T.tensor_view(T.ptr_of(a), layout=(2, 8 @ _warp.lane, 2, 4 @ _warp.warp, 2))
            b_view = T.tensor_view(T.ptr_of(b), layout=(2, 4 @ _warp.warp, 2, 8 @ _warp.lane))
            a_frag = T.alloc_tensor(
                tensor_type=Tensor[
                    (16, 16),
                    "bf16",
                    ((2, 8 @ _warp.lane, 2, 4 @ _warp.warp, 2), (128, 16, 8, 2, 1)),
                    "rmem",
                ]
            )
            b_frag = T.alloc_tensor(
                tensor_type=Tensor[
                    (16, 8),
                    "bf16",
                    ((2, 4 @ _warp.warp, 2, 8 @ _warp.lane), (64, 16, 8, 1)),
                    "rmem",
                ]
            )
            acc = T.alloc_tensor(
                tensor_type=Tensor[
                    (16, 8), "f32", ((2, 8 @ _warp.lane, 4 @ _warp.warp, 2), (64, 8, 2, 1)), "rmem"
                ]
            )
            T.copy(a_view, a_frag)
            T.copy(b_view, b_frag)
            T.fill(acc, 0.0)
            T.tiled_mma(acc, a_frag, b_frag, atom=T.cuda.sm80.Mma())
            c_view = T.tensor_view(T.ptr_of(c), layout=(2, 8 @ _warp.lane, 4 @ _warp.warp, 2))
            T.copy(acc, c_view)

    @prim_func(target=CpuTarget())
    def mm_host(a: Tensor[(16, 16), "bf16"], b: Tensor[(16, 8), "bf16"], c: Tensor[(16, 8), "f32"]):
        launch(mm_device, a, b, c, grid=(
        1,
        1,
        1,
    ), block=(
        32,
        1,
        1,
    ))  # noqa: F821
