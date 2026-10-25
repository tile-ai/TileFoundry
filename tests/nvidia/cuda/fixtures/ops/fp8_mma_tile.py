"""One sm89 E4M3 MMA atom on per-lane fragments gathered straight from global memory."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget
from tilefoundry_nvidia import CudaTarget

_CUDA = CudaTarget("nvidia.h200_sxm")
_MESH_LAYOUT = Layout(shape=(4, 8), strides=(1, 4))


@module(entry="fp8_host", target=_CUDA, topologies=(Topology("thread", 32),))
class Fp8MmaTile:
    """A 16x32 by 32x8 E4M3 product done as one atom, accumulating in F32."""

    @prim_func(target=_CUDA)
    def fp8_device(
        a: Tensor[(16, 32), "fp8e4m3"],
        b: Tensor[(32, 8), "fp8e4m3"],
        c: Tensor[(16, 8), "f32"],
    ):
        with Mesh((Topology("thread", 32),), _MESH_LAYOUT, names=("warp", "lane")) as m:
            a_frag = T.alloc_tensor(
                tensor_type=Tensor[
                    (16, 32),
                    "fp8e4m3",
                    ((2, 8 @ m.lane, 2, 4 @ m.warp, 4), (256, 32, 16, 4, 1)),
                    "rmem",
                ]
            )
            b_frag = T.alloc_tensor(
                tensor_type=Tensor[
                    (32, 8),
                    "fp8e4m3",
                    ((2, 4 @ m.warp, 4, 8 @ m.lane), (128, 32, 8, 1)),
                    "rmem",
                ]
            )
            acc = T.alloc_tensor(
                tensor_type=Tensor[
                    (16, 8), "f32", ((2, 8 @ m.lane, 4 @ m.warp, 2), (64, 8, 2, 1)), "rmem"
                ]
            )
            a_view = T.tensor_view(
                T.ptr_of(a), layout=((2, 8 @ m.lane, 2, 4 @ m.warp, 4), (256, 32, 16, 4, 1))
            )
            b_view = T.tensor_view(
                T.ptr_of(b), layout=((2, 4 @ m.warp, 4, 8 @ m.lane), (128, 32, 8, 1))
            )
            T.copy(a_view, a_frag)
            T.copy(b_view, b_frag)
            T.fill(acc, 0.0)
            T.tiled_mma(acc, a_frag, b_frag, atom=T.cuda.sm89.Mma())
            c_view = T.tensor_view(
                T.ptr_of(c), layout=((2, 8 @ m.lane, 4 @ m.warp, 2), (64, 8, 2, 1))
            )
            T.copy(acc, c_view)

    @prim_func(target=CpuTarget())
    def fp8_host(
        a: Tensor[(16, 32), "fp8e4m3"],
        b: Tensor[(32, 8), "fp8e4m3"],
        c: Tensor[(16, 8), "f32"],
    ):
        launch(fp8_device, a, b, c, grid=(1, 1, 1), block=(32, 1, 1))
