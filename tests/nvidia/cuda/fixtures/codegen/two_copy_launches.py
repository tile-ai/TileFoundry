"""A host entry that launches two copy kernels, one after the other."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget
from tilefoundry_nvidia import CudaTarget


@module(entry="host", target=CudaTarget("nvidia.h200_sxm"))
class _TwoCopyLaunches:
    """Two concrete kernels used by the end-to-end multi-launch check."""

    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def first(x: Tensor[(1, 128), "f32"], out: Tensor[(1, 128), "f32"]):
        with Mesh((Topology("thread", 1),), Layout((1,), (1,))) as thread:
            view = T.tensor_view(
                T.ptr_of(x),
                layout=ShardLayout(
                    layout=Layout((1, 128), (128, 1)),
                    attrs=(B(),),
                    mesh=thread,
                ),
            )
            out_view = T.tensor_view(
                T.ptr_of(out),
                layout=ShardLayout(
                    layout=Layout((1, 128), (128, 1)),
                    attrs=(B(),),
                    mesh=thread,
                ),
            )
            T.copy(view, out_view)
            T.sync(thread)

    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def second(x: Tensor[(1, 128), "f32"], out: Tensor[(1, 128), "f32"]):
        with Mesh((Topology("thread", 1),), Layout((1,), (1,))) as thread:
            view = T.tensor_view(
                T.ptr_of(x),
                layout=ShardLayout(
                    layout=Layout((1, 128), (128, 1)),
                    attrs=(B(),),
                    mesh=thread,
                ),
            )
            out_view = T.tensor_view(
                T.ptr_of(out),
                layout=ShardLayout(
                    layout=Layout((1, 128), (128, 1)),
                    attrs=(B(),),
                    mesh=thread,
                ),
            )
            T.copy(view, out_view)
            T.sync(thread)

    @prim_func(target=CpuTarget())
    def host(
        x0: Tensor[(1, 128), "f32"],
        out0: Tensor[(1, 128), "f32"],
        x1: Tensor[(1, 128), "f32"],
        out1: Tensor[(1, 128), "f32"],
    ):
        launch(first, x0, out0, grid=(1, 1, 1), block=(1, 1, 1))  # noqa: F821
        launch(second, x1, out1, grid=(1, 1, 1), block=(1, 1, 1))  # noqa: F821
