"""One (2, 4) parameter copied into another, each side seen through a different layout."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu
from tilefoundry_nvidia import cuda

_CUDA = cuda.Target("nvidia.h200_sxm")


@module(
    entry="view_into_parameter_host",
    target=_CUDA,
    topologies=(Topology("thread", 1),),
)
class ViewIntoParameter:
    """Copy a rank-2 view into a parameter that CUDA holds as one flat mode."""

    @prim_func(target=_CUDA)
    def view_into_parameter_device(a: Tensor[(2, 4), "f32"], out: Tensor[(2, 4), "f32"]):
        with Mesh((Topology("thread", 1),), layout=(1,), names=("t",)) as _thread:
            rows = T.tensor_view(T.ptr_of(a), layout=((2, 4), (4, 1)))
            T.copy(rows, out)

    @prim_func(target=cpu.Target())
    def view_into_parameter_host(a: Tensor[(2, 4), "f32"], out: Tensor[(2, 4), "f32"]):
        launch(view_into_parameter_device, a, out, grid=(1, 1, 1), block=(1, 1, 1))


@module(
    entry="columns_into_rows_host",
    target=_CUDA,
    topologies=(Topology("thread", 1),),
)
class ColumnsIntoRows:
    """Copy a parameter stored by columns into a parameter stored by rows."""

    @prim_func(target=_CUDA)
    def columns_into_rows_device(
        a: Tensor[(2, 4), "f32", Layout((2, 4), (1, 2))], out: Tensor[(2, 4), "f32"]
    ):
        with Mesh((Topology("thread", 1),), layout=(1,), names=("t",)) as _thread:
            T.copy(a, out)

    @prim_func(target=cpu.Target())
    def columns_into_rows_host(
        a: Tensor[(2, 4), "f32", Layout((2, 4), (1, 2))], out: Tensor[(2, 4), "f32"]
    ):
        launch(columns_into_rows_device, a, out, grid=(1, 1, 1), block=(1, 1, 1))
