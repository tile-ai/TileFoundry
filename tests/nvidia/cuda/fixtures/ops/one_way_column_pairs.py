"""A plain (4, 64) tile copied into a column-pair register shard, one way only."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget
from tilefoundry_nvidia import CudaTarget

_CUDA = CudaTarget("nvidia.h200_sxm")


@module(
    target=_CUDA,
    topologies=(Topology("thread", 32),),
)
class _OneWay:
    """A plain (4, 64) tile entering a column-pair shard, one direction only.

    Lane ``t`` of 32 owns columns ``2t`` and ``2t+1`` of every row: the
    plain->shard leg takes the ``local_as`` projection, the shard->shard leg
    the path that predates it, so a slice placed wrong on the way in lands in
    the wrong output columns directly.
    """

    @prim_func(target=_CUDA)
    def one_way(a: Tensor[(4, 64), "f32"], out: Tensor[(4, 64), "f32"]):
        with Mesh((Topology("thread", 32),), layout=(32,), names=("t",)) as m:
            src = T.tensor_view(T.ptr_of(a), layout=((4, 64), (64, 1)))
            reg = T.alloc_tensor(
                tensor_type=Tensor[(4, 64), "f32", ((4, 32 @ m.t, 2), (64, 2, 1)), "rmem"]
            )
            dst = T.tensor_view(T.ptr_of(out), layout=((4, 32 @ m.t, 2), (64, 2, 1)))
            T.copy(src, reg)
            T.copy(reg, dst)


@module(entry="one_way_host", target=_CUDA)
class OneWay:
    """Launch the one warp the shard is written for."""

    one_way = _OneWay

    @prim_func(target=CpuTarget())
    def one_way_host(a: Tensor[(4, 64), "f32"], out: Tensor[(4, 64), "f32"]):
        launch(  # noqa: F821
            one_way.one_way,  # noqa: F821
            a,
            out,
            grid=(1, 1, 1),
            block=(32, 1, 1),
        )
