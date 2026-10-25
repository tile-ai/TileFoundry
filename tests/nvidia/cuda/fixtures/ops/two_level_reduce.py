"""A mean reduce on a composed cta x thread mesh, launched one CTA per two rows."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.ir.core.kinds import ReduceKind
from tilefoundry.ir.types.shard_layout import Broadcast
from tilefoundry.target import CpuTarget
from tilefoundry_nvidia import CudaTarget

_CUDA = CudaTarget("nvidia.h200_sxm")


@module(topologies=(Topology("cta", 4), Topology("thread", 128)))
class _TwoLevel:
    """A reduce sharded over a cta×thread mesh, as a multi-CTA finalize writes it.

    The CTA level takes its rows through coordinate arithmetic on the view, so
    both operands leave it broadcast; the thread level alone splits a reduced
    axis. That is the operand shape every nested cta/thread scope produces.
    """

    @prim_func(target=_CUDA)
    def two_level(a_two: Tensor[(8, 4, 128, 4), "f32"], out_two: Tensor[(8, 4, 1, 1), "f32"]):
        with Mesh(
            (Topology("cta", 4), Topology("thread", 128)),
            Layout(shape=(4, 128), strides=(128, 1)),
            ("c", "t"),
        ) as ml:
            two_src = T.tensor_view(
                T.ptr_of(a_two[ml.c * 2 : ml.c * 2 + 2, 0:4, 0:128, 0:4]),
                layout=ShardLayout(
                    Layout((2, 4, 128, 4), (2048, 512, 4, 1)), (Broadcast(), Split(2)), ml
                ),
                shape=(2, 4, 128, 4),
            )
            two_dst = T.tensor_view(
                T.ptr_of(out_two[ml.c * 2 : ml.c * 2 + 2, 0:4, 0:1, 0:1]),
                layout=ShardLayout(
                    Layout((2, 4, 1, 1), (4, 1, 1, 1)), (Broadcast(), Broadcast()), ml
                ),
                shape=(2, 4, 1, 1),
            )
            two_ws = T.alloc_tensor(Tensor[(4,), "f32", None, "smem"])
            T.reduce(
                two_src,
                two_dst,
                two_ws,
                axes=(2, 3),
                keepdim=True,
                kind=ReduceKind.MEAN,
            )


@module(entry="two_level_host", target=_CUDA)
class TwoLevelReduce:
    """Launch one CTA per two rows, 128 threads each."""

    two_level = _TwoLevel

    @prim_func(target=CpuTarget())
    def two_level_host(
        a_two: Tensor[(8, 4, 128, 4), "f32"], out_two: Tensor[(8, 4, 1, 1), "f32"]
    ):
        launch(two_level.two_level, a_two, out_two, grid=(4, 1, 1), block=(128, 1, 1))
