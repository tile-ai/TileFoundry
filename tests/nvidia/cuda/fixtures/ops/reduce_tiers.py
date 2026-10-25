"""Every reduce tier in one kernel, each forced there by its layouts alone."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.ir.core.kinds import ReduceKind
from tilefoundry.ir.types.shard_layout import Broadcast
from tilefoundry.target import CpuTarget
from tilefoundry_nvidia import CudaTarget

_CUDA = CudaTarget("nvidia.h200_sxm")


@module(topologies=(Topology("thread", 128),))
class _Plain:
    @prim_func(target=_CUDA)
    def plain(a_plain: Tensor[(128, 8), "f32"], out_plain: Tensor[(128,), "f32"]):
        with Mesh((Topology("thread", 128),), Layout(shape=(128,), strides=(1,)), ("t",)) as mp:
            plain_src = T.tensor_view(
                T.ptr_of(a_plain), layout=ShardLayout(Layout((128, 8), (8, 1)), (Split(0),), mp)
            )
            plain_dst = T.tensor_view(
                T.ptr_of(out_plain), layout=ShardLayout(Layout((128,), (1,)), (Split(0),), mp)
            )
            T.reduce(
                plain_src,
                plain_dst,
                axes=(1,),
                keepdim=False,
                kind=ReduceKind.MEAN,
            )


@module(topologies=(Topology("thread", 32),))
class _Warp:
    @prim_func(target=_CUDA)
    def intra_warp(a_warp: Tensor[(32, 4), "f32"], out_warp: Tensor[(1,), "f32"]):
        with Mesh((Topology("thread", 32),), Layout(shape=(32,), strides=(1,)), ("t",)) as mw:
            warp_src = T.tensor_view(
                T.ptr_of(a_warp), layout=ShardLayout(Layout((32, 4), (4, 1)), (Split(0),), mw)
            )
            warp_dst = T.tensor_view(
                T.ptr_of(out_warp), layout=ShardLayout(Layout((1,), (1,)), (Broadcast(),), mw)
            )
            T.reduce(
                warp_src,
                warp_dst,
                axes=(1,),
                keepdim=False,
                kind=ReduceKind.ABS_MAX,
            )


@module(topologies=(Topology("thread", 128),))
class _Cta:
    @prim_func(target=_CUDA)
    def intra_cta(a_cta: Tensor[(4, 32, 8), "f32"], out_cta: Tensor[(1,), "f32"]):
        with Mesh(
            (Topology("thread", 128),), Layout(shape=(4, 32), strides=(32, 1)), ("w", "t")
        ) as mc:
            cta_src = T.tensor_view(
                T.ptr_of(a_cta),
                layout=ShardLayout(Layout((4, 32, 8), (256, 8, 1)), (Split(0), Split(1)), mc),
            )
            cta_dst = T.tensor_view(
                T.ptr_of(out_cta),
                layout=ShardLayout(Layout((1,), (1,)), (Broadcast(), Broadcast()), mc),
            )
            cta_ws = T.alloc_tensor(Tensor[(4,), "f32", None, "smem"])
            T.reduce(
                cta_src,
                cta_dst,
                cta_ws,
                axes=(2,),
                keepdim=False,
                kind=ReduceKind.MEAN,
            )


@module(topologies=(Topology("thread", 128),))
class _Cross:
    @prim_func(target=_CUDA)
    def cross_warp(a_cross: Tensor[(4, 32), "f32"], out_cross: Tensor[(1, 32), "f32"]):
        with Mesh(
            (Topology("thread", 128),), Layout(shape=(4, 32), strides=(32, 1)), ("w", "t")
        ) as mx:
            cross_src = T.tensor_view(
                T.ptr_of(a_cross),
                layout=ShardLayout(Layout((4, 32), (32, 1)), (Split(0), Split(1)), mx),
            )
            cross_dst = T.tensor_view(
                T.ptr_of(out_cross),
                layout=ShardLayout(Layout((1, 32), (32, 1)), (Broadcast(), Split(1)), mx),
            )
            cross_ws = T.alloc_tensor(Tensor[(128,), "f32", None, "smem"])
            T.reduce(
                cross_src,
                cross_dst,
                cross_ws,
                axes=(0,),
                keepdim=True,
                kind=ReduceKind.ABS_MAX,
            )


@module(entry="reduce_tiers_host", target=_CUDA)
class ReduceTiers:
    plain = _Plain
    warp = _Warp
    cta = _Cta
    cross = _Cross

    @prim_func(target=CpuTarget())
    def reduce_tiers_host(
        a_plain: Tensor[(128, 8), "f32"],
        out_plain: Tensor[(128,), "f32"],
        a_warp: Tensor[(32, 4), "f32"],
        out_warp: Tensor[(1,), "f32"],
        a_cta: Tensor[(4, 32, 8), "f32"],
        out_cta: Tensor[(1,), "f32"],
        a_cross: Tensor[(4, 32), "f32"],
        out_cross: Tensor[(1, 32), "f32"],
    ):
        launch(plain.plain, a_plain, out_plain, grid=(1, 1, 1), block=(128, 1, 1))
        launch(warp.intra_warp, a_warp, out_warp, grid=(1, 1, 1), block=(32, 1, 1))
        launch(cta.intra_cta, a_cta, out_cta, grid=(1, 1, 1), block=(128, 1, 1))
        launch(cross.cross_warp, a_cross, out_cross, grid=(1, 1, 1), block=(128, 1, 1))
