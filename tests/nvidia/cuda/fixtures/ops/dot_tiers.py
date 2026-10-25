"""Both dot tiers in one host entry: a warp butterfly and a block contraction."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.ir.types.shard_layout import Broadcast
from tilefoundry.target import CpuTarget
from tilefoundry_nvidia import CudaTarget


@module(topologies=(Topology("thread", 32),))
class DotWarp:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def dot_warp(
        warp_a: Tensor[(32, 32), "f32"],
        warp_b: Tensor[(32,), "f32"],
        warp_c: Tensor[(32,), "f32"],
    ):
        with Mesh((Topology("thread", 32),), Layout(shape=(32,), strides=(1,)), ("t",)) as mw:
            warp_a_view = T.tensor_view(
                T.ptr_of(warp_a),
                layout=ShardLayout(
                    layout=Layout(shape=(32, 32), strides=(32, 1)),
                    attrs=(Split(0),),
                    mesh=mw,
                ),
            )
            warp_b_view = T.tensor_view(
                T.ptr_of(warp_b),
                layout=ShardLayout(
                    layout=Layout(shape=(32,), strides=(1,)), attrs=(Broadcast(),), mesh=mw
                ),
            )
            warp_c_view = T.tensor_view(
                T.ptr_of(warp_c),
                layout=ShardLayout(
                    layout=Layout(shape=(32,), strides=(1,)), attrs=(Split(0),), mesh=mw
                ),
            )
            T.dot(warp_a_view, warp_b_view, warp_c_view)


@module(topologies=(Topology("thread", 128),))
class DotCta:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def dot_cta(
        cta_a: Tensor[(128,), "f32"], cta_b: Tensor[(128,), "f32"], cta_c: Tensor[(1,), "f32"]
    ):
        with Mesh((Topology("thread", 128),), Layout(shape=(128,), strides=(1,)), ("t",)) as mc:
            cta_a_view = T.tensor_view(
                T.ptr_of(cta_a), layout=ShardLayout(Layout((128,), (1,)), (Split(0),), mc)
            )
            cta_b_view = T.tensor_view(
                T.ptr_of(cta_b), layout=ShardLayout(Layout((128,), (1,)), (Split(0),), mc)
            )
            cta_c_view = T.tensor_view(
                T.ptr_of(cta_c), layout=ShardLayout(Layout((1,), (1,)), (Broadcast(),), mc)
            )
            cta_ws = T.alloc_tensor(Tensor[(4,), "f32", None, "smem"])
            T.dot(cta_a_view, cta_b_view, cta_c_view, cta_ws)


@module(entry="dot_tiers_host", target=CudaTarget("nvidia.h200_sxm"))
class DotTiers:
    warp = DotWarp
    cta = DotCta

    @prim_func(target=CpuTarget())
    def dot_tiers_host(
        warp_a: Tensor[(32, 32), "f32"],
        warp_b: Tensor[(32,), "f32"],
        warp_c: Tensor[(32,), "f32"],
        cta_a: Tensor[(128,), "f32"],
        cta_b: Tensor[(128,), "f32"],
        cta_c: Tensor[(1,), "f32"],
    ):
        launch(warp.dot_warp, warp_a, warp_b, warp_c, grid=(1, 1, 1), block=(32, 1, 1))
        launch(cta.dot_cta, cta_a, cta_b, cta_c, grid=(1, 1, 1), block=(128, 1, 1))
