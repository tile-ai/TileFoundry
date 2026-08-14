from __future__ import annotations

from tilefoundry.module import module
from tilefoundry import func
from tilefoundry.dsl.tf import *  # noqa: F401, F403
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.storage import gmem, host, rmem, smem, tmem  # noqa: F401
from tilefoundry.ir.types.shard import (
    B, S, P, ComposedLayout, Layout, Mesh, ShardLayout, Topology,
)
from tilefoundry.ir.types.dim import DimVar, ceildiv

seq_len = DimVar("seq_len", 1, 4)

gpu = Mesh((Topology("gpu", 8192),), Layout((32, 2, 8, 32), (2048, 1024, 32, 1)), names=('cluster', 'cta', 'warp', 'lane'))
thread = Mesh((Topology("thread", 192),), Layout((6, 32), (32, 1)), names=('w', 't'))
thread_2 = Mesh((Topology("thread", 128),), Layout((4, 32), (32, 1)), names=('y', 't'))
cta = Mesh((Topology("cta", 128),), Layout((128,), (1,)), names=('cta',))
cta_2 = Mesh((Topology("cta", 8),), Layout((8,), (1,)), names=('w',))
cta_3 = Mesh((Topology("cta", 8),), Layout((8,), (1,)), names=())

@module(entry="split_inline_and_default_broadcast", topologies=(Topology("cta", 8),))
class HirSharded:
    @func
    def partial_brace_value_state(
        a: Tensor[(64, 128), "bf16", ((32 @ gpu.cluster, 64), {gpu.warp @ P("sum")}), "smem"]
    ) -> Tensor[(64, 128), "f32"]:
        return a

    @func
    def multi_axis_split_with_remainder(
        a: Tensor[(1, 1536), "f32", (1, 6 @ thread.w, 32 @ thread.t, 8), "smem"]
    ) -> Tensor[(1, 1536), "f32"]:
        return a

    @func
    def explicit_strides(
        a: Tensor[(12, 4), "f32", (12 @ thread_2.y, 4), "smem"]
    ) -> Tensor[(12, 4), "f32"]:
        return a

    @func
    def int_at_a_single_axis_mesh(
        a: Tensor[(1, 8192), "f32", (1, 128 @ cta.cta, 64), "smem"]
    ) -> Tensor[(1, 8192), "f32"]:
        return a

    @func
    def mesh_axis_as_a_position_coordinate(
    ) -> Tensor[(), "i64"]:
        v0 = arange(end=8, start=0, step=1, dtype="i64")
        v1 = reshard(v0, layout=ShardLayout(
            layout=Layout((8,), (1,)),
            attrs=(S(0),),
            mesh=cta_2,
        ), storage=rmem)
        v2 = local(v1)
        v3 = reshape(v2, new_shape=())
        return v3

    @func
    def reshard_with_a_dynamic_and_a_closure_axis(
        q: Tensor[(1, seq_len, 32, 128), "bf16"]
    ) -> Tensor[(1, seq_len, 32, 128), "bf16"]:
        v0 = reshard(q, layout=ShardLayout(
            layout=Layout((1, seq_len, 8, 4, 128), None),
            attrs=(S(2),),
            mesh=Mesh((Topology("cta", 8),), Layout((8,), (1,))),
        ))
        return v0

    @func
    def split_inline_and_default_broadcast(
        a: Tensor[(32, 128), "bf16", (32 @ gpu.cluster, 2 @ gpu.cta, 64), "smem"]
    ) -> Tensor[(32, 128), "f32"]:
        return a
