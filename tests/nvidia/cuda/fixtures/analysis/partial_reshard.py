"""Complete replicated partials while retaining an independent tensor split."""

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

REPLICAS = DimVar("replicas", 1, 4)
SHARDS = DimVar("shards", 1, 4)
GPU = Topology("gpu", REPLICAS * SHARDS)
MESH = Mesh((GPU,), Layout((REPLICAS, SHARDS), (SHARDS, 1)), ("replica", "shard"))
LOCAL = Layout(((SHARDS, 8), 8), ((0, 8), 1))
PARTIAL = ShardLayout(LOCAL, (Partial("sum"), Split(0)), MESH)


@module(
    entry="complete",
    target=cuda.Target("nvidia.h200_sxm", device_count=16),
    topologies=(GPU, Topology("cta", 1)),
)
class PartialReshard:
    @func
    def complete(x: Tensor[(SHARDS * 8, 8), "f32", PARTIAL]):
        with Mesh(("gpu",), (REPLICAS, SHARDS), names=("replica", "shard")) as mesh:
            completed = tf.reshard(x, (SHARDS * 8 @ mesh.shard, 8), "gmem")
            unchanged = tf.reshard(
                x, (SHARDS * 8 @ mesh.shard, 8) @ {mesh.replica @ P("sum")}, "gmem"
            )
            return completed, unchanged
