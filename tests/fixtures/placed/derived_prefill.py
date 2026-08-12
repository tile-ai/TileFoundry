"""A placed prefill whose execution geometry derives from authored dimensions."""

from tilefoundry import func, module
from tilefoundry.dsl import DimVar, Mesh, Tensor, ceildiv, tf
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

PREFILL_N = DimVar("prefill_n", 1, 65)
TOPOLOGY_ONLY = DimVar("topology_only", 1, 1025)
PREFILL_TILES = ceildiv(PREFILL_N, 8)


@module(
    entry="prefill",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", PREFILL_TILES), Topology("thread", TOPOLOGY_ONLY)),
)
class DerivedPrefill:
    @func
    def prefill(
        x: Tensor[(PREFILL_TILES, 8), "f32"]
    ) -> Tensor[(PREFILL_TILES, 8), "f32"]:
        with Mesh(
            ("cta",), layout=(PREFILL_TILES,), names=("tile",)
        ) as cta:
            local = tf.reshard(
                x, (PREFILL_TILES @ cta.tile, 8), "gmem"
            )
            return tf.relu(local)
