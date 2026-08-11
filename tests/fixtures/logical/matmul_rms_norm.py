"""A matmul followed by an RMS norm, repeated in a child Module.

Selectors, dimension parsing, topology rejection, and report rendering consume
this program. It deliberately carries no Mesh, Reshard, or ShardLayout.
"""

from tilefoundry import func, module
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import matmul, rms_norm
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget


@module(entry="root", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1), Topology("thread", 128)))
class CMine:
    @func
    def root(
        x: Tensor[(16, 16), "bf16"],
        w: Tensor[(16, 16), "bf16"],
        weight: Tensor[(16,), "f32"],
    ) -> Tensor[(16, 16), "bf16"]:
        h = matmul(x, w)
        return rms_norm(h, weight)

    @module(entry="inner")
    class child:
        @func
        def inner(
            x: Tensor[(16, 16), "bf16"],
            w: Tensor[(16, 16), "bf16"],
            weight: Tensor[(16,), "f32"],
        ) -> Tensor[(16, 16), "bf16"]:
            h = matmul(x, w)
            return rms_norm(h, weight)
