"""One feature-dense Module tree for context resolution and printer round trips."""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Tensor, tf
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

CONTEXT_CTA = Topology("cta", 132)
CONTEXT_WARP = Topology("warp", 4)
CONTEXT_THREAD = Topology("thread", 32)


@module(
    entry="forward",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(CONTEXT_CTA, CONTEXT_WARP),
)
class ContextTree:
    @func
    def forward(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return tf.relu(x)

    @func
    def spare(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return tf.square(x)

    @module(entry="step")
    class inherits:
        @func
        def step(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
            return tf.relu(x)

    @module(entry="step", topologies=(CONTEXT_THREAD,))
    class replaces:
        @func
        def step(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
            return tf.square(x)

    @module(topologies=())
    class topology_free:
        @func
        def helper(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
            return tf.relu(x)

    @module
    class nominates_nothing:
        @func
        def helper(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
            return tf.relu(x)
