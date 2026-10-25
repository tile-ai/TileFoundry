"""An accumulator carried across sibling thread regions of one mesh."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(
    entry="run",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 256)),
)
class SiblingRegionChain:
    @func
    def run(
        a: Tensor[(64, 32), "bf16"],
        c: Tensor[(64, 32), "bf16"],
    ) -> Tensor[(64, 32), "bf16"]:
        with Mesh(("cta",), (1,), names=("block",)) as _cta:
            with Mesh(
                ("thread",),
                (2, 128),
                names=("role", "participant"),
            ) as threads:
                with threads[1, :] as _compute:
                    acc = tf.zeros(Tensor[(64, 32), "bf16"])
                with threads[0, :32] as _loader:
                    lhs = tf.cast(a, dtype="bf16")
                with threads[1, :] as _compute:
                    acc = acc + lhs
                with threads[0, :32] as _loader:
                    rhs = tf.cast(c, dtype="bf16")
                with threads[1, :] as _compute:
                    acc = acc + rhs
                return acc
