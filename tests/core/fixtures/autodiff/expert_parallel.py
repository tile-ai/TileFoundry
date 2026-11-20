"""Expert parallel on two GPUs, one expert each: bucket locally, all-to-all, apply, all-to-all back.

Each GPU holds its own tokens and routes `n` of them to every expert. Swapping the
split axis of the bucketed tokens is the all-to-all: before it a GPU holds what it
sends, after it what its expert receives.
"""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu

P, NL, C, F = 2, 8, 8, 16
n = DimVar("n", 1, NL)


@module(entry="forward", target=cpu.Target(), topologies=(Topology("gpu", P),))
class ExpertParallel:
    mesh = Mesh(("gpu",), (P,), names=("ep",))

    @func
    def forward(tokens: Tensor[(P @ mesh.ep, NL, C), "f32", "gmem"], route: Tensor[(P * n,), "i32", "gmem"],
                w: Tensor[(C, F), "f32", "gmem"]):
        with mesh:
            sent = tf.index_select(tokens, route, dim=1)
            received = tf.reshard(sent, layout=(P, P * n @ mesh.ep, C), storage="gmem")
            applied = tf.matmul(received, w)
            return tf.reshard(applied, layout=(P @ mesh.ep, P * n, F), storage="gmem")
