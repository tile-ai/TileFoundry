"""A target-less child whose f32 shared-memory matmul no CUDA MMA reads.

It parses on its own; a CUDA root that calls it is refused where it makes the call.
"""

from __future__ import annotations

from tests.nvidia.cuda.fixtures.logical.child_matmul_inherited_target import K, M, N
from tilefoundry.dsl import *


@module(entry="run", topologies=(Topology("cta", 1),))
class RefusedChildMatmul:
    @func
    def run(a: Tensor[(M, K), "f32"], b: Tensor[(K, N), "f32"]):
        with Mesh(("cta",), layout=(1,), names=("g",)) as _cta:
            acc = tf.matmul(
                tf.reshard(a[:, 0:16], (M, 16), "smem"),
                tf.reshard(b[0:16, :], (16, N), "smem"),
            )
            for k in tf.tile(16, K, 16):  # noqa: F405 -- authored tile loop
                acc = acc + tf.matmul(
                    tf.reshard(a[:, k], (M, 16), "smem"),
                    tf.reshard(b[k, :], (16, N), "smem"),
                )
            return tf.reshard(acc, (M, N), "gmem")
