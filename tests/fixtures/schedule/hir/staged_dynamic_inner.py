"""Reject a staged ring whose inner HIR loop depends on its executing thread."""

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 32)),
)
class StagedDynamicInner:
    @func
    def gemm(x: Tensor[(1, 8), "bf16"]) -> Tensor[(1, 8), "bf16", "rmem"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(32,), names=("lane",)) as threads:
                acc = tf.zeros(Tensor[(1, 8), "bf16", "rmem"])
                for i in range(2):
                    for j in range(threads.lane):
                        ring = tf.empty(
                            Tensor[(1, 8), "bf16", Layout((1, 8), (8, 1)), "smem"], buffers=2
                        )
                        row = tf.schedule((x, ring), op=T.copy_async())
                        held = tf.schedule(
                            (row, tf.empty(Tensor[(1, 8), "bf16", (1, 8), "rmem"])), op=T.copy()
                        )
                        acc = tf.insert_slice(acc, held, (0, 0))
                return acc
