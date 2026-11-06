"""A tensor-map store between two plain reads of the global tile it overwrites.

The plain read before the store is published to the async proxy and met before
the store issues. The read after it needs a barrier alone: the store's
completion already orders its writes before later plain accesses.
"""

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(
    entry="kernel",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 32)),
)
class TmaStoreBetweenReads:
    @func
    def kernel(x: Tensor[(32, 8), "bf16"]) -> Tensor[(64, 8), "bf16", "umat"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(32,), names=("lane",)) as threads:
                scratch = tf.schedule((x, tf.empty(Tensor[(32, 8), "bf16"])), op=T.copy())
                before = tf.schedule(
                    (
                        scratch,
                        tf.empty(Tensor[(32, 8), "bf16", ((32 @ threads.lane, 8), (0, 1)), "rmem"]),
                    ),
                    op=T.copy(),
                )
                staged = tf.schedule(
                    (x, tf.empty(Tensor[(32, 8), "bf16", (32, 8), "smem"])),
                    op=T.copy_async_tensor(),
                )
                stored = tf.schedule((staged, scratch), op=T.copy_async_tensor())
                after = tf.schedule(
                    (
                        stored,
                        tf.empty(Tensor[(32, 8), "bf16", ((32 @ threads.lane, 8), (0, 1)), "rmem"]),
                    ),
                    op=T.copy(),
                )
                output = tf.zeros(Tensor[(64, 8), "bf16"])
                output = tf.insert_slice(output, before, (0, 0))
                return tf.insert_slice(output, after, (32, 0))
