"""A tensor-map round trip whose staging tile alone exceeds shared memory.

The memory family reports the placed peak above the 227 KiB a CTA may use; the
mbarrier word after it adds no second error.
"""

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda


@module(
    entry="round_trip",
    target=cuda.Target("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 32)),
)
class TmaPeakOverCapacity:
    @func
    def round_trip(x: Tensor[(2, 228, 256), "bf16"]) -> Tensor[(2, 228, 256), "bf16"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(32,), names=("lane",)) as _warp:
                staged = tf.schedule(
                    (x, tf.empty(Tensor[(2, 228, 256), "bf16", (2, 228, 256), "smem"])),
                    op=T.copy_async_tensor(),
                )
                return tf.schedule(
                    (staged, tf.empty(Tensor[(2, 228, 256), "bf16", (2, 228, 256)])),
                    op=T.copy_async_tensor(),
                )
