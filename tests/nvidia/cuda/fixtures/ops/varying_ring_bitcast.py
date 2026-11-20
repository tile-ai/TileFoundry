"""Each CTA stages its own weight rows in a swizzled ring, then reads them K-major.

The rows start at ``cta.n * 80``, so the ring is Varying on ``n``. The TMA
copy matches its swizzle through the stated ``@ {cta.n @ V()}``. The bitcast
re-lays each CTA's copy alone: neither end splits, so shared memory is
accepted and the result keeps ``V`` on ``n``.
"""

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda


@module(entry="run", target=cuda.Target("nvidia.h200_sxm"), topologies=(Topology("cta", 2), Topology("thread", 128)))
class RingBitcast:
    @func
    def run(w: Tensor[(160, 64), "bf16"]) -> Tensor[(160, 64), "bf16"]:
        with Mesh(("cta",), layout=(2,), names=("n",)) as cta:
            with Mesh(("thread",), layout=(128,), names=("t",)) as threads:
                out = tf.empty(Tensor[(160, 64), "bf16"])
                row = cta.n * 80
                with threads[0:32] as _loader:
                    ring = tf.empty(Tensor[(80, 64), "bf16", (Layout(((10, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3)) @ {cta.n @ V()}, "smem"])
                    rows = tf.schedule((w[row:row + 80, :], ring), op=T.copy_async_tensor())
                k_major = tf.bitcast(rows, Layout(((4, 16), (10, 8)), ((16, 1), (512, 64))) | Swizzle(3, 4, 3))
                back = tf.bitcast(k_major, Layout(((10, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3))
                return tf.insert_slice(out, back, (row, 0))
