"""One thread of each CTA gathers that CTA's row into shared memory; all threads store it.

The row is Varying on ``row`` (one per CTA). It is written in the ``th[0:1]``
slice and read in the whole thread scope: shared memory is one copy per CTA,
so the writer slice does not limit which threads read it.
"""

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

M = 4
H = 256


@module(entry="embed", target=cuda.Target("nvidia.h200_sxm"), topologies=(Topology("cta", M), Topology("thread", 128)))
class TileOtherSlice:
    @func
    def embed(ids: Tensor[(M,), "i64"], table: Tensor[(16, H), "bf16"]) -> Tensor[(M, H), "bf16"]:
        with Mesh(("cta",), layout=(M,), names=("row",)) as cta:
            with Mesh(("thread",), layout=(128,), names=("t",)) as th:
                out = tf.zeros(Tensor[(M, H), "bf16"])
                one = ids[cta.row:cta.row + 1]
                with th[0:1] as _writer:
                    row = tf.schedule((table, tf.empty(Tensor[(1, H), "bf16", (1, H) @ {cta.row @ V()}, "smem"]), one), op=T.copy_async(fill=0.0))
                stored = tf.schedule((row, tf.empty(Tensor[(1, H), "bf16", (1, H)])), op=T.copy())
                return tf.insert_slice(out, stored, (cta.row, 0))
