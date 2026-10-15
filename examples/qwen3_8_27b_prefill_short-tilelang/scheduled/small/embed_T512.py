"""Scheduled HIR: embedding gather for M = 512 prompt tokens: out[t] = table[ids[t]].
Runtime kernel: runtime/kernels/small.py embed_kernel(M=512): one CTA per token, the row staged through shared memory by cp.async, then stored."""
from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

M = 512
V = 248320
H = 5120
TH = 128
PER = H // TH


@module(entry="embed", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", M), Topology("thread", TH)))
class SCHED_EMBED_T512:
    @func
    def embed(ids: Tensor[(M,), "i64"], table: Tensor[(V, H), "bf16"]) -> Tensor[(M, H), "bf16"]:
        with Mesh(("cta",), layout=(M,), names=("row",)) as cta:
            out = tf.zeros(Tensor[(M, H), "bf16"])
            with Mesh(("thread",), layout=(TH,), names=("t",)) as th:
                one = ids[cta.row:cta.row + 1]
                row = tf.schedule((table, tf.empty(Tensor[(1, H), "bf16", (1, H), "smem"]), one), op=T.copy_async(fill=0.0))
                ro = tf.schedule((row, tf.empty(Tensor[(1, H), "bf16", (1, H)])), op=T.copy())
                out = tf.insert_slice(out, ro, (cta.row, 0))
            return out
