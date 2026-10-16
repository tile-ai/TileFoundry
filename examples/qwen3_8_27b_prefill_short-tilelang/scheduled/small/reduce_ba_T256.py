"""Scheduled HIR: finish of the GDN b|a projections at M = 256, 8 K slices:
ba = bf16(sum_s P[s]) for the 96 columns [b (48) | a (48)].
Implements: model.py Qwen3_8DeltaNetLayer.gated_delta_net, matmul(x, w_b, NK) and
matmul(x, w_a, NK) (second half: the slices come from scheduled/gemm/gdn_ba_T256.py).
Runtime kernel: runtime/kernels/gemm.py ba_reduce_kernel(M=256, S=8, 16640, 16384):
one CTA per row, thread n owns column n.
Differences from the kernel: the kernel adds the slices from the last to the first
(cuBLAS's split-K order, see gemm.py BA_N); this program states the sum without an
order. The kernel writes the 96 values into columns 16384..16479 of the (M, 16640)
GDN projection buffer; this program's result is the (M, 96) block itself.
"""
from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

M = 256
S = 8
N = 96


@module(entry="reduce_ba", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", M), Topology("thread", N)))
class SCHED_REDUCE_BA_T256:
    @func
    def reduce_ba(p: Tensor[(S, M, N), "f32"]) -> Tensor[(M, N), "bf16"]:
        with Mesh(("cta",), layout=(M,), names=("row",)) as cta:
            with Mesh(("thread",), layout=(N,), names=("n",)) as th:
                out = tf.zeros(Tensor[(M, N), "bf16"])
                pp = tf.schedule((tf.reshape(p[:, cta.row:cta.row + 1, :], new_shape=(S, N)), tf.empty(Tensor[(S, N), "f32", ((S, N @ th.n), (1, 0)), "rmem"])), op=T.copy())
                total = tf.reduce(pp, axes=(0,), keepdim=True, kind="sum")
                c = tf.cast(total, dtype="bf16")
                co = tf.schedule((c, tf.empty(Tensor[(1, N), "bf16", (1, N)])), op=T.copy())
                out = tf.insert_slice(out, co, (cta.row, 0))
            return out
