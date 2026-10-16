"""Scheduled HIR: split-K finish of o_proj/down at M = 128, 2 slices: C = bf16(R + bf16(sum_s P[s])).
Runtime kernel: runtime/kernels/gemm.py splitk_reduce_kernel(M=128, N=5120, S=2, residual=True): 1024 threads x 8 elements per CTA."""
from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

M = 128
N = 5120
S = 2
BLOCK = 1024
PER = 8
SPAN = BLOCK * PER
TILES = M * N // SPAN


@module(entry="reduce_sk", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", TILES), Topology("thread", BLOCK)))
class SCHED_REDUCE_SK_T128:
    @func
    def reduce_sk(p: Tensor[(S, M * N), "f32"], r: Tensor[(M * N,), "bf16"]) -> Tensor[(M * N,), "bf16"]:
        with Mesh(("cta",), layout=(TILES,), names=("tile",)) as cta:
            base = cta.tile * SPAN
            with Mesh(("thread",), layout=(BLOCK,), names=("t",)) as th:
                out = tf.zeros(Tensor[(M * N,), "bf16"])
                pp = tf.schedule((p[:, base:base + SPAN], tf.empty(Tensor[(S, SPAN), "f32", ((S, BLOCK @ th.t, PER), (PER, 0, 1)), "rmem"])), op=T.copy())
                rr = tf.schedule((r[base:base + SPAN], tf.empty(Tensor[(SPAN,), "bf16", ((BLOCK @ th.t, PER), (0, 1)), "rmem"])), op=T.copy())
                total = tf.reduce(pp, axes=(0,), keepdim=False, kind="sum")
                c = tf.cast(tf.cast(rr, dtype="f32") + tf.cast(tf.cast(total, dtype="bf16"), dtype="f32"), dtype="bf16")
                co = tf.schedule((c, tf.empty(Tensor[(SPAN,), "bf16", (SPAN,)])), op=T.copy())
                out = tf.insert_slice(out, co, base)
            return out
