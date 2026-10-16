"""Scheduled HIR: lm_head over the last row, (1, 5120) x (248320, 5120)^T, as a GEMV (no MMA:
one row reads every weight byte once, 2.54 GB, memory-bound at every length).
Runtime kernel: runtime/kernels/small.py gemv_kernel(N=248320, K=5120, rows=16, threads=256)."""
from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

N = 248320
K = 5120
ROWS = 16
WARPS = 8
RPW = ROWS // WARPS
VEC = 8
CHUNKS = K // (32 * VEC)


@module(entry="lm_head", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", N // ROWS), Topology("thread", 256)))
class SCHED_LM_HEAD:
    @func
    def lm_head(x: Tensor[(1, K), "bf16"], w: Tensor[(N, K), "bf16"]) -> Tensor[(1, N), "bf16"]:
        with Mesh(("cta",), layout=(N // ROWS,), names=("blk",)) as cta:
            n = cta.blk * ROWS
            with Mesh(("thread",), layout=(WARPS, 32), names=("warp", "lane")) as th:
                out = tf.zeros(Tensor[(1, N), "bf16"])
                wt = tf.schedule((w[n:n + ROWS, :], tf.empty(Tensor[(ROWS, K), "bf16", ((WARPS @ th.warp, RPW, CHUNKS, 32 @ th.lane, VEC), (0, CHUNKS * VEC, VEC, 0, 1)), "rmem"])), op=T.copy())
                xt = tf.schedule((x, tf.empty(Tensor[(1, K), "bf16", ((1, CHUNKS, 32 @ th.lane, VEC), (0, VEC, 0, 1)), "rmem"])), op=T.copy())
                prod = tf.cast(wt, dtype="f32") * tf.cast(xt, dtype="f32")
                logit = tf.cast(tf.reduce(prod, axes=(-1,), keepdim=False, kind="sum"), dtype="bf16")
                lo = tf.schedule((tf.reshape(logit, new_shape=(1, ROWS)), tf.empty(Tensor[(1, ROWS), "bf16", (1, ROWS)])), op=T.copy())
                out = tf.insert_slice(out, lo, (0, n))
            return out
