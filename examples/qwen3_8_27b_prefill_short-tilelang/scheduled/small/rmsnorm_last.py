"""Scheduled HIR: final_norm of the last row. Rows M = 1.
Runtime kernel: runtime/kernels/small.py rmsnorm_kernel(M=1, H=5120, rows=1).

Decomposition (as the kernel): CTA = 1 row(s), one warp per row; lane l holds the
4-element chunks l, l + 32, ..., l + 32 * (NCH - 1) of its row (register index 4c + i);
the mean of squares is one cross-lane reduce. The kernel fixes the f32 summation order
to torch's (per-lane accumulators per chunk position, ((a0+a1)+a2)+a3, shfl_down
16..1, times f32(1/H)) so its output is bitwise equal to the HIR; this program states
the reduce without an order.
"""
from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

M = 1
H = 5120
ROWS = 1
NCH = H // 128
EPS = 1e-6


@module(entry="rmsnorm", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", M // ROWS), Topology("thread", 32 * ROWS)))
class SCHED_RMSNORM_LAST:
    @func
    def rmsnorm(x: Tensor[(M, H), "bf16"], gamma: Tensor[(H,), "bf16"]) -> Tensor[(M, H), "bf16"]:
        with Mesh(("cta",), layout=(M // ROWS,), names=("blk",)) as cta:
            out = tf.zeros(Tensor[(M, H), "bf16"])
            with Mesh(("thread",), layout=(ROWS, 32), names=("w", "lane")) as th:
                xr = tf.schedule((x[cta.blk * ROWS:cta.blk * ROWS + ROWS, :], tf.empty(Tensor[(ROWS, H), "bf16", ((ROWS @ th.w, NCH, 32 @ th.lane, 4), (0, 4, 0, 1)), "rmem"])), op=T.copy())
                gr = tf.schedule((tf.reshape(gamma, new_shape=(1, H)), tf.empty(Tensor[(1, H), "bf16", ((1, NCH, 32 @ th.lane, 4), (0, 4, 0, 1)), "rmem"])), op=T.copy())
                xf = tf.cast(xr, dtype="f32")
                ms = tf.reduce(xf * xf, axes=(-1,), keepdim=True, kind="mean")
                y = tf.cast(xf * tf.rsqrt(ms + EPS) * (tf.cast(gr, dtype="f32") + 1.0), dtype="bf16")
                yo = tf.schedule((y, tf.empty(Tensor[(ROWS, H), "bf16", (ROWS, H)])), op=T.copy())
                out = tf.insert_slice(out, yo, (cta.blk * ROWS, 0))
            return out
