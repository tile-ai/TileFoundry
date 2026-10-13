"""Scheduled HIR for the RMSNorms: model.py's `input_norm` / `post_norm`
(Qwen3_5RMSNorm, weight 1 + gamma in f32, one bf16 landing).

One Module per sequence length; `schedule finalize` lowers `rmsnorm` to
tir/rmsnorm_{n}.py and kernels/elementwise.py `rmsnorm_kernel` implements it:

  grid      M / R CTAs (R = 2 rows each), 256 threads; every thread works
            on both rows of its CTA's pair.
  ownership a 5120-wide row is (5, 256, 4): thread t holds columns
            (k * 256 + t) * 4 + j, k < 5, j < 4 -- five 8-byte vectors per
            row, so each load is one contiguous 2 KB run per 256 threads.
            (16-byte vectors do not divide 5120 over 256 threads; 128 threads
            per row with 16-byte vectors measured 54.0 vs 41.7 us at 8192.)
  gamma     the same (5, 256, 4) slice of gamma, per thread, in registers.
  reduce    mean of squares over (k, t, j) in f32, per row: per thread over
            (k, j), then across the 256 threads (`T.reduce` with its smem
            workspace; the runtime does a warp butterfly + one smem slot per
            warp, summed in warp order), the result held by every thread.
  store     bf16(x * rsqrt(ms + eps) * (1 + gamma)), the same 8-byte vectors.
"""
from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

_H200 = CudaTarget("nvidia.h200_sxm")
H, R, EPS = 5120, 2, 1e-6
KV, TV, VEC = 5, 256, 4          # H = KV * TV * VEC


def rmsnorm(M: int):
    @module(entry="entry", target=_H200, topologies=(Topology("cta", M // R), Topology("thread", TV)))
    class RMSNorm:
        @func
        def rmsnorm(hidden: Tensor[(1, M, H), "bf16"], gamma: ConstTensor[(H,), "bf16"]):
            with Mesh(("cta",), layout=(M // R,), names=("rb",)) as cta:
                with Mesh(("thread",), layout=(TV,), names=("t",)) as th:
                    x4 = tf.reshape(hidden, (M, KV, TV, VEC))
                    m = cta.rb * R
                    x = tf.schedule((x4[m:m + R, :, :, :],), op=T.copy(rmem_layout=(R, KV, TV @ th.t, VEC)))
                    g = tf.schedule((tf.reshape(gamma, (1, KV, TV, VEC)),), op=T.copy(rmem_layout=(1, KV, TV @ th.t, VEC)))
                    x32 = tf.cast(x, "f32")
                    sq = tf.schedule((tf.square(x32),), op=T.copy(rmem_layout=((R, KV, TV @ th.t, VEC), (KV * VEC, VEC, 0, 1))))
                    ms = tf.schedule((tf.reduce(sq, (1, 2, 3), True, ReduceKind.MEAN),),
                                     op=T.copy(rmem_layout=(R, 1, 1, 1)))
                    y = tf.cast(x32 * tf.rsqrt(ms + EPS) * (tf.cast(g, "f32") + 1.0), "bf16")
                    return tf.insert_slice(tf.zeros(Tensor[(M, KV, TV, VEC), "bf16"]), y, (m, 0, 0, 0))

        @func
        def entry(hidden: Tensor[(1, M, H), "bf16"], gamma: ConstTensor[(H,), "bf16"]) -> Tensor[(1, M, H), "bf16"]:
            return tf.reshape(rmsnorm(hidden, gamma), (1, M, H))  # noqa: F821

    RMSNorm.__doc__ = f"input_norm / post_norm at seq_len={M}: {M // R} CTAs x {TV} threads"
    return RMSNorm


for _M in (1024, 4096, 8192):
    globals()[f"RMSNorm{_M}"] = rmsnorm(_M)
