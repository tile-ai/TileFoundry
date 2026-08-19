"""A CTA-scoped two-row RMS norm followed by blockwise FP8 quantization."""

from tilefoundry import module
from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@module(
    entry="rmsnorm_quant_seq_2",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 2 * 4 * 32)),
)
class RmsnormQuantSeq2Module:
    @func
    def rmsnorm_quant_seq_2(a: Tensor[(2, 1536), "bf16"]):
        with Mesh(("cta",), (1,), ("tile",)) as _cta:
            with Mesh(("thread",), (2, 4, 32), ("x", "y", "t")) as m:
                a_reg = tf.reshard(a, (2 @ m.x, 12 @ m.y, 128 @ m.t), "rmem")
                a_f32 = tf.cast(a_reg, "f32")
                a_sq = tf.square(a_f32)
                a_mean = tf.reduce(a_sq, (-1,), True, ReduceKind.MEAN)
                a_inv = tf.rsqrt(a_mean + 1e-6)
                a_norm_f32 = a_f32 * a_inv
                a_norm = tf.cast(a_norm_f32, "bf16")
                a_norm_f32_for_quant = tf.cast(a_norm, "f32")
                a_reshaped = tf.reshape(a_norm_f32_for_quant, (2, 12, 128))
                a_amax = tf.reduce(a_reshaped, (-1,), True, ReduceKind.ABS_MAX)
                a_scale = a_amax * (0.002232142857142857)
                a_quant = tf.cast(tf.clamp(a_reshaped / a_scale, -448.0, 448.0), "fp8e4m3")
                return (
                    tf.reshard(a_quant, (2 @ m.x, 12 @ m.y, 128 @ m.t), "gmem"),
                    tf.reshard(a_scale, (2 @ m.x, 12 @ m.y), "gmem"),
                )
