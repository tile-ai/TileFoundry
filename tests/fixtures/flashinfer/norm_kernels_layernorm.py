"""Placed CuTe LayerNorm kernel with explicit distributed statistics.

upstream: flashinfer-ai/flashinfer@2ab910c58fdd2392914ea05e2a8714946ac0eef6 \
flashinfer/norm/kernels/layernorm.py
license: Apache-2.0 (no upstream source is vendored)
The hidden dimension stays vectorized. Separate sum and sumsq reductions carry
the two statistics needed to combine sharded LayerNorm state.
"""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Tensor, tf
from tilefoundry.ir.types.shard import Mesh, Topology
from tilefoundry.target import CudaTarget

ROWS, HIDDEN = 256, 1024
TARGET = CudaTarget("nvidia.h200_sxm")
TOPOLOGIES = (Topology("cta", ROWS), Topology("thread", 4 * 32))


@module(entry="layernorm_cute", target=TARGET, topologies=TOPOLOGIES)
class LayerNormKernel:
    """FlashInfer CuTe DSL LayerNorm kernel.

    predicted-ns: 748
    waves: 2
    measured-ns: not taken
    note: recorded evidence, not an assertion; open a ledger row if predicted exceeds measured.
    """

    @func
    def layernorm_cute(
        x: Tensor[(ROWS, HIDDEN), "bf16"],
        gamma: ConstTensor[(HIDDEN,), "f32"],
        beta: ConstTensor[(HIDDEN,), "f32"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(ROWS, 4, 32),
            names=("row", "warp", "lane"),
        ) as mesh:
            x_reg = tf.reshard(
                x, (ROWS @ mesh.row, 8 @ mesh.warp, 128 @ mesh.lane), "rmem"
            )
            x_smem = tf.reshard(
                x_reg, (ROWS @ mesh.row, 8 @ mesh.warp, 128 @ mesh.lane), "smem"
            )
            gamma_smem = tf.reshard(gamma, (HIDDEN,), "smem")
            beta_smem = tf.reshard(beta, (HIDDEN,), "smem")
            x32 = tf.cast(x_smem, "f32")
            count = tf.reduce(
                tf.full_like(x32, value=1.0),
                axes=(-1,),
                keepdim=True,
                kind="sum",
            )
            summed = tf.reduce(x32, axes=(-1,), keepdim=True, kind="sum")
            sumsq = tf.reduce(x32 * x32, axes=(-1,), keepdim=True, kind="sum")
            mean = summed / count
            variance = sumsq / count - mean * mean
            normalized = (x32 - mean) * tf.rsqrt(variance + 1e-6)
            affine = normalized * gamma_smem + beta_smem
            return tf.reshard(
                tf.cast(affine, "bf16"),
                (ROWS @ mesh.row, 8 @ mesh.warp, 128 @ mesh.lane),
                "gmem",
            )


__all__ = ["LayerNormKernel"]
