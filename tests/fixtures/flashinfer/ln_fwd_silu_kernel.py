"""Placed SM100 RMSNorm and SiLU specialization.

upstream: flashinfer-ai/flashinfer@2ab910c58fdd2392914ea05e2a8714946ac0eef6 \
include/flashinfer/norm/ln_fwd_silu_kernel.cuh
license: Apache-2.0 (no upstream source is vendored)

This is the upstream LUT specialization for 1,560 tokens, hidden size 1,024,
and bf16 output: 390 CTAs, four rows per CTA, and one warp per row.
"""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Tensor, tf
from tilefoundry.ir.types.shard import Mesh, Topology
from tilefoundry.target import CudaTarget

ROWS, HIDDEN = 1560, 1024
CTA_COUNT, ROWS_PER_CTA = 390, 4
TARGET = CudaTarget("nvidia.b200_sxm")
TOPOLOGIES = (Topology("cta", CTA_COUNT), Topology("thread", ROWS_PER_CTA * 32))


@module(entry="ln_fwd_kernel", target=TARGET, topologies=TOPOLOGIES)
class LnFwdSiluKernel:
    """FlashInfer SM100 ln_fwd_kernel RMSNorm-SiLU bf16 specialization.

    predicted-ns: 1272  waves: 3
    measured-ns: not taken
    note: Recorded evidence, not an assertion; open a ledger row if predicted exceeds measured.
    """

    @func
    def ln_fwd_kernel(
        x: Tensor[(ROWS, HIDDEN), "bf16"],
        gamma: ConstTensor[(HIDDEN,), "bf16"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(CTA_COUNT, ROWS_PER_CTA, 32),
            names=("block", "row", "lane"),
        ) as mesh:
            x_reg = tf.reshard(
                x,
                (ROWS @ (mesh.block, mesh.row), HIDDEN @ mesh.lane),
                "rmem",
            )
            x_smem = tf.reshard(
                x_reg,
                (ROWS @ (mesh.block, mesh.row), HIDDEN @ mesh.lane),
                "smem",
            )
            gamma_smem = tf.reshard(gamma, (HIDDEN,), "smem")
            normalized = tf.rms_norm(x_smem, gamma_smem, eps=1e-6)
            activated = tf.silu(normalized)
            return tf.reshard(
                activated,
                (ROWS @ (mesh.block, mesh.row), HIDDEN @ mesh.lane),
                "gmem",
            )


__all__ = ["LnFwdSiluKernel"]
