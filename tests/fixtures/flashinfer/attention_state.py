"""Placed A07 online attention-state merge boundary.

upstream: flashinfer-ai/flashinfer@2ab910c58fdd2392914ea05e2a8714946ac0eef6 \
include/flashinfer/attention/state.cuh:state_t.merge
license: Apache-2.0 (no upstream source is vendored)

CTAs own independent state rows. Four warps and 32 lanes split the output
vector, matching the upstream vectorized merge while scalar maxima and sums
remain broadcast within each row.
"""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Tensor, tf
from tilefoundry.ir.types.shard import Mesh, Topology
from tilefoundry.target import CudaTarget

ROWS, WIDTH = 256, 128
TARGET = CudaTarget("nvidia.h200_sxm")
TOPOLOGIES = (Topology("cta", ROWS), Topology("thread", 4 * 32))


@module(entry="merge", target=TARGET, topologies=TOPOLOGIES)
class AttentionStateMerge:
    """FlashInfer attention state merge kernel.

    predicted-ns: 126  waves: 2
    measured-ns: not taken
    note: Recorded evidence, not an assertion; open a ledger row if predicted exceeds measured.
    """

    @func
    def merge(
        o0: Tensor[(ROWS, WIDTH), "f32"],
        m0: Tensor[(ROWS, 1), "f32"],
        d0: Tensor[(ROWS, 1), "f32"],
        o1: Tensor[(ROWS, WIDTH), "f32"],
        m1: Tensor[(ROWS, 1), "f32"],
        d1: Tensor[(ROWS, 1), "f32"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(ROWS, 4, 32),
            names=("state", "warp", "lane"),
        ) as mesh:
            o0_reg = tf.reshard(
                o0,
                (ROWS @ mesh.state, 4 @ mesh.warp, 32 @ mesh.lane),
                "rmem",
            )
            o1_reg = tf.reshard(
                o1,
                (ROWS @ mesh.state, 4 @ mesh.warp, 32 @ mesh.lane),
                "rmem",
            )
            m0_reg = tf.reshard(m0, (ROWS @ mesh.state, 1), "rmem")
            m1_reg = tf.reshard(m1, (ROWS @ mesh.state, 1), "rmem")
            d0_reg = tf.reshard(d0, (ROWS @ mesh.state, 1), "rmem")
            d1_reg = tf.reshard(d1, (ROWS @ mesh.state, 1), "rmem")
            a = tf.reshard(
                o0_reg,
                (ROWS @ mesh.state, 4 @ mesh.warp, 32 @ mesh.lane),
                "smem",
            )
            b = tf.reshard(
                o1_reg,
                (ROWS @ mesh.state, 4 @ mesh.warp, 32 @ mesh.lane),
                "smem",
            )
            ml0 = tf.reshard(m0_reg, (ROWS @ mesh.state, 1), "smem")
            ml1 = tf.reshard(m1_reg, (ROWS @ mesh.state, 1), "smem")
            dl0 = tf.reshard(d0_reg, (ROWS @ mesh.state, 1), "smem")
            dl1 = tf.reshard(d1_reg, (ROWS @ mesh.state, 1), "smem")
            maximum = tf.maximum(ml0, ml1)
            w0 = tf.exp2(ml0 - maximum)
            w1 = tf.exp2(ml1 - maximum)
            denominator = w0 * dl0 + w1 * dl1
            output = (w0 * a + w1 * b) / denominator
            return (
                tf.reshard(
                    output,
                    (ROWS @ mesh.state, 4 @ mesh.warp, 32 @ mesh.lane),
                    "gmem",
                ),
                tf.reshard(maximum, (ROWS @ mesh.state, 1), "gmem"),
                tf.reshard(denominator, (ROWS @ mesh.state, 1), "gmem"),
            )


__all__ = ["AttentionStateMerge"]
