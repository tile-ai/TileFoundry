"""Reusable proj_qkv parser model definitions."""

from __future__ import annotations

from tilefoundry import func
from tilefoundry.dsl import Mesh, Tensor, Topology
from tilefoundry.dsl.tf import *  # noqa: F401, F403


@func(topologies=(Topology("cta", 128),))
def proj_qkv_subset(
    x: Tensor[(1, 2048), "bf16"],
    wqkv: Tensor[(2048, 8192), "bf16"],
) -> Tensor[(1, 8192), "bf16"]:
    with Mesh(topology="cta", layout=(128,)) as cta:  # noqa: F841
        o_smem = zeros((1, 64), "bf16", storage="smem")
        for ok in tile(2048, 512):
            o_smem = relu(o_smem)
        return reshard(o_smem, (1, 64), "smem")


@func(topologies=(Topology("thread", 32),))
def proj_qkv_with_mma(
    x: Tensor[(16, 2048), "bf16"],
    w: Tensor[(2048, 8), "bf16"],
) -> Tensor[(16, 8), "f32"]:
    with Mesh(topology="thread", layout=(4, 8), names=("x", "y")) as warp:
        acc = zeros((16, 8), "f32", storage="rmem")
        for ok in tile(2048, 16):
            x_frag = reshard(
                x[:, ok],
                ((2, 4 @ warp.x, 2, 8 @ warp.y, 2), (1, 2, 8, 16, 128)),
                "rmem",
            )
            w_frag = reshard(
                w[ok, :],
                ((8 @ warp.y, 2, 4 @ warp.x, 2), (1, 8, 16, 64)),
                "rmem",
            )
            acc = add(
                acc,
                mma_sm80_16x8x16(
                    x_frag,
                    w_frag,
                    dtype_a="bf16",
                    dtype_b="bf16",
                    dtype_acc="f32",
                ),
            )
        return reshard(
            acc,
            ((2, 4 @ warp.x, 8 @ warp.y, 2), (1, 2, 8, 64)),
            "gmem",
        )
