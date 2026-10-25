"""Placement ``@`` and matmul ``@`` in one function."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(
    entry="placed_matmul",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 2),),
)
class PlacedMatMul:
    @func
    def placed_matmul(x: Tensor[(2, 4), "bf16"], b: Tensor[(4, 3), "bf16"]):
        with Mesh(("cta",), (2,), ("tile",)) as mesh:
            x_local = tf.reshard(x, (2 @ mesh.tile, 4), "gmem")
            b_local = tf.reshard(b, (4, 3), "gmem")
            return x_local @ b_local
