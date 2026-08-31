"""Parser normalization of authored scalar expressions used as dimensions."""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, Topology, tf
from tilefoundry.ir.core import Call
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.target import CudaTarget


def test_body_local_integer_arithmetic_is_normalized_before_layout_construction() -> None:
    @module(
        entry="f",
        target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 128),),
    )
    class Model:
        @func
        def f(
            x: Tensor[(1, 16, 8192), "f32"],
        ):
            with Mesh(("cta",), layout=(128,), names=("unit",)) as mesh:
                width = 4096 + 4096
                return tf.reshard(
                    x,
                    (1, 16, width @ mesh.unit),
                    "smem",
                )

    call = Model.functions[0].body
    assert isinstance(call, Call) and isinstance(call.target, Reshard)
    assert call.target.layout.layout.shape == (1, 16, 128, 64)
