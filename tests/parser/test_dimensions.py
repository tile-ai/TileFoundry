"""Parser normalization of authored scalar expressions used as dimensions."""

from __future__ import annotations

import pytest

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, Topology, tf
from tilefoundry.ir.core import Call
from tilefoundry.ir.hir.mesh_scope import MeshScope
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.parser import ParseError
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

    scope = Model.functions[0].body
    assert isinstance(scope, MeshScope)
    call = scope.body
    assert isinstance(call, Call) and isinstance(call.target, Reshard)
    assert call.target.layout.layout.shape == (1, 16, 128, 64)


def test_one_layout_cannot_split_the_same_topology_through_two_meshes() -> None:
    """The layout parser reports duplicate levels before composing mesh axes."""
    with pytest.raises(ParseError, match="a layout can split one level once"):

        @module(
            entry="f",
            target=CudaTarget("nvidia.h200_sxm"),
            topologies=(Topology("cta", 128),),
        )
        class Model:
            @func
            def f(x: Tensor[(4, 64), "f32"]):
                with Mesh(("cta",), layout=(4, 32), names=("x", "y")) as first:
                    with Mesh(("cta",), layout=(2, 64), names=("p", "q")) as second:
                        return tf.reshard(
                            x,
                            (4 @ first.x, 64 @ second.q),
                            "gmem",
                        )
