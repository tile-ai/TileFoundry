"""Parser normalization of authored scalar expressions used as dimensions."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.core._corpus import load_program, reload_program
from tilefoundry.dsl import *
from tilefoundry.inspection import as_script
from tilefoundry.ir.core import Call
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.parser import ParseError
from tilefoundry_nvidia import CudaTarget


@pytest.mark.parametrize(
    ("fixture", "message"),
    [
        ("empty", "layout states must be"),
        ("bare_split", "layout states must be"),
        ("mixed", "layout states must be"),
        ("empty_partial", "layout states must be"),
        ("scalar_shape", "layout states require"),
    ],
)
def test_layout_states_reject_invalid_forms(fixture: str, message: str) -> None:
    source = Path(__file__).parents[1] / f"fixtures/diagnostics/parser_layout_states_{fixture}.py"
    with pytest.raises(ParseError, match=message):
        load_program(source)


def test_composed_layout_states_parse_round_trip_and_reshard() -> None:
    @module(entry="f", target=CudaTarget("nvidia.h200_sxm"))
    class Model:
        @func(mesh=Mesh((Topology("thread", 8),), layout=(8,), names=("lane",)))
        def f(
            x: Tensor[
                (8, 16), "f32",
                ((8, 16) + 4 | Swizzle(2, 4, 3)) @ {mesh.lane @ P("sum")},
                "rmem",
            ],
        ):
            return tf.reshard(
                x, ((8, 16) + 4 | Swizzle(2, 4, 3)) @ {B()}, "smem"
            )

    function = Model.entry_function()
    execution_mesh = Mesh((Topology("thread", 8),), (8,), ("lane",))
    composed = ComposedLayout(Swizzle(2, 4, 3), 4, Layout((8, 16), (16, 1)))
    assert function.params[0].type.layout == ShardLayout(
        composed, (P("sum"),), execution_mesh
    )
    assert function.body.type.layout == ShardLayout(composed, (B(),), execution_mesh)
    assert function.body.type.storage is StorageKind.SMEM
    printed = as_script(Model)
    assert as_script(reload_program(printed, "Model")) == printed


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
    assert isinstance(scope, MeshRegion)
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
