"""Parser normalization of authored scalar expressions used as dimensions."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.core._corpus import reload
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.dsl import *
from tilefoundry.inspection import as_script
from tilefoundry.ir.core import Call
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.parser import ParseError

FIXTURES = Path(__file__).parents[1] / "fixtures"
PARSER = FIXTURES / "parser"


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
        load_authored_ir(str(source))


def test_composed_layout_states_parse_round_trip_and_reshard() -> None:
    Model = load_authored_ir(str(PARSER / "composed_swizzled_layout_states.py"))

    function = Model.entry_function()
    execution_mesh = Mesh((Topology("thread", 8),), (8,), ("lane",))
    composed = ComposedLayout(Swizzle(2, 4, 3), 4, Layout((8, 16), (16, 1)))
    assert function.params[0].type.layout == ShardLayout(
        composed, (P("sum"),), execution_mesh
    )
    assert function.body.type.layout == ShardLayout(composed, (B(),), execution_mesh)
    assert function.body.type.storage is StorageKind.SMEM
    printed = as_script(Model)
    assert as_script(reload(printed, "Model")) == printed


def test_body_local_integer_arithmetic_is_normalized_before_layout_construction() -> None:
    Model = load_authored_ir(str(PARSER / "body_local_width_arithmetic.py"))

    scope = Model.functions[0].body
    assert isinstance(scope, MeshRegion)
    call = scope.body
    assert isinstance(call, Call) and isinstance(call.target, Reshard)
    assert call.target.layout.layout.shape == (1, 16, 128, 64)


def test_one_layout_cannot_split_the_same_topology_through_two_meshes() -> None:
    """The layout parser reports duplicate levels before composing mesh axes."""
    with pytest.raises(ParseError, match="a layout can split one level once"):
        load_authored_ir(str(FIXTURES / "diagnostics" / "parser_layout_split_level_twice.py"))
