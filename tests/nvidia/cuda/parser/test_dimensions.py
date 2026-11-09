"""Parser normalization of authored scalar expressions used as dimensions."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.core._corpus import Case, case, check_case, reload, special
from tilefoundry import cli, inspection, parser
from tilefoundry.dsl import *
from tilefoundry.ir import core, expr
from tilefoundry.ir.hir import sharding

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
    with pytest.raises(parser.ParseError, match=message):
        cli.load_authored_ir(str(source))


class DimensionCase(Case):
    """A program checked by the layouts its dimensions build."""


@case
class ComposedLayoutStates(DimensionCase):
    path: Path = PARSER / "composed_swizzled_layout_states.py"

    def guard(self, state, problems):
        function = state.entry_function()
        execution_mesh = Mesh((Topology("thread", 8),), (8,), ("lane",))
        composed = ComposedLayout(Swizzle(2, 4, 3), 4, Layout((8, 16), (16, 1)))
        assert function.params[0].type.layout == ShardLayout(composed, (P("sum"),), execution_mesh)
        assert function.body.type.layout == ShardLayout(composed, (B(),), execution_mesh)
        assert function.body.type.storage is StorageKind.SMEM
        printed = inspection.as_script(state)
        assert inspection.as_script(reload(printed, "Model")) == printed


@case
class BodyLocalWidthArithmetic(DimensionCase):
    path: Path = PARSER / "body_local_width_arithmetic.py"

    def guard(self, state, problems):
        scope = state.functions[0].body
        assert isinstance(scope, core.MeshRegion)
        call = scope.body
        assert isinstance(call, expr.Call) and isinstance(call.target, sharding.Reshard)
        assert call.target.layout.layout.shape == (1, 16, 128, 64)


CASES = special(DimensionCase)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_dimension_case(case: DimensionCase) -> None:
    check_case(case)


def test_layout_rejects_level_split_twice() -> None:
    """The layout parser reports duplicate levels before composing mesh axes."""
    with pytest.raises(parser.ParseError, match="a layout can split one level once"):
        cli.load_authored_ir(str(FIXTURES / "diagnostics" / "parser_layout_split_level_twice.py"))
