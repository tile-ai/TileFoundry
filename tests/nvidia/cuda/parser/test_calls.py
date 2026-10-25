"""Parser contracts for explicit variadic input sequences."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.core._corpus import Case, case, check_case, special
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.ir.core import Call, Constant, Tuple, VerifyError
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.parser import ParseError

FIXTURES = Path(__file__).parents[1] / "fixtures"
PARSER = FIXTURES / "parser"
DIAGNOSTICS = FIXTURES / "diagnostics"


class MeshScopeCase(Case):
    """A program checked by the mesh scopes its calls parse into."""


@case
class PlacementAndMatmulAt(MeshScopeCase):
    """The two spellings of ``@`` are told apart by position, not by shape."""

    path: Path = PARSER / "placed_matmul.py"

    def guard(self, state, problems):
        scope = state.entry_function().body
        assert isinstance(scope, MeshRegion)
        body = scope.body
        assert isinstance(body, Call)
        assert isinstance(body.target, MatMul)
        assert any(isinstance(argument.target, Reshard) for argument in body.args)


@case
class ValueLessRegionTuple(MeshScopeCase):
    """A value-less with exports only its live values through one region result."""

    path: Path = PARSER / "mesh_region_tuple_escape.py"

    def guard(self, state, problems):
        body = state.entry_function().body
        assert isinstance(body, Tuple)
        first, second = body.elements
        assert isinstance(first, Call) and isinstance(second, Call)
        assert isinstance(first.target, TupleGetItem)
        assert isinstance(second.target, TupleGetItem)
        scope = first.args[0]
        assert isinstance(scope, MeshRegion)
        assert isinstance(scope.body, Tuple)
        assert len(scope.body.elements) == 2
        assert isinstance(first.args[1], Constant) and first.args[1].value == 0
        assert isinstance(second.args[1], Constant) and second.args[1].value == 1


@case
class ValuefulRegionEscape(MeshScopeCase):
    """A with result and an escaping assignment keep both region edges."""

    path: Path = PARSER / "mesh_region_valueful_escape.py"

    def guard(self, state, problems):
        scoped = state.entry_function().body
        assert isinstance(scoped, MeshRegion)
        assert isinstance(scoped.body, LoopRegion)
        assert isinstance(scoped.body.body, Call)


CASES = special(MeshScopeCase)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_mesh_scope_case(case: MeshScopeCase) -> None:
    check_case(case)


def test_mesh_binding_does_not_escape_its_with_scope() -> None:
    """A mesh alias is removed with its lexical frame after the with body."""
    with pytest.raises(ParseError, match="'mesh' is not a lexical Mesh binding"):
        load_authored_ir(str(DIAGNOSTICS / "parser_mesh_binding_escapes_scope.py"))


def test_a_boundary_rejects_a_different_split_axis() -> None:
    with pytest.raises(VerifyError, match="type mismatch.*param 'x'"):
        load_authored_ir(str(DIAGNOSTICS / "parser_boundary_split_axis_mismatch.py"))


def test_a_boundary_rejects_a_different_storage() -> None:
    with pytest.raises(VerifyError, match="type mismatch.*param 'x'"):
        load_authored_ir(str(DIAGNOSTICS / "parser_boundary_storage_mismatch.py"))


def test_a_layout_undefined_boundary_still_constrains_storage() -> None:
    with pytest.raises(VerifyError, match="type mismatch.*param 'x'"):
        load_authored_ir(str(DIAGNOSTICS / "parser_boundary_layout_undefined_storage.py"))


def test_a_boundary_rejects_a_bare_undeclared_mesh_name() -> None:
    with pytest.raises(ParseError, match="'nope' is not a lexical Mesh binding"):
        load_authored_ir(str(DIAGNOSTICS / "parser_boundary_undeclared_mesh.py"))


def test_placement_rejects_an_external_mesh_axis_binding() -> None:
    with pytest.raises(
        ParseError, match="'_EXTERNAL_PLACEMENT_MESH' is not a lexical Mesh binding"
    ):
        load_authored_ir(str(DIAGNOSTICS / "parser_placement_external_mesh.py"))
