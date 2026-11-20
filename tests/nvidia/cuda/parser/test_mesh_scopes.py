"""Parser contracts for mesh scopes and the boundaries they check."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.core._corpus import Case, case, check_case, reload, special
from tilefoundry import cli, inspection, parser
from tilefoundry.ir import core, expr, types, utils
from tilefoundry.ir.hir import nn, sharding, tensor

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
        assert isinstance(scope, core.MeshRegion)
        body = scope.body
        assert isinstance(body, expr.Call)
        assert isinstance(body.target, nn.MatMul)
        assert any(isinstance(argument.target, sharding.Reshard) for argument in body.args)


@case
class ValueLessRegionTuple(MeshScopeCase):
    """A value-less with exports only its live values through one region result."""

    path: Path = PARSER / "mesh_region_tuple_escape.py"

    def guard(self, state, problems):
        body = state.entry_function().body
        assert isinstance(body, expr.Tuple)
        first, second = body.elements
        assert isinstance(first, expr.Call) and isinstance(second, expr.Call)
        assert isinstance(first.target, tensor.TupleGetItem)
        assert isinstance(second.target, tensor.TupleGetItem)
        scope = first.args[0]
        assert isinstance(scope, core.MeshRegion)
        assert isinstance(scope.body, expr.Tuple)
        assert len(scope.body.elements) == 2
        assert isinstance(first.args[1], expr.Constant) and first.args[1].value == 0
        assert isinstance(second.args[1], expr.Constant) and second.args[1].value == 1


@case
class ValuefulRegionEscape(MeshScopeCase):
    """A with result and an escaping assignment keep both region edges."""

    path: Path = PARSER / "mesh_region_valueful_escape.py"

    def guard(self, state, problems):
        scoped = state.entry_function().body
        assert isinstance(scoped, core.MeshRegion)
        assert isinstance(scoped.body, core.LoopRegion)
        assert isinstance(scoped.body.body, expr.Call)


@case
class RootMeshWithoutWith(MeshScopeCase):
    """A body that opens no with: a Mesh value and an external layout stay legal, unscoped."""

    path: Path = PARSER / "root_mesh_without_with.py"

    def guard(self, state, problems):
        assert not isinstance(state.entry_function().body, core.MeshRegion)


@case
class ScopeNameLayout(MeshScopeCase):
    """'@ warp' in cta x a thread slice keeps cta and takes warp's threads: B on that scope."""

    path: Path = PARSER / "scope_name_layout.py"

    def guard(self, state, problems):
        regions = [region.mesh for region in core.collect_exprs(state.entry_function().body) if isinstance(region, core.MeshRegion)]
        scope = utils.from_meshes(*reversed(regions))
        tiles = smem_types(state)
        plain = types.ShardLayout(types.Layout((16, 16)), (types.Broadcast(),) * len(utils.axis_keys(scope)), scope)
        assert tiles and all(tile.layout == plain for tile in tiles)
        printed = inspection.as_script(state)
        assert '(16, 16) @ thread_2, "smem"' in printed
        assert smem_types(reload(printed, state.name)) == tiles


@case
class ClassMeshPrimFunc(MeshScopeCase):
    """prim_func signatures see the meshes the class body declares, as @func signatures do."""

    path: Path = PARSER / "class_mesh_prim_func.py"

    def guard(self, state, problems):
        (function,) = state.functions
        lanes = types.Mesh((types.Topology("thread", 8),), types.Layout((8,), (1,)), ("lane",))
        assert function.params[0].type.layout == types.ShardLayout(types.Layout((8, 16), (0, 1)), (types.Split(0),), lanes)


def smem_types(module) -> list:
    body = module.entry_function().body
    return [
        node.type
        for node in core.collect_exprs(body)
        if isinstance(node, expr.Call) and node.type.storage is expr.StorageKind.SMEM
    ]


CASES = special(MeshScopeCase)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_mesh_scope_case(case: MeshScopeCase) -> None:
    check_case(case)


def test_mesh_binding_stays_in_scope() -> None:
    """A mesh alias is removed with its lexical frame after the with body."""
    with pytest.raises(parser.ParseError, match="'mesh' is not a lexical Mesh binding"):
        cli.load_authored_ir(str(DIAGNOSTICS / "parser_mesh_binding_escapes_scope.py"))


def test_boundary_rejects_split_axis_mismatch() -> None:
    with pytest.raises(expr.VerifyError, match="type mismatch.*param 'x'"):
        cli.load_authored_ir(str(DIAGNOSTICS / "parser_boundary_split_axis_mismatch.py"))


def test_boundary_rejects_storage_mismatch() -> None:
    with pytest.raises(expr.VerifyError, match="type mismatch.*param 'x'"):
        cli.load_authored_ir(str(DIAGNOSTICS / "parser_boundary_storage_mismatch.py"))


def test_layoutless_boundary_constrains_storage() -> None:
    with pytest.raises(expr.VerifyError, match="type mismatch.*param 'x'"):
        cli.load_authored_ir(str(DIAGNOSTICS / "parser_boundary_layout_undefined_storage.py"))


def test_boundary_rejects_undeclared_mesh() -> None:
    with pytest.raises(parser.ParseError, match="'nope' is not a lexical Mesh binding"):
        cli.load_authored_ir(str(DIAGNOSTICS / "parser_boundary_undeclared_mesh.py"))


def test_placement_rejects_external_mesh() -> None:
    with pytest.raises(
        parser.ParseError, match="'_EXTERNAL_PLACEMENT_MESH' is not a lexical Mesh binding"
    ):
        cli.load_authored_ir(str(DIAGNOSTICS / "parser_placement_external_mesh.py"))
