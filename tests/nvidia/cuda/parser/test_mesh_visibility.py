"""Placed values are consumed only where their positions and storage reach."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.core._corpus import Case, case, check_case, special
from tilefoundry import analysis, cli, parser
from tilefoundry.interfaces import typeinfer
from tilefoundry.ir import core, expr

PLACED = Path(__file__).parents[1] / "fixtures" / "placed"


class MeshVisibilityCase(Case):
    """A program checked by the mesh regions that hold its values."""


@case
class FunctionMeshOuterRegion(MeshVisibilityCase):
    path: Path = PLACED / "fused_boundary.py"
    name: str | None = "FusedBoundary"

    def guard(self, state, problems):
        stage = state.stage
        assert isinstance(stage.body, core.MeshRegion)
        assert [topology.name for topology in stage.body.mesh.topologies] == ["cta"]
        assert isinstance(stage.body.body, core.MeshRegion)
        assert [topology.name for topology in stage.body.body.mesh.topologies] == ["thread"]


@case
class InnerLayoutCoveredByScope(MeshVisibilityCase):
    path: Path = PLACED / "rmsnorm.py"

    def guard(self, state, problems):
        body = state.rmsnorm.body
        assert isinstance(body, core.MeshRegion)
        typeinfer.TypeInferVisitor().visit(body, typeinfer.TypeInferContext())


@case
class SiblingChainCapturesAtBoundary(MeshVisibilityCase):
    """An accumulator crosses each sibling region through fresh parameters."""

    path: Path = Path(__file__).parents[1] / "fixtures" / "parser" / "sibling_region_chain.py"

    def guard(self, state, problems):
        entry = state.entry_function()
        analysis.check_program(state, entry)
        regions = [node for node in core.collect_exprs(entry.body) if isinstance(node, core.MeshRegion)]
        leaves = [region for region in regions if expr.binding_name(region.body) is not None]
        assert all(
            not any(isinstance(node, core.MeshRegion) for node in core.collect_exprs(region.body))
            for region in leaves
        )
        final = next(
            region for region in leaves if [param.name for param in region.params] == ["acc", "rhs"]
        )
        previous = final.args[0]
        assert isinstance(previous, core.MeshRegion)
        assert [param.name for param in previous.params] == ["acc", "lhs"]
        assert isinstance(previous.args[0], core.MeshRegion)


CASES = special(MeshVisibilityCase)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_mesh_visibility_case(case: MeshVisibilityCase) -> None:
    check_case(case)


def test_region_boundary_captures_sibling_value() -> None:
    """A sibling region receives an earlier region value through its boundary."""
    RegionBoundaries = cli.load_authored_ir(f"{PLACED / 'region_boundaries.py'}:RegionBoundaries")
    body = RegionBoundaries.entry_function().body
    scopes = [node for node in core.collect_exprs(body) if isinstance(node, core.MeshRegion)]
    helper_call = next(
        node
        for node in core.collect_exprs(body)
        if isinstance(node, expr.Call)
        and isinstance(node.target, core.Function)
        and node.target.name == "helper"
    )
    assert any(
        scope.mesh.topologies[0].name == "cta"
        and helper_call in core.collect_exprs(scope.body)
        for scope in scopes
    )
    producer = next(scope for scope in scopes if expr.binding_name(scope.body) == "v2")
    parents = [
        node
        for node in core.collect_exprs(body)
        if producer in core.expr_children(node)
    ]
    assert len(parents) == 2
    assert any(isinstance(parent, core.MeshRegion) and producer in parent.args for parent in parents)

    core_diagnostics = Path(__file__).parents[3] / "core" / "fixtures" / "diagnostics"
    cli.load_authored_ir(str(core_diagnostics / "region_rebind.py"))
    cli.load_authored_ir(str(core_diagnostics / "region_tuple_rebind.py"))


def test_unscoped_init_rejected_at_call() -> None:
    path = Path(__file__).parents[1] / "fixtures" / "diagnostics" / "unscoped_call_loop.py"
    source = path.read_text()
    line = next(i for i, text in enumerate(source.splitlines(), 1) if "out = tf.zeros" in text)
    with pytest.raises(parser.ParseError, match="runs outside every mesh scope") as raised:
        cli.load_authored_ir(str(path))
    assert f"unscoped_call_loop.py:{line}:" in str(raised.value)
    assert "tf.zeros" in str(raised.value)
    assert "inside with Mesh(...)" in str(raised.value)
