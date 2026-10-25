"""Placed values are consumed only where their positions and storage reach."""

from __future__ import annotations

from pathlib import Path

import pytest

from tilefoundry.analysis.check import check_program
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.ir.core import Call, binding_name
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.visitor import collect_exprs, expr_children
from tilefoundry.parser import ParseError
from tilefoundry.visitor_registry.contexts import TypeInferContext
from tilefoundry.visitor_registry.typeinfer import TypeInferVisitor

PLACED = Path(__file__).parents[1] / "fixtures" / "placed"


def test_a_function_mesh_is_an_outer_region_around_its_body() -> None:
    stage = load_authored_ir(f"{PLACED / 'fused_boundary.py'}:FusedBoundary").stage
    assert isinstance(stage.body, MeshRegion)
    assert [topology.name for topology in stage.body.mesh.topologies] == ["cta"]
    assert isinstance(stage.body.body, MeshRegion)
    assert [topology.name for topology in stage.body.body.mesh.topologies] == ["thread"]


def test_an_inner_level_layout_is_covered_by_the_whole_running_scope() -> None:
    body = load_authored_ir(f"{PLACED / 'rmsnorm.py'}:RmsnormModule").rmsnorm.body
    assert isinstance(body, MeshRegion)
    TypeInferVisitor().visit(body, TypeInferContext())


def test_region_boundaries_capture_external_regions_through_args() -> None:
    """A sibling region receives an earlier region value through its boundary."""
    RegionBoundaries = load_authored_ir(f"{PLACED / 'region_boundaries.py'}:RegionBoundaries")
    body = RegionBoundaries.entry_function().body
    scopes = [expr for expr in collect_exprs(body) if isinstance(expr, MeshRegion)]
    helper_call = next(
        expr
        for expr in collect_exprs(body)
        if isinstance(expr, Call)
        and isinstance(expr.target, Function)
        and expr.target.name == "helper"
    )
    assert any(
        scope.mesh.topologies[0].name == "cta"
        and helper_call in collect_exprs(scope.body)
        for scope in scopes
    )
    producer = next(scope for scope in scopes if binding_name(scope.body) == "v2")
    parents = [
        expr
        for expr in collect_exprs(body)
        if producer in expr_children(expr)
    ]
    assert len(parents) == 2
    assert any(isinstance(parent, MeshRegion) and producer in parent.args for parent in parents)

    core_diagnostics = Path(__file__).parents[3] / "core" / "fixtures" / "diagnostics"
    load_authored_ir(str(core_diagnostics / "region_rebind.py"))
    load_authored_ir(str(core_diagnostics / "region_tuple_rebind.py"))


def test_sibling_region_chain_captures_each_value_at_the_next_boundary() -> None:
    """An accumulator crosses each sibling region through fresh parameters."""
    SiblingRegionChain = load_authored_ir(
        str(Path(__file__).parents[1] / "fixtures" / "parser" / "sibling_region_chain.py")
    )
    entry = SiblingRegionChain.entry_function()
    check_program(SiblingRegionChain, entry)
    regions = [expr for expr in collect_exprs(entry.body) if isinstance(expr, MeshRegion)]
    leaves = [region for region in regions if binding_name(region.body) is not None]
    assert all(
        not any(isinstance(expr, MeshRegion) for expr in collect_exprs(region.body))
        for region in leaves
    )
    final = next(
        region for region in leaves if [param.name for param in region.params] == ["acc", "rhs"]
    )
    previous = final.args[0]
    assert isinstance(previous, MeshRegion)
    assert [param.name for param in previous.params] == ["acc", "lhs"]
    assert isinstance(previous.args[0], MeshRegion)


def test_unscoped_initialization_is_rejected_at_the_authored_call() -> None:
    path = Path(__file__).parents[1] / "fixtures" / "diagnostics" / "unscoped_call_loop.py"
    source = path.read_text()
    line = next(i for i, text in enumerate(source.splitlines(), 1) if "out = tf.zeros" in text)
    with pytest.raises(ParseError, match="runs outside every mesh scope") as raised:
        load_authored_ir(str(path))
    assert f"unscoped_call_loop.py:{line}:" in str(raised.value)
    assert "tf.zeros" in str(raised.value)
    assert "inside with Mesh(...)" in str(raised.value)
