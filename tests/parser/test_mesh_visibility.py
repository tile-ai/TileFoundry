"""Placed values are consumed only where their positions and storage reach."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._source import import_dsl
from tests.fixtures.placed.fused_boundary import FusedBoundary
from tests.fixtures.placed.region_boundaries import RegionBoundaries
from tests.fixtures.placed.rmsnorm import RmsnormModule
from tilefoundry import module
from tilefoundry.dsl import *
from tilefoundry.ir.core import Call, VerifyError, binding_name
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.visitor import collect_exprs, expr_children
from tilefoundry.visitor_registry.contexts import TypeInferContext
from tilefoundry.visitor_registry.visitors import TypeInferVisitor

_DIAGNOSTICS = Path(__file__).parents[1] / "fixtures" / "diagnostics"


def _diagnostic(name: str) -> str:
    return (_DIAGNOSTICS / f"{name}.py").read_text()


def test_a_layout_finer_than_the_running_scope_is_rejected() -> None:
    with pytest.raises(
        VerifyError,
        match=(
            "input 0 is laid out more finely than the scope it runs in; "
            "write it inside that scope, or reshard it back first"
        ),
    ):
        import_dsl(_diagnostic("value_escapes"), "ValueEscapes")


def test_rmem_does_not_reach_finer_units_than_its_layout() -> None:
    with pytest.raises(
        VerifyError,
        match=(
            "input 0 is laid out more coarsely and kept in rmem, which does not "
            "reach the units this runs on; reshard it to smem or gmem first"
        ),
    ):
        import_dsl(_diagnostic("coarse_rmem"), "CoarseRmem")


def test_reshard_is_the_one_consumer_allowed_to_move_an_invisible_value() -> None:
    @module(
        entry="run",
        topologies=(Topology("cta", 2), Topology("thread", 4)),
    )
    class ReshardEscape:
        @func
        def run(x: Tensor[(8,), "f32"]):
            with Mesh(("thread",), (4,), names=("lane",)) as lanes:
                value = tf.reshard(x, (8 @ lanes.lane,), "rmem")
            with Mesh(("cta",), (2,), names=("block",)) as blocks:
                return tf.reshard(value, (8 @ blocks.block,), "smem")

    assert ReshardEscape.entry_function().body is not None


def test_a_function_mesh_is_an_outer_region_around_its_body() -> None:
    stage = FusedBoundary.stage
    assert isinstance(stage.body, MeshRegion)
    assert [topology.name for topology in stage.body.mesh.topologies] == ["cta"]
    assert isinstance(stage.body.body, MeshRegion)
    assert [topology.name for topology in stage.body.body.mesh.topologies] == ["thread"]


def test_an_inner_level_layout_is_covered_by_the_whole_running_scope() -> None:
    body = RmsnormModule.rmsnorm.body
    assert isinstance(body, MeshRegion)
    TypeInferVisitor().visit(body, TypeInferContext())


def test_region_boundaries_capture_external_regions_through_args() -> None:
    """A sibling region receives an earlier region value through its boundary."""
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

    import_dsl(_diagnostic("region_rebind"), "RegionRebind")
    import_dsl(_diagnostic("region_tuple_rebind"), "RegionTupleRebind")
