"""Placed values are consumed only where their positions and storage reach."""

from __future__ import annotations

from tests.core._source import import_dsl
from tests.core.parser.test_mesh_visibility import (
    _diagnostic,
)
from tests.nvidia.cuda.fixtures.placed.fused_boundary import FusedBoundary
from tests.nvidia.cuda.fixtures.placed.region_boundaries import RegionBoundaries
from tests.nvidia.cuda.fixtures.placed.rmsnorm import RmsnormModule
from tilefoundry.analysis.check import check_program
from tilefoundry.dsl import *
from tilefoundry.ir.core import Call, binding_name
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.visitor import collect_exprs, expr_children
from tilefoundry.visitor_registry.contexts import TypeInferContext
from tilefoundry.visitor_registry.typeinfer import TypeInferVisitor
from tilefoundry_nvidia import CudaTarget


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


def test_sibling_region_chain_captures_each_value_at_the_next_boundary() -> None:
    """An accumulator crosses each sibling region through fresh parameters."""

    @module(
        entry="run",
        target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 256)),
    )
    class SiblingRegionChain:
        @func
        def run(
            a: Tensor[(64, 32), "bf16"],
            c: Tensor[(64, 32), "bf16"],
        ) -> Tensor[(64, 32), "bf16"]:
            with Mesh(("cta",), (1,), names=("block",)) as _cta:
                with Mesh(
                    ("thread",),
                    (2, 128),
                    names=("role", "participant"),
                ) as threads:
                    with threads[1, :] as _compute:
                        acc = tf.zeros(Tensor[(64, 32), "bf16"])
                    with threads[0, :32] as _loader:
                        lhs = tf.cast(a, dtype="bf16")
                    with threads[1, :] as _compute:
                        acc = acc + lhs
                    with threads[0, :32] as _loader:
                        rhs = tf.cast(c, dtype="bf16")
                    with threads[1, :] as _compute:
                        acc = acc + rhs
                    return acc

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
