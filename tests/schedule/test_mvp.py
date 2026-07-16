from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from tilefoundry import Call, DType, VerifyError
from tilefoundry.dsl import Tensor
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.hir.verify import verify_function
from tilefoundry.ir.target import CudaTarget
from tilefoundry.ir.target.storage import StorageKind
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.shard import Layout, Mesh, S, ShardLayout, Topology
from tilefoundry.parser import parse_func, parse_schedule_func
from tilefoundry.schedule import (
    EdgeKind,
    ScheduleContext,
    problem_fingerprint,
    run_schedule,
)
from tilefoundry.visitor_registry.contexts import TypeInferContext

from .fixtures.dsv4_moe_mvp import moe_entry, route_func


def _context() -> ScheduleContext:
    mesh = Mesh(
        topology=Topology("cta", 8),
        layout=Layout(shape=(8,), strides=(1,)),
        names=("cta",),
    )
    return ScheduleContext(target=CudaTarget(), level="cta", mesh=mesh)


def _walk_calls(expr):
    if isinstance(expr, Call):
        yield expr
        for arg in expr.args:
            yield from _walk_calls(arg)
    elif hasattr(expr, "elements"):
        for element in expr.elements:
            yield from _walk_calls(element)


def _typed_call(target, args):
    call = Call(type=TensorType.scalar(DType.f32), target=target, args=args)
    return dataclasses.replace(call, type=TypeInferContext().type_of(call))


def test_dsv4_schedule_mvp_end_to_end():
    original = parse_schedule_func(moe_entry)
    result = run_schedule(original, _context())

    assert len(result.graph.nodes) == 4
    assert [node.callee.name for node in result.graph.nodes] == [
        "route_func",
        "routed_func",
        "shared_func",
        "combine_func",
    ]
    assert all(math.isfinite(cost.duration) for cost in result.costs.all())
    assert len(result.graph.constraints) == 1
    authored_value = result.graph.value(result.graph.constraints[0].target)
    assert authored_value.consumers == (result.graph.nodes[1].id,)

    by_name = {
        result.graph.node(item.node).callee.name: item
        for item in result.solution.node_assignments
    }
    routed = by_name["routed_func"]
    shared = by_name["shared_func"]
    assert routed.start_time < shared.end_time
    assert shared.start_time < routed.end_time
    assert routed.axis_starts[0] + routed.axis_extents[0] <= shared.axis_starts[0] or (
        shared.axis_starts[0] + shared.axis_extents[0] <= routed.axis_starts[0]
    )
    serial_duration = sum(
        result.costs.node(item.option).duration
        for item in result.solution.node_assignments
    ) + sum(
        result.costs.edge(item.option).duration
        for item in result.solution.edge_assignments
    )
    assert result.solution.makespan < serial_duration
    assert result.solution.problem_fingerprint == problem_fingerprint(
        result.graph,
        result.space,
        result.costs,
        original.constraints,
    )

    verify_function(result.output)
    TypeInferContext().type_of(result.output.body)

    inputs = (torch.arange(8, dtype=torch.float32),)
    torch.testing.assert_close(
        evaluate(result.output, *inputs, device="cpu"),
        evaluate(original.function, *inputs, device="cpu"),
    )
    assert sum(
        isinstance(call.target, Reshard) for call in _walk_calls(result.output.body)
    ) >= 2
    assert any(
        edge.kind is EdgeKind.RESHARD for edge in result.solution.edge_assignments
    )


def test_direct_cross_slice_binary_combination_fails():
    context = _context()
    parent = context.mesh
    param = parse_schedule_func(moe_entry).function.params[0]
    left = ShardLayout(
        layout=Layout(shape=(8,), strides=None),
        attrs=(S(0),),
        mesh=parent[:4],
    )
    right = ShardLayout(
        layout=Layout(shape=(8,), strides=None),
        attrs=(S(0),),
        mesh=parent[4:8],
    )
    left_value = _typed_call(
        Reshard(layout=left, storage=StorageKind.GMEM), (param,)
    )
    right_value = _typed_call(
        Reshard(layout=right, storage=StorageKind.GMEM), (param,)
    )
    with pytest.raises((ValueError, VerifyError), match="mesh|incompatible|different"):
        _typed_call(Binary(kind=BinaryKind.ADD), (left_value, right_value))


def test_schedule_parser_uses_hir_identity_and_rejects_alias_duplicates():
    def annotated_entry(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
        value: where(storage="rmem") = route_func(x)
        return value

    with pytest.raises(VerifyError, match="parse_schedule_func"):
        parse_func(annotated_entry)
    parsed = parse_schedule_func(annotated_entry)
    assert len(parsed.constraints) == 1
    assert parsed.constraints[0].target is parsed.function.body
    assert parsed.constraints[0].provenance.value == "author"

    def duplicate_entry(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
        value = route_func(x)
        _alias: where(storage="rmem") = value
        value: where(storage="rmem") = value
        return value

    with pytest.raises(VerifyError, match="already constrained|alias"):
        parse_schedule_func(duplicate_entry)
