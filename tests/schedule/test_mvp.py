from __future__ import annotations

import dataclasses
import math

import pytest

from tests.models.deepseek_v4 import DIM, dsv4_moe_layer, pre_moe_rms_norm
from tilefoundry import Call, DType, VerifyError
from tilefoundry.dsl import Tensor
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.hir.verify import verify_function
from tilefoundry.ir.target import CudaTarget
from tilefoundry.ir.target.storage import StorageKind
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.shard import Layout, Mesh, S, ShardLayout, Topology
from tilefoundry.parser import parse_func, parse_schedule_func
from tilefoundry.schedule import ScheduleContext, problem_fingerprint, run_schedule
from tilefoundry.visitor_registry.contexts import TypeInferContext

from .fixtures.dsv4_moe_mvp import moe_entry


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
        for argument in expr.args:
            yield from _walk_calls(argument)
    elif hasattr(expr, "elements"):
        for element in expr.elements:
            yield from _walk_calls(element)


def _typed_call(target, args):
    call = Call(type=TensorType.scalar(DType.f32), target=target, args=args)
    return dataclasses.replace(call, type=TypeInferContext().type_of(call))


def _node(result, name_fragment: str):
    matches = [
        node
        for node in result.graph.nodes
        if name_fragment in node.callee.name.lower()
    ]
    assert len(matches) == 1, f"expected one node containing {name_fragment!r}"
    return matches[0]


def _producer_ids(result, consumer_id: int) -> set[int]:
    producers = set()
    for use in result.graph.uses:
        if use.consumer != consumer_id:
            continue
        producer = result.graph.producer_of(use.value)
        if producer is not None:
            producers.add(producer.id)
    return producers


def _placements_overlap(left, right) -> bool:
    left_end = left.axis_starts[0] + left.axis_extents[0]
    right_end = right.axis_starts[0] + right.axis_extents[0]
    return left.axis_starts[0] < right_end and right.axis_starts[0] < left_end


def test_dsv4_schedule_mvp_accepts_real_moe_layer():
    schedule_input = parse_schedule_func(moe_entry)
    assert schedule_input.function.params[0].type.shape == (1, 1, DIM)
    result = run_schedule(schedule_input, _context())

    assert len(result.graph.nodes) == 4
    pre = _node(result, "pre_moe_rms_norm")
    routed = _node(result, "routed_expert")
    shared = _node(result, "shared_expert")
    combine = _node(result, "combine_expert_outputs")
    assert _producer_ids(result, routed.id) == {pre.id}
    assert _producer_ids(result, shared.id) == {pre.id}
    assert _producer_ids(result, combine.id) == {routed.id, shared.id}

    assert len(result.graph.constraints) == 1
    constrained = result.graph.value(result.graph.constraints[0].target)
    assert constrained.producer == routed.id
    assert constrained.consumers == (combine.id,)
    assert all(math.isfinite(cost.duration) for cost in result.costs.all())

    node_assignments = {item.node: item for item in result.solution.node_assignments}
    edge_assignments = {item.use: item for item in result.solution.edge_assignments}
    options_by_id = {option.id: option for option in result.space.node_options}
    for assignment in result.solution.node_assignments:
        option = options_by_id[assignment.option]
        assert option.node == assignment.node
        assert assignment.start_time <= assignment.end_time
        assert assignment.placement.axis_starts[0] >= 0
        assert (
            assignment.placement.axis_starts[0]
            + assignment.placement.axis_extents[0]
            <= _context().mesh.shape[0]
        )
        assert math.isfinite(result.costs.node(assignment.option).duration)

    for use in result.graph.uses:
        edge_assignment = edge_assignments[use.id]
        assert math.isfinite(result.costs.edge(edge_assignment.option).duration)
        producer = result.graph.producer_of(use.value)
        if producer is not None:
            edge_end = node_assignments[producer.id].end_time + result.costs.edge(
                edge_assignment.option
            ).duration
            assert node_assignments[use.consumer].start_time >= edge_end

    assignments = tuple(result.solution.node_assignments)
    for left_index, left in enumerate(assignments):
        for right in assignments[left_index + 1 :]:
            if _placements_overlap(left.placement, right.placement):
                assert left.end_time <= right.start_time or right.end_time <= left.start_time

    assert result.solution.problem_fingerprint == problem_fingerprint(
        result.graph, result.space, result.costs, schedule_input.constraints
    )
    verify_function(result.output)
    TypeInferContext().type_of(result.output.body)
    assert any(
        isinstance(call.target, Reshard) for call in _walk_calls(result.output.body)
    )


def test_direct_cross_slice_binary_combination_fails():
    context = _context()
    parent = context.mesh
    param = parse_schedule_func(moe_entry).function.params[0]
    left = ShardLayout(
        layout=Layout(shape=(1, 1, DIM), strides=None),
        attrs=(S(2),),
        mesh=parent[:4],
    )
    right = ShardLayout(
        layout=Layout(shape=(1, 1, DIM), strides=None),
        attrs=(S(2),),
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
    def annotated_entry(
        x: Tensor[(1, 1, DIM), "bf16"],
        gamma: Tensor[(DIM,), "f32"],
    ) -> Tensor[(1, 1, DIM), "bf16"]:
        value: where(storage="rmem") = pre_moe_rms_norm(x, gamma)
        return value

    with pytest.raises(VerifyError, match="parse_schedule_func"):
        parse_func(annotated_entry)
    parsed = parse_schedule_func(annotated_entry)
    assert len(parsed.constraints) == 1
    assert parsed.constraints[0].target is parsed.function.body
    assert parsed.constraints[0].provenance.value == "author"

    def duplicate_entry(
        x: Tensor[(1, 1, DIM), "bf16"],
        gamma: Tensor[(DIM,), "f32"],
    ) -> Tensor[(1, 1, DIM), "bf16"]:
        value = pre_moe_rms_norm(x, gamma)
        _alias: where(storage="rmem") = value
        value: where(storage="rmem") = value
        return value

    with pytest.raises(VerifyError, match="already constrained|alias"):
        parse_schedule_func(duplicate_entry)


assert dsv4_moe_layer is moe_entry
