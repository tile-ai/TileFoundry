from __future__ import annotations

from tilefoundry import func
from tilefoundry.analysis import AnalysisOptions, FootprintMetadata, analyze
from tilefoundry.analysis.analyzer import _postorder, _timeline_for_function
from tilefoundry.dsl import Mesh, Tensor, Topology, tf
from tilefoundry.ir.core import Call, get_metadata
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.types.shard import Layout
from tilefoundry.visitor_registry.contexts import Cost


def _costs(*calls: Call, duration_ns: int = 10):
    return {id(call): (Cost({}, 0), (), duration_ns) for call in calls}


def _calls(function) -> tuple[Call, ...]:
    return tuple(expr for expr in _postorder(function.body) if isinstance(expr, Call))


@func(topologies=(Topology("cta", 168),))
def _large_grid(source: Tensor[(1,), "f32"]):
    return tf.add(source, source)


@func(topologies=(Topology("cta", 64),))
def _independent_branches(source: Tensor[(1,), "f32"]):
    return tf.add(source, source), tf.mul(source, source)


@func(topologies=(Topology("thread", 1),))
def _reshard_boundary(source: Tensor[(1,), "f32"]):
    with Mesh(topology="thread", layout=(1,), names=("lane",)) as thread:
        local = tf.reshard(source, (1 @ thread.lane,), "rmem")
        moved = tf.reshard(local, (1 @ thread.lane,), "rmem")
        return tf.add(moved, moved)


@func(topologies=(Topology("cta", 2), Topology("thread", 4)))
def _thread_sharded(source: Tensor[(8,), "f32"]):
    with Mesh(topology="thread", layout=(4,), names=("lane",)) as thread:
        local = tf.reshard(source, (8 @ thread.lane,), "rmem")
        return tf.add(local, local)


_THREAD_MESH = Mesh(
    Topology("thread", 4),
    Layout((4,), (1,)),
    names=("lane",),
)


@func(topologies=(Topology("thread", 4),))
def _footprint_helper(
    source: Tensor[(8,), "f32", (8 @ _THREAD_MESH.lane,), "rmem"],
):
    return tf.add(source, source)


@func(topologies=(Topology("thread", 4),))
def _footprint_caller(
    source: Tensor[(8,), "f32", (8 @ _THREAD_MESH.lane,), "rmem"],
):
    return _footprint_helper(source)


def test_fixed_grid_larger_than_capacity_unfolds_into_waves() -> None:
    (call,) = _calls(_large_grid)

    makespan, metadata = _timeline_for_function(
        _large_grid, _costs(call), capacity=132
    )

    assert metadata[id(call)].grid_ctas == 168
    assert metadata[id(call)].waves == 2
    assert metadata[id(call)].start_ns == 0
    assert metadata[id(call)].end_ns == makespan


def test_plain_gmem_layout_is_a_resolved_authored_type() -> None:
    analysis = analyze(
        _large_grid,
        options=AnalysisOptions(roofline=True, footprint=False, timeline=False),
    )

    assert "flops f32=168" in analysis.summary_lines


def test_independent_units_overlap_when_their_cta_demands_fit() -> None:
    routed, shared = _calls(_independent_branches)

    makespan, metadata = _timeline_for_function(
        _independent_branches, _costs(routed, shared), capacity=132
    )

    routed_timeline = metadata[id(routed)]
    shared_timeline = metadata[id(shared)]
    assert routed_timeline.waves == shared_timeline.waves == 1
    assert routed_timeline.start_ns == shared_timeline.start_ns == 0
    assert routed_timeline.end_ns == shared_timeline.end_ns == 10
    assert makespan == 10


def test_explicit_reshard_is_a_fusion_boundary_on_both_sides() -> None:
    local, moved, consumer = _calls(_reshard_boundary)
    assert isinstance(local.target, Reshard)
    assert isinstance(moved.target, Reshard)

    makespan, metadata = _timeline_for_function(
        _reshard_boundary, _costs(local, moved, consumer), capacity=132
    )

    assert metadata[id(local)].end_ns == metadata[id(moved)].start_ns
    assert metadata[id(moved)].end_ns == metadata[id(consumer)].start_ns
    assert metadata[id(consumer)].end_ns == makespan == 30


def test_roofline_scales_leaf_cost_by_the_full_execution_mesh() -> None:
    analysis = analyze(
        _thread_sharded,
        options=AnalysisOptions(roofline=True, footprint=False, timeline=False),
    )

    assert "flops f32=16" in analysis.summary_lines


def test_transparent_function_footprint_does_not_double_count_return() -> None:
    analyze(
        _footprint_caller,
        options=AnalysisOptions(roofline=False, footprint=True, timeline=False),
    )

    call = _footprint_caller.body
    assert isinstance(call, Call)
    footprint = get_metadata(call, FootprintMetadata)
    assert footprint is not None
    assert dict(footprint.live_bytes) == {"rmem": 32}
