from __future__ import annotations

import ast

from tilefoundry.analysis.timeline import _fusable
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard import (
    Layout,
    Mesh,
    Partial,
    Split,
    Topology,
    make_mesh,
    product,
)
from tilefoundry.ir.types.shard.layout_algebra import size
from tilefoundry.ir.types.shard.scope_match import (
    mesh_scope_matches_required_scope,
    states_consistent_positions,
)
from tilefoundry.ir.types.shard.shard_layout import ShardLayout
from tilefoundry.parser.sugar import parse_shard_layout_sugar
from tilefoundry.schedule.partition.problem import _placement_relation


def test_mesh_position_consistency_is_an_explicit_predicate() -> None:
    matching = make_mesh((32,), topology="thread")
    mismatching = make_mesh((32,), topology=Topology("thread", 64))
    launch_provided = make_mesh((8,), topology=Topology("cta", None))

    assert product(matching.topologies) == 32
    assert product((2, None)) is None
    assert states_consistent_positions(matching)
    assert not states_consistent_positions(mismatching)
    assert product(launch_provided.topologies) is None
    assert size(launch_provided.layout) == 8
    assert states_consistent_positions(launch_provided)
    assert not mesh_scope_matches_required_scope(mismatching, matching)


def test_mesh_is_an_unmodified_record_without_axis_attributes() -> None:
    topologies = (Topology("thread", 32),)
    layout = Layout(shape=(4, 8), strides=(8, 1))

    mesh = Mesh(topologies, layout, ("warp", "lane"))

    assert mesh.topologies is topologies
    assert mesh.layout is layout
    assert mesh.names == ("warp", "lane")
    assert "__post_init__" not in Mesh.__dict__
    assert not hasattr(mesh, "topology")
    assert not hasattr(mesh, "axes")

    normalized = make_mesh((4, 8), topology="cta")
    assert normalized.topologies == (Topology("cta", 32),)
    assert normalized.layout == Layout(shape=(4, 8), strides=(8, 1))


def test_mesh_slice_keeps_the_parent_topologies() -> None:
    mesh = make_mesh((4, 32), topology="thread")

    sliced = mesh[0, :]

    assert sliced.topologies is mesh.topologies
    assert sliced.layout.shape == (1, 32)


def test_named_mesh_axis_sugar_carries_a_layout_index() -> None:
    mesh = make_mesh((8,), names=("cta",), topology="cta")
    node = ast.parse("(8 @ cta.cta,)", mode="eval").body

    layout = parse_shard_layout_sugar(node, lambda name: mesh if name == "cta" else None)
    partial_node = ast.parse('((8,), {cta.cta @ P("sum")})', mode="eval").body
    partial_layout = parse_shard_layout_sugar(
        partial_node, lambda name: mesh if name == "cta" else None
    )

    assert layout.attrs == (Split(0),)
    assert partial_layout.attrs == (Partial("sum"),)


def test_mesh_value_equality_is_usable_by_timeline_and_partition() -> None:
    left = make_mesh((8,), topology="thread")
    right = make_mesh((8,), topology="thread")
    layout = Layout(shape=(8,), strides=(1,))
    producer = TensorType(
        shape=(8,),
        dtype=DType.f32,
        layout=ShardLayout(layout, (Split(0),), left),
        storage="rmem",
    )
    consumer = TensorType(
        shape=(8,),
        dtype=DType.f32,
        layout=ShardLayout(layout, (Split(0),), right),
        storage="rmem",
    )

    assert left == right
    assert hash(left) == hash(right)
    assert _fusable(producer, consumer)
    assert _placement_relation(consumer, left) == "SAME_INTERVAL"
