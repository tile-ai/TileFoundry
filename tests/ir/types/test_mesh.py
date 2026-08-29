from __future__ import annotations

import pytest

from tilefoundry.ir.types.shard import (
    Layout,
    Mesh,
    Topology,
    make_mesh,
    product,
)
from tilefoundry.ir.types.shard.layout_algebra import size
from tilefoundry.ir.types.shard.scope_match import (
    mesh_scope_matches_required_scope,
    states_consistent_positions,
)


def test_mesh_position_consistency_is_an_explicit_predicate() -> None:
    matching = make_mesh((32,), topology="thread")
    mismatching = make_mesh((32,), topology=Topology("thread", 64))
    explicit = make_mesh((8,), topology=Topology("cta", 8))

    assert product(matching.topologies) == 32
    assert states_consistent_positions(matching)
    assert not states_consistent_positions(mismatching)
    assert product(explicit.topologies) == 8
    assert size(explicit.layout) == 8
    assert states_consistent_positions(explicit)
    assert not mesh_scope_matches_required_scope(mismatching, matching)


def test_topology_and_mesh_require_explicit_extents() -> None:
    with pytest.raises(ValueError, match="extent must be explicit"):
        Topology("cta", None)
    with pytest.raises(ValueError, match="layout axis 0 must have an explicit extent"):
        Mesh((Topology("cta", 8),), Layout(shape=(None,), strides=(1,)))


def test_mesh_is_a_frozen_record_without_axis_attributes() -> None:
    topologies = (Topology("thread", 32),)
    layout = Layout(shape=(4, 8), strides=(8, 1))

    mesh = Mesh(topologies, layout, ("warp", "lane"))

    assert mesh.topologies is topologies
    assert mesh.layout is layout
    assert mesh.names == ("warp", "lane")
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


def test_mesh_value_equality_is_by_value() -> None:
    left = make_mesh((8,), topology="thread")
    right = make_mesh((8,), topology="thread")

    assert left == right
    assert hash(left) == hash(right)
