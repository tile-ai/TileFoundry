from __future__ import annotations

import pytest

from tests.fixtures.meshes import CT, CTA, RUN, THR
from tilefoundry.ir.mesh_scope import (
    check_topology,
    covered_by_scope,
    mesh_scope_matches_required_scope,
    states_consistent_positions,
)
from tilefoundry.ir.types import ComposedLayout, Layout, Mesh, Topology, make_mesh
from tilefoundry.ir.types.int_tuple import product
from tilefoundry.ir.types.layout_algebra import size
from tilefoundry.ir.types.mesh import separate


def test_mesh_position_consistency_is_an_explicit_predicate() -> None:
    matching = Mesh((Topology("thread", 32),), Layout((32,), (1,)), ("g",))
    mismatching = Mesh((Topology("thread", 64),), Layout((32,), (1,)), ("g",))
    explicit = Mesh((Topology("cta", 8),), Layout((8,), (1,)), ("g",))

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
    assert mesh.layout == Layout(shape=((4, 8),), strides=((8, 1),))
    assert mesh.names == ("warp", "lane")
    assert not hasattr(mesh, "topology")
    assert not hasattr(mesh, "axes")

    normalized = Mesh((Topology("cta", 32),), Layout((4, 8), (8, 1)), ("a", "b"))
    assert normalized.topologies == (Topology("cta", 32),)
    assert normalized.layout == Layout(shape=((4, 8),), strides=((8, 1),))


def test_mesh_slice_keeps_the_parent_topologies() -> None:
    mesh = Mesh((Topology("thread", 128),), Layout((4, 32), (32, 1)), ("a", "b"))

    sliced = mesh[0, :]

    assert sliced.topologies is mesh.topologies
    assert sliced.layout.shape == ((1, 32),)


def test_check_topology_rejects_positions_beyond_a_declared_extent() -> None:
    oversized = Mesh((Topology("cta", 128),), Layout((256,), (1,)), ("cta",))

    with pytest.raises(
        ValueError,
        match="mesh level 'cta' has 256 positions, exceeding declared extent 128",
    ):
        check_topology(oversized)


def test_mesh_value_equality_is_by_value() -> None:
    left = Mesh((Topology("thread", 8),), Layout((8,), (1,)), ("g",))
    right = Mesh((Topology("thread", 8),), Layout((8,), (1,)), ("g",))

    assert left == right
    assert hash(left) == hash(right)


def test_make_mesh_appends_a_sliced_scope_with_its_offset() -> None:
    assert make_mesh(CTA, THR[128:256]).layout == RUN
    assert not covered_by_scope(make_mesh(CTA, THR[0:128]), make_mesh(CTA, THR[128:256]))


def test_make_mesh_refuses_a_sliced_suffix_replacement() -> None:
    with pytest.raises(ValueError):
        make_mesh(CT, THR[128:256])


def test_mesh_with_several_levels_slices_in_device_numbering() -> None:
    assert CT[:, 128:256].layout == RUN
    assert CT[:, 128:256] == make_mesh(CTA, THR[128:256])
    assert CT[1:3, 128:256].layout == ComposedLayout(
        None, 1 * 384 + 128, Layout(((2,), (128,)), ((1,), (1,)))
    )


def test_separate_undoes_make_mesh() -> None:
    mesh = CT[1:3, 128:256]
    assert separate(mesh) == (
        Mesh(
            (Topology("cta", 4),),
            ComposedLayout(None, 1, Layout((2,), (1,))),
        ),
        Mesh(
            (Topology("thread", 384),),
            ComposedLayout(None, 128, Layout((128,), (1,))),
        ),
    )
    assert make_mesh(*separate(mesh)) == mesh


def test_mesh_refuses_a_repeated_topology_name() -> None:
    with pytest.raises(ValueError):
        Mesh(
            (Topology("thread", 4), Topology("thread", 32)),
            Layout(((4,), (32,)), ((1,), (1,))),
        )
