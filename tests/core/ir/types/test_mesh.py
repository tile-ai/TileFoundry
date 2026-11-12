from __future__ import annotations

import pytest

from tests.core.fixtures.meshes import CT, CTA, RUN, THR
from tilefoundry.ir import expr, types, utils
from tilefoundry.ir.types import int_tuple


def test_mesh_position_consistency_is_an_explicit_predicate() -> None:
    matching = types.Mesh((types.Topology("thread", 32),), types.Layout((32,), (1,)), ("g",))
    mismatching = types.Mesh((types.Topology("thread", 64),), types.Layout((32,), (1,)), ("g",))
    explicit = types.Mesh((types.Topology("cta", 8),), types.Layout((8,), (1,)), ("g",))

    assert int_tuple.product(matching.topologies) == 32
    assert utils.states_consistent_positions(matching)
    assert not utils.states_consistent_positions(mismatching)
    assert int_tuple.product(explicit.topologies) == 8
    assert types.size(explicit.layout) == 8
    assert utils.states_consistent_positions(explicit)
    assert not utils.mesh_scope_matches_required_scope(mismatching, matching)


def test_topology_and_mesh_require_explicit_extents() -> None:
    with pytest.raises(ValueError, match="extent must be explicit"):
        types.Topology("cta", None)
    with pytest.raises(ValueError, match="layout axis 0 must have an explicit extent"):
        types.Mesh((types.Topology("cta", 8),), types.Layout(shape=(None,), strides=(1,)))


def test_mesh_is_a_frozen_record_without_axis_attributes() -> None:
    topologies = (types.Topology("thread", 32),)
    layout = types.Layout(shape=(4, 8), strides=(8, 1))

    mesh = types.Mesh(topologies, layout, ("warp", "lane"))

    assert mesh.topologies is topologies
    assert mesh.layout == types.Layout(shape=((4, 8),), strides=((8, 1),))
    assert mesh.names == ("warp", "lane")
    assert not hasattr(mesh, "topology")
    assert not hasattr(mesh, "axes")

    normalized = types.Mesh((types.Topology("cta", 32),), types.Layout((4, 8), (8, 1)), ("a", "b"))
    assert normalized.topologies == (types.Topology("cta", 32),)
    assert normalized.layout == types.Layout(shape=((4, 8),), strides=((8, 1),))


def test_mesh_slice_keeps_the_parent_topologies() -> None:
    mesh = types.Mesh((types.Topology("thread", 128),), types.Layout((4, 32), (32, 1)), ("a", "b"))

    sliced = mesh[0, :]

    assert sliced.topologies is mesh.topologies
    assert sliced.layout.shape == ((1, 32),)


def test_placements_keep_each_sliced_level_origin():
    mesh = types.Mesh(
        (types.Topology("node", 3), types.Topology("gpu", 4)),
        types.Layout((3, 4)), ("replica", "local"),
    )
    assert types.Placement.from_mesh(mesh[1:3, 2:4]) == tuple(
        types.Placement({"node": node, "gpu": gpu})
        for node in (1, 2) for gpu in (2, 3)
    )


def test_mesh_coordinates_keep_zero_stride_singleton_axes():
    mesh = types.Mesh((types.Topology("gpu", 8),), types.Layout((1, 2, 1, 4)),
                      ("data", "ep", "replica", "cp"))
    for rank in range(8):
        assert types.mesh_coords(mesh, (rank,)) == (0, rank // 4, 0, rank % 4)


def test_check_topology_rejects_positions_beyond_a_declared_extent() -> None:
    oversized = types.Mesh((types.Topology("cta", 128),), types.Layout((256,), (1,)), ("cta",))

    with pytest.raises(
        ValueError,
        match="mesh level 'cta' has 256 positions, exceeding declared extent 128",
    ):
        types.check_topology(oversized)


def test_mesh_value_equality_is_by_value() -> None:
    left = types.Mesh((types.Topology("thread", 8),), types.Layout((8,), (1,)), ("g",))
    right = types.Mesh((types.Topology("thread", 8),), types.Layout((8,), (1,)), ("g",))

    assert left == right
    assert hash(left) == hash(right)


def test_make_mesh_appends_a_sliced_scope_with_its_offset() -> None:
    assert types.Mesh.from_meshes(CTA, THR[128:256]).layout == RUN
    assert not utils.covered_by_scope(types.Mesh.from_meshes(CTA, THR[0:128]), types.Mesh.from_meshes(CTA, THR[128:256]))


def test_make_mesh_replaces_a_sliced_suffix_by_component() -> None:
    cta = types.Mesh(
        (types.Topology("cta", 4352),),
        types.Layout((64, 68), (68, 1)),
    )
    current = types.Mesh.from_meshes(cta[2:3, :], THR)
    inner = types.Mesh(
        (types.Topology("thread", 384),),
        types.ComposedLayout(None, 128, types.Layout((4, 8, 4), (32, 4, 1))),
    )

    replaced = types.Mesh.from_meshes(current, inner)

    assert replaced.layout == types.ComposedLayout(
        None,
        52352,
        types.Layout(((1, 68), (4, 8, 4)), ((68, 1), (32, 4, 1))),
    )
    assert types.separate(replaced) == (
        types.Mesh(
            (types.Topology("cta", 4352),),
            types.ComposedLayout(None, 136, types.Layout((1, 68), (68, 1))),
        ),
        inner,
    )


def test_mesh_with_several_levels_slices_in_device_numbering() -> None:
    assert CT[:, 128:256].layout == RUN
    assert CT[:, 128:256] == types.Mesh.from_meshes(CTA, THR[128:256])
    assert CT[1:3, 128:256].layout == types.ComposedLayout(
        None, 1 * 384 + 128, types.Layout(((2,), (128,)), ((1,), (1,)))
    )


def test_separate_undoes_make_mesh() -> None:
    mesh = CT[1:3, 128:256]
    assert types.separate(mesh) == (
        types.Mesh(
            (types.Topology("cta", 4),),
            types.ComposedLayout(None, 1, types.Layout((2,), (1,))),
        ),
        types.Mesh(
            (types.Topology("thread", 384),),
            types.ComposedLayout(None, 128, types.Layout((128,), (1,))),
        ),
    )
    assert types.Mesh.from_meshes(*types.separate(mesh)) == mesh


def test_mesh_refuses_a_repeated_topology_name() -> None:
    with pytest.raises(ValueError):
        types.Mesh(
            (types.Topology("thread", 4), types.Topology("thread", 32)),
            types.Layout(((4,), (32,)), ((1,), (1,))),
        )


_SEQUENCE = expr.DimVar("mesh_sequence", 64, 2048)
_CHUNKS = _SEQUENCE // 64
_TOPOLOGY = (types.Topology("cta", 132),)
_FULL = types.Mesh(_TOPOLOGY, types.Layout((2, _CHUNKS, 2), (2 * _CHUNKS, 2, 1)))
_FLAT = types.Mesh(_TOPOLOGY, types.Layout((4 * _CHUNKS,), (1,)))
_HALF = types.Mesh(_TOPOLOGY, types.Layout((2 * _CHUNKS,), (1,)))
_HOLES = types.Mesh(_TOPOLOGY, types.Layout((2, _CHUNKS), (2 * _CHUNKS, 1)))
_STRIDED = types.Mesh(_TOPOLOGY, types.Layout((_CHUNKS,), (3,)))


@pytest.mark.parametrize(
    ("inner", "outer", "covered", "within"),
    (
        (_FULL, _FULL, True, True),
        (_FULL, _FLAT, True, True),
        (_HALF, _FULL, False, True),
        (_FULL, _HALF, False, False),
        (_FULL, THR, False, False),
    ),
    ids=("same", "equivalent", "contained", "exceeds", "different-topology"),
)
def test_symbolic_scopes_compare_by_selected_positions(inner, outer, covered, within) -> None:
    assert utils.covered_by_scope(inner, outer) is covered
    assert types.within_scope(inner, outer) is within


@pytest.mark.parametrize("selection", (_HOLES, _STRIDED), ids=("holes", "strided"))
def test_a_scope_with_gaps_is_refused(selection) -> None:
    assert not types.within_scope(selection, selection)
    assert not types.within_scope(selection, _FULL)
    with pytest.raises(ValueError, match="both must be continuous"):
        types.Mesh.from_meshes(types.Mesh.from_meshes(THR, selection), selection)
