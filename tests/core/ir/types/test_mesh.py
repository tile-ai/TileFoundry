from __future__ import annotations

import pytest

from tests.core.fixtures.meshes import CT, CTA, RUN, THR
from tilefoundry.ir import expr, tir, types, utils


def test_topology_and_mesh_require_explicit_extents() -> None:
    with pytest.raises(ValueError, match="extent must be explicit"):
        types.Topology("cta", None)
    with pytest.raises(ValueError, match="mesh axis 0 must have an explicit extent"):
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


def test_check_topology_rejects_positions_beyond_a_declared_extent() -> None:
    oversized = types.Mesh((types.Topology("cta", 128),), types.Layout((256,), (1,)), ("cta",))

    with pytest.raises(
        ValueError,
        match="mesh topology 'cta' has 256 units, exceeding declared extent 128",
    ):
        types.check_topology(oversized)


def test_mesh_value_equality_is_by_value() -> None:
    left = types.Mesh((types.Topology("thread", 8),), types.Layout((8,), (1,)), ("g",))
    right = types.Mesh((types.Topology("thread", 8),), types.Layout((8,), (1,)), ("g",))

    assert left == right
    assert hash(left) == hash(right)


def test_make_mesh_appends_a_sliced_scope_with_its_offset() -> None:
    assert utils.from_meshes(CTA, THR[128:256]).layout == RUN
    assert not utils.equals_mesh_scope(utils.from_meshes(CTA, THR[0:128]), utils.from_meshes(CTA, THR[128:256]))


def test_make_mesh_replaces_a_sliced_suffix_by_component() -> None:
    cta = types.Mesh(
        (types.Topology("cta", 4352),),
        types.Layout((64, 68), (68, 1)),
    )
    current = utils.from_meshes(cta[2:3, :], THR)
    inner = types.Mesh(
        (types.Topology("thread", 384),),
        types.ComposedLayout(None, 128, types.Layout((4, 8, 4), (32, 4, 1))),
    )

    replaced = utils.from_meshes(current, inner)

    assert replaced.layout == types.ComposedLayout(
        None,
        52352,
        types.Layout(((1, 68), (4, 8, 4)), ((68 * 384, 384), (32, 4, 1))),
    )
    assert tuple(utils.topology_mesh(replaced, name) for name in utils.topology_names(replaced)) == (
        types.Mesh(
            (types.Topology("cta", 4352),),
            types.ComposedLayout(None, 136, types.Layout((1, 68), (68, 1))),
        ),
        inner,
    )


def test_mesh_with_several_levels_slices_in_device_numbering() -> None:
    assert CT[:, 128:256].layout == RUN
    assert CT[:, 128:256] == utils.from_meshes(CTA, THR[128:256])
    assert CT[1:3, 128:256].layout == types.ComposedLayout(
        None, 1 * 384 + 128, types.Layout(((2,), (128,)), ((384,), (1,)))
    )


def test_separate_undoes_make_mesh() -> None:
    mesh = CT[1:3, 128:256]
    assert tuple(utils.topology_mesh(mesh, name) for name in utils.topology_names(mesh)) == (
        types.Mesh(
            (types.Topology("cta", 4),),
            types.ComposedLayout(None, 1, types.Layout((2,), (1,))),
        ),
        types.Mesh(
            (types.Topology("thread", 384),),
            types.ComposedLayout(None, 128, types.Layout((128,), (1,))),
        ),
    )
    assert utils.from_meshes(*(utils.topology_mesh(mesh, name) for name in utils.topology_names(mesh))) == mesh



def test_the_first_index_counts_over_every_topology() -> None:
    """The first index ``L(0)`` counts over every topology; ``idx2crd`` splits it into unit ids."""
    cta, thread = types.Topology("cta", 4), types.Topology("thread", 128)
    sliced = utils.from_meshes(
        types.Mesh((cta,), types.Layout((4,), (1,)), ("c",)),
        types.Mesh((thread,), types.Layout((128,), (1,)), ("t",)),
    )[1, 32:64]

    assert sliced.layout(0) == 1 * 128 + 32
    assert utils.topology_mesh(sliced, "cta") == types.Mesh(
        (cta,), types.ComposedLayout(None, 1, types.Layout((1,), (1,))), ("c",)
    )
    assert utils.topology_mesh(sliced, "thread") == types.Mesh(
        (thread,), types.ComposedLayout(None, 32, types.Layout((32,), (1,))), ("t",)
    )
    assert utils.topology_mesh(sliced, "gpu") is None


def test_a_slice_counts_its_unit_ids_from_its_start() -> None:
    """Unit ids are absolute in the topology; coords count from the slice start."""
    mesh = types.Mesh((types.Topology("thread", 128),), types.Layout((128,), (1,)), ("t",))[16:48]

    ids = [placement.ids["thread"] for placement in types.Placement.from_mesh(mesh)]
    assert ids == list(range(16, 48))
    assert [types.mesh_coords(mesh, (unit_id,)) for unit_id in (16, 23, 47)] == [(0,), (7,), (31,)]
    assert types.mesh_coords(mesh, (None,)) == (None,)

def test_mesh_refuses_a_repeated_topology_name() -> None:
    with pytest.raises(ValueError):
        types.Mesh(
            (types.Topology("thread", 4), types.Topology("thread", 32)),
            types.Layout(((4,), (32,)), ((32,), (1,))),
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
    assert utils.equals_mesh_scope(inner, outer) is covered
    assert utils.within_mesh_scope(inner, outer) is within


@pytest.mark.parametrize("selection", (_HOLES, _STRIDED), ids=("holes", "strided"))
def test_a_scope_with_gaps_is_refused(selection) -> None:
    assert not utils.within_mesh_scope(selection, selection)
    assert not utils.within_mesh_scope(selection, _FULL)
    with pytest.raises(ValueError, match="both must be continuous"):
        utils.from_meshes(utils.from_meshes(THR, selection), selection)


_CTA4, _THREAD128 = types.Topology("cta", 4), types.Topology("thread", 128)


def test_a_mesh_of_several_topologies_maps_a_coord_to_the_index_over_all() -> None:
    """Stored strides are in the index over both topologies; ``idx2crd`` reads the unit ids back."""
    grouped = types.Mesh((_CTA4, _THREAD128), types.Layout(((4,), (128,)), None))
    joined = utils.from_meshes(
        types.Mesh((_CTA4,), types.Layout((4,), (1,))),
        types.Mesh((_THREAD128,), types.Layout((128,), (1,))),
    )
    assert grouped == joined
    assert grouped.layout.strides == ((128,), (1,))
    assert tuple(types.get(grouped.layout, t) for t in range(2)) == (
        types.Layout((4,), (128,)),
        types.Layout((128,), (1,)),
    )
    assert grouped.layout(1 * 128 + 40) == types.crd2idx((1, 40), (4, 128), (128, 1)) == 168
    sliced = grouped[1, 32:64]
    assert sliced.layout(0) == 160
    assert utils.first_unit_ids(sliced) == (1, 32)


def test_an_outer_stride_counts_every_declared_inner_unit() -> None:
    """Thread declares 256 units and the mesh selects 32; the CTA strides scale by 256."""
    mesh = utils.from_meshes(
        types.Mesh((types.Topology("cta", 8),), types.Layout((4, 2), (2, 1))),
        types.Mesh((types.Topology("thread", 256),), types.Layout((32,), (1,))),
    )
    assert mesh.layout.strides == ((512, 256), (1,))
    assert utils.unit_layout(mesh, 0) == types.Layout((4, 2), (2, 1))


@pytest.mark.parametrize(
    "layout",
    (
        (4, 128),
        types.Layout(((4,), (128,)), ((128,), (1,))),
        types.ComposedLayout(None, 0, types.Layout(((4,), (128,)), ((128,), (1,)))),
    ),
    ids=("ungrouped", "grouped", "composed"),
)
def test_an_inner_topology_needs_a_static_size(layout) -> None:
    threads = types.Topology("thread", expr.DimVar("threads", 1, 128))
    with pytest.raises(ValueError, match="unit count of every inner topology"):
        types.Mesh((_CTA4, threads), layout)


def test_the_outermost_size_may_be_symbolic() -> None:
    blocks = types.Topology("cta", expr.DimVar("blocks", 1, 64))
    mesh = types.Mesh((blocks, _THREAD128), types.Layout(((4,), (128,)), ((128,), (1,))))
    assert utils.unit_layout(mesh, 0) == types.Layout((4,), (1,))


def test_flat_layout_keeps_the_slice_wrapper() -> None:
    whole = utils.from_meshes(types.Mesh((_CTA4,), types.Layout((4,))), types.Mesh((_THREAD128,), types.Layout((128,))))
    assert utils.flat_layout(whole) == types.Layout((4, 128), (128, 1))
    assert utils.flat_layout(whole[1, 32:64]) == types.ComposedLayout(None, 160, types.Layout((1, 32), (128, 1)))
    assert utils.flat_layout(whole[0, :]) == types.ComposedLayout(None, 0, types.Layout((1, 128), (128, 1)))
    swizzled = types.Mesh((_THREAD128,), types.ComposedLayout(types.Swizzle(1, 0, 1), 0, types.Layout((32,), (1,))))
    assert utils.flat_layout(swizzled).inner == types.Swizzle(1, 0, 1)


def _local_mesh(mesh: types.Mesh, coords: tuple) -> types.Mesh:
    attrs = tuple(types.Split(0) if axis == 0 else types.Broadcast() for axis in range(len(coords)))
    layout, _ = types.local_layout(types.ShardLayout(types.Layout((2, 4)), attrs, mesh), (2, 4), coords)
    return layout.mesh


def test_a_local_view_reencodes_the_mesh_it_keeps() -> None:
    cta, thread = types.Topology("cta", 2), types.Topology("thread", 4)
    whole = types.Mesh((cta, thread), types.Layout(((2,), (4,)), None), ("c", "t"))
    assert _local_mesh(whole[1, 2:4], (None, 0)) == types.Mesh(
        (cta,), types.ComposedLayout(None, 1, types.Layout((1,), (1,))), ("c",)
    )
    assert _local_mesh(whole[1, 2:4], (1, None)) == types.Mesh(
        (thread,), types.ComposedLayout(None, 2, types.Layout((2,), (1,))), ("t",)
    )
    split = types.Mesh((cta, thread), types.Layout(((2,), (2, 2)), None), ("c", "a", "b"))
    assert _local_mesh(split, (None, 1, None)) == types.Mesh(
        (cta, thread), types.Layout(((2,), (2,)), ((4,), (1,))), ("c", "b")
    )


def test_a_constant_slice_is_proved_by_its_sub_box() -> None:
    whole = utils.from_meshes(types.Mesh((_CTA4,), types.Layout((4,))), types.Mesh((_THREAD128,), types.Layout((128,))))
    assert tir.sync.legal_slice_of(whole[:, :], whole)
    assert tir.sync.legal_slice_of(whole[1:3, 32:64], whole)
    assert not tir.sync.legal_slice_of(whole[1:3, 32:64], whole[1:3, :])
    assert not tir.sync.legal_slice_of(
        types.Mesh(whole.topologies, types.ComposedLayout(None, 160, types.Layout(((2,), (32,)), ((256,), (1,))))),
        whole,
    )
    with pytest.raises(ValueError, match="'cta' steps by 1, which is not a multiple of its unit 128"):
        types.Mesh(whole.topologies, types.ComposedLayout(None, 160, types.Layout(((2,), (32,)), ((1,), (1,)))))
