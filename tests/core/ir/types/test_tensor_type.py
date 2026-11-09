"""Spec 002 TensorType — equality over a symbolic shape, and DimVar validation."""

from __future__ import annotations

import pytest

from tilefoundry.ir import dim, expr, types, utils


def test_tensor_type_equality_over_a_dim_var_shape_entry() -> None:
    """Test tensor type equality over a dim var shape entry.

    A TensorType carrying a bounded ``DimVar(name, lo, hi)`` stays hashable and
    compares equal to an independently constructed one: the DimVar is interned by
    ``(name, lo, hi)``, so an independently built signature uses the same key.
    """
    s = expr.DimVar("S_a", 1, 7)
    t = types.TensorType(shape=(s, 8), dtype=expr.DType.f32, layout=None, storage="gmem")
    assert t.shape == (s, 8)
    t2 = types.TensorType(shape=(expr.DimVar("S_a", 1, 7), 8), dtype=expr.DType.f32, layout=None, storage="gmem")
    assert t == t2
    assert hash(t) == hash(t2)


def test_dim_var_range_validation() -> None:
    """``lo <= hi`` is required (closed [lo, hi]); a single value is [k, k].

    Same name with different bounds constructs distinct objects rather than
    raising: cross-instance scoping is a signature-level rule enforced by HIR
    ``verify_function``, not by construction.
    """
    point = expr.DimVar("S_point", 4, 4)
    assert (point.lo, point.hi) == (4, 4)
    with pytest.raises(ValueError, match="require lo <= hi"):
        expr.DimVar("S_empty", 4, 3)
    with pytest.raises(ValueError, match="require lo <= hi"):
        expr.DimVar("S_inv", 5, 0)

    a = expr.DimVar("S_conflict", 1, 3)
    b = expr.DimVar("S_conflict", 1, 7)
    assert a is not b
    assert ((a.lo, a.hi), (b.lo, b.hi)) == ((1, 3), (1, 7))


def test_zero_extent_has_zero_logical_and_local_size() -> None:
    type = types.TensorType(shape=(0, 8), dtype=expr.DType.f32, layout=None, storage="gmem")

    assert utils.numel(type) == 0
    assert utils.tensor_bytes(type) == 0

    sharded = utils.make_shard_tensor_type(
        (0,), mesh=types.Mesh((types.Topology("gpu", 2),), types.Layout((2,), (1,)), ("g",)), attrs=(types.Split(0),)
    )
    assert utils.local_type_of(sharded, topology_level="gpu", topologies=(types.Topology("gpu", 2),)).shape == (
        1,
        0,
    )


def test_size_rejects_symbolic_and_negative_extents() -> None:
    ctx_len = expr.DimVar("ctx_len", 0, 4095)
    symbolic = types.TensorType(shape=(ctx_len,), dtype=expr.DType.f32, layout=None, storage="gmem")
    compound = types.TensorType(shape=(ctx_len + 1,), dtype=expr.DType.f32, layout=None, storage="gmem")
    negative = types.TensorType(shape=(-1,), dtype=expr.DType.f32, layout=None, storage="gmem")

    with pytest.raises(ValueError, match=r"ctx_len.*bind it with --dim ctx_len=EXTENT"):
        utils.numel(symbolic)
    with pytest.raises(ValueError, match=r"ctx_len.*bind it with --dim ctx_len=EXTENT"):
        utils.numel(compound)
    with pytest.raises(ValueError, match="tensor extent -1 is negative"):
        utils.numel(negative)


def test_local_type_rejects_a_zero_mesh_extent() -> None:
    mesh = types.Mesh((types.Topology("gpu", 0),), types.Layout((0,), (1,)), ("g",))
    layout = types.ShardLayout(types.Layout(shape=(0,), strides=(1,)), (types.Split(0),), mesh)
    type = types.TensorType(shape=(0,), dtype=expr.DType.f32, layout=layout, storage="gmem")

    with pytest.raises(ValueError, match="mesh extent is not a concrete positive integer"):
        utils.local_type_of(type, topology_level="gpu", topologies=(types.Topology("gpu", 0),))


def _cta_mesh(extent: int) -> types.Mesh:
    return types.Mesh(
        topologies=(types.Topology("cta", extent),),
        layout=types.Layout(shape=(extent,), strides=(1,)),
    )


@pytest.mark.parametrize(
    ("shape", "extent", "expected"),
    [
        ((1, 128, 128, 2048), 128, (1, 1, 128, 2048)),
        ((1, 1024, 128, 2048), 128, (1, 1, 8, 128, 2048)),
    ],
)
def test_local_type_preserves_canonical_split_projection(
    shape: tuple[int, ...], extent: int, expected: tuple[int, ...]
) -> None:
    tensor = utils.make_shard_tensor_type(shape, mesh=_cta_mesh(extent), attrs=(types.Split(1),))

    assert (
        utils.local_type_of(tensor, topology_level="cta", topologies=(types.Topology("cta", extent),)).shape
        == expected
    )


def test_local_type_projects_a_static_offset_sharded_view():
    sharded = utils.make_shard_tensor_type(
        (64, 8, 1, 16), mesh=_cta_mesh(8), attrs=(types.Split(1),), storage="smem"
    )
    view = types.TensorType(
        shape=sharded.shape,
        dtype=sharded.dtype,
        layout=types.ComposedLayout(inner=None, offset=64, outer=sharded.layout),
        storage=sharded.storage,
    )

    local = utils.local_type_of(view, topology_level="cta", topologies=(types.Topology("cta", 8),))

    assert local.shape == (64, 1, 1, 16)
    assert local.layout is view.layout


@pytest.mark.parametrize(
    ("shape", "mesh_shape", "attrs", "expected"),
    [
        ((16, 64), (4, 8), (types.Split(0), types.Split(1)), (4, 8)),
        ((128,), (4, 32), (types.Split(0), types.Split(0)), (1,)),
    ],
)
def test_local_type_projects_every_axis_of_a_single_topology_mesh(
    shape: tuple[int, ...],
    mesh_shape: tuple[int, ...],
    attrs: tuple[types.Split, ...],
    expected: tuple[int, ...],
) -> None:
    topology_extent = mesh_shape[0] * mesh_shape[1]
    mesh = types.Mesh(
        topologies=(types.Topology("cta", topology_extent),),
        layout=types.Layout(shape=mesh_shape, strides=(mesh_shape[1], 1)),
    )
    layout = types.ShardLayout(
        layout=types.Layout(shape=shape, strides=None),
        attrs=attrs,
        mesh=mesh,
    )
    tensor = types.TensorType(shape=shape, dtype=expr.DType.f32, layout=layout, storage="gmem")

    local = utils.local_type_of(
        tensor,
        topology_level="cta",
        topologies=(types.Topology("cta", topology_extent),),
    )

    assert local.shape == expected


@pytest.mark.parametrize(("shape", "extent"), [(1024, 132), (64, 128)])
def test_default_shard_layout_keeps_rejecting_non_divisible_splits(
    shape: int, extent: int
) -> None:
    with pytest.raises(ValueError, match="logical axis 1 size .* is not divisible"):
        utils.make_shard_tensor_type((1, shape, 128, 2048), mesh=_cta_mesh(extent), attrs=(types.Split(1),))


@pytest.mark.parametrize(
    "axis", [expr.DimVar("S_fixed", 1, 8192), dim.ceildiv(expr.DimVar("S_fixed_tiles", 1, 8192), 128)]
)
def test_local_type_rejects_dynamic_split_against_fixed_mesh(axis: object) -> None:
    tensor = utils.make_shard_tensor_type((1, axis, 128, 2048), mesh=_cta_mesh(132), attrs=(types.Split(1),))

    with pytest.raises(ValueError, match="bind the axis before local projection"):
        utils.local_type_of(tensor, topology_level="cta", topologies=(types.Topology("cta", 132),))


def test_local_type_rejects_unfactorized_inexact_split_with_two_loop_form() -> None:
    layout = types.ShardLayout(
        layout=types.Layout(shape=(10,), strides=(1,)), attrs=(types.Split(0),), mesh=_cta_mesh(3)
    )
    tensor = types.TensorType(shape=(10,), dtype=expr.DType.f32, layout=layout, storage="gmem")

    with pytest.raises(ValueError, match=r"two loops out as \(ceildiv\(N, T\), T\)"):
        utils.local_type_of(type=tensor, topology_level="cta", topologies=(types.Topology("cta", 3),))


def test_local_type_stops_before_a_finer_split() -> None:
    base = types.Layout(shape=(2, 4, 8), strides=(32, 8, 1))
    thread = types.ShardLayout(
        layout=base,
        attrs=(types.Split(1),),
        mesh=types.Mesh(
            topologies=(types.Topology("thread", 4),),
            layout=types.Layout(shape=(4,), strides=(1,)),
        ),
    )
    layout = types.ShardLayout(
        layout=thread,
        attrs=(types.Split(0),),
        mesh=types.Mesh(
            topologies=(types.Topology("cta", 2),),
            layout=types.Layout(shape=(2,), strides=(1,)),
        ),
    )
    tensor = types.TensorType(shape=(64,), dtype=expr.DType.f32, layout=layout, storage="gmem")

    cta = utils.local_type_of(
        tensor,
        topology_level="cta",
        topologies=(types.Topology("cta", 2), types.Topology("thread", 4)),
    )
    thread = utils.local_type_of(
        tensor,
        topology_level="thread",
        topologies=(types.Topology("cta", 2), types.Topology("thread", 4)),
    )

    assert utils.numel(cta) == 32
    assert utils.numel(thread) == 8


@pytest.mark.parametrize("attr", [types.Broadcast(), types.Partial()])
def test_local_type_does_not_divide_replicated_or_partial_values(attr) -> None:
    tensor = utils.make_shard_tensor_type((256,), mesh=_cta_mesh(4), attrs=(attr,))

    local = utils.local_type_of(tensor, topology_level="cta", topologies=(types.Topology("cta", 4),))

    assert utils.numel(local) == 256


@pytest.mark.parametrize(
    ("declared_layout", "actual_layout", "expected"),
    [
        pytest.param(types.Layout((8, 16), (16, 1)), types.Layout((8, 16), (1, 8)), False, id="written-strides"),
        pytest.param(types.Layout((8, 16), (16, 1)), None, False, id="plain-to-none"),
        pytest.param(None, types.Layout((8, 16), (16, 1)), True, id="none-to-plain"),
        pytest.param(
            types.ShardLayout(types.Layout((8, 16), (16, 1)), (types.Broadcast(),), types.Mesh((types.Topology("thread", 8),), (8,), ("lane",))),
            types.Layout((8, 16), (16, 1)), False, id="broadcast-to-plain",
        ),
        pytest.param(
            types.ShardLayout(types.Layout((8, 16), (16, 1)), (types.Broadcast(),), types.Mesh((types.Topology("thread", 8),), (8,), ("lane",))),
            types.Layout((8, 16), (1, 8)), False, id="broadcast-written-strides",
        ),
        pytest.param(
            types.ComposedLayout(types.Swizzle(2, 4, 3), 0, types.Layout((8, 16), (16, 1))),
            None, False, id="composed-to-none",
        ),
        pytest.param(
            types.ShardLayout(types.Layout((8, 16), (16, 1)), (types.Split(0),), types.Mesh((types.Topology("thread", 8),), (8,), ("lane",))),
            types.Layout((8, 16), (16, 1)), False, id="split-to-plain",
        ),
    ],
)
def test_unsplit_layout_forms_preserve_written_constraints(
    declared_layout, actual_layout, expected: bool,
) -> None:
    declared = types.TensorType((8, 16), expr.DType.f32, declared_layout, "rmem")
    actual = types.TensorType((8, 16), expr.DType.f32, actual_layout, "rmem")
    assert utils.types_compatible(declared, actual) is expected
