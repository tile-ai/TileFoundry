"""Spec 002 TensorType — equality over a symbolic shape, and DimVar validation."""

from __future__ import annotations

import pytest

from tilefoundry.ir.core import Call, Var
from tilefoundry.ir.core.kinds import UnaryKind
from tilefoundry.ir.hir.math.unary import Unary
from tilefoundry.ir.types import (
    DType,
    TensorType,
    local_type_of,
    make_shard_tensor_type,
    numel,
    tensor_bytes,
)
from tilefoundry.ir.types.dim import DimVar, ceildiv
from tilefoundry.ir.types.shard import (
    Broadcast,
    Layout,
    Mesh,
    Partial,
    ShardLayout,
    Split,
    Topology,
    make_mesh,
)
from tilefoundry.visitor_registry.contexts import CostContext
from tilefoundry.visitor_registry.visitors import CostEvaluator


def test_tensor_type_equality_over_a_dim_var_shape_entry() -> None:
    """A TensorType carrying a bounded ``DimVar(name, lo, hi)`` stays hashable and
    compares equal to an independently constructed one: the DimVar is interned by
    ``(name, lo, hi)``, so an independently built signature uses the same key."""
    s = DimVar("S_a", 1, 8)
    t = TensorType(shape=(s, 8), dtype=DType.f32, layout=None, storage="gmem")
    assert t.shape == (s, 8)
    t2 = TensorType(shape=(DimVar("S_a", 1, 8), 8), dtype=DType.f32, layout=None, storage="gmem")
    assert t == t2
    assert hash(t) == hash(t2)


def test_dim_var_range_validation() -> None:
    """``lo < hi`` is required (half-open [lo, hi)); a single value is [k, k+1).

    Same name with different bounds constructs distinct objects rather than
    raising: cross-instance scoping is a signature-level rule enforced by HIR
    ``verify_function``, not by construction.
    """
    DimVar("S_point", 4, 5)  # single value 4 as [4, 5) — no raise
    with pytest.raises(ValueError, match="require lo < hi"):
        DimVar("S_empty", 4, 4)  # empty half-open range
    with pytest.raises(ValueError, match="require lo < hi"):
        DimVar("S_inv", 5, 1)

    a = DimVar("S_conflict", 1, 4)
    b = DimVar("S_conflict", 1, 8)
    assert a is not b
    assert ((a.lo, a.hi), (b.lo, b.hi)) == ((1, 4), (1, 8))


def test_zero_extent_has_zero_logical_and_local_size() -> None:
    type = TensorType(shape=(0, 8), dtype=DType.f32, layout=None, storage="gmem")

    assert numel(type) == 0
    assert tensor_bytes(type) == 0

    sharded = make_shard_tensor_type((0,), mesh=make_mesh((2,)), attrs=(Split(0),))
    assert local_type_of(
        sharded, level="gpu", topologies=(Topology("gpu", 2),)
    ).shape == (1, 0)


def test_size_rejects_symbolic_and_negative_extents() -> None:
    ctx_len = DimVar("ctx_len", 0, 4096)
    symbolic = TensorType(shape=(ctx_len,), dtype=DType.f32, layout=None, storage="gmem")
    compound = TensorType(shape=(ctx_len + 1,), dtype=DType.f32, layout=None, storage="gmem")
    negative = TensorType(shape=(-1,), dtype=DType.f32, layout=None, storage="gmem")

    with pytest.raises(ValueError, match=r"ctx_len.*bind it with --dim ctx_len=EXTENT"):
        numel(symbolic)
    with pytest.raises(ValueError, match=r"ctx_len.*bind it with --dim ctx_len=EXTENT"):
        numel(compound)
    with pytest.raises(ValueError, match="tensor extent -1 is negative"):
        numel(negative)


def test_local_type_rejects_a_zero_mesh_extent() -> None:
    mesh = make_mesh((0,))
    layout = ShardLayout(Layout(shape=(0,), strides=(1,)), (Split(0),), mesh)
    type = TensorType(shape=(0,), dtype=DType.f32, layout=layout, storage="gmem")

    with pytest.raises(ValueError, match="mesh extent is not a concrete positive integer"):
        local_type_of(type, level="gpu", topologies=(Topology("gpu", 0),))


def _cta_mesh(extent: int | None) -> Mesh:
    return Mesh(
        topologies=(Topology("cta", extent),),
        layout=Layout(shape=(extent,), strides=(1,)),
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
    tensor = make_shard_tensor_type(shape, mesh=_cta_mesh(extent), attrs=(Split(1),))

    assert local_type_of(
        tensor, level="cta", topologies=(Topology("cta", extent),)
    ).shape == expected


@pytest.mark.parametrize(
    ("shape", "mesh_shape", "attrs", "expected"),
    [
        ((16, 64), (4, 8), (Split(0), Split(1)), (4, 8)),
        ((128,), (4, 32), (Split(0), Split(0)), (1,)),
    ],
)
def test_local_type_projects_every_axis_of_a_single_topology_mesh(
    shape: tuple[int, ...],
    mesh_shape: tuple[int, ...],
    attrs: tuple[Split, ...],
    expected: tuple[int, ...],
) -> None:
    topology_extent = mesh_shape[0] * mesh_shape[1]
    mesh = Mesh(
        topologies=(Topology("cta", topology_extent),),
        layout=Layout(shape=mesh_shape, strides=(mesh_shape[1], 1)),
    )
    layout = ShardLayout(
        layout=Layout(shape=shape, strides=None),
        attrs=attrs,
        mesh=mesh,
    )
    tensor = TensorType(shape=shape, dtype=DType.f32, layout=layout, storage="gmem")

    local = local_type_of(
        tensor,
        level="cta",
        topologies=(Topology("cta", topology_extent),),
    )

    assert local.shape == expected


@pytest.mark.parametrize(("shape", "extent"), [(1024, 132), (64, 128)])
def test_canonical_shard_layout_keeps_rejecting_non_divisible_splits(
    shape: int, extent: int
) -> None:
    with pytest.raises(ValueError, match="logical axis 1 size .* is not divisible"):
        make_shard_tensor_type((1, shape, 128, 2048), mesh=_cta_mesh(extent), attrs=(Split(1),))


@pytest.mark.parametrize(("axis", "expected"), [(1056, 8), (1000, 8), (8, 1)])
def test_local_type_uses_ceildiv_for_a_resolved_launch_extent(
    axis: int, expected: int
) -> None:
    tensor = make_shard_tensor_type(
        (1, axis, 128, 2048), mesh=_cta_mesh(None), attrs=(Split(1),)
    )

    assert local_type_of(
        tensor, level="cta", topologies=(Topology("cta", 132),)
    ).shape == (1, expected, 128, 2048)


def test_local_type_rejects_multiple_launch_provided_split_axes() -> None:
    mesh = Mesh(
        topologies=(Topology("cta", None),),
        layout=Layout(shape=(None, None), strides=None),
    )
    tensor = make_shard_tensor_type(
        (1056, 128), mesh=mesh, attrs=(Split(0), Split(1))
    )

    with pytest.raises(
        ValueError,
        match=r"mesh Mesh\(.*Split axes \(0, 1\).*one parallel width with no per-axis source",
    ):
        local_type_of(
            tensor, level="cta", topologies=(Topology("cta", 132),)
        )


@pytest.mark.parametrize(
    "axis", [DimVar("S_fixed", 1, 8193), ceildiv(DimVar("S_fixed_tiles", 1, 8193), 128)]
)
def test_local_type_rejects_dynamic_split_against_fixed_mesh(axis: object) -> None:
    tensor = make_shard_tensor_type((1, axis, 128, 2048), mesh=_cta_mesh(132), attrs=(Split(1),))

    with pytest.raises(ValueError, match="launch-provided"):
        local_type_of(tensor, level="cta", topologies=(Topology("cta", 132),))


def test_local_type_rejects_unfactorized_inexact_split_with_two_loop_form() -> None:
    layout = ShardLayout(
        layout=Layout(shape=(10,), strides=(1,)), attrs=(Split(0),), mesh=_cta_mesh(3)
    )
    tensor = TensorType(shape=(10,), dtype=DType.f32, layout=layout, storage="gmem")

    with pytest.raises(ValueError, match=r"two loops out as \(ceildiv\(N, T\), T\)"):
        local_type_of(type=tensor, level="cta", topologies=(Topology("cta", 3),))


def test_cost_evaluator_uses_the_resolved_launch_extent() -> None:
    tile_count = 132
    tensor = make_shard_tensor_type(
        (1, tile_count, 128, 2048), mesh=_cta_mesh(None), attrs=(Split(1),)
    )
    source = Var(type=tensor, name="source")
    call = Call(type=tensor, target=Unary(kind=UnaryKind.NEG), args=(source,))

    cost = CostEvaluator(
        CostContext(level="cta", topologies=(Topology("cta", 132),))
    ).visit_Call(call)

    assert cost.flops[DType.f32] == 128 * 2048


def test_local_type_stops_before_a_finer_split() -> None:
    base = Layout(shape=(2, 4, 8), strides=(32, 8, 1))
    thread = ShardLayout(
        layout=base,
        attrs=(Split(1),),
        mesh=Mesh(
            topologies=(Topology("thread", 4),),
            layout=Layout(shape=(4,), strides=(1,)),
        ),
    )
    layout = ShardLayout(
        layout=thread,
        attrs=(Split(0),),
        mesh=Mesh(
            topologies=(Topology("cta", 2),),
            layout=Layout(shape=(2,), strides=(1,)),
        ),
    )
    tensor = TensorType(shape=(64,), dtype=DType.f32, layout=layout, storage="gmem")

    cta = local_type_of(
        tensor,
        level="cta",
        topologies=(Topology("cta", 2), Topology("thread", 4)),
    )
    thread = local_type_of(
        tensor,
        level="thread",
        topologies=(Topology("cta", 2), Topology("thread", 4)),
    )

    assert numel(cta) == 32
    assert numel(thread) == 8


@pytest.mark.parametrize("attr", [Broadcast(), Partial()])
def test_local_type_does_not_divide_replicated_or_partial_values(attr) -> None:
    tensor = make_shard_tensor_type((256,), mesh=_cta_mesh(4), attrs=(attr,))

    local = local_type_of(
        tensor, level="cta", topologies=(Topology("cta", 4),)
    )

    assert numel(local) == 256
