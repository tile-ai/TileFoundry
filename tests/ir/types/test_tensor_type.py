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
from tilefoundry.ir.types.shard import Layout, Mesh, ShardLayout, Split, Topology, make_mesh
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
    assert local_type_of(sharded).shape == (1, 0)


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
        local_type_of(type)


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

    assert local_type_of(tensor).shape == expected


@pytest.mark.parametrize(("shape", "extent"), [(1024, 132), (64, 128)])
def test_canonical_shard_layout_keeps_rejecting_non_divisible_splits(
    shape: int, extent: int
) -> None:
    with pytest.raises(ValueError, match="logical axis 1 size .* is not divisible"):
        make_shard_tensor_type((1, shape, 128, 2048), mesh=_cta_mesh(extent), attrs=(Split(1),))


@pytest.mark.parametrize(
    ("axis", "expected"),
    [
        (DimVar("S_local", 1, 8193), (1, 1, 128, 2048)),
        (ceildiv(DimVar("S_tiles", 1, 8193), 128), (1, 1, 128, 2048)),
        (1024, (1, 1, 128, 2048)),
    ],
)
def test_local_type_collapses_launch_provided_split_before_concrete_check(
    axis: object, expected: tuple[int, ...]
) -> None:
    tensor = make_shard_tensor_type(
        (1, axis, 128, 2048), mesh=_cta_mesh(None), attrs=(Split(1),)
    )

    assert local_type_of(tensor).shape == expected


@pytest.mark.parametrize(
    "axis", [DimVar("S_fixed", 1, 8193), ceildiv(DimVar("S_fixed_tiles", 1, 8193), 128)]
)
def test_local_type_rejects_dynamic_split_against_fixed_mesh(axis: object) -> None:
    tensor = make_shard_tensor_type((1, axis, 128, 2048), mesh=_cta_mesh(132), attrs=(Split(1),))

    with pytest.raises(ValueError, match="launch-provided"):
        local_type_of(tensor)


def test_local_type_rejects_unfactorized_inexact_split_with_two_loop_form() -> None:
    layout = ShardLayout(
        layout=Layout(shape=(10,), strides=(1,)), attrs=(Split(0),), mesh=_cta_mesh(3)
    )
    tensor = TensorType(shape=(10,), dtype=DType.f32, layout=layout, storage="gmem")

    with pytest.raises(ValueError, match=r"two loops out as \(ceildiv\(N, T\), T\)"):
        local_type_of(tensor)


def test_cost_evaluator_uses_local_type_of_for_launch_provided_tile_count() -> None:
    tile_count = ceildiv(DimVar("S_cost", 1, 8193), 128)
    tensor = make_shard_tensor_type(
        (1, tile_count, 128, 2048), mesh=_cta_mesh(None), attrs=(Split(1),)
    )
    source = Var(type=tensor, name="source")
    call = Call(type=tensor, target=Unary(kind=UnaryKind.NEG), args=(source,))

    cost = CostEvaluator(CostContext()).visit_Call(call)

    assert cost.flops[DType.f32] == 128 * 2048
