"""Spec 002 TensorType — equality over a symbolic shape, and DimVar validation."""

from __future__ import annotations

import pytest

from tilefoundry.ir.types import (
    DType,
    TensorType,
    local_type_of,
    make_shard_tensor_type,
    numel,
    tensor_bytes,
)
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.shard import Layout, ShardLayout, Split, make_mesh


def test_tensor_type_equality_over_a_dim_var_shape_entry() -> None:
    """A TensorType carrying a bounded ``DimVar(name, lo, hi)`` stays hashable and
    compares equal to an independently constructed one: the DimVar is interned by
    ``(name, lo, hi)``, so a re-parsed signature is the same key as the original."""
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
