"""Spec 002 TensorType smoke coverage."""

from __future__ import annotations

import pytest

from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.dim import DimVar


def test_scalar_builds_rank0_tensor_type():
    t = TensorType.scalar(DType.f32)
    assert t.shape == ()
    assert t.dtype == DType.f32


def test_tensor_type_accepts_dim_var_shape_entry():
    """A TensorType can carry a bounded ``DimVar(name, lo, hi)`` in its shape."""
    s = DimVar("S_a", 1, 8)
    t = TensorType(shape=(s, 8), dtype=DType.f32, layout=None, storage="gmem")
    assert t.shape == (s, 8)
    # Same (name, lo, hi) is cached: equality + hashability via the TensorType.
    t2 = TensorType(shape=(DimVar("S_a", 1, 8), 8), dtype=DType.f32, layout=None, storage="gmem")
    assert t == t2
    assert hash(t) == hash(t2)


def test_dim_var_same_name_distinct_bounds_constructs_distinct_objects():
    """Same name with different (lo, hi) produces distinct canonical objects.

    Cross-instance scoping lives in HIR ``verify_function`` (within a
    single function signature); construction itself never raises on
    same-name distinct bounds.
    """
    a = DimVar("S_conflict", 1, 4)
    b = DimVar("S_conflict", 1, 8)
    assert a is not b
    assert (a.lo, a.hi) == (1, 4)
    assert (b.lo, b.hi) == (1, 8)


def test_dim_var_rejects_non_positive_range():
    """``lo < hi`` is required (half-open [lo, hi)); a single point is [k, k+1)."""
    DimVar("S_point", 4, 5)  # single value 4 as [4, 5) — no raise
    with pytest.raises(ValueError, match="require lo < hi"):
        DimVar("S_empty", 4, 4)  # empty half-open range
    with pytest.raises(ValueError, match="require lo < hi"):
        DimVar("S_inv", 5, 1)
