"""Spec 002 TensorType — equality over a symbolic shape, and DimVar validation."""

from __future__ import annotations

import pytest

from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.dim import DimVar


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
