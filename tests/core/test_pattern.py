"""Pattern — minimal contract."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from tilefoundry.ir.core.pattern import (
    AndPat,
    DimVarRangePat,
    Scalar,
    Tensor,
    TensorPat,
)


@dataclass(frozen=True)
class FakeTy:
    shape: tuple[int, ...]
    dtype: str = "f32"


def test_pattern_match_contract() -> None:
    """Singletons + parametric patterns + And combinator share one contract."""
    assert Scalar.match(FakeTy(shape=()))
    assert not Scalar.match(FakeTy(shape=(3,)))
    assert Tensor.match(FakeTy(shape=(3, 4)))
    assert not Tensor.match(FakeTy(shape=()))

    rank2_bf16 = TensorPat(rank=2, dtype="bf16")
    assert rank2_bf16.match(FakeTy(shape=(3, 4), dtype="bf16"))
    assert not rank2_bf16.match(FakeTy(shape=(3,), dtype="bf16"))
    assert not rank2_bf16.match(FakeTy(shape=(3, 4), dtype="f32"))

    combined = AndPat(parts=(TensorPat(rank=2), TensorPat(dtype="f16")))
    assert combined.match(FakeTy(shape=(3, 4), dtype="f16"))
    assert not combined.match(FakeTy(shape=(3,), dtype="f16"))
    assert AndPat(parts=()).match(FakeTy(shape=()))  # empty AND is a tautology


def test_dim_var_range_pat_contract() -> None:
    """Half-open ``[lo, hi)`` match semantics and the ``lo < hi`` rule.

    A single point is spelled ``[k, k+1)``; ``lo >= hi`` is an empty range
    and rejected at construction. Non-int values (incl. ``bool``, which
    subclasses ``int`` but is not a shape value) never match.
    """
    p = DimVarRangePat("S", 1, 4)
    assert p.match(1) and p.match(3)
    assert not p.match(4)  # hi exclusive
    assert not p.match(0)
    assert not p.match(2.0)
    assert not p.match(True)

    single = DimVarRangePat("S", 3, 4)
    assert single.match(3)
    assert not single.match(2) and not single.match(4)

    with pytest.raises(ValueError, match="lo < hi"):
        DimVarRangePat("S", 4, 4)
