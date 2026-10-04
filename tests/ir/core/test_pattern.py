"""Pattern — core scalar, tensor, composition, and range contracts."""

import pytest

from tilefoundry.ir.pattern import (
    AndPattern,
    RangePattern,
    TensorPattern,
    WildcardPattern,
    is_ranked_tensor,
    is_scalar_tensor,
)
from tilefoundry.ir.types import DType, StorageKind, TensorType


def _tensor(shape: tuple[int, ...], dtype: DType = DType.f32) -> TensorType:
    return TensorType.umat_tensor(shape, dtype)


def test_pattern_match_contract() -> None:
    """Singletons + parametric patterns + And combinator share one contract."""
    assert is_scalar_tensor().match(TensorType.umat_scalar())
    assert not is_scalar_tensor().match(_tensor((3,)))
    assert is_ranked_tensor().match(_tensor((3, 4)))
    scalar = TensorType.umat_scalar()
    assert not is_ranked_tensor().match(scalar)
    assert TensorPattern().match(scalar)
    assert TensorPattern(shape=()).match(scalar)
    assert not TensorPattern(shape=()).match(_tensor((3,)))
    for storage in StorageKind:
        assert TensorPattern(storage=storage).match(scalar)
    assert not is_ranked_tensor().match(type("FakeTy", (), {"shape": (3, 4)})())

    rank2_bf16 = TensorPattern(shape=(WildcardPattern(),) * 2, dtype=DType.bf16)
    assert rank2_bf16.match(_tensor((3, 4), DType.bf16))
    assert not rank2_bf16.match(_tensor((3,), DType.bf16))
    assert not rank2_bf16.match(_tensor((3, 4), DType.f32))

    combined = AndPattern(
        parts=(
            TensorPattern(shape=(WildcardPattern(),) * 2),
            TensorPattern(dtype=DType.f16),
        )
    )
    assert combined.match(_tensor((3, 4), DType.f16))
    assert not combined.match(_tensor((3,), DType.f16))
    assert AndPattern(parts=()).match(TensorType.umat_scalar())


def test_range_pattern_contract() -> None:
    """Specialization ranges are closed and reject non-integer values."""
    p = RangePattern("S", 1, 4)
    assert p.match(1) and p.match(3)
    assert p.match(4)
    assert not p.match(0)
    assert not p.match(2.0)
    assert not p.match(True)

    closed = RangePattern("S", 3, 4)
    assert closed.match(3)
    assert not closed.match(2) and closed.match(4)

    assert RangePattern("S", 4, 4).match(4)
    assert RangePattern(lo=3).match(3) and RangePattern(lo=3).match(30)
    assert RangePattern(hi=3).match(-3) and RangePattern(hi=3).match(3)
    with pytest.raises(ValueError, match="lower or upper"):
        RangePattern()
    with pytest.raises(ValueError, match="both lo and hi"):
        RangePattern("S", lo=1)
    with pytest.raises(ValueError, match="lo <= hi"):
        RangePattern("S", 5, 4)
