"""Pattern — core scalar, tensor, composition, and range contracts."""

import pytest

from tilefoundry.ir.pattern import (
    AndPattern,
    RangePattern,
    Scalar,
    Tensor,
    TensorPattern,
)
from tilefoundry.ir.types import DType, TensorType


def _tensor(shape: tuple[int, ...], dtype: DType = DType.f32) -> TensorType:
    return TensorType.umat_tensor(shape, dtype)


def test_pattern_match_contract() -> None:
    """Singletons + parametric patterns + And combinator share one contract."""
    assert Scalar.match(TensorType.umat_scalar())
    assert not Scalar.match(_tensor((3,)))
    assert Tensor.match(_tensor((3, 4)))
    assert not Tensor.match(TensorType.umat_scalar())
    assert not Tensor.match(type("FakeTy", (), {"shape": (3, 4)})())

    rank2_bf16 = TensorPattern(rank=2, dtype=DType.bf16)
    assert rank2_bf16.match(_tensor((3, 4), DType.bf16))
    assert not rank2_bf16.match(_tensor((3,), DType.bf16))
    assert not rank2_bf16.match(_tensor((3, 4), DType.f32))

    combined = AndPattern(parts=(TensorPattern(rank=2), TensorPattern(dtype=DType.f16)))
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
