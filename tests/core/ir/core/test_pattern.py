"""Pattern — core scalar, tensor, composition, and range contracts."""

import pytest

from tilefoundry import inspection
from tilefoundry.ir import expr, pattern, types


def _tensor(shape: tuple[int, ...], dtype: expr.DType = expr.DType.f32) -> types.TensorType:
    return types.TensorType.from_umat_tensor(shape, dtype)


def test_pattern_match_contract() -> None:
    """Singletons + parametric patterns + And combinator share one contract."""
    assert pattern.is_scalar_tensor().match(types.TensorType.from_umat_scalar())
    assert not pattern.is_scalar_tensor().match(_tensor((3,)))
    assert pattern.is_ranked_tensor().match(_tensor((3, 4)))
    scalar = types.TensorType.from_umat_scalar()
    assert not pattern.is_ranked_tensor().match(scalar)
    assert pattern.Tensor().match(scalar)
    assert pattern.Tensor(shape=()).match(scalar)
    assert not pattern.Tensor(shape=()).match(_tensor((3,)))
    for storage in expr.StorageKind:
        assert pattern.Tensor(storage=storage).match(scalar)
    assert not pattern.is_ranked_tensor().match(type("FakeTy", (), {"shape": (3, 4)})())

    rank2_bf16 = pattern.Tensor(shape=(pattern.Wildcard(),) * 2, dtype=expr.DType.bf16)
    assert rank2_bf16.match(_tensor((3, 4), expr.DType.bf16))
    assert not rank2_bf16.match(_tensor((3,), expr.DType.bf16))
    assert not rank2_bf16.match(_tensor((3, 4), expr.DType.f32))

    combined = pattern.And(
        parts=(
            pattern.Tensor(shape=(pattern.Wildcard(),) * 2),
            pattern.Tensor(dtype=expr.DType.f16),
        )
    )
    assert combined.match(_tensor((3, 4), expr.DType.f16))
    assert not combined.match(_tensor((3,), expr.DType.f16))
    assert pattern.And(parts=()).match(types.TensorType.from_umat_scalar())


def test_range_pattern_contract() -> None:
    """Specialization ranges are closed and reject non-integer values."""
    p = pattern.Range("S", 1, 4)
    assert p.match(1) and p.match(3)
    assert p.match(4)
    assert not p.match(0)
    assert not p.match(2.0)
    assert not p.match(True)

    closed = pattern.Range("S", 3, 4)
    assert closed.match(3)
    assert not closed.match(2) and closed.match(4)

    assert pattern.Range("S", 4, 4).match(4)
    assert pattern.Range(lo=3).match(3) and pattern.Range(lo=3).match(30)
    assert pattern.Range(hi=3).match(-3) and pattern.Range(hi=3).match(3)
    with pytest.raises(ValueError, match="lower or upper"):
        pattern.Range()
    with pytest.raises(ValueError, match="both lo and hi"):
        pattern.Range("S", lo=1)
    with pytest.raises(ValueError, match="lo <= hi"):
        pattern.Range("S", 5, 4)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        (lambda a, b, c: a % (8 * b) == 0, "a % (8 * b) == 0"),
        (lambda a, b, c: a * (b + c) == 0, "a * (b + c) == 0"),
        (lambda a, b, c: a - (b - c) == 0, "a - (b - c) == 0"),
        (lambda a, b, c: a // (b * c) == 0, "a // (b * c) == 0"),
        (lambda a, b, c: (a < b) & (b < c), "(a < b) & (b < c)"),
    ],
)
def test_pattern_rules_preserve_expression_grouping(expression, expected):
    a, b, c = (pattern.Wildcard(name) for name in ("a", "b", "c"))
    assert inspection.PatternPrinter().rules(expression(a, b, c)) == (expected,)
