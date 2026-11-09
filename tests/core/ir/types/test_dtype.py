"""DType descriptor declarations and hardware-relevant facts."""

from __future__ import annotations

from tilefoundry.ir import expr

_FACTS = (
    ("f32", expr.FloatDType("f32", 32, 8, 23)),
    ("f16", expr.FloatDType("f16", 16, 5, 10)),
    ("bf16", expr.FloatDType("bf16", 16, 8, 7)),
    ("fp8e4m3", expr.FloatDType("fp8e4m3", 8, 4, 3)),
    ("f8e8m0", expr.FloatDType("f8e8m0", 8, 8, 0)),
    ("f4e2m1", expr.FloatDType("f4e2m1", 4, 2, 1)),
    ("i32", expr.IntegerDType("i32", 32, True)),
    ("i64", expr.IntegerDType("i64", 64, True)),
    ("bool", expr.BoolDType("bool", 1)),
)


def test_dtype_facts_match_declarations() -> None:
    for name, expected in _FACTS:
        member = getattr(expr.DType, name)
        assert member == expected
        assert hash(member) == hash(expected)
        assert repr(member) == repr(expected)


def test_dtype_members_are_singletons() -> None:
    """A dtype is addressed by exactly one name and interned by its facts.

    A dtype is addressed by exactly one name and interned by its facts, so a
    parsed ``"f32"`` and a constructed ``FloatDType`` are one key, and no
    alternate spelling of the low-precision names exists to diverge from.
    """
    assert expr.DType.f32 is expr.DType.f32
    assert expr.DType.f32 is not expr.DType.f16
    assert {expr.DType.f32, expr.FloatDType("f32", 32, 8, 23)} == {expr.DType.f32}
    assert not hasattr(expr.DType.f32, "value")
    assert not hasattr(expr.DType, "__members__")

    assert not hasattr(expr.DType, "f8e4m3")
