"""DType descriptor declarations and hardware-relevant facts."""
from __future__ import annotations

from tilefoundry.ir.types import BoolDType, DType, FloatDType, IntegerDType

# The declared facts of every builtin dtype. Exponent / mantissa widths are what
# codegen and the numerical oracles read, so a silent edit here must fail a test.
_FACTS = (
    ("f32", FloatDType("f32", 32, 8, 23)),
    ("f16", FloatDType("f16", 16, 5, 10)),
    ("bf16", FloatDType("bf16", 16, 8, 7)),
    ("fp8e4m3", FloatDType("fp8e4m3", 8, 4, 3)),
    ("f8e8m0", FloatDType("f8e8m0", 8, 8, 0)),
    ("f4e2m1", FloatDType("f4e2m1", 4, 2, 1)),
    ("i32", IntegerDType("i32", 32, True)),
    ("i64", IntegerDType("i64", 64, True)),
    ("bool", BoolDType("bool", 1)),
)


def test_builtin_dtype_facts() -> None:
    for name, expected in _FACTS:
        member = getattr(DType, name)
        assert member == expected
        assert hash(member) == hash(expected)
        assert repr(member) == repr(expected)


def test_dtype_members_are_canonically_named_singletons() -> None:
    """A dtype is addressed by exactly one name and interned by its facts, so a
    parsed ``"f32"`` and a constructed ``FloatDType`` are one key, and no
    alternate spelling of the low-precision names exists to diverge from."""
    assert DType.f32 is DType.f32
    assert DType.f32 is not DType.f16
    assert {DType.f32, FloatDType("f32", 32, 8, 23)} == {DType.f32}
    assert not hasattr(DType.f32, "value")
    assert not hasattr(DType, "__members__")
    # fp8e4m3 is the sole canonical fp8 spelling; no alternate is introduced.
    assert not hasattr(DType, "f8e4m3")
