"""``parser.overload`` — F3 first-match contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from tilefoundry.ir.core.op_schema import OpSchema
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Scalar, Tensor, TensorPat
from tilefoundry.parser.overload import OverloadError, filter_candidates, resolve


@dataclass(frozen=True)
class _FakeType:
    shape: tuple[int, ...]
    dtype: str = "f32"


_S = _FakeType(shape=())
_T1 = _FakeType(shape=(8,))
_T2 = _FakeType(shape=(4, 8))


def _schema(name: str, *patterns: Any, defaults: tuple = ()) -> OpSchema:
    sig: list[ParamDef] = []
    for i, p in enumerate(patterns):
        kw: dict[str, Any] = {"kind": "input", "pattern": p}
        if i < len(defaults) and defaults[i] is not None:
            kw["default"] = defaults[i]
        pd = ParamDef(**kw)
        pd._attr_name = f"x{i}"
        sig.append(pd)

    class _Builder:
        def __init__(self, **kw: Any) -> None: ...

    return OpSchema(
        name=name, dialect="tf", category="test",
        signature=tuple(sig), builder=_Builder, op_class=_Builder,
    )


def test_resolve_picks_first_matching_candidate() -> None:
    """Arity + pattern filter; first-match wins; raises when no match."""
    rank2 = _schema("matmul", TensorPat(rank=2), TensorPat(rank=2))
    any_t = _schema("matmul", Tensor, Tensor)
    # rank-2 inputs → specific schema wins (first in list).
    assert resolve([rank2, any_t], [_T2, _T2]) is rank2
    # rank-1 inputs → fall through to generic Tensor schema.
    assert resolve([rank2, any_t], [_T1, _T1]) is any_t

    only_scalar = _schema("relu", Scalar)
    with pytest.raises(OverloadError, match="No OpSchema candidate"):
        resolve([only_scalar], [_T1])


def test_arity_uses_default_not_optional() -> None:
    """``optional=True`` is nullable, not omittable. Only ``default`` lowers n_min."""
    nullable_required = ParamDef(kind="input", pattern=Tensor, optional=True)
    nullable_required._attr_name = "y"
    s_nullable = OpSchema(
        name="op", dialect="tf", category="test",
        signature=(ParamDef(kind="input", pattern=Tensor), nullable_required),
        builder=type, op_class=type,
    )
    s_nullable.signature[0]._attr_name = "x"
    # nullable but no default → still 2 args required.
    assert filter_candidates([s_nullable], [_T1]) == []
    assert filter_candidates([s_nullable], [_T1, _T2]) == [s_nullable]

    # default=None → omittable.
    s_omittable = _schema("op2", Tensor, Tensor, defaults=(None, "X"))
    assert filter_candidates([s_omittable], [_T1]) == [s_omittable]
