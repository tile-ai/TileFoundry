"""Expression basics: what metadata may be, and what it may never change.

The positive metadata path (a binding name, a source span) is carried by every
printed model, so only the invariants a model happy path cannot localise stay
here: identity is blind to metadata, malformed metadata is rejected at
construction, and a no-attribute Op is a cached singleton.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from tilefoundry.ir import expr, types


@dataclass(frozen=True)
class _Label(expr.IRMetadata):
    value: str


def _type() -> types.TensorType:
    return types.TensorType.from_scalar(expr.DType.f32)


def test_source_labels_do_not_change_expr_identity() -> None:
    plain = expr.Var(type=_type(), name="x")
    located = expr.Var(
        type=_type(),
        name="x",
        metadata=(
            expr.BindingMetadata("x"),
            expr.SourceSpanMetadata("model.py", 7, 3, 7, 9),
        ),
    )

    assert located == plain
    assert hash(located) == hash(plain)
    assert expr.get_metadata(located, expr.BindingMetadata) == expr.BindingMetadata("x")

    assert expr.get_metadata(located, expr.IRMetadata) is None


def test_expr_rejects_malformed_metadata() -> None:
    """Two entries of one concrete class would make ``get_metadata`` ambiguous.

    Two entries of one concrete class would make ``get_metadata`` ambiguous,
    and a non-``IRMetadata`` entry has no class to look up by. Both are
    construction-time errors, and the first reports the span it was given.
    """
    with pytest.raises(expr.VerifyError, match=r"duplicate _Label metadata") as exc_info:
        expr.Var(
            type=_type(),
            name="x",
            metadata=(
                expr.SourceSpanMetadata("model.py", 7, 3),
                _Label("first"),
                _Label("second"),
            ),
        )
    assert "at model.py:7:3" in str(exc_info.value)

    with pytest.raises(expr.VerifyError, match="must be IRMetadata, got object"):
        expr.Var(type=_type(), name="x", metadata=(object(),))


def test_op_attribute_singleton_cache() -> None:
    """No-attribute Ops are cached — ``Foo() is Foo()`` (spec 001)."""

    class _OpB(expr.Op):
        pass

    assert _OpB() is _OpB()
