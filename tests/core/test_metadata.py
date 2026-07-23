from __future__ import annotations

from dataclasses import dataclass

import pytest

from tilefoundry.ir.core import (
    BindingMetadata,
    IRMetadata,
    SourceSpanMetadata,
    Var,
    VerifyError,
    get_metadata,
    remove_metadata,
    replace_metadata,
)
from tilefoundry.ir.types import DType, TensorType


@dataclass(frozen=True)
class _Label(IRMetadata):
    value: str


@dataclass(frozen=True)
class _Ordinal(IRMetadata):
    value: int


def _type() -> TensorType:
    return TensorType.scalar(DType.f32)


def test_metadata_defaults_and_base_comment() -> None:
    expr = Var(type=_type(), name="x")

    assert expr.metadata == ()
    assert IRMetadata().format_comment() is None


def test_metadata_does_not_change_expr_semantics_or_repr() -> None:
    plain = Var(type=_type(), name="x")
    annotated = Var(type=_type(), name="x", metadata=(_Label("selected"),))

    assert annotated == plain
    assert hash(annotated) == hash(plain)
    assert "metadata" not in repr(annotated)


def test_source_labels_do_not_change_expr_identity() -> None:
    plain = Var(type=_type(), name="x")
    located = Var(
        type=_type(), name="x",
        metadata=(
            BindingMetadata("x"),
            SourceSpanMetadata("model.py", 7, 3, 7, 9),
        ),
    )

    assert located == plain
    assert hash(located) == hash(plain)
    assert get_metadata(located, BindingMetadata) == BindingMetadata("x")


def test_expr_rejects_duplicate_concrete_metadata_class() -> None:
    with pytest.raises(VerifyError, match=r"duplicate _Label metadata") as exc_info:
        Var(
            type=_type(),
            name="x",
            metadata=(
                BindingMetadata("x"),
                SourceSpanMetadata("model.py", 7, 3),
                _Label("first"),
                _Label("second"),
            ),
        )
    assert "at model.py:7:3" in str(exc_info.value)


def test_expr_rejects_untyped_metadata_entry() -> None:
    with pytest.raises(VerifyError, match="must be IRMetadata, got object"):
        Var(type=_type(), name="x", metadata=(object(),))  # type: ignore[arg-type]


def test_metadata_helpers_preserve_order_and_source_expr() -> None:
    label = _Label("old")
    ordinal = _Ordinal(3)
    expr = Var(type=_type(), name="x", metadata=(label, ordinal))

    assert get_metadata(expr, _Label) is label
    assert get_metadata(expr, IRMetadata) is None

    replacement = _Label("new")
    replaced = replace_metadata(expr, replacement)
    assert replaced.metadata == (replacement, ordinal)
    assert expr.metadata == (label, ordinal)

    removed = remove_metadata(replaced, _Label)
    assert removed.metadata == (ordinal,)
    assert remove_metadata(removed, _Label) is removed
