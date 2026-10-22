"""Evaluator value model."""
from __future__ import annotations

from dataclasses import dataclass, replace

from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard_layout import ShardLayout
from tilefoundry.utils.lazy import torch


class EvalError(Exception):
    """Raised when a program cannot be evaluated (unsupported op / dtype / shape)."""


class Value:
    """Base of every evaluated value."""


@dataclass(frozen=True)
class TensorValue(Value):
    """A logical tensor value paired with its HIR type."""

    data: torch.Tensor
    type: TensorType


@dataclass(frozen=True)
class TupleValue(Value):
    """An aggregate of values projected by ``tuple_get_item``."""

    elements: tuple[Value, ...]


_TORCH_DTYPE = {
    DType.f32: "float32",
    DType.f16: "float16",
    DType.bf16: "bfloat16",
    DType.fp8e4m3: "float8_e4m3fn",
    DType.f8e8m0: "float8_e8m0fnu",
    DType.i32: "int32",
    DType.i64: "int64",
    DType.bool: "bool",
}


def to_torch_dtype(dtype: DType) -> torch.dtype:
    try:
        return getattr(torch, _TORCH_DTYPE[dtype])
    except KeyError:
        raise EvalError(f"evaluator: unsupported dtype {dtype}") from None


def from_torch_dtype(dtype: torch.dtype) -> DType:
    """The declared dtype a torch dtype stands for.

    The declared dtype a torch dtype stands for -- ``to_torch_dtype`` inverted,
    off the same table, so a name reported back cannot drift from one accepted.
    """
    for declared, name in _TORCH_DTYPE.items():
        if getattr(torch, name) == dtype:
            return declared
    raise EvalError(f"evaluator: no declared dtype for {dtype}")


def tensor_type_of(data: torch.Tensor, *, like: TensorType) -> TensorType:
    """Describe a concrete tensor using the declaration's non-shape fields."""
    return replace(like, shape=tuple(data.shape), dtype=from_torch_dtype(data.dtype))


def _flatten_ints(shape) -> tuple[int, ...]:
    """Flatten a (possibly nested) IntTuple of static dims into plain ints."""
    out: list[int] = []
    for d in shape:
        if isinstance(d, (tuple, list)):
            out.extend(_flatten_ints(d))
        else:
            out.append(int(d))
    return tuple(out)


def _layout_shape(type) -> tuple[int, ...] | None:
    """Layout shape.

    The element organisation of ``type.layout`` as a flat int tuple, or
    ``None`` when its shape is not statically projectable.
    """
    layout = type.layout
    base = layout.layout if isinstance(layout, ShardLayout) else layout
    shape = base.shape
    try:
        return _flatten_ints(shape)
    except (TypeError, ValueError):
        return None


def as_layout_view(value: TensorValue) -> torch.Tensor:
    """Project ``value.data`` from its logical shape to the layout-domain element organisation.

    Project ``value.data`` from its logical shape to the layout-domain
    element organisation. Returns ``data`` unchanged when there is no layout
    or the element count does not match.
    """
    shape = _layout_shape(value.type)
    if shape is None:
        return value.data
    numel = 1
    for d in shape:
        numel *= d
    if value.data.numel() != numel:
        return value.data
    return value.data.reshape(shape)


def from_layout_view(data: torch.Tensor, type: TensorType) -> torch.Tensor:
    """Inverse of :func:`as_layout_view`.

    Inverse of :func:`as_layout_view`: reshape a layout-domain tensor back
    to the logical shape of ``type``.
    """
    return data.reshape(tuple(int(d) for d in type.shape))
