"""Evaluator value model."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard.shard_layout import ShardLayout


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
    DType.f32: torch.float32,
    DType.f16: torch.float16,
    DType.bf16: torch.bfloat16,
    DType.i32: torch.int32,
    DType.i64: torch.int64,
    DType.bool: torch.bool,
}


def to_torch_dtype(dtype: DType) -> torch.dtype:
    try:
        return _TORCH_DTYPE[dtype]
    except KeyError:
        raise EvalError(f"evaluator: unsupported dtype {dtype}") from None


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
    """The element organisation of ``type.layout`` as a flat int tuple, or
    ``None`` when there is no layout to project onto."""
    layout = getattr(type, "layout", None)
    if layout is None:
        return None
    base = layout.layout if isinstance(layout, ShardLayout) else layout
    shape = getattr(base, "shape", None)
    if shape is None:
        return None
    try:
        return _flatten_ints(shape)
    except (TypeError, ValueError):
        return None


def as_layout_view(value: TensorValue) -> torch.Tensor:
    """Project ``value.data`` from its logical shape to the layout-domain
    element organisation. Returns ``data`` unchanged when there is no layout
    or the element count does not match."""
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
    """Inverse of :func:`as_layout_view`: reshape a layout-domain tensor back
    to the logical shape of ``type``."""
    return data.reshape(tuple(int(d) for d in type.shape))
