"""Evaluator value model."""
from __future__ import annotations

from dataclasses import dataclass, replace

from tilefoundry.core import ShardTensor
from tilefoundry.ir.types import DType, TensorType, TupleType
from tilefoundry.ir.types.shard_layout import partial_reductions_by_axis
from tilefoundry.utils.lazy import torch


class EvalError(Exception):
    """Raised when a program cannot be evaluated (unsupported op / dtype / shape)."""


class Value:
    """Base of every evaluated value."""


@dataclass(frozen=True)
class TensorValue(Value):
    """One ``ShardTensor``; its type is the tensor's."""

    tensor: ShardTensor

    @property
    def type(self) -> TensorType:
        return self.tensor.type

    @property
    def data(self) -> torch.Tensor:
        """The one global logical tensor; a value that differs per unit has none."""
        if not self.tensor.packed:
            raise EvalError("value differs per unit, so it has no single tensor")
        return self.tensor.tensor


def refuse_reduced_partial(tensor: ShardTensor) -> None:
    """A packed Partial holds its reduced value, so the part each unit holds is gone."""
    if tensor.packed and any(partial_reductions_by_axis(tensor.type.layout)):
        raise EvalError(
            "a Partial value is held reduced, so the part each unit holds is gone; "
            "reshard it before it meets a value that differs per unit"
        )


@dataclass(frozen=True)
class TupleValue(Value):
    """An aggregate of values projected by ``tuple_get_item``."""

    elements: tuple[Value, ...]

    @property
    def type(self) -> TupleType:
        """The element types, which a Function boundary checks a tuple argument against."""
        return TupleType(tuple(element.type for element in self.elements))


_TORCH_DTYPE = {
    DType.f64: "float64",
    DType.f32: "float32",
    DType.f16: "float16",
    DType.bf16: "bfloat16",
    DType.fp8e4m3: "float8_e4m3fn",
    DType.f8e8m0: "float8_e8m0fnu",
    DType.i8: "int8",
    DType.u8: "uint8",
    DType.i16: "int16",
    DType.u16: "uint16",
    DType.i32: "int32",
    DType.u32: "uint32",
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
