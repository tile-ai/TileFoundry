"""Evaluator value model."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType

from tilefoundry import core, utils
from tilefoundry.ir import expr, types


class EvalError(Exception):
    """Raised when a program cannot be evaluated (unsupported op / dtype / shape)."""


class Value:
    """Base of every evaluated value."""


@dataclass(frozen=True)
class TensorValue(Value):
    """One ``ShardTensor``; its type is the tensor's."""

    tensor: core.ShardTensor

    @property
    def type(self) -> types.TensorType:
        return self.tensor.type

    @property
    def data(self) -> utils.torch.Tensor:
        """The one global logical tensor; a local view has none."""
        if not self.tensor.is_global:
            raise EvalError("value is a local view; this op reads only the global view")
        return self.tensor.tensor


@dataclass(frozen=True)
class TupleValue(Value):
    """An aggregate of values projected by ``tuple_get_item``."""

    elements: tuple[Value, ...]

    @property
    def type(self) -> types.TupleType:
        """The element types, which a Function boundary checks a tuple argument against."""
        return types.TupleType(tuple(element.type for element in self.elements))


TORCH_DTYPE = MappingProxyType({
    expr.DType.f64: "float64",
    expr.DType.f32: "float32",
    expr.DType.f16: "float16",
    expr.DType.bf16: "bfloat16",
    expr.DType.fp8e4m3: "float8_e4m3fn",
    expr.DType.f8e8m0: "float8_e8m0fnu",
    expr.DType.i8: "int8",
    expr.DType.u8: "uint8",
    expr.DType.i16: "int16",
    expr.DType.u16: "uint16",
    expr.DType.i32: "int32",
    expr.DType.u32: "uint32",
    expr.DType.i64: "int64",
    expr.DType.bool: "bool",
})


def to_torch_dtype(dtype: expr.DType) -> utils.torch.dtype:
    try:
        return getattr(utils.torch, TORCH_DTYPE[dtype])
    except KeyError:
        raise EvalError(f"evaluator: unsupported dtype {dtype}") from None


def from_torch_dtype(dtype: utils.torch.dtype) -> expr.DType:
    """The declared dtype a torch dtype stands for.

    The declared dtype a torch dtype stands for -- ``to_torch_dtype`` inverted,
    off the same table, so a name reported back cannot drift from one accepted.
    """
    for declared, name in TORCH_DTYPE.items():
        if getattr(utils.torch, name) == dtype:
            return declared
    raise EvalError(f"evaluator: no declared dtype for {dtype}")


def tensor_type_of(data: utils.torch.Tensor, *, like: types.TensorType) -> types.TensorType:
    """Describe a concrete tensor using the declaration's non-shape fields."""
    return replace(like, shape=tuple(data.shape), dtype=from_torch_dtype(data.dtype))


__all__ = [
    "EvalError",
    "TensorValue",
    "TupleValue",
    "Value",
    "from_torch_dtype",
    "tensor_type_of",
    "to_torch_dtype",
]
