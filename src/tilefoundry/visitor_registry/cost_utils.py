"""Shared helpers for operation work and operand traffic."""

from __future__ import annotations

from tilefoundry.ir.core import Call
from tilefoundry.ir.types import BoolDType, DType, IntegerDType, TensorType, Type
from tilefoundry.ir.types.utils import numel, tensor_bytes

from .contexts import Cost, CostContext, TrafficBytes


def operation_cost(dtype: DType, count: int, traffic: tuple[TrafficBytes, ...]) -> Cost:
    """Count floating-point arithmetic or an integer/predicate service."""
    if isinstance(dtype, IntegerDType):
        return Cost({}, traffic, {"integer": count})
    if isinstance(dtype, BoolDType):
        return Cost({}, traffic, {"predicate": count})
    return Cost({dtype: count}, traffic)


def input_types(call: Call, ctx: CostContext) -> tuple[Type, ...]:
    return tuple(ctx.local_type_of(arg) for arg in call.args)


def output_type(call: Call, ctx: CostContext) -> Type:
    return ctx.local_output_type(call)


def traffic(inputs: tuple[Type, ...], output: Type) -> tuple[TrafficBytes, ...]:
    """One entry per operand: every input read whole, the result written whole.

    The default an Op gets by saying nothing about which part it touches.
    """
    return (
        *(TrafficBytes(read=tensor_bytes(type)) for type in inputs),
        TrafficBytes(write=tensor_bytes(output)),
    )


def row(table: Type) -> int:
    """One position's worth of a cache whose leading axis is the position."""
    return tensor_bytes(table) // table.shape[0]


def idle(call: Call) -> tuple[TrafficBytes, ...]:
    """No operand moves, but every operand still has a slot."""
    return tuple(TrafficBytes() for _ in range(len(call.args) + 1))


def elementwise(call: Call, ctx: CostContext, *, dtype: DType | None = None) -> Cost:
    inputs = input_types(call, ctx)
    output = output_type(call, ctx)
    result_dtype = dtype
    if result_dtype is None:
        if isinstance(output, TensorType):
            result_dtype = output.dtype
        else:
            result_dtype = next(
                type.dtype for type in inputs if isinstance(type, TensorType)
            )
    return operation_cost(result_dtype, numel(output), traffic(inputs, output))


def serviced(call: Call, ctx: CostContext, kind: str) -> Cost:
    """One result of *kind* per element, and no floating-point work at all."""
    inputs = input_types(call, ctx)
    output = output_type(call, ctx)
    return Cost({}, traffic(inputs, output), {kind: numel(output)})


__all__ = [
    "operation_cost",
    "input_types",
    "output_type",
    "traffic",
    "row",
    "idle",
    "elementwise",
    "serviced",
]
