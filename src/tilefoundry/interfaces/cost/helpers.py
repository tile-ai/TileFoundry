"""Shared helpers for operation work and operand traffic."""

from __future__ import annotations

from tilefoundry.ir import expr, types, utils

from .context import Context, Cost, TrafficBytes


def operation(dtype: expr.DType, count: int, traffic: tuple[TrafficBytes, ...]) -> Cost:
    """Count floating-point arithmetic or an integer/predicate service."""
    if isinstance(dtype, expr.IntegerDType):
        return Cost({}, traffic, {"integer": count})
    if isinstance(dtype, expr.BoolDType):
        return Cost({}, traffic, {"predicate": count})
    return Cost({dtype: count}, traffic)


def input_types(call: expr.Call, ctx: Context) -> tuple[types.Type, ...]:
    return tuple(ctx.local_type_of(arg) for arg in call.args)


def output_type(call: expr.Call, ctx: Context) -> types.Type:
    return ctx.local_output_type(call)


def traffic(inputs: tuple[types.Type, ...], output: types.Type) -> tuple[TrafficBytes, ...]:
    """One entry per operand: every input read whole, the result written whole.

    The default an Op gets by saying nothing about which part it touches.
    """
    return (
        *(TrafficBytes(read=utils.tensor_bytes(type)) for type in inputs),
        TrafficBytes(write=utils.tensor_bytes(output)),
    )


def row(table: types.Type) -> int:
    """One position's worth of a cache whose leading axis is the position."""
    return utils.tensor_bytes(table) // table.shape[0]


def idle(call: expr.Call) -> tuple[TrafficBytes, ...]:
    """No operand moves, but every operand still has a slot."""
    return tuple(TrafficBytes() for _ in range(len(call.args) + 1))


def elementwise(call: expr.Call, ctx: Context, *, dtype: expr.DType | None = None) -> Cost:
    inputs = input_types(call, ctx)
    output = output_type(call, ctx)
    result_dtype = dtype
    if result_dtype is None:
        if isinstance(output, types.TensorType):
            result_dtype = output.dtype
        else:
            result_dtype = next(
                type.dtype for type in inputs if isinstance(type, types.TensorType)
            )
    return operation(result_dtype, utils.numel(output), traffic(inputs, output))


def serviced(call: expr.Call, ctx: Context, kind: str) -> Cost:
    """One result of *kind* per element, and no floating-point work at all."""
    inputs = input_types(call, ctx)
    output = output_type(call, ctx)
    return Cost({}, traffic(inputs, output), {kind: utils.numel(output)})


def metadata_view(call: expr.Call, ctx: Context) -> Cost:
    """A view of metadata: no operand moves."""
    return Cost({}, idle(call))


def structure(call: expr.Call, ctx: Context) -> Cost:
    """Pure data placement: every input read whole, the result written whole."""
    return Cost({}, traffic(input_types(call, ctx), output_type(call, ctx)))


__all__ = [
    "operation",
    "input_types",
    "output_type",
    "traffic",
    "row",
    "idle",
    "elementwise",
    "serviced",
    "metadata_view",
    "structure",
]
