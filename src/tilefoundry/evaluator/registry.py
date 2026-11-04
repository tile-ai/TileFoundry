"""Registries for HIR evaluation and instruction-backed schedule values."""

from __future__ import annotations

import functools
from dataclasses import replace
from typing import Callable

from tilefoundry.core import ShardTensor, local_type
from tilefoundry.evaluator.value import EvalError, TensorValue, TupleValue
from tilefoundry.ir.types import TupleType
from tilefoundry.visitor_registry.registries import DispatchRegistry

eval_registry: DispatchRegistry = DispatchRegistry("eval")
schedule_eval_registry: DispatchRegistry = DispatchRegistry("schedule_eval")


def register_eval(op_cls: type) -> Callable[[Callable], Callable]:
    """Register *fn* as the evaluator for ``op_cls``."""
    return eval_registry.decorator()(op_cls)


def register_schedule_eval(op_cls: type) -> Callable[[Callable], Callable]:
    """Register the SSA-value semantics of scheduling *op_cls*."""
    return schedule_eval_registry.decorator()(op_cls)


def per_unit(handler: Callable) -> Callable:
    """Let *handler* take inputs that differ per unit.

    With every input packed the handler runs once. Otherwise it runs once per
    placement on what that unit holds -- a scattered input's piece, a packed
    input's ``local`` -- under the local result type, and the results are held
    scattered. ``arg_types`` keeps the global input types throughout.
    """

    @functools.wraps(handler)
    def run(ctx):
        scattered = [leaf for leaf in _tensors(ctx.args) if not leaf.tensor.is_global]
        if not scattered:
            return handler(ctx)
        placements = scattered[0].tensor.placements
        if any(leaf.tensor.placements != placements for leaf in scattered[1:]):
            raise EvalError("scattered inputs are held by different units")
        result_type = _local(ctx.result_type)
        pieces = [
            handler(
                replace(
                    ctx,
                    args=tuple(_unit_value(arg, placement) for arg in ctx.args),
                    result_type=result_type,
                )
            )
            for placement in placements
        ]
        return _scattered(ctx.result_type, placements, pieces)

    return run


def _tensors(values):
    for value in values:
        if isinstance(value, TensorValue):
            yield value
        elif isinstance(value, TupleValue):
            yield from _tensors(value.elements)


def _local(type):
    if isinstance(type, TupleType):
        return TupleType(tuple(_local(field) for field in type.fields))
    return local_type(type)


def _unit_value(value, placement):
    """What *placement* holds of *value*."""
    if isinstance(value, TupleValue):
        return TupleValue(tuple(_unit_value(element, placement) for element in value.elements))
    if not isinstance(value, TensorValue):
        return value
    return TensorValue(ShardTensor(local_type(value.type), value.tensor.to_local(placement)))


def _scattered(type, placements, pieces):
    """Hold per-placement results as one value, piece by piece."""
    if isinstance(pieces[0], TupleValue):
        return TupleValue(
            tuple(
                _scattered(field, placements, [piece.elements[index] for piece in pieces])
                for index, field in enumerate(type.fields)
            )
        )
    held = tuple((placement, piece.data) for placement, piece in zip(placements, pieces))
    return TensorValue(ShardTensor(type, held))
