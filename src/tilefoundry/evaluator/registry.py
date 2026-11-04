"""Registries for HIR evaluation and instruction-backed schedule values."""

from __future__ import annotations

from typing import Callable

from tilefoundry.visitor_registry.registries import DispatchRegistry

eval_registry: DispatchRegistry = DispatchRegistry("eval")
schedule_eval_registry: DispatchRegistry = DispatchRegistry("schedule_eval")


def register_eval(op_cls: type) -> Callable[[Callable], Callable]:
    """Register *fn* as the evaluator for ``op_cls``."""
    return eval_registry.decorator()(op_cls)


def register_schedule_eval(op_cls: type) -> Callable[[Callable], Callable]:
    """Register the SSA-value semantics of scheduling *op_cls*."""
    return schedule_eval_registry.decorator()(op_cls)
