"""Visitor that dispatches cost evaluators."""

from __future__ import annotations

from tilefoundry import dispatch
from tilefoundry.interfaces import valuation
from tilefoundry.ir import expr

from .context import Context, Cost
from .registry import registry


class Visitor(valuation.Visitor[Cost]):
    """Dispatch the registered recursive-local Cost Evaluator per Op class.

    A missing evaluator fails closed — it is a construction error, not a
    zero-Cost default.
    """

    def __init__(
        self,
        registry: dispatch.Registry = registry,
    ) -> None:
        super().__init__()
        self.registry = registry

    def visit_Call(self, call: expr.Call, ctx: Context) -> Cost:
        ctx = ctx.enter_call(call)
        fn = self.registry.get(type(call.target), None)
        if fn is None:
            ctx.error(call, f"no cost evaluator registered for {type(call.target).__name__}")
        return fn(call, ctx)



__all__ = ["Visitor"]
