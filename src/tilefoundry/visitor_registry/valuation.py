"""Context transitions around the ordinary expression visitor dispatch."""

from __future__ import annotations

from tilefoundry.ir.core import Call, Expr
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.visitor import ExprVisitor

from .contexts import ValuationContext

__all__ = ["ValuationContext", "ValuationVisitor"]


class ValuationVisitor[T](ExprVisitor[T]):
    """Visit children in their lexical environment, with overridable transitions."""

    def __init__(self, *, contexts=None, **kwargs):
        super().__init__(**kwargs)
        self.contexts = contexts

    def on_call_start(self, call, ctx):
        return ctx.enter_call(call) if ctx is not None else None

    def on_call_end(self, call, result, ctx):
        return result

    def on_region_start(self, region, ctx):
        inner = ctx.enter_region(region) if ctx is not None else None
        if self.contexts is not None:
            self.contexts[id(region)] = inner
        return inner

    def on_region_end(self, region, operands, ctx):
        return operands

    def dispatch_visit(self, expr, ctx):
        if id(expr) in self._memo:
            return self._memo[id(expr)][1]
        if isinstance(expr, Call):
            ctx = self.on_call_start(expr, ctx)
        if self.contexts is not None:
            self.contexts[id(expr)] = ctx
        result = super().dispatch_visit(expr, ctx)
        if isinstance(expr, Call):
            result = self.on_call_end(expr, result, ctx)
            self._memo[id(expr)] = (expr, result)
        return result

    def visit_operands(self, expr, ctx):
        if not isinstance(expr, (LoopRegion, MeshRegion)):
            return super().visit_operands(expr, ctx)
        if isinstance(expr, LoopRegion):
            for bound in (expr.start, expr.extent, expr.step):
                if isinstance(bound, Expr):
                    self.visit(bound, ctx)
        operands = self.visit_operands_outside(expr, ctx)
        inner = self.on_region_start(expr, ctx)
        if isinstance(expr, LoopRegion):
            self.visit(expr.induction_var, inner)
        for param in expr.params:
            self.visit(param, inner)
        operands = (*operands, self.visit(expr.body, inner))
        if isinstance(expr, LoopRegion):
            operands += tuple(self.visit(value, inner) for value in expr.yield_values)
        return self.on_region_end(expr, operands, inner)

    def visit_operands_outside(self, region, ctx):
        return tuple(self.visit(arg, ctx) for arg in region.args)

    def default_visit_leaf(self, expr, operands, ctx):
        return None

    @staticmethod
    def index_contexts(function, ctx):
        contexts = {id(function): ctx, **{id(param): ctx for param in function.params}}
        ValuationVisitor(contexts=contexts).visit_function_body(function, ctx)
        return contexts
