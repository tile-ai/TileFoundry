"""Context transitions around the ordinary expression visitor dispatch."""

from __future__ import annotations

import typing

from tilefoundry.ir import core, expr

from .context import Context

__all__ = ["Visitor", "definitions", "index_contexts"]


class Visitor[T](core.ExprVisitor[T]):
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

    def dispatch_visit(self, node, ctx):
        if id(node) in self.memo:
            return self.memo[id(node)][1]
        if isinstance(node, expr.Call):
            ctx = self.on_call_start(node, ctx)
        if self.contexts is not None:
            self.contexts[id(node)] = ctx
        result = super().dispatch_visit(node, ctx)
        if isinstance(node, expr.Call):
            result = self.on_call_end(node, result, ctx)
            self.memo[id(node)] = (node, result)
        return result

    def visit_operands(self, node, ctx):
        if not isinstance(node, (core.LoopRegion, core.MeshRegion)):
            if isinstance(node, expr.Call) and ctx is not None and ctx.call_site is node:
                ctx = ctx.caller_context or ctx
            return super().visit_operands(node, ctx)
        if isinstance(node, core.LoopRegion):
            for bound in (node.start, node.extent, node.step):
                if isinstance(bound, expr.Expr):
                    self.visit(bound, ctx)
        operands = self.visit_operands_outside(node, ctx)
        inner = self.on_region_start(node, ctx)
        if isinstance(node, core.LoopRegion):
            self.visit(node.induction_var, inner)
        for param in node.params:
            self.visit(param, inner)
        operands = (*operands, self.visit(node.body, inner))
        if isinstance(node, core.LoopRegion):
            operands += tuple(self.visit(value, inner) for value in node.yield_values)
        return self.on_region_end(node, operands, inner)

    def visit_operands_outside(self, region, ctx):
        return tuple(self.visit(arg, ctx) for arg in region.args)

    @typing.override
    def default_visit_leaf(self, node, operands, ctx):
        return None


class Definitions(Visitor):
    """Visit one definition's body, handing each callee's environment to *reach*."""

    def __init__(self, reach, **kwargs):
        super().__init__(**kwargs)
        self.reach = reach

    @typing.override
    def on_call_start(self, call, ctx):
        inner = super().on_call_start(call, ctx)
        callee = inner.callee_context()
        if callee is not None:
            self.reach(callee)
        return inner


def index_contexts(function, ctx):
    """The environment each node of *function* is visited in, by node identity."""
    contexts = {id(function): ctx, **{id(param): ctx for param in function.params}}
    Visitor(contexts=contexts).visit_function_body(function, ctx)
    return contexts


def definitions(context, *, postorder=False):
    """Reach definitions once through the shared call dispatch."""
    context = Context(
        module=context.module,
        function=context.function,
        target=context.target,
        topologies=context.topologies,
    )
    reached, seen = [], set()

    def visit(ctx):
        if id(ctx.function) in seen:
            return
        seen.add(id(ctx.function))
        if not postorder:
            reached.append(ctx)
        Definitions(visit).visit_function_body(ctx.function, ctx)
        if postorder:
            reached.append(ctx)

    visit(context)
    return tuple(reached)
