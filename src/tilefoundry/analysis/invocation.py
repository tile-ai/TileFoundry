"""Invocation-local records and traversal through retained calls."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from tilefoundry.ir.core import Call, Expr, attach_metadata, get_metadata
from tilefoundry.ir.core.module import ModuleList
from tilefoundry.ir.visitor import collect_exprs, expr_children, function_values
from tilefoundry.visitor_registry.call_frame import CallFrameVisitor


def invocation_key(invocation):
    """The invocation key, with None representing the root invocation."""
    return None if invocation is None else invocation.key


def call_labels(frame):
    """Sibling call labels in operand-first order within one definition."""
    groups = {}
    for call in collect_exprs(frame.function.body):
        if not isinstance(call, Call):
            continue
        child = CallFrameVisitor.callee_frame(call, frame)
        if child is None:
            continue
        target = call.target
        stem = child.function.name
        if isinstance(target, ModuleList):
            stem = target.key
        elif isinstance(target, Call):
            index = target.args[1]
            value = getattr(index, "value", getattr(index, "name", None) or "i")
            stem = f"{target.args[0].key}[{value}]"
        groups.setdefault(stem, []).append(call)
    return {
        id(call): stem if len(calls) == 1 else f"{stem}#{index}"
        for stem, calls in groups.items() for index, call in enumerate(calls)
    }


class InvocationRecords:
    """Root metadata lives on IR; callee metadata lives in an invocation side table."""

    def __init__(self):
        self.table = {}
        self.invocations: list = []

    def get(self, invocation, node, kind):
        if invocation is None or invocation.key is None:
            return get_metadata(node, kind)
        return self.table.get((invocation, id(node), kind))

    def attach(self, invocation, node, record):
        if invocation is None or invocation.key is None:
            attach_metadata(node, record)
        else:
            self.table[invocation, id(node), type(record)] = record

    def reader(self, invocation) -> Callable:
        """Return a metadata lookup bound to one invocation."""
        return lambda node, kind: self.get(invocation, node, kind)

    def definition_reader(self, body):
        invocation = next(invocation for invocation in self.invocations if invocation.function is body)
        return self.reader(invocation)

    def snapshot(self, functions):
        snapshot = dict((key, id(value)) for key, value in self.table.items())
        for body in functions:
            for node in function_values(body):
                for record in node.metadata:
                    snapshot[None, id(node), type(record)] = id(record)
        return snapshot


class InvocationWalker:
    """Step into retained calls while swapping the visitor's DAG memo."""

    def attach(self, node, record, ctx):
        if ctx.records is None:
            attach_metadata(node, record)
        else:
            ctx.records.attach(ctx.current.invocation, node, record)

    def record(self, node, kind, ctx):
        return (
            get_metadata(node, kind)
            if ctx.records is None
            else ctx.records.get(ctx.current.invocation, node, kind)
        )

    def visit_Call(self, expr, ctx):
        invocation = getattr(ctx.current, "invocation", None)
        child = None if invocation is None else invocation.callees.get(id(expr))
        if child is None:
            return self.default_visit_leaf(expr, self.visit_operands(expr, ctx), ctx)
        for arg in expr.args:
            self.visit(arg, ctx)
        if isinstance(child.index, Expr):
            self.visit(child.index, ctx)
        memo, self._memo = self._memo, {}
        try:
            self.visit(child.function.body, replace(ctx, current=child, module=child.frame.module, target=child.frame.target))
        finally:
            self._memo = memo
        self.call_result(expr, ctx)
        return None

    def call_result(self, expr, ctx):
        """A retained call contributes no extra operation-level work."""


def invocation_values(invocation):
    """Walk values in operand-first order, entering callees at their call sites."""
    seen = set()

    def visit(expr):
        if id(expr) in seen:
            return
        seen.add(id(expr))
        child = invocation.callees.get(id(expr))
        if child is not None:
            for arg in expr.args:
                yield from visit(arg)
            if isinstance(child.index, Expr):
                yield from visit(child.index)
            yield from invocation_values(child)
        else:
            for arg in expr_children(expr):
                yield from visit(arg)
        yield invocation, expr

    yield from visit(invocation.function.body)
