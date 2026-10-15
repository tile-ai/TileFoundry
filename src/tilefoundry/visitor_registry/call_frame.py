"""Call-path context and the shared entry into a called definition."""

from __future__ import annotations

from dataclasses import dataclass, replace

from tilefoundry.ir.core import Call, Constant, Expr
from tilefoundry.ir.core.module import ModuleList
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.visitor import ExprVisitor, collect_exprs


@dataclass(frozen=True)
class CallFrame:
    """Current definition and its effective execution context on a call path."""

    module: Module
    function: Function | None
    target: Target | None = None
    topologies: tuple[Topology, ...] | None = None

    def __post_init__(self):
        if self.target is None:
            object.__setattr__(self, "target", self.module.target)
        if self.topologies is None:
            object.__setattr__(self, "topologies", tuple(self.module.topologies or ()))


class CallFrameVisitor[T](ExprVisitor[T]):
    @staticmethod
    def callee_frame(call, frame) -> CallFrame | None:
        """Enter a definition through the current module's instance slots.

        Function and PrimFunction targets preserve the caller's module/context,
        or return None when there is no caller frame. Such unframed direct calls
        follow the caller's existing path; None therefore does not imply an Op.
        Instance targets require a frame. Op calls always return None.
        """
        target = call.target
        if isinstance(target, (Function, PrimFunction)):
            return None if frame is None else replace(frame, function=target)
        slot = target if isinstance(target, ModuleList) else (
            target.args[0] if isinstance(target, Call) and isinstance(target.target, TupleGetItem)
            and isinstance(target.args[0], ModuleList) else None
        )
        if slot is None:
            return None
        if frame is None:
            raise ValueError("instance call requires a CallFrame")
        slot = frame.module.children[slot.key]
        index = target.args[1] if isinstance(target, Call) else None
        index = index.value if isinstance(index, Constant) else 0
        child = slot.elements[index]
        return CallFrame(
            child, child.entry_function(),
            child.target if child.target is not None else frame.target,
            child.topologies if child.topologies is not None else frame.topologies,
        )

    @classmethod
    def walk_frames(cls, frame, *, postorder=False):
        """Reach HIR definitions once, in DFS first-arrival or dependency order.

        This walks calls from the supplied function, not all declarations in
        its Module. Per-call signature checks still inspect every call site.
        """
        seen = set()

        def visit(current):
            if id(current.function) in seen:
                return
            seen.add(id(current.function))
            if not postorder:
                yield current
            for call in collect_exprs(current.function.body):
                if isinstance(call, Call) and (child := cls.callee_frame(call, current)) is not None:
                    yield from visit(child)
            if postorder:
                yield current

        yield from visit(frame)


class ValuationVisitor[T](CallFrameVisitor[T]):
    """Visit lexical regions with the same context transitions in every family."""

    def __init__(self, *, contexts=None, **kwargs):
        super().__init__(**kwargs)
        self.contexts = contexts

    def dispatch_visit(self, expr, ctx):
        if isinstance(expr, Call):
            ctx = ctx.for_call(expr)
        if self.contexts is not None and id(expr) not in self._memo:
            self.contexts[id(expr)] = ctx
        return super().dispatch_visit(expr, ctx)

    def region_context(self, region, ctx):
        inner = ctx.for_region(region)
        if self.contexts is not None:
            self.contexts[id(region)] = inner
        return inner

    def visit_MeshRegion(self, region, ctx):
        operands = self.visit_operands_outside(region, ctx)
        inner = self.region_context(region, ctx)
        for param in region.params:
            self.visit(param, inner)
        return self.default_visit_leaf(region, (*operands, self.visit(region.body, inner)), inner)

    def visit_LoopRegion(self, region, ctx):
        for bound in (region.start, region.extent, region.step):
            if isinstance(bound, Expr):
                self.visit(bound, ctx)
        operands = self.visit_operands_outside(region, ctx)
        inner = self.region_context(region, ctx)
        for param in (region.induction_var, *region.params):
            self.visit(param, inner)
        body = self.visit(region.body, inner)
        yielded = tuple(self.visit(value, inner) for value in region.yield_values)
        return self.default_visit_leaf(region, (*operands, body, *yielded), inner)

    def visit_operands_outside(self, region, ctx):
        return tuple(self.visit(arg, ctx) for arg in region.args)

    def default_visit_leaf(self, expr, operands, ctx):
        return None

    @staticmethod
    def index_contexts(function, ctx):
        """Index references to one definition's contexts for planning consumers."""
        contexts = {id(function): ctx, **{id(param): ctx for param in function.params}}
        ValuationVisitor(contexts=contexts).visit_function_body(function, ctx)
        return contexts
