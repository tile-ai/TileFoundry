"""Target-independent definition/use intervals for structured HIR SSA."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import ClassVar

from tilefoundry.ir.core import Call, Constant, Expr, Var
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.tensor.empty import Empty
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.visitor import collect_exprs, expr_children
from tilefoundry.visitor_registry.access_relation import leaf_span, leaves_of
from tilefoundry.visitor_registry.call_frame import CallFrameVisitor, ValuationVisitor
from tilefoundry.visitor_registry.contexts import ValuationContext

from .access import SSAValue
from .errors import AnalysisError
from .metadata import TimingValuation, ValueLifetime


@dataclass(frozen=True)
class UseEvent:
    """One SSA use and whether it exists only to extend structured liveness."""

    value: SSAValue
    at: int
    synthetic: bool = False


@dataclass(frozen=True)
class RegionInterval:
    """One mesh region's entry and exit events on the structured timeline."""

    region: MeshRegion
    entered_at: int
    exited_at: int


@dataclass(frozen=True)
class Liveness:
    """Definition-ordered intervals on one function-wide event timeline."""

    intervals: tuple[ValueLifetime, ...]
    uses: tuple[UseEvent, ...]
    regions: tuple[RegionInterval, ...]
    timeline_end: int
    loop_windows: dict[int, tuple[int, int]] = field(default_factory=dict)
    requires: ClassVar[tuple[type, ...]] = (TimingValuation,)


    def interval_of(self, value: SSAValue) -> ValueLifetime | None:
        """Return *value*'s interval when it belongs to this timeline."""
        return next(
            (item for item in self.intervals if item.value == value),
            None,
        )


def result_copies(expr: Expr) -> int:
    """Physical slots declared for one tensor storage value."""
    return expr.target.buffers if isinstance(expr, Call) and isinstance(expr.target, Empty) else 1


class LivenessVisitor(ValuationVisitor[None]):
    """Build definition/use intervals while preserving structured SSA edges."""

    def __init__(self, function: Function, *, callee=False) -> None:
        super().__init__(root_function=function)
        self.loop_windows = {}
        self.point = -1
        self.states: dict[SSAValue, ValueLifetime] = {}
        self.definition_order: list[SSAValue] = []
        self.uses: list[UseEvent] = []
        self.regions: list[RegionInterval] = []
        self.loop_entries: list[tuple[int, set[SSAValue], set[SSAValue]]] = []
        if callee:
            definition = self.next_event()
        for parameter in function.params:
            self.define(parameter, definition if callee else self.next_event())
        values = collect_exprs(function.body) if function.body is not None else ()
        bound_ids = {id(parameter) for parameter in function.params}
        for value in values:
            if isinstance(value, MeshRegion):
                bound_ids.update(id(parameter) for parameter in value.params)
            elif isinstance(value, LoopRegion):
                bound_ids.add(id(value.induction_var))
                bound_ids.update(id(phi) for phi in value.params)
        free_vars = tuple(
            value for value in values if isinstance(value, Var) and id(value) not in bound_ids
        )
        for free in free_vars:
            self.define(free, self.next_event())

    def next_event(self) -> int:
        """Advance and return the function-wide event position."""
        self.point += 1
        return self.point

    def define(self, value: Expr, point: int) -> None:
        """Define each flattened leaf once, including scalar event identities."""
        for index in range(len(leaves_of(value.type)) or 1):
            key = SSAValue(value, index)
            if key in self.states:
                raise ValueError(f"liveness: {type(value).__name__} is defined more than once")
            self.states[key] = ValueLifetime(key, point, point)
            self.definition_order.append(key)

    def use(self, value: Expr | SSAValue, point: int, *, synthetic: bool = False) -> None:
        """Extend just the named leaf, or all leaves of an ordinary operand."""
        keys = (
            (value,)
            if isinstance(value, SSAValue)
            else tuple(SSAValue(value, index) for index in range(len(leaves_of(value.type)) or 1))
        )
        for key in keys:
            state = self.states.get(key)
            if state is None:
                raise ValueError(f"liveness: {type(key.expr).__name__} is used before its definition")
            for entry, outside in (loop_entry[:2] for loop_entry in self.loop_entries):
                if state.defined_at < entry:
                    outside.add(key)
            self.states[key] = replace(state, last_used_at=max(state.last_used_at, point))
            self.uses.append(UseEvent(key, point, synthetic))

    def finish(self) -> Liveness:
        """Freeze the definition-ordered result."""
        for value in dict.fromkeys(
            SSAValue(key.expr.args[0], index)
            for key in self.definition_order
            if key.index == 0
            and isinstance(key.expr, Call)
            and isinstance(key.expr.target, TupleGetItem)
            and not (isinstance(key.expr.args[1], Constant) and type(key.expr.args[1].value) is int)
            for index in range(len(leaves_of(key.expr.args[0].type)))
        ):
            self.use(value, self.point, synthetic=True)
        states = (self.states[key] for key in self.definition_order)
        return Liveness(
            intervals=tuple(states),
            uses=tuple(self.uses),
            regions=tuple(self.regions),
            timeline_end=self.point,
            loop_windows=self.loop_windows,
        )

    def visit_Var(self, value: Var, ctx=None) -> None:
        """A binding site defines a Var; encountering a use does not."""
        del value, ctx

    def default_visit_leaf(self, value: Expr, operands: tuple[None, ...], ctx=None) -> None:
        del operands, ctx
        point = self.next_event()
        for operand in expr_children(value):
            if (
                isinstance(value, Call)
                and isinstance(value.target, TupleGetItem)
                and operand is value.args[0]
                and isinstance(value.args[1], Constant)
                and type(value.args[1].value) is int
            ):
                begin, count = leaf_span(operand.type, value.args[1].value)
                for index in range(begin, begin + count):
                    self.use(SSAValue(operand, index), point)
            else:
                self.use(operand, point)
        self.define(value, point)
        if result_copies(value) > 1 and self.loop_entries:
            self.loop_entries[-1][2].update(
                SSAValue(value, index) for index in range(len(leaves_of(value.type)) or 1)
            )

    def visit_Call(self, call: Call, ctx=None) -> None:
        context = ctx
        callee = None if context is None else CallFrameVisitor.callee_frame(call, context.frame)
        if callee is None:
            return self.default_visit_leaf(call, self.visit_operands(call, ctx), ctx)
        self.visit_operands(call, ctx)
        argument_use = self.next_event()
        for argument in call.args:
            self.use(argument, argument_use)
        template = context.definitions[id(callee.function)].liveness
        if template is None:
            raise AnalysisError(f"liveness: missing local template for {callee.function.name!r}")
        start = self.next_event()
        context.call_events[id(call)] = start
        binding = context.call_context(call, event_base=start)
        for formal, argument in binding.bindings:
            for use in template.uses:
                if use.value.expr is formal:
                    self.use(
                        SSAValue(argument, use.value.index),
                        binding.event(use.at),
                        synthetic=use.synthetic,
                    )
        self.point = start + template.timeline_end
        self.define(call, self.next_event())

    def visit_MeshRegion(self, region: MeshRegion, ctx=None) -> None:
        """Make the argument/parameter and body/result binding edges explicit."""
        for argument in region.args:
            self.visit(argument, ctx)
        argument_use = self.next_event()
        for argument in region.args:
            self.use(argument, argument_use)
        parameter_definition = self.next_event()
        for parameter in region.params:
            self.define(parameter, parameter_definition)

        inner = self.region_context(region, ctx)
        self.visit(region.body, inner)
        body_use = self.next_event()
        self.use(region.body, body_use)
        self.regions.append(RegionInterval(region, argument_use, body_use))
        self.define(region, self.next_event())

    def visit_LoopRegion(self, region: LoopRegion, ctx=None) -> None:
        """Represent entry, one body iteration, backedge, and exit events."""
        for initial in region.args:
            self.visit(initial, ctx)
        entry_use = self.next_event()
        for initial in region.args:
            self.use(initial, entry_use)
        for parameter in region.params[len(region.yield_values) :]:
            self.define(parameter, entry_use)

        phi_definition = self.next_event()
        self.define(region.induction_var, phi_definition)
        for phi in region.params[: len(region.yield_values)]:
            self.define(phi, phi_definition)

        self.loop_entries.append((phi_definition, set(), set()))
        inner = self.region_context(region, ctx)
        self.visit(region.body, inner)
        for yielded in region.yield_values:
            self.visit(yielded, inner)
        backedge = self.next_event()
        for yielded in region.yield_values:
            self.use(yielded, backedge)
        self.loop_windows[id(region)] = (phi_definition, backedge)
        outside, staged = self.loop_entries.pop()[1:]
        for key in outside:
            self.use(self.states[key].value, backedge, synthetic=True)
        for key in staged:
            self.states[key] = replace(
                self.states[key], defined_at=phi_definition, last_used_at=backedge
            )
        if self.loop_entries:
            self.loop_entries[-1][2].update(staged)

        exit_use = self.next_event()
        carried = region.params[: len(region.yield_values)]
        for source in carried or (region.body,):
            self.use(source, exit_use, synthetic=bool(carried))
        self.define(region, self.next_event())


def analyze_liveness(
    function: Function, context: AnalyzeContext | None = None, *, callee: bool = False
) -> Liveness:
    """Build target-independent intervals for a checked HIR function."""
    if function.body is None:
        raise ValueError(f"liveness: function {function.name!r} has no body")
    visitor = LivenessVisitor(function, callee=callee)
    visitor.visit_function_body(function, context or ValuationContext(owner=function))
    return visitor.finish()


__all__ = [
    "Liveness",
    "RegionInterval",
    "UseEvent",
    "analyze_liveness",
    "result_copies",
]
