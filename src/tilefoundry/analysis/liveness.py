"""Target-independent definition/use intervals for structured HIR SSA."""

from __future__ import annotations

from dataclasses import dataclass, replace

from tilefoundry.ir.core import Call, Expr, Var
from tilefoundry.ir.core.param_def import MemoryEffect
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.schedule import ScheduleOp
from tilefoundry.ir.visitor import ExprVisitor, collect_exprs, expr_children


@dataclass(frozen=True)
class LiveInterval:
    """One SSA value's definition and greatest use event."""

    value: Expr
    defined_at: int
    last_used_at: int


@dataclass(frozen=True)
class UseEvent:
    """One SSA use and whether it exists only to extend structured liveness."""

    value: Expr
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

    intervals: tuple[LiveInterval, ...]
    uses: tuple[UseEvent, ...]
    regions: tuple[RegionInterval, ...]
    timeline_end: int

    def interval_of(self, value: Expr) -> LiveInterval | None:
        """Return *value*'s interval when it belongs to this timeline."""
        return next((item for item in self.intervals if item.value is value), None)


def result_copies(expr: Expr) -> int:
    """Physical result slots represented by one scheduled SSA value."""
    if not isinstance(expr, Call) or not isinstance(expr.target, ScheduleOp):
        return 1
    op = expr.target.op
    schema = getattr(type(op), "_op_schema", None)
    if schema is None:
        return 1
    result = next(
        (
            param
            for param in schema.signature
            if param.kind == "input"
            and param.effect is not None
            and param.effect & MemoryEffect.WRITE
        ),
        None,
    )
    if result is None or result.effect & MemoryEffect.READ:
        return 1
    return expr.target.buffers


class LivenessVisitor(ExprVisitor[None]):
    """Build definition/use intervals while preserving structured SSA edges."""

    def __init__(self, function: Function) -> None:
        super().__init__(root_function=function)
        self.point = -1
        self.states: dict[int, LiveInterval] = {}
        self.definition_order: list[int] = []
        self.uses: list[UseEvent] = []
        self.regions: list[RegionInterval] = []
        self.loop_entries: list[tuple[int, set[int], set[int]]] = []
        for parameter in function.params:
            self.define(parameter, self.next_event())
        values = collect_exprs(function.body) if function.body is not None else ()
        bound_ids = {id(parameter) for parameter in function.params}
        for value in values:
            if isinstance(value, MeshRegion):
                bound_ids.update(id(parameter) for parameter in value.params)
            elif isinstance(value, LoopRegion):
                bound_ids.add(id(value.induction_var))
                bound_ids.update(id(phi) for phi in value.carried_args)
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
        """Record the one definition of *value*."""
        key = id(value)
        if key in self.states:
            raise ValueError(f"liveness: {type(value).__name__} is defined more than once")
        self.states[key] = LiveInterval(value, point, point)
        self.definition_order.append(key)

    def use(self, value: Expr, point: int, *, synthetic: bool = False) -> None:
        """Extend *value* through one consumer event."""
        state = self.states.get(id(value))
        if state is None:
            raise ValueError(f"liveness: {type(value).__name__} is used before its definition")
        for entry, outside in (loop_entry[:2] for loop_entry in self.loop_entries):
            if state.defined_at < entry:
                outside.add(id(value))
        self.states[id(value)] = replace(state, last_used_at=max(state.last_used_at, point))
        self.uses.append(UseEvent(value, point, synthetic))

    def finish(self) -> Liveness:
        """Freeze the definition-ordered result."""
        states = (self.states[key] for key in self.definition_order)
        return Liveness(
            intervals=tuple(states),
            uses=tuple(self.uses),
            regions=tuple(self.regions),
            timeline_end=self.point,
        )

    def visit_Var(self, value: Var, ctx=None) -> None:
        """A binding site defines a Var; encountering a use does not."""
        del value, ctx

    def default_visit_leaf(self, value: Expr, operands: tuple[None, ...], ctx=None) -> None:
        del operands, ctx
        point = self.next_event()
        for operand in expr_children(value):
            self.use(operand, point)
        self.define(value, point)
        if result_copies(value) > 1 and self.loop_entries:
            self.loop_entries[-1][2].add(id(value))

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

        self.visit(region.body, ctx)
        body_use = self.next_event()
        self.use(region.body, body_use)
        self.regions.append(RegionInterval(region, argument_use, body_use))
        self.define(region, self.next_event())

    def visit_LoopRegion(self, region: LoopRegion, ctx=None) -> None:
        """Represent entry, one body iteration, backedge, and exit events."""
        for initial in region.init_args:
            self.visit(initial, ctx)
        entry_use = self.next_event()
        for initial in region.init_args:
            self.use(initial, entry_use)

        phi_definition = self.next_event()
        self.define(region.induction_var, phi_definition)
        for phi in region.carried_args:
            self.define(phi, phi_definition)

        self.loop_entries.append((phi_definition, set(), set()))
        self.visit(region.body, ctx)
        for yielded in region.yield_values:
            self.visit(yielded, ctx)
        backedge = self.next_event()
        for yielded in region.yield_values:
            self.use(yielded, backedge)
        outside, staged = self.loop_entries.pop()[1:]
        for key in outside:
            self.use(self.states[key].value, backedge, synthetic=True)
        for key in staged:
            self.states[key] = replace(
                self.states[key], defined_at=phi_definition, last_used_at=backedge
            )

        exit_use = self.next_event()
        for source in region.carried_args or (region.body,):
            self.use(source, exit_use, synthetic=bool(region.carried_args))
        self.define(region, self.next_event())


def analyze_liveness(function: Function) -> Liveness:
    """Build target-independent intervals for a checked HIR function."""
    if function.body is None:
        raise ValueError(f"liveness: function {function.name!r} has no body")
    visitor = LivenessVisitor(function)
    visitor.visit_function_body(function)
    return visitor.finish()


__all__ = [
    "LiveInterval",
    "Liveness",
    "RegionInterval",
    "UseEvent",
    "analyze_liveness",
    "result_copies",
]
