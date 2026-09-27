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
class Liveness:
    """Definition-ordered intervals on one function-wide event timeline."""

    intervals: tuple[LiveInterval, ...]
    uses: tuple[UseEvent, ...]
    timeline_end: int


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


def _free_vars(function: Function) -> tuple[Var, ...]:
    """Boundary Vars reached as uses but not owned by a structural binding site."""
    if function.body is None:
        return ()
    values = collect_exprs(function.body)
    bound_ids = {id(parameter) for parameter in function.params}
    for value in values:
        if isinstance(value, MeshRegion):
            bound_ids.update(id(parameter) for parameter in value.params)
        elif isinstance(value, LoopRegion):
            bound_ids.add(id(value.induction_var))
            bound_ids.update(id(phi) for phi in value.carried_args)
    return tuple(value for value in values if isinstance(value, Var) and id(value) not in bound_ids)


class LivenessVisitor(ExprVisitor[None]):
    """Build definition/use intervals while preserving structured SSA edges."""

    def __init__(self, function: Function) -> None:
        super().__init__(root_function=function)
        self._point = -1
        self._states: dict[int, LiveInterval] = {}
        self._definition_order: list[int] = []
        self._uses: list[UseEvent] = []
        self._loop_entries: list[tuple[int, set[int], set[int]]] = []
        for parameter in function.params:
            self.define(parameter, self.next_event())
        for free in _free_vars(function):
            self.define(free, self.next_event())

    def next_event(self) -> int:
        """Advance and return the function-wide event position."""
        self._point += 1
        return self._point

    def define(self, value: Expr, point: int) -> None:
        """Record the one definition of *value*."""
        key = id(value)
        if key in self._states:
            raise ValueError(f"liveness: {type(value).__name__} is defined more than once")
        self._states[key] = LiveInterval(value, point, point)
        self._definition_order.append(key)

    def use(self, value: Expr, point: int, *, synthetic: bool = False) -> None:
        """Extend *value* through one consumer event."""
        state = self._states.get(id(value))
        if state is None:
            raise ValueError(f"liveness: {type(value).__name__} is used before its definition")
        for entry, outside, _staged in self._loop_entries:
            if state.defined_at < entry:
                outside.add(id(value))
        self._states[id(value)] = replace(state, last_used_at=max(state.last_used_at, point))
        self._uses.append(UseEvent(value, point, synthetic))

    def finish(self) -> Liveness:
        """Freeze the definition-ordered result."""
        states = (self._states[key] for key in self._definition_order)
        return Liveness(
            intervals=tuple(states),
            uses=tuple(self._uses),
            timeline_end=self._point,
        )

    def visit_Var(self, value: Var, _ctx=None) -> None:
        """A binding site defines a Var; encountering a use does not."""

    def default_visit_leaf(self, value: Expr, _operands: tuple[None, ...], _ctx=None) -> None:
        point = self.next_event()
        for operand in expr_children(value):
            self.use(operand, point)
        self.define(value, point)
        if result_copies(value) > 1 and self._loop_entries:
            self._loop_entries[-1][2].add(id(value))

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

        self._loop_entries.append((phi_definition, set(), set()))
        self.visit(region.body, ctx)
        for yielded in region.yield_values:
            self.visit(yielded, ctx)
        backedge = self.next_event()
        for yielded in region.yield_values:
            self.use(yielded, backedge)
        _, outside, staged = self._loop_entries.pop()
        for key in outside:
            self.use(self._states[key].value, backedge, synthetic=True)
        for key in staged:
            self._states[key] = replace(
                self._states[key], defined_at=phi_definition, last_used_at=backedge
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


__all__ = ["LiveInterval", "Liveness", "UseEvent", "analyze_liveness", "result_copies"]
