"""Target-independent definition/use intervals for structured HIR SSA."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.core import Expr, Var
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.visitor import ExprVisitor, collect_exprs, expr_children


@dataclass(frozen=True)
class LiveInterval:
    """One SSA value's definition and greatest use event."""

    value: Expr
    defined_at: int
    last_used_at: int


@dataclass(frozen=True)
class Liveness:
    """Definition-ordered intervals on one function-wide event timeline."""

    intervals: tuple[LiveInterval, ...]
    timeline_end: int


@dataclass
class _IntervalState:
    value: Expr
    defined_at: int
    last_used_at: int


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
        self._states: dict[int, _IntervalState] = {}
        self._definition_order: list[int] = []
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
        self._states[key] = _IntervalState(value, point, point)
        self._definition_order.append(key)

    def use(self, value: Expr, point: int) -> None:
        """Extend *value* through one consumer event."""
        state = self._states.get(id(value))
        if state is None:
            raise ValueError(f"liveness: {type(value).__name__} is used before its definition")
        state.last_used_at = max(state.last_used_at, point)

    def finish(self) -> Liveness:
        """Freeze the definition-ordered result."""
        states = (self._states[key] for key in self._definition_order)
        return Liveness(
            intervals=tuple(
                LiveInterval(state.value, state.defined_at, state.last_used_at) for state in states
            ),
            timeline_end=self._point,
        )

    def visit_Var(self, value: Var, _ctx=None) -> None:
        """A binding site defines a Var; encountering a use does not."""

    def default_visit_leaf(self, value: Expr, _operands: tuple[None, ...], _ctx=None) -> None:
        point = self.next_event()
        for operand in expr_children(value):
            self.use(operand, point)
        self.define(value, point)

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

        self.visit(region.body, ctx)
        for yielded in region.yield_values:
            self.visit(yielded, ctx)
        backedge = self.next_event()
        for yielded in region.yield_values:
            self.use(yielded, backedge)

        exit_use = self.next_event()
        for source in region.carried_args or (region.body,):
            self.use(source, exit_use)
        self.define(region, self.next_event())


def analyze_liveness(function: Function) -> Liveness:
    """Build target-independent intervals for a checked HIR function."""
    if function.body is None:
        raise ValueError(f"liveness: function {function.name!r} has no body")
    visitor = LivenessVisitor(function)
    visitor.visit_function_body(function)
    return visitor.finish()


__all__ = ["LiveInterval", "Liveness", "analyze_liveness"]
