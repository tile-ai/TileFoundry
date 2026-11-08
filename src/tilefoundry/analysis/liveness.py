"""Target-independent definition/use intervals for structured HIR SSA."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from heapq import heapify, heappop, heappush
from typing import ClassVar

import isl

from tilefoundry.ir.core import Call, Constant, Expr, IRMetadata, Var, describe_expr, get_metadata
from tilefoundry.ir.core.expr import SSAValue
from tilefoundry.ir.core.param_def import MemoryEffect
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.schedule import ScheduleOp, instruction_schema
from tilefoundry.ir.hir.tensor.empty import Empty
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.isl_interop import shape_to_isl_set
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.utils import is_literal_shape, leaf_span, leaves_of, static_dim_value
from tilefoundry.ir.visitor import collect_exprs, expr_children
from tilefoundry.visitor_registry.buffer_alias import aliased_operand
from tilefoundry.visitor_registry.contexts import TypeInferContext
from tilefoundry.visitor_registry.valuation import ValuationVisitor

from .errors import AnalysisError
from .loop_domain import span_bounds
from .metadata import ValueLifetime, shifted
from .visitor import AnalyzeContext


@dataclass(frozen=True)
class UseEvent:
    events: ClassVar = ("at",)

    """One SSA use and whether it exists only to extend structured liveness."""

    value: SSAValue
    at: int
    synthetic: bool = False


@dataclass(frozen=True)
class RegionInterval:
    events: ClassVar = ("entered_at", "exited_at")

    """One mesh region's entry and exit events on the structured timeline."""

    region: MeshRegion
    entered_at: int
    exited_at: int


@dataclass(frozen=True)
class LivenessMetadata(IRMetadata):
    """Definition-ordered intervals on one function-wide event timeline.

    ``orders`` maps a scope's owner, by id, to the calls and regions of its body
    in the order they take events, for each scope a destination overwrite
    reorders; lowering emits them in that order.
    """

    intervals: tuple[ValueLifetime, ...]
    uses: tuple[UseEvent, ...]
    regions: tuple[RegionInterval, ...]
    timeline_end: int
    loop_windows: dict[int, tuple[int, int]] = field(default_factory=dict)
    orders: dict[int, tuple[Expr, ...]] = field(default_factory=dict)

    def at(self, origin):
        return replace(
            self,
            intervals=tuple(shifted(item, origin) for item in self.intervals),
            uses=tuple(shifted(item, origin) for item in self.uses),
            regions=tuple(shifted(item, origin) for item in self.regions),
            timeline_end=self.timeline_end + origin,
            loop_windows={
                key: tuple(point + origin for point in window)
                for key, window in self.loop_windows.items()
            },
        )

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

    def __init__(self, function: Function, *, is_callee=False, orders=None) -> None:
        super().__init__(root_function=function)
        self.orders = orders or {}
        self.loop_windows = {}
        self.point = -1
        self.states: dict[SSAValue, ValueLifetime] = {}
        self.definition_order: list[SSAValue] = []
        self.uses: list[UseEvent] = []
        self.regions: list[RegionInterval] = []
        self.loop_entries: list[tuple[int, set[SSAValue], set[SSAValue]]] = []
        if is_callee:
            definition = self.next_event()
        for parameter in function.params:
            self.define(parameter, definition if is_callee else self.next_event())
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
                raise ValueError(
                    f"liveness: {type(key.expr).__name__} is used before its definition"
                )
            for entry, outside in (loop_entry[:2] for loop_entry in self.loop_entries):
                if state.defined_at < entry:
                    outside.add(key)
            self.states[key] = replace(state, last_used_at=max(state.last_used_at, point))
            self.uses.append(UseEvent(key, point, synthetic))

    def visit_ordered(self, owner, ctx) -> None:
        """Visit the children of *owner*'s body first in the order ``orders`` gives them."""
        for expr in self.orders.get(id(owner), ()):
            self.visit(expr, ctx)

    def on_region_start(self, region, ctx):
        inner = super().on_region_start(region, ctx)
        self.visit_ordered(region, inner)
        return inner

    def finish(self) -> LivenessMetadata:
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
        return LivenessMetadata(
            intervals=tuple(states),
            uses=tuple(self.uses),
            regions=tuple(self.regions),
            timeline_end=self.point,
            loop_windows=self.loop_windows,
            orders=self.orders,
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
        callee = None if context is None else context.callee
        if callee is None:
            return self.default_visit_leaf(call, self.visit_operands(call, ctx), ctx)
        self.visit_operands(call, ctx)
        argument_use = self.next_event()
        for argument in call.args:
            self.use(argument, argument_use)
        template = get_metadata(callee, LivenessMetadata)
        if template is None:
            raise AnalysisError(
                f"liveness: missing local template for {callee.name!r}"
            )
        start = self.next_event()
        template = template.at(start)
        self.regions.extend(template.regions)
        arguments = context.callee_scope.provenance.arguments
        for formal in callee.params:
            if id(formal) not in arguments:
                continue
            argument = arguments[id(formal)]
            for use in template.uses:
                if use.value.expr is formal:
                    self.use(
                        SSAValue(argument, use.value.index),
                        use.at,
                        synthetic=use.synthetic,
                    )
        self.point = template.timeline_end
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

        inner = self.on_region_start(region, ctx)
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
        inner = self.on_region_start(region, ctx)
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
    function: Function, context: AnalyzeContext | None = None, *, is_callee: bool = False
) -> LivenessMetadata:
    """Build target-independent intervals for a checked HIR function.

    Given an iteration scope, events put the uses of data a schedule overwrites
    in place before the overwrite where the analysis can order them, and
    ``AnalysisError`` is raised where it cannot.
    """
    if function.body is None:
        raise ValueError(f"liveness: function {function.name!r} has no body")
    visitor = LivenessVisitor(function, is_callee=is_callee)
    if context is None:
        context = TypeInferContext(owner=function)
    visitor.visit_function_body(function, context)
    first = visitor.finish()
    scope = getattr(context, "scope", None)
    if scope is None:
        return first
    ranks = {}
    for item in first.intervals:
        ranks.setdefault(id(item.value.expr), len(ranks))
    orders, unordered = _scope_orders(function, scope, ranks)
    if unordered:
        raise AnalysisError(
            "liveness: cannot order every use of the data these schedules overwrite "
            "before them: " + "; ".join(describe_expr(write) for write in unordered)
        )
    if not orders:
        return first
    visitor = LivenessVisitor(function, is_callee=is_callee, orders=orders)
    visitor.visit_ordered(function, context)
    visitor.visit_function_body(function, context)
    return visitor.finish()


def _scope_orders(function: Function, scope, ranks):
    """The scopes a destination overwrite reorders, each with its children's order.

    A read of data a schedule replaces comes first, or the loop whose carry took
    that data out of the written storage does. In the innermost scope holding
    both, the child holding the first precedes the child holding the write; a
    scope sorts these with its data edges, ties depth-first, and keeps the order
    of its calls and regions, call targets aside, where it differs. The writes of
    a cyclic scope, and those replacing data a hand-off passes on, come back
    unordered.
    """
    definition = scope.snapshot.definition
    locations, snapshot = definition.locations, definition.values
    exprs = [expr for expr in collect_exprs(function.body) if id(expr) in locations]
    loops = {id(expr): expr for expr in exprs if isinstance(expr, LoopRegion)}
    carries = {
        id(param): (loop, initial, yielded)
        for loop in loops.values()
        for param, initial, yielded in loop.carries()
    }
    writes, shared = _writes(scope, exprs, snapshot, ranks)
    if not writes:
        return {}, ()
    constraints, unordered = {}, []

    def precede(first: Expr, write: Expr) -> None:
        common, before, after = _meeting(locations, first, write)
        constraints.setdefault(id(common), (common, []))[1].append((before, after, write))

    for earlier, later in shared:
        precede(earlier, later)
    for expr in exprs:
        for index in _reads(expr, snapshot):
            replaced = [
                found
                for leaf in SSAValue.leaves(expr.args[index])
                for found in _replaced(leaf, snapshot, carries, writes, locations, loops)
                if found[0] is not expr
            ]
            window = _window(scope, expr, index) if replaced else None
            for write, written, loop, source in replaced:
                if _meet(window if loop is None else _held(scope, locations, source), written):
                    precede(expr if loop is None else loop, write)
    for write, loop in _handed_on(function, scope, carries, writes, locations, snapshot, loops):
        if loop is None:
            unordered.append(write)
        else:
            precede(loop, write)
    targets = {id(expr.target) for expr in exprs if isinstance(expr, Call)}
    orders = {}
    for common, pairs in constraints.values():
        members = [
            expr
            for expr in exprs
            if locations[id(expr)] is common and not isinstance(expr, Var)
        ]
        order = _sorted(members, [(before, after) for before, after, _ in pairs], ranks)
        if order is None:
            unordered.extend(write for _, _, write in pairs)
            continue
        order, default = (
            tuple(
                expr
                for expr in sequence
                if isinstance(expr, (MeshRegion, LoopRegion))
                or (isinstance(expr, Call) and id(expr) not in targets)
            )
            for sequence in (order, _sorted(members, (), ranks))
        )
        if any(expr is not other for expr, other in zip(order, default, strict=True)):
            orders[id(common.owner)] = order
    return orders, tuple({id(write): write for write in unordered}.values())


def _handed_on(function, scope, carries, writes, locations, snapshot, loops):
    """The writes replacing data a hand-off passes on, each with the loop that must precede it.

    A yield passes its data to the next round after every child of the body,
    and the function returns its result after every child, so no order puts
    them before a write in the loop or the function. They need no order where
    a carry took that data first, and only that loop must precede the write.
    """
    handoffs = [
        (id(loop), leaf)
        for loop, _initial, yielded in carries.values()
        for leaf in SSAValue.leaves(yielded)
    ]
    handoffs.extend((None, leaf) for leaf in SSAValue.leaves(function.body))
    for holder, leaf in handoffs:
        for write, written, loop, source in _replaced(
            leaf, snapshot, None, writes, locations, loops
        ):
            if holder is not None and not _within(locations, write, holder):
                continue
            if _meet(_held(scope, locations, leaf if loop is None else source), written):
                yield write, loop


def _replaced(leaf: SSAValue, snapshot, carries, writes, locations, loops):
    """The writes replacing data *leaf* may hold, each with the loop that must precede it.

    The loop is None where *leaf* holds that data in the storage the write
    overwrites, so the reader itself must precede the write; otherwise it comes
    with the value its carry took. A write in that loop needs no order.
    """
    for version, entries in _versions(leaf, snapshot, carries).items():
        for write, written in writes.get(version, ()):
            for entry in entries:
                if entry is None:
                    yield write, written, None, None
                elif not _within(locations, write, entry[0]):
                    yield write, written, loops[entry[0]], entry[1]


def _writes(scope, exprs, snapshot, ranks):
    """Each schedule's in-place write, under the values whose data it replaces.

    A write replaces the data its result operand holds in the storage it
    writes, where no carry took that data out, and, in the same provenance
    root, the result of each earlier write whose elements it meets; those pairs
    keep their depth-first order and come back earlier first.
    """
    writes, roots, shared = {}, {}, []
    for expr in exprs:
        if not isinstance(expr, Call) or not isinstance(expr.target, ScheduleOp):
            continue
        index = aliased_operand(expr)
        operand = SSAValue(expr.args[index])
        window = _window(scope, expr, index)
        for version, entries in _versions(operand, snapshot).items():
            if None in entries:
                writes.setdefault(version, []).append((expr, window))
        entry = snapshot.get(operand)
        roots.setdefault(operand if entry is None else entry.root, []).append((expr, window))
    for group in roots.values():
        group.sort(key=lambda item: ranks[id(item[0])])
        for position, (later, written) in enumerate(group):
            for earlier, other in group[:position]:
                if not _meet(other, written):
                    continue
                shared.append((earlier, later))
                replaced = writes.setdefault(SSAValue(earlier), [])
                if all(write is not later for write, _ in replaced):
                    replaced.append((later, written))
    return writes, shared


def _held(scope, locations, leaf: SSAValue):
    """The elements *leaf* holds, as a set in its provenance root; None when not known."""
    leaf_type = leaf.leaf()
    if not isinstance(leaf_type, TensorType) or not is_literal_shape(leaf_type.shape):
        return None
    entry = locations.get(id(leaf.expr), scope).provenance.lookup(leaf)
    if entry.device is None:
        return None
    box = shape_to_isl_set(tuple(leaf_type.shape), {})
    try:
        elements = entry.device.relation.intersect_domain(box).range()
    except isl.Error:
        return None
    return ((entry.root, elements),)


def _versions(leaf: SSAValue, snapshot, carries=None, opened=frozenset()):
    """The values whose data *leaf* may hold, each with the loops whose carries took it.

    Forwards and views write nothing; a schedule's result is new data. A call's
    result sharing an argument's provenance root holds that argument's data. A
    loop's result holds what its body yields, and its initial data where the body
    passes that on or the loop may run no times; without carries it forwards its
    body. Given *carries*, a parameter holds its initial data and its yield. Each
    way to a value names the loop whose carry took the data out of that value's
    storage, or None where *leaf* holds it there.
    """
    while True:
        entry = snapshot.get(leaf)
        if isinstance(leaf.expr, LoopRegion):
            found = _carried(leaf, entry, snapshot, carries, opened)
            if found is not None:
                return found
        if carries is not None and id(leaf.expr) in carries and leaf not in opened:
            loop, initial, yielded = carries[id(leaf.expr)]
            found = {leaf: {None}}
            for source in (SSAValue(initial, leaf.index), SSAValue(yielded, leaf.index)):
                _passed(found, loop, source, _versions(source, snapshot, carries, opened | {leaf}))
            return found
        if entry is not None and entry.forward is not None:
            leaf = entry.forward
            continue
        expr = leaf.expr
        if not isinstance(expr, Call) or isinstance(expr.target, ScheduleOp):
            return {leaf: {None}}
        index = aliased_operand(expr)
        if index is None:
            return _shared(leaf, entry, snapshot, carries, opened)
        leaf = SSAValue(expr.args[index])


def _shared(leaf: SSAValue, entry, snapshot, carries, opened):
    """The data a call's result *leaf* holds: its own, and its arguments' in the same root."""
    found = {leaf: {None}}
    if entry is None or entry.root == leaf:
        return found
    for argument in leaf.expr.args:
        for operand in SSAValue.leaves(argument):
            bound = snapshot.get(operand)
            if bound is not None and bound.root == entry.root:
                for version, entries in _versions(operand, snapshot, carries, opened).items():
                    found.setdefault(version, set()).update(entries)
    return found


def _carried(leaf: SSAValue, entry, snapshot, carries, opened):
    """The data loop result *leaf* may hold, by what its body yields; None without a carry."""
    loop = leaf.expr
    phi = None if entry is None else entry.forward
    for param, initial, yielded in loop.carries():
        if phi is not None and param is phi.expr:
            found = {leaf: {None}}
            handed, start = SSAValue(yielded, phi.index), SSAValue(initial, phi.index)
            through = _versions(handed, snapshot, carries, opened)
            _passed(found, loop, handed, through)
            if phi in through or _may_skip(loop):
                _passed(found, loop, start, _versions(start, snapshot, carries, opened))
            return found
    return None


def _passed(found, loop: LoopRegion, source: SSAValue, versions) -> None:
    """Add *versions* to *found*; where no carry took them yet, *loop*'s took *source*."""
    for version, entries in versions.items():
        found.setdefault(version, set()).update(
            (id(loop), source) if entry is None else entry for entry in entries
        )


def _within(locations, write: Expr, loop: int) -> bool:
    """Whether *write* sits in the body of the loop whose id is *loop*."""
    return any(id(place.owner) == loop for place in locations[id(write)].ancestors())


def _may_skip(loop: LoopRegion) -> bool:
    """Whether *loop* may run no iterations: its step or least span is not known positive."""
    step = static_dim_value(loop.step)
    span = span_bounds(loop)
    return step is None or step <= 0 or span is None or span[0] <= 0


def _window(scope, expr: Call, index: int):
    """The elements *expr* reaches through argument *index*, each set with its provenance root.

    They are the accesses the iteration scope resolved; None when it resolved none.
    """
    resolved = scope.access.accesses.get("device", {}).get((SSAValue(expr),))
    reached = () if resolved is None else [a for a in resolved[1] if a.input_index == index]
    if not reached:
        return None
    return tuple((access.target, access.relation.relation.range()) for access in reached)


def _meet(first, second) -> bool:
    """Whether two windows may share an element; one not known may meet anything."""
    if first is None or second is None:
        return True
    try:
        return any(
            root != other or not elements.intersect(those).is_empty()
            for root, elements in first
            for other, those in second
        )
    except isl.Error:
        return True


def _reads(expr: Expr, snapshot) -> tuple[int, ...]:
    """The arguments *expr* reads data from, by position; forwards and views read nothing.

    A schedule does not read an argument its instruction only writes.
    """
    if not isinstance(expr, Call) or SSAValue(expr) not in _versions(SSAValue(expr), snapshot):
        return ()
    if isinstance(expr.target, ScheduleOp):
        params, _ = instruction_schema(expr.target.op, len(expr.args))
        return tuple(
            index
            for index, param in enumerate(params)
            if param.effect & MemoryEffect.READ
        )
    return tuple(range(len(expr.args)))


def _meeting(locations, first: Expr, write: Expr):
    """The innermost scope holding both, and the child of it holding each."""
    chain, node = {}, write
    for place in locations[id(write)].ancestors():
        chain[id(place)] = node
        node = place.owner
    node = first
    for place in locations[id(first)].ancestors():
        if id(place) in chain:
            return place, node, chain[id(place)]
        node = place.owner


def _sorted(members, constraints, ranks) -> tuple[Expr, ...] | None:
    """*members* after their operands and *constraints*, ties by rank; None for a cycle."""
    known = {id(member): member for member in members}
    priority = {id(member): (ranks.get(id(member), -1), at) for at, member in enumerate(members)}
    before = {id(member): set() for member in members}
    for member in members:
        before[id(member)].update(
            id(child) for child in expr_children(member) if id(child) in known
        )
    for first, then in constraints:
        before[id(then)].add(id(first))
    after = {key: [] for key in known}
    for key, sources in before.items():
        for source in sources:
            after[source].append(key)
    pending = {key: len(sources) for key, sources in before.items()}
    ready = [(priority[key], key) for key, count in pending.items() if count == 0]
    heapify(ready)
    order = []
    while ready:
        _priority, key = heappop(ready)
        order.append(known[key])
        for target in after[key]:
            pending[target] -= 1
            if pending[target] == 0:
                heappush(ready, (priority[target], target))
    return tuple(order) if len(order) == len(members) else None


__all__ = [
    "LivenessMetadata",
    "RegionInterval",
    "UseEvent",
    "analyze_liveness",
    "result_copies",
]
