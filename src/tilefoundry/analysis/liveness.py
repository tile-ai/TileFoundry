"""Target-independent definition/use intervals for structured HIR SSA."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from heapq import heapify, heappop, heappush
from typing import ClassVar

import isl

from tilefoundry.interfaces import access, typeinfer, valuation
from tilefoundry.ir import core, dim, expr, hir, types, utils
from tilefoundry.ir.hir import tensor

from .errors import Error
from .loop_domain import span_bounds
from .metadata import ValueLifetime, shifted
from .visitor import AnalyzeContext


@dataclass(frozen=True)
class UseEvent:
    events: ClassVar = ("at",)

    """One SSA use and whether it exists only to extend structured liveness."""

    value: core.SSAValue
    at: int
    synthetic: bool = False


@dataclass(frozen=True)
class RegionInterval:
    events: ClassVar = ("entered_at", "exited_at")

    """One mesh region's entry and exit events on the structured timeline."""

    region: core.MeshRegion
    entered_at: int
    exited_at: int


@dataclass(frozen=True)
class LivenessMetadata(expr.IRMetadata):
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
    orders: dict[int, tuple[expr.Expr, ...]] = field(default_factory=dict)

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

    def interval_of(self, value: core.SSAValue) -> ValueLifetime | None:
        """Return *value*'s interval when it belongs to this timeline."""
        return next(
            (item for item in self.intervals if item.value == value),
            None,
        )


def result_copies(node: expr.Expr) -> int:
    """Physical slots declared for one tensor storage value."""
    return node.target.buffers if isinstance(node, expr.Call) and isinstance(node.target, tensor.Empty) else 1


class LivenessVisitor(valuation.Visitor[None]):
    """Build definition/use intervals while preserving structured SSA edges."""

    def __init__(self, function: core.Function, *, is_callee=False, orders=None) -> None:
        super().__init__(root_function=function)
        self.orders = orders or {}
        self.loop_windows = {}
        self.point = -1
        self.states: dict[core.SSAValue, ValueLifetime] = {}
        self.definition_order: list[core.SSAValue] = []
        self.uses: list[UseEvent] = []
        self.regions: list[RegionInterval] = []
        self.loop_entries: list[tuple[int, set[core.SSAValue], set[core.SSAValue]]] = []
        if is_callee:
            definition = self.next_event()
        for parameter in function.params:
            self.define(parameter, definition if is_callee else self.next_event())
        values = core.collect_exprs(function.body) if function.body is not None else ()
        bound_ids = {id(parameter) for parameter in function.params}
        for value in values:
            if isinstance(value, core.MeshRegion):
                bound_ids.update(id(parameter) for parameter in value.params)
            elif isinstance(value, core.LoopRegion):
                bound_ids.add(id(value.induction_var))
                bound_ids.update(id(phi) for phi in value.params)
        free_vars = tuple(
            value for value in values if isinstance(value, expr.Var) and id(value) not in bound_ids
        )
        for free in free_vars:
            self.define(free, self.next_event())

    def next_event(self) -> int:
        """Advance and return the function-wide event position."""
        self.point += 1
        return self.point

    def define(self, value: expr.Expr, point: int) -> None:
        """Define each flattened leaf once, including scalar event identities."""
        for index in range(len(utils.leaves_of(value.type)) or 1):
            key = core.SSAValue(value, index)
            if key in self.states:
                raise ValueError(f"liveness: {type(value).__name__} is defined more than once")
            self.states[key] = ValueLifetime(key, point, point)
            self.definition_order.append(key)

    def use(self, value: expr.Expr | core.SSAValue, point: int, *, synthetic: bool = False) -> None:
        """Extend just the named leaf, or all leaves of an ordinary operand."""
        keys = (
            (value,)
            if isinstance(value, core.SSAValue)
            else tuple(core.SSAValue(value, index) for index in range(len(utils.leaves_of(value.type)) or 1))
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
        for node in self.orders.get(id(owner), ()):
            self.visit(node, ctx)

    def on_region_start(self, region, ctx):
        inner = super().on_region_start(region, ctx)
        self.visit_ordered(region, inner)
        return inner

    def finish(self) -> LivenessMetadata:
        """Freeze the definition-ordered result."""
        for value in dict.fromkeys(
            core.SSAValue(key.expr.args[0], index)
            for key in self.definition_order
            if key.index == 0
            and isinstance(key.expr, expr.Call)
            and isinstance(key.expr.target, tensor.TupleGetItem)
            and not (isinstance(key.expr.args[1], expr.Constant) and type(key.expr.args[1].value) is int)
            for index in range(len(utils.leaves_of(key.expr.args[0].type)))
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

    def visit_Var(self, value: expr.Var, ctx=None) -> None:
        """A binding site defines a Var; encountering a use does not."""
        del value, ctx

    def default_visit_leaf(self, value: expr.Expr, operands: tuple[None, ...], ctx=None) -> None:
        del operands, ctx
        point = self.next_event()
        for operand in core.expr_children(value):
            if (
                isinstance(value, expr.Call)
                and isinstance(value.target, tensor.TupleGetItem)
                and operand is value.args[0]
                and isinstance(value.args[1], expr.Constant)
                and type(value.args[1].value) is int
            ):
                begin, count = utils.leaf_span(operand.type, value.args[1].value)
                for index in range(begin, begin + count):
                    self.use(core.SSAValue(operand, index), point)
            else:
                self.use(operand, point)
        self.define(value, point)
        if result_copies(value) > 1 and self.loop_entries:
            self.loop_entries[-1][2].update(
                core.SSAValue(value, index) for index in range(len(utils.leaves_of(value.type)) or 1)
            )

    def visit_Call(self, call: expr.Call, ctx=None) -> None:
        context = ctx
        callee = None if context is None else context.callee
        if callee is None:
            return self.default_visit_leaf(call, self.visit_operands(call, ctx), ctx)
        self.visit_operands(call, ctx)
        argument_use = self.next_event()
        for argument in call.args:
            self.use(argument, argument_use)
        template = expr.get_metadata(callee, LivenessMetadata)
        if template is None:
            raise Error(f"liveness: missing local template for {callee.name!r}")
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
                        core.SSAValue(argument, use.value.index),
                        use.at,
                        synthetic=use.synthetic,
                    )
        self.point = template.timeline_end
        self.define(call, self.next_event())

    def visit_MeshRegion(self, region: core.MeshRegion, ctx=None) -> None:
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

    def visit_LoopRegion(self, region: core.LoopRegion, ctx=None) -> None:
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
    function: core.Function, context: AnalyzeContext | None = None, *, is_callee: bool = False
) -> LivenessMetadata:
    """Build target-independent intervals for a checked HIR function.

    Given an iteration scope, events put the uses of data a schedule overwrites
    in place before the overwrite where the analysis can order them, and
    ``Error`` is raised where it cannot.
    """
    if function.body is None:
        raise ValueError(f"liveness: function {function.name!r} has no body")
    visitor = LivenessVisitor(function, is_callee=is_callee)
    if context is None:
        context = typeinfer.TypeInferContext(owner=function)
    visitor.visit_function_body(function, context)
    first = visitor.finish()
    scope = getattr(context, "scope", None)
    if scope is None:
        return first
    ranks = {}
    for item in first.intervals:
        ranks.setdefault(id(item.value.expr), len(ranks))
    orders, unordered = scope_orders(function, scope, ranks)
    if unordered:
        raise Error(
            "liveness: cannot order every use of the data these schedules overwrite "
            "before them: " + "; ".join(expr.describe_value(write) for write in unordered)
        )
    if not orders:
        return first
    visitor = LivenessVisitor(function, is_callee=is_callee, orders=orders)
    visitor.visit_ordered(function, context)
    visitor.visit_function_body(function, context)
    return visitor.finish()


def scope_orders(function: core.Function, scope, ranks):
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
    nodes = [node for node in core.collect_exprs(function.body) if id(node) in locations]
    loops = {id(node): node for node in nodes if isinstance(node, core.LoopRegion)}
    carries = {
        id(param): (loop, initial, yielded)
        for loop in loops.values()
        for param, initial, yielded in loop.carries()
    }
    writes, shared = schedule_writes(scope, nodes, snapshot, ranks)
    if not writes:
        return {}, ()
    constraints, unordered = {}, []

    def precede(first: expr.Expr, write: expr.Expr) -> None:
        common, before, after = meeting_scope(locations, first, write)
        constraints.setdefault(id(common), (common, []))[1].append((before, after, write))

    for earlier, later in shared:
        precede(earlier, later)
    for node in nodes:
        for index in read_positions(node, snapshot):
            replaced = [
                found
                for leaf in core.ssa_leaves(node.args[index])
                for found in replaced_by_writes(leaf, snapshot, carries, writes, locations, loops)
                if found[0] is not node
            ]
            window = reached_window(scope, node, index) if replaced else None
            for write, written, loop, source in replaced:
                if windows_meet(window if loop is None else held_elements(scope, locations, source), written):
                    precede(node if loop is None else loop, write)
    for write, loop in handed_on(function, scope, carries, writes, locations, snapshot, loops):
        if loop is None:
            unordered.append(write)
        else:
            precede(loop, write)
    targets = {id(node.target) for node in nodes if isinstance(node, expr.Call)}
    orders = {}
    for common, pairs in constraints.values():
        members = [
            node
            for node in nodes
            if locations[id(node)] is common and not isinstance(node, expr.Var)
        ]
        order = sort_members(members, [(before, after) for before, after, _ in pairs], ranks)
        if order is None:
            unordered.extend(write for _, _, write in pairs)
            continue
        order, default = (
            tuple(
                node
                for node in sequence
                if isinstance(node, (core.MeshRegion, core.LoopRegion))
                or (isinstance(node, expr.Call) and id(node) not in targets)
            )
            for sequence in (order, sort_members(members, (), ranks))
        )
        if any(node is not other for node, other in zip(order, default, strict=True)):
            orders[id(common.owner)] = order
    return orders, tuple({id(write): write for write in unordered}.values())


def handed_on(function, scope, carries, writes, locations, snapshot, loops):
    """The writes replacing data a hand-off passes on, each with the loop that must precede it.

    A yield passes its data to the next round after every child of the body,
    and the function returns its result after every child, so no order puts
    them before a write in the loop or the function. They need no order where
    a carry took that data first, and only that loop must precede the write.
    """
    handoffs = [
        (id(loop), leaf)
        for loop, start, yielded in carries.values()
        for leaf in core.ssa_leaves(yielded)
    ]
    handoffs.extend((None, leaf) for leaf in core.ssa_leaves(function.body))
    for holder, leaf in handoffs:
        for write, written, loop, source in replaced_by_writes(
            leaf, snapshot, None, writes, locations, loops
        ):
            if holder is not None and not within_loop(locations, write, holder):
                continue
            if windows_meet(held_elements(scope, locations, leaf if loop is None else source), written):
                yield write, loop


def replaced_by_writes(leaf: core.SSAValue, snapshot, carries, writes, locations, loops):
    """The writes replacing data *leaf* may hold, each with the loop that must precede it.

    The loop is None where *leaf* holds that data in the storage the write
    overwrites, so the reader itself must precede the write; otherwise it comes
    with the value its carry took. A write in that loop needs no order.
    """
    for version, entries in versions_of(leaf, snapshot, carries).items():
        for write, written in writes.get(version, ()):
            for entry in entries:
                if entry is None:
                    yield write, written, None, None
                elif not within_loop(locations, write, entry[0]):
                    yield write, written, loops[entry[0]], entry[1]


def schedule_writes(scope, nodes, snapshot, ranks):
    """Each schedule's in-place write, under the values whose data it replaces.

    A write replaces the data its result operand holds in the storage it
    writes, where no carry took that data out, and, in the same provenance
    root, the result of each earlier write whose elements it meets; those pairs
    keep their depth-first order and come back earlier first.
    """
    writes, roots, shared = {}, {}, []
    for node in nodes:
        if not isinstance(node, expr.Call) or not isinstance(node.target, hir.schedule.ScheduleOp):
            continue
        index = access.aliased_operand(node)
        operand = core.SSAValue(node.args[index])
        window = reached_window(scope, node, index)
        for version, entries in versions_of(operand, snapshot).items():
            if None in entries:
                writes.setdefault(version, []).append((node, window))
        entry = snapshot.get(operand)
        roots.setdefault(operand if entry is None else entry.root, []).append((node, window))
    for group in roots.values():
        group.sort(key=lambda item: ranks[id(item[0])])
        for position, (later, written) in enumerate(group):
            for earlier, other in group[:position]:
                if not windows_meet(other, written):
                    continue
                shared.append((earlier, later))
                replaced = writes.setdefault(core.SSAValue(earlier), [])
                if all(write is not later for write, _ in replaced):
                    replaced.append((later, written))
    return writes, shared


def held_elements(scope, locations, leaf: core.SSAValue):
    """The elements *leaf* holds, as a set in its provenance root; None when not known."""
    leaf_type = leaf.leaf()
    if not isinstance(leaf_type, types.TensorType) or not expr.is_literal_shape(leaf_type.shape):
        return None
    entry = locations.get(id(leaf.expr), scope).provenance.lookup(leaf)
    if entry.device is None:
        return None
    box = dim.shape_to_isl_set(tuple(leaf_type.shape), {})
    try:
        elements = entry.device.relation.intersect_domain(box).range()
    except isl.Error:
        return None
    return ((entry.root, elements),)


def versions_of(leaf: core.SSAValue, snapshot, carries=None, opened=frozenset()):
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
        if isinstance(leaf.expr, core.LoopRegion):
            found = carried_versions(leaf, entry, snapshot, carries, opened)
            if found is not None:
                return found
        if carries is not None and id(leaf.expr) in carries and leaf not in opened:
            loop, initial, yielded = carries[id(leaf.expr)]
            found = {leaf: {None}}
            for source in (core.SSAValue(initial, leaf.index), core.SSAValue(yielded, leaf.index)):
                add_versions(found, loop, source, versions_of(source, snapshot, carries, opened | {leaf}))
            return found
        if entry is not None and entry.forward is not None:
            leaf = entry.forward
            continue
        node = leaf.expr
        if not isinstance(node, expr.Call) or isinstance(node.target, hir.schedule.ScheduleOp):
            return {leaf: {None}}
        index = access.aliased_operand(node)
        if index is None:
            return shared_versions(leaf, entry, snapshot, carries, opened)
        leaf = core.SSAValue(node.args[index])


def shared_versions(leaf: core.SSAValue, entry, snapshot, carries, opened):
    """The data a call's result *leaf* holds: its own, and its arguments' in the same root."""
    found = {leaf: {None}}
    if entry is None or entry.root == leaf:
        return found
    for argument in leaf.expr.args:
        for operand in core.ssa_leaves(argument):
            bound = snapshot.get(operand)
            if bound is not None and bound.root == entry.root:
                for version, entries in versions_of(operand, snapshot, carries, opened).items():
                    found.setdefault(version, set()).update(entries)
    return found


def carried_versions(leaf: core.SSAValue, entry, snapshot, carries, opened):
    """The data loop result *leaf* may hold, by what its body yields; None without a carry."""
    loop = leaf.expr
    phi = None if entry is None else entry.forward
    for param, initial, yielded in loop.carries():
        if phi is not None and param is phi.expr:
            found = {leaf: {None}}
            handed, start = core.SSAValue(yielded, phi.index), core.SSAValue(initial, phi.index)
            through = versions_of(handed, snapshot, carries, opened)
            add_versions(found, loop, handed, through)
            if phi in through or may_skip(loop):
                add_versions(found, loop, start, versions_of(start, snapshot, carries, opened))
            return found
    return None


def add_versions(found, loop: core.LoopRegion, source: core.SSAValue, versions) -> None:
    """Add *versions* to *found*; where no carry took them yet, *loop*'s took *source*."""
    for version, entries in versions.items():
        found.setdefault(version, set()).update(
            (id(loop), source) if entry is None else entry for entry in entries
        )


def within_loop(locations, write: expr.Expr, loop: int) -> bool:
    """Whether *write* sits in the body of the loop whose id is *loop*."""
    return any(id(place.owner) == loop for place in locations[id(write)].ancestors())


def may_skip(loop: core.LoopRegion) -> bool:
    """Whether *loop* may run no iterations: its step or least span is not known positive."""
    step = expr.static_dim_value(loop.step)
    span = span_bounds(loop)
    return step is None or step <= 0 or span is None or span[0] <= 0


def reached_window(scope, node: expr.Call, index: int):
    """The elements *node* reaches through argument *index*, each set with its provenance root.

    They are the accesses the iteration scope resolved; None when it resolved none.
    """
    resolved = scope.access.accesses.get("device", {}).get((core.SSAValue(node),))
    reached = () if resolved is None else [item for item in resolved[1] if item.input_index == index]
    if not reached:
        return None
    return tuple((item.target, item.relation.relation.range()) for item in reached)


def windows_meet(first, second) -> bool:
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


def read_positions(node: expr.Expr, snapshot) -> tuple[int, ...]:
    """The arguments *node* reads data from, by position; forwards and views read nothing.

    A schedule does not read an argument its instruction only writes.
    """
    if not isinstance(node, expr.Call) or core.SSAValue(node) not in versions_of(core.SSAValue(node), snapshot):
        return ()
    if isinstance(node.target, hir.schedule.ScheduleOp):
        params, _ = hir.schedule.instruction_schema(node.target.op, len(node.args))
        return tuple(
            index
            for index, param in enumerate(params)
            if param.effect & expr.MemoryEffect.READ
        )
    return tuple(range(len(node.args)))


def meeting_scope(locations, first: expr.Expr, write: expr.Expr):
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


def sort_members(members, constraints, ranks) -> tuple[expr.Expr, ...] | None:
    """*members* after their operands and *constraints*, ties by rank; None for a cycle."""
    known = {id(member): member for member in members}
    priority = {id(member): (ranks.get(id(member), -1), at) for at, member in enumerate(members)}
    before = {id(member): set() for member in members}
    for member in members:
        before[id(member)].update(
            id(child) for child in core.expr_children(member) if id(child) in known
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
        rank, key = heappop(ready)
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
