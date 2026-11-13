"""Internal values and solver results for physical memory placement."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from functools import cached_property
from itertools import combinations
from math import prod
from typing import Protocol

import isl
from ortools.sat.python import cp_model

from tilefoundry import target, utils
from tilefoundry.interfaces import access, cost, valuation
from tilefoundry.ir import core, dim, expr, types
from tilefoundry.ir import utils as ir_utils

from .access import (
    SEMANTIC_KINDS,
    STRUCTURAL_KINDS,
    AliasKind,
    BufferAlias,
    eliminate_parameters,
)
from .errors import Error
from .liveness import LivenessMetadata, result_copies
from .metadata import ValueLifetime, shifted
from .precision import Precision
from .reach import Footprint, ReachedAddresses
from .visitor import AnalyzeContext


class SolverOptions(Protocol):
    """Placement limits consumed independently of the memory front end."""

    timeout_seconds: float
    workers: int
    random_seed: int


@dataclass(eq=False)
class AllocationValue:
    """One SSA result's own lifetime and accepted storage links."""

    lifetime: ValueLifetime
    aliases: tuple[BufferAlias, ...] = ()

    @property
    def value(self) -> core.SSAValue:
        return self.lifetime.value

    @property
    def bytes(self) -> int:
        return self.lifetime.bytes

    @property
    def alignment(self) -> int:
        return alignment_of(self.value.expr)

    def add_alias(self, alias: BufferAlias) -> None:
        """Accept one more storage link."""
        self.aliases += (alias,)


def identity_of(value: AllocationValue) -> access.Relation:
    held = ir_utils.local_type_of(ir_utils.leaves_of(value.value.expr.type)[value.value.index])
    box = dim.shape_to_isl_set(tuple(held.shape), {})
    return access.Relation(isl.map.identity(box.get_space().map_from_set()).intersect_domain(box))


def alias_edge(value, by_value, *, storage_path=False):
    return next(
        (
            alias
            for alias in value.aliases
            if alias.target != value.value
            and alias.target in by_value
            and (not storage_path or alias.kind <= AliasKind.CARRY_INIT)
        ),
        None,
    )


def root_value(value: AllocationValue, by_value, *, storage_path=False) -> AllocationValue:
    """Follow accepted value links without modifying or compressing them."""
    seen = set()
    while True:
        if value in seen:
            raise Error("allocation: storage links contain a cycle")
        seen.add(value)
        edge = alias_edge(value, by_value, storage_path=storage_path)
        if edge is None:
            return value
        value = by_value[edge.target]


def address_alias_of(value, by_value, *, alias=None, precisions=None):
    """Compose accepted address links without merging SSA values or lifetimes."""
    structural = alias is not None
    alias = alias or BufferAlias(value.value, identity_of(value), AliasKind.FORWARD)
    seen = set()
    while alias.target in by_value:
        current = by_value[alias.target]
        edge = alias_edge(current, by_value)
        if edge is None or (structural and edge.kind not in STRUCTURAL_KINDS):
            return alias
        if precisions is not None:
            precision = precisions.get(current.value, Precision.UNKNOWN)
            if precision is not Precision.EXACT:
                raise Error(
                    f"non-EXACT address for {expr.value_label(current.value.expr)}: {edge.relation.relation}"
                )
        if current in seen:
            raise Error("allocation: storage links contain a cycle")
        seen.add(current)
        relation = alias.relation.compose(edge.relation)
        alias = replace(
            alias,
            target=edge.target,
            relation=relation,
            writes=alias.writes or edge.writes or edge.kind is AliasKind.INSERT_SLICE,
        )
    return alias


@dataclass(frozen=True)
class AllocationSummaryMetadata(expr.IRMetadata):
    """Definition facts interpreted at a leaf call, never an execution context."""

    values: tuple[AllocationValue, ...]
    in_place: tuple[tuple[int, BufferAlias], ...] = ()
    address_decisions: tuple[tuple[core.SSAValue, BufferAlias, core.SSAValue], ...] = ()

    windows: tuple[ValueLifetime, ...] = field(default=(), repr=False)
    sizes: dict[AllocationValue, int] = field(default_factory=dict, repr=False)

    def resolve(self, ctx):
        if ctx.callee_scope is None:
            return self
        scope = ctx.callee_scope
        cache = scope.snapshot.resolved.allocations
        if id(self) in cache:
            return cache[id(self)][1]
        function = ctx.callee
        caller_liveness = expr.get_metadata(ctx.function, LivenessMetadata)
        callee_liveness = expr.get_metadata(function, LivenessMetadata)
        origin = (
            caller_liveness.interval_of(core.SSAValue(ctx.call_site)).defined_at
            - callee_liveness.timeline_end
            - 1
        )
        boundary = {
            core.SSAValue(param, leaf.index): leaf
            for param in function.params
            if id(param) in scope.provenance.arguments
            for leaf in core.ssa_leaves(scope.provenance.arguments[id(param)])
        }
        boundary.update(
            {core.SSAValue(function.body, leaf.index): leaf for leaf in core.ssa_leaves(ctx.call_site)}
        )
        mapped = {self.root(key): value for key, value in boundary.items() if key in self.by_value}
        windows = tuple(
            replace(
                shifted(window, origin),
                value=mapped.get(self.root(window.value), window.value),
                persistent=window.persistent and self.root(window.value) not in mapped,
            )
            for window in self.windows
        )
        result = replace(
            self,
            values=tuple(value for value in self.values if self.root(value) not in mapped),
            windows=windows,
        )
        cache[id(self)] = self, result
        return result

    @cached_property
    def by_value(self):
        return {item.value: item for item in self.values}

    def root(self, value):
        return root_value(
            self.by_value[value] if isinstance(value, core.SSAValue) else value, self.by_value
        )

    def address_alias(self, value):
        return address_alias_of(
            self.by_value[value] if isinstance(value, core.SSAValue) else value, self.by_value
        )


@dataclass(frozen=True)
class AllocationResult:
    peak_bytes: int
    solver_status: str
    offsets: dict[AllocationValue, int] = field(default_factory=dict)


def bound_relation(relation, scope, held):
    coordinates = relation.relation.gist_params(relation.relation.params())
    irrelevant = tuple(
        value
        for name, value in relation.values.items()
        if not coordinates.involves_dims(
            isl.dim_type.PARAM, coordinates.find_dim_by_name(isl.dim_type.PARAM, name), 1
        )
    )
    relation = relation.project(irrelevant)
    bound, precision = eliminate_parameters(
        relation, scope, held, narrow=True, formals=scope.root.owner.params
    )
    return bound if precision is Precision.EXACT else None


def storage_source(value, operand, ctx):
    source = ctx.scope.provenance.lookup(value).narrow
    actual = ctx.scope.provenance.lookup(operand).narrow
    inverse = actual.aligned(actual.relation.reverse())
    return BufferAlias(operand, source.compose(inverse), AliasKind.VIEW)


def value_coordinates(model, value, scope, *, override=None, storage_path=False):
    seen, composed = set(), None
    while value not in seen:
        seen.add(value)
        edge = (
            override[1]
            if override is not None and override[0] is value
            else alias_edge(value, model.by_value, storage_path=storage_path)
        )
        relation = bound_relation(
            identity_of(value) if edge is None else edge.relation, scope, value.value.leaf()
        )
        if relation is None:
            return None
        composed = relation if composed is None else composed.compose(relation)
        if edge is None:
            return value, composed
        value = model.by_value[edge.target]
    return None


def visible_at(model, value, producer):
    context = model.context_of(producer.value.expr)
    scope = context.scope
    if isinstance(value, (int, expr.Constant, expr.DimVar)):
        return True
    if isinstance(value, expr.Call) and isinstance(value.target, dim.MeshCoord):
        mesh = scope.mesh
        if mesh is None:
            return False
        available = {getattr(held, "name", held) for held in mesh.topologies}
        axes = dim.literal_axes(value.args[0])
        offset = 0
        for topology, layout in zip(
            value.target.mesh.topologies, types.levels(value.target.mesh), strict=True
        ):
            width = len(types.flatten(layout.shape))
            if axes is None or any(offset <= axis < offset + width for axis in axes):
                if getattr(topology, "name", topology) not in available:
                    return False
            offset += width
        return True
    interval = model.liveness.interval_of(core.SSAValue(value))
    location = model.context.scope.snapshot.definition.locations.get(id(value))
    return (
        location is not None
        and any(location.owner is c.owner for c in context.scope.ancestors())
        and interval is not None
        and interval.defined_at < producer.lifetime.defined_at
    )


def landing_of(model, holder, alias, *, window=False):
    try:
        representative = root_value(holder, model.by_value, storage_path=window)
        scope = model.context.scope.snapshot.definition.locations[id(representative.value.expr)]
        source = value_coordinates(model, holder, scope, storage_path=window)
        tgt = bound_relation(alias.relation, scope, alias.target.leaf())
        if source is None or tgt is None:
            return None
        relation = source[1].aligned(source[1].relation.reverse()).compose(tgt)
        if window:
            held = ir_utils.local_type_of(representative.value.leaf())
            expected = dim.shape_to_isl_set(tuple(held.shape), {}).intersect_params(
                relation.relation.params()
            )
            if (
                not relation.relation.domain().is_equal(expected)
                or relation.relation.dim(isl.dim_type.IN) != relation.relation.dim(isl.dim_type.OUT)
                or relation.relation.is_translation() != 1
            ):
                return None
            if not all(
                visible_at(model, value, representative) for value in relation.values.values()
            ):
                return None
        return representative, replace(alias, relation=relation)
    except (isl.Error, Error):
        return None


def mapped_pointwise(model, left, right, start, landing):
    touching = {
        window
        for member in right
        for window in model.occupancy(member)
        if window is not member.lifetime and window.last_used_at == start
    }
    covered, proved = set(), set()
    for output in left:
        context = model.context_of(output.value.expr)
        if (
            context is None
            or id(output.value.expr) not in model.context.scope.snapshot.resolved.allocations
        ):
            continue
        written = value_coordinates(model, output, context.scope, override=landing)
        for key, actual, windows in model.in_place_operands(output.value.expr, context):
            if key != output.value:
                continue
            member = model.by_value.get(actual.target)
            read = None if member is None else value_coordinates(model, member, context.scope)
            if read is None or written is None or read[0] is not written[0]:
                continue
            reached = actual.relation.compose(read[1]).relation
            if reached.is_equal(written[1].relation):
                covered.update(touching & windows)
                proved.add(member.value)
    return proved if touching <= covered else None


def pointwise(model, left, right, start, landing, use_events):
    proved = mapped_pointwise(model, left, right, start, landing)
    if proved is None:
        return False
    read = {
        member.value
        for member in right
        if any(use.at == start for use in use_events.get(member.value, ()))
    }
    for producer in left:
        if producer.lifetime.defined_at != start or not isinstance(producer.value.expr, expr.Call):
            continue
        node = producer.value.expr
        context = model.context_of(node)
        scope = context.scope
        relations = local_relations(node, scope)
        outputs = relations[len(node.args) :]
        if not outputs:
            continue
        positioned = value_coordinates(model, producer, scope, override=landing)
        if positioned is None:
            continue
        written = outputs[producer.value.index].compose(positioned[1]).relation
        for holder, alias, _ in model.candidates:
            if holder == producer.value and alias.kind is AliasKind.INSERT_SLICE:
                for index, operand in enumerate(node.args):
                    if alias.target == core.SSAValue(operand):
                        written = written.union(relations[index].compose(positioned[1]).relation)
        for member in right:
            candidates = []
            for index, operand in enumerate(node.args):
                for leaf in core.ssa_leaves(operand):
                    value = model.by_value.get(leaf)
                    if value is None:
                        continue
                    reached = value_coordinates(model, value, scope)
                    if reached is not None and reached[0] is positioned[0]:
                        candidates.append(relations[index].compose(reached[1]).relation)
            if candidates and all(
                image.is_equal(written.intersect_domain(image.domain())) for image in candidates
            ):
                proved.add(member.value)
    return read <= proved


def is_zero_offset(relation: isl.map) -> bool:
    """Whether every result coordinate equals the operand coordinate."""
    output_dims = relation.dim(isl.dim_type.OUT)
    loop_dims = relation.dim(isl.dim_type.IN) - output_dims
    if loop_dims < 0:
        return False
    try:
        return all(
            utils.equates(relation, output_axis, loop_dims + output_axis)
            for output_axis in range(output_dims)
        )
    except isl.Error:
        return False


def call_result_address(ctx, alias):
    bound = ctx.callee_scope
    argument = bound.provenance.arguments.get(id(alias.target.expr))
    if argument is None:
        return None
    relation = bound.access.project(alias.relation)
    return replace(alias, target=core.SSAValue(argument, alias.target.index), relation=relation)


def local_relations(node, scope):
    key = (id(node), "storage", None)
    if key in scope.access.projections:
        return scope.access.projections[key]
    scope.access.cache_projection(key, ())
    stated = scope.access.relations.get(id(node))
    if stated is None or len(stated[1]) != len(node.args) + 1:
        return ()
    result = []
    for value, relation in zip((*node.args, node), stated[1], strict=True):
        held = ir_utils.local_type_of(value.type)
        relation, precision = eliminate_parameters(
            relation, scope, held, narrow=True, formals=scope.root.owner.params
        )
        if precision is not Precision.EXACT:
            return ()
        if isinstance(held, types.TensorType) and expr.is_literal_shape(held.shape):
            relation = relation.aligned(
                relation.relation.intersect_range(dim.shape_to_isl_set(tuple(held.shape), {}))
            )
        result.append(relation)
    scope.access.cache_projection(key, tuple(result))
    return scope.access.projections[key]


def operand_to_result_relation(node, scope, insert_accesses=None):
    """Read all operation boundaries in the same iteration parameter space."""
    relations = local_relations(node, scope)
    outputs = relations[len(node.args) :]
    if len(outputs) != 1:
        return ()
    written = outputs[0]
    leaves = core.ssa_leaves(node)
    if len(leaves) != 1 or not expr.is_literal_shape(leaves[0].leaf().shape):
        return ()
    box = dim.shape_to_isl_set(tuple(ir_utils.local_type_of(leaves[0].leaf()).shape), {})
    full = box.intersect_params(written.relation.params())
    if not covers_result(
        written.relation.range(), full, written.relation, written.relation.params(), box
    ):
        return ()
    result = []
    for index, operand in enumerate(node.args):
        read = relations[index]
        if not read.relation.is_equal(written.relation):
            continue
        mapped = read.aligned(read.relation.reverse()).compose(written)
        if mapped.relation.is_single_valued() and mapped.relation.is_injective():
            for value in core.ssa_leaves(operand):
                if value.index == 0:
                    result.append((value, mapped))
    return tuple(result)


@dataclass(kw_only=True)
class AllocationConstraintVisitor(valuation.Visitor[None]):
    """Allocation working state over an existing analysis context."""

    context: AnalyzeContext | None = None

    address_decisions: list[tuple[core.SSAValue, BufferAlias, core.SSAValue]] = field(default_factory=list)

    liveness: LivenessMetadata | None = None
    quantities: dict[core.SSAValue, tuple[str, int]] = field(default_factory=dict)
    persistent: frozenset[core.SSAValue] = frozenset()
    by_value: dict[core.SSAValue, AllocationValue] = field(default_factory=dict)
    windows: list[ValueLifetime] = field(default_factory=list)
    candidates: list[tuple[core.SSAValue, BufferAlias, core.LoopRegion | None]] = field(default_factory=list)
    instance_groups: list = field(default_factory=list)
    member_groups: dict | None = field(default=None, init=False, repr=False)
    occupancy_index: dict | None = field(default=None, init=False, repr=False)

    def forget_indexes(self) -> None:
        """Drop the lookups derived from ``by_value`` and ``windows``."""
        self.member_groups = self.occupancy_index = None

    def context_of(self, node):
        scope = self.context.scope.snapshot.definition.locations.get(id(node))
        if scope is None:
            return None
        return replace(
            self.context,
            scope=scope,
            owner=scope.owner,
            current_mesh=scope.mesh,
            call_site=node if isinstance(node, expr.Call) else None,
        )

    def occupancy(self, value):
        """Native lifetime plus translated occupancy, without changing either."""
        if self.occupancy_index is None:
            self.occupancy_index = defaultdict(list)
            for window in self.windows:
                self.occupancy_index[window.value].append(window)
        return tuple(self.occupancy_index[value.value])

    def span(self, values):
        windows = tuple(window for value in values for window in self.occupancy(value))
        return min(w.defined_at for w in windows), max(w.last_used_at for w in windows)

    def members(self, value):
        if self.member_groups is None:
            self.member_groups = defaultdict(list)
            for item in self.by_value.values():
                self.member_groups[root_value(item, self.by_value)].append(item)
        return tuple(self.member_groups[root_value(value, self.by_value)])

    def root(self, value):
        return root_value(value, self.by_value)

    def roots(self):
        return tuple(
            dict.fromkeys(root_value(value, self.by_value) for value in self.by_value.values())
        )

    def size(self, value):
        members = self.members(value)
        size = max(window.bytes for item in members for window in self.occupancy(item))
        if self.instance_groups and any(item.lifetime.persistent for item in members):
            keys = {item.value for item in members}
            totals = defaultdict(int)
            for reached in Footprint(tuple(self.instance_groups)).reached:
                if reached.target in keys:
                    totals[reached.target] += reached.counted[0]
            size = max(size, max(totals.values(), default=0))
        return size

    def is_persistent(self, value):
        return any(item.lifetime.persistent for item in self.members(value))

    def tie_analysis(self, ctx):
        """All ties read occupancy; global interference keeps individual windows."""
        use_events = defaultdict(list)
        for use in self.liveness.uses:
            use_events[use.value].append(use)
        local = {item.value for item in self.liveness.intervals}
        for holder, alias, loop in sorted(
            self.candidates,
            key=lambda item: (
                item[1].kind not in SEMANTIC_KINDS,
                item[1].kind,
                self.by_value[item[0]].lifetime.defined_at,
            ),
        ):
            value = self.by_value[holder]
            tgt = self.by_value.get(alias.target)
            if tgt is None or holder not in local:
                continue
            a = root_value(value, self.by_value)
            b = root_value(tgt, self.by_value)
            if a is b:
                continue
            if a.value not in local:
                continue
            left = self.members(a)
            right = self.members(b)
            semantic = alias.kind in SEMANTIC_KINDS
            if not semantic and (
                self.is_persistent(b)
                or any(
                    any(item.value.expr is param for param in self.context.function.params)
                    for item in right
                )
            ):
                continue
            if (alias.kind is AliasKind.INSERT_SLICE or alias.writes) and any(
                not use.synthetic and use.at > value.lifetime.defined_at
                for item in right
                for use in use_events.get(item.value, ())
            ):
                continue
            if a.lifetime.memory_level != b.lifetime.memory_level or self.size(a) > self.size(b):
                continue
            start, _ = self.span(left)
            _, end = self.span(right)
            window = alias.kind is AliasKind.WINDOW
            if window:
                node = loop
                destination = self.by_value.get(core.SSAValue(node.args[0]))
                if destination is None or root_value(destination, self.by_value) is not b:
                    continue
                event = self.liveness.interval_of(core.SSAValue(node)).defined_at
                if self.is_persistent(a) or end < 0 or self.span(left)[1] > event:
                    continue
                if any(
                    start < use.at < event
                    for item in right
                    for use in use_events.get(item.value, ())
                ):
                    continue
                end = start
            elif alias.kind is AliasKind.CARRY_YIELD:
                phi_event, backedge = self.liveness.loop_windows[id(loop)]
                if not phi_event < a.lifetime.defined_at < backedge:
                    continue
                end = max(
                    (
                        use.at
                        for item in right
                        for use in use_events.get(item.value, ())
                        if phi_event < use.at < backedge
                    ),
                    default=phi_event,
                )
                end = max(
                    end,
                    max(
                        (
                            min(window.last_used_at, backedge - 1)
                            for item in right
                            for window in self.occupancy(item)
                            if window is not item.lifetime
                            and phi_event < window.last_used_at
                            and window.defined_at < backedge
                        ),
                        default=phi_event,
                    ),
                )
                start = min(
                    window.defined_at
                    for item in left
                    for window in self.occupancy(item)
                    if phi_event < window.defined_at < backedge
                )
            if not window and not semantic and end > start:
                continue
            landed = landing_of(self, value, alias, window=window)
            physical = landed if not window or landed is None else landing_of(self, value, alias)
            if physical is None or landed is None:
                continue
            new_root, edge = physical
            if alias.kind is AliasKind.IN_PLACE:
                scope = self.context.scope.snapshot.definition.locations[id(new_root.value.expr)]
                positioned = value_coordinates(self, new_root, scope, override=physical)
                if positioned is None or not is_zero_offset(positioned[1].relation):
                    continue
            if (
                not semantic
                and end == start
                and not pointwise(self, left, right, start, physical, use_events)
            ):
                continue
            new_root.add_alias(edge)
            self.member_groups = None
            self.address_decisions.append((holder, alias, new_root.value))
            if value is not new_root and alias not in value.aliases:
                value.add_alias(alias)
            if window and landed[0] is not new_root:
                landed[0].add_alias(landed[1])
        for holder, alias, _ in self.candidates:
            if alias.kind in (AliasKind.CARRY_INIT, AliasKind.CARRY_YIELD, AliasKind.INSERT_SLICE):
                value = self.by_value[holder]
                tgt = self.by_value.get(alias.target)
                if (
                    tgt is not None
                    and root_value(value, self.by_value) is root_value(tgt, self.by_value)
                    and alias not in value.aliases
                ):
                    value.add_alias(alias)

    def in_place_operands(self, node, ctx):
        callee = ctx.callee
        if callee is None:
            for operand, relation in operand_to_result_relation(node, ctx.scope):
                if is_zero_offset(relation.relation):
                    yield (
                        core.SSAValue(node),
                        BufferAlias(operand, relation, AliasKind.IN_PLACE),
                        frozenset(),
                    )
            return
        template = expr.get_metadata(callee, AllocationSummaryMetadata)
        resolved = self.context.scope.snapshot.resolved.allocations.get(id(node))
        windows = () if resolved is None else resolved.windows
        for index, capability in template.in_place:
            actual = call_result_address(ctx, capability)
            if actual is not None and is_zero_offset(actual.relation.relation):
                yield (
                    core.SSAValue(node, index),
                    actual,
                    frozenset(w for w in windows if w.value == actual.target),
                )

    def summary(self):
        capabilities = []
        for leaf in core.ssa_leaves(self.context.function.body):
            value = self.by_value.get(leaf)
            if value is None:
                continue
            for member in self.members(value):
                context = self.context_of(member.value.expr)
                if context is None or not isinstance(member.value.expr, expr.Call):
                    continue
                for operand, relation in operand_to_result_relation(
                    member.value.expr, context.scope
                ):
                    actual = self.by_value.get(operand)
                    if actual is None:
                        continue
                    address = address_alias_of(actual, self.by_value)
                    relation = relation.compose(address.relation)
                    if (
                        any(address.target.expr is param for param in self.context.function.params)
                        and is_zero_offset(relation.relation)
                        and self.span(self.members(actual))[1] <= self.span(self.members(value))[0]
                    ):
                        capabilities.append(
                            (leaf.index, BufferAlias(address.target, relation, AliasKind.IN_PLACE))
                        )
        return AllocationSummaryMetadata(
            tuple(self.by_value.values()),
            tuple(capabilities),
            tuple(self.address_decisions),
            tuple(self.windows),
            {value: self.size(value) for value in self.roots()},
        )

    def run(self, function, context):
        self.context = context
        liveness = self.liveness = expr.get_metadata(function, LivenessMetadata, ctx=context)
        facts = context.target.get_facts(target.MemoryHierarchyFacts)
        quantities = {}
        persistent = set()
        placement = cost.Context(
            module=context.module,
            function=context.function,
            target=context.target,
            topology_level=context.topology_level,
            topologies=context.topologies,
        )
        for lifetime in liveness.intervals:
            key = lifetime.value
            leaves = ir_utils.leaves_of(placement.local_type_of(key.expr))
            if not leaves:
                continue
            held = leaves[key.index]
            quantity = next(
                (
                    (level, amount)
                    for level, amount in ir_utils.bytes_by_storage(held).items()
                    if facts.explicit(level) is not None
                ),
                None,
            )
            if quantity is not None:
                level, amount = quantity
                quantities[key] = level, amount * result_copies(key.expr) * partial_copies(
                    held, context.target.storage_owners
                )
            if any(key.expr is param for param in function.params):
                persistent.add(key)
        self.quantities, self.persistent = quantities, frozenset(persistent)
        super().__init__(root_function=function)
        for param in function.params:
            self.visit(param, context)
        self.visit_function_body(function, context)
        borrowed = tuple(
            value
            for value in self.by_value.values()
            if id(value.value.expr) not in self.context.scope.snapshot.definition.locations
        )
        aliases = tuple(value.aliases for value in borrowed)
        for path, (_, accesses) in context.scope.access.accesses.get("unit", {}).items():
            if not any(
                isinstance(site.expr.target, core.FunctionRef) and site.expr.target.name is None
                for site in path[:-1]
            ):
                continue
            for acc in accesses:
                if not getattr(acc.target.expr, "is_const", False):
                    continue
                held = acc.target.leaf(context.local_type_of(acc.target.expr))
                image = acc.relation.relation.range()
                rank = len(held.shape)
                prefix = image.dim(isl.dim_type.SET) - rank
                if prefix <= 0:
                    continue
                if prefix > 0:
                    image = image.project_out(isl.dim_type.SET, prefix, rank).flat_product(
                        dim.shape_to_isl_set(tuple(held.shape), {})
                    )
                image = image.project_out(isl.dim_type.PARAM, 0, image.dim(isl.dim_type.PARAM))
                self.instance_groups.append(
                    (
                        1,
                        (ReachedAddresses(acc.target, None, held.dtype, image, Precision.EXACT),),
                    )
                )
        self.tie_analysis(context)
        assert tuple(value.aliases for value in borrowed) == aliases, (
            "caller ties mutated callee aliases"
        )
        for resolved in context.scope.snapshot.resolved.allocations.values():
            for value in resolved.values:
                self.by_value.setdefault(value.value, value)
            imported = {value.value for value in resolved.values}
            self.windows.extend(window for window in resolved.windows if window.value in imported)
        self.member_groups = self.occupancy_index = None
        return self.summary()

    def add_value(self, lifetime, ctx, aliases=()):
        value = AllocationValue(lifetime, aliases)
        self.by_value[value.value] = value
        self.windows.append(value.lifetime)
        return value

    def allocate(self, node, ctx):
        if isinstance(node.type, types.TupleType) and result_copies(node) > 1:
            raise Error("allocation: staged tuple results are not supported")
        view = access.aliased_operand(node) if isinstance(node, expr.Call) else None
        for index, leaf in enumerate(ir_utils.leaves_of(node.type)):
            key = core.SSAValue(node, index)
            cached = ctx.scope.provenance.snapshot.get(key)
            source = None
            if cached is not None and cached.forward is not None:
                source = BufferAlias(
                    cached.forward, access.identity(len(leaf.shape)), AliasKind.FORWARD
                )
            elif view is not None and cached is not None and cached.root != key:
                source = storage_source(key, core.SSAValue(node.args[view]), ctx)
            forwarding = source is not None and source.target != key
            tgt = self.by_value.get(source.target) if forwarding else None
            quantity = self.quantities.get(key)
            if forwarding:
                if tgt is None:
                    continue
                level = tgt.lifetime.memory_level
                amount = tgt.lifetime.bytes
            elif quantity is not None:
                level, amount = quantity
                if level == "smem":
                    copies = result_copies(node)
                    size = max(
                        amount // copies,
                        storage_bytes(
                            leaf, topology_level=ctx.topology_level, topologies=ctx.topologies
                        ),
                    )
                    if copies > 1:
                        size = aligned(size, alignment_of(node))
                    amount = size * copies
            else:
                continue
            lifetime = replace(
                self.liveness.interval_of(key),
                memory_level=level,
                bytes=amount,
                persistent=key in self.persistent,
            )
            self.add_value(lifetime, ctx, (source,) if forwarding else ())
            if getattr(node, "is_const", False):
                held = ir_utils.leaves_of(ctx.local_type_of(node))[index]
                if isinstance(held, types.TensorType) and expr.is_literal_shape(held.shape):
                    image = dim.shape_to_isl_set(tuple(held.shape), {})
                    self.instance_groups.append(
                        (
                            1,
                            (ReachedAddresses(key, None, held.dtype, image, Precision.EXACT),),
                        )
                    )
        return None

    def visit_Var(self, node, ctx):
        return self.allocate(node, ctx)

    def visit_leaf_Tuple(self, node, operands, ctx):
        return self.allocate(node, ctx)

    def default_visit_leaf(self, node, operands, ctx):
        self.allocate(node, ctx)

    def visit_leaf_LoopRegion(self, node, operands, ctx):
        self.allocate(node, ctx)
        for phi, initial, yielded in node.carries():
            for formal, start, end in zip(
                core.ssa_leaves(phi),
                core.ssa_leaves(initial),
                core.ssa_leaves(yielded),
                strict=True,
            ):
                if formal not in self.by_value:
                    continue
                for holder, operand, kind in (
                    (formal, start, AliasKind.CARRY_INIT),
                    (end, formal, AliasKind.CARRY_YIELD),
                ):
                    if holder in self.by_value and operand in self.by_value:
                        self.candidates.append(
                            (
                                holder,
                                BufferAlias(operand, identity_of(self.by_value[holder]), kind),
                                node,
                            )
                        )

    def visit_Call(self, node, ctx):
        self.visit_operands(node, ctx)
        if ctx.callee is not None:
            self.call_storage(node, ctx)
        else:
            handler = getattr(self, f"visit_{type(node.target).__name__}", None)
            if handler is not None:
                handler(node, ctx)
                return
            self.allocate(node, ctx)
        self.in_place(node, ctx)

    def visit_TupleGetItem(self, node, ctx):
        cached = ctx.scope.provenance.snapshot.get(core.SSAValue(node))
        if cached is not None and cached.forward is not None:
            self.allocate(node, ctx)

    def visit_InsertSlice(self, node, ctx):
        self.allocate(node, ctx)
        value = self.by_value.get(core.SSAValue(node))
        if value is None:
            return
        destination = self.by_value.get(core.SSAValue(node.args[0]))
        if destination is not None:
            self.candidates.append(
                (
                    value.value,
                    BufferAlias(destination.value, identity_of(value), AliasKind.INSERT_SLICE),
                    None,
                )
            )
        update = self.by_value.get(core.SSAValue(node.args[1]))
        relations = ctx.scope.access.projected_relations(node, replace(ctx, topology_level=None))
        if update is not None and len(relations) > len(node.args):
            mapping = (
                relations[1]
                .aligned(relations[1].relation.reverse())
                .compose(relations[len(node.args)])
            )
            if mapping.relation.is_single_valued() and mapping.relation.is_injective():
                self.candidates.append(
                    (update.value, BufferAlias(value.value, mapping, AliasKind.WINDOW), node)
                )

    def in_place(self, node, ctx):
        for key, operand, _ in self.in_place_operands(node, ctx):
            value, tgt = self.by_value.get(key), self.by_value.get(operand.target)
            if value is not None and tgt is not None and not tgt.lifetime.persistent:
                self.candidates.append(
                    (
                        key,
                        replace(operand, relation=identity_of(value), kind=AliasKind.IN_PLACE),
                        None,
                    )
                )

    def call_storage(self, node, ctx):
        template = expr.get_metadata(ctx.callee, AllocationSummaryMetadata)
        if template is None:
            raise Error(f"allocation: missing summary for {ctx.callee.name!r}")
        local = {value.value: value for value in template.values}
        self.allocate(node, ctx)
        for leaf in core.ssa_leaves(node):
            returned = local.get(core.SSAValue(ctx.callee.body, leaf.index))
            if returned is not None:
                source = call_result_address(ctx, address_alias_of(returned, local))
                if source is not None:
                    self.candidates.append((leaf, source, None))
        resolved = expr.get_metadata(ctx.callee, AllocationSummaryMetadata, ctx=ctx)
        self.context.scope.keep_allocation(node, resolved)
        self.windows.extend(window for window in resolved.windows if window.value in self.by_value)


def build_interference_graph(values, windows, by_value):
    """Interference uses translated windows, never the local lifetime envelope."""
    intervals = defaultdict(list)
    for window in windows:
        if window.value in by_value:
            intervals[root_value(by_value[window.value], by_value)].append(window)
    return {
        (i, j)
        for i, j in combinations(range(len(values)), 2)
        if any(
            max(a.defined_at, b.defined_at) <= min(a.last_used_at, b.last_used_at)
            for a in intervals[values[i]]
            for b in intervals[values[j]]
        )
    }


def partial_copies(held: types.TensorType, owners: expr.StorageOwners) -> int:
    """Count Partial contributions not already separated by storage ownership."""
    layout = types.shard_layout_of(held.layout)
    if layout is None:
        return 1
    copies = 1
    for (level, _), reduction, extent in zip(
        types.axis_keys(layout.mesh),
        types.partial_reductions_by_axis(layout),
        types.flatten(layout.mesh.layout).shape,
    ):
        if reduction is None or owners.private(held.storage, level):
            continue
        amount = expr.static_dim_value(extent)
        if amount is None:
            raise Error(
                f"allocation: Partial copies at level {level!r} need a static extent, got {extent!r}"
            )
        copies *= amount
    return copies


def storage_bytes(held: types.TensorType, *, topology_level=None, topologies=()) -> int:
    """A unit's safe arena span, including holes and all reachable swizzle phases."""
    if not expr.is_literal_shape(held.shape):
        raise Error(f"allocation: unresolved physical shape {held.shape!r}")
    if any(extent == 0 for extent in held.shape):
        return 0

    def bounds(layout, low, high, shape=None):
        if isinstance(layout, types.ShardLayout):
            if topology_level is not None:
                shape = ir_utils.local_type_of(
                    replace(held, layout=layout),
                    topology_level=topology_level,
                    topologies=topologies,
                ).shape
            return bounds(layout.layout, low, high, shape)
        if isinstance(layout, types.ComposedLayout):
            if layout.outer is not None:
                low, high = bounds(layout.outer, low, high, shape)
            offset = expr.static_dim_value(layout.offset)
            if offset is None:
                raise Error(f"allocation: unresolved physical offset {layout.offset!r}")
            if layout.inner is None:
                return low + offset, high + offset
            return bounds(layout.inner, low + offset, high + offset)
        if isinstance(layout, types.Layout):
            flat = types.flatten(layout)
            if not all(isinstance(value, int) for value in (*flat.shape, *flat.strides)):
                raise Error(f"allocation: unresolved physical layout {layout!r}")
            contributions = tuple(
                (extent - 1) * stride
                for extent, stride in zip(
                    flat.shape if shape is None else shape, flat.strides, strict=True
                )
            )
            return sum(min(0, value) for value in contributions), sum(
                max(0, value) for value in contributions
            )
        if isinstance(layout, types.Swizzle):
            if layout.bits == 0:
                return low, high
            period = 1 << (layout.bits + layout.base + abs(layout.shift))
            return low // period * period, (high // period + 1) * period - 1
        raise Error(f"allocation: unsupported physical layout {layout!r}")

    low, high = bounds(held.layout, 0, prod(held.shape) - 1)
    if not isinstance(low, int) or not isinstance(high, int) or low < 0:
        raise Error(f"allocation: invalid physical address range [{low}, {high}]")
    return -(-((high + 1) * held.dtype.bit_width) // 8)


def alignment_of(value: expr.Expr) -> int:
    """Required byte alignment for one allocation."""
    try:
        widths = tuple(
            -(-tensor.dtype.bit_width // 8) for tensor in ir_utils.tensor_types(ir_utils.local_type_of(value.type))
        )
    except (TypeError, ValueError, NotImplementedError):
        widths = ()
    return max((16, *widths))


def aligned(value: int, alignment: int) -> int:
    """Round a byte offset up to its required alignment."""
    return -(-value // alignment) * alignment


def first_fit(
    order: list[int],
    values: tuple[AllocationValue, ...],
    interference: set[tuple[int, int]],
    alignments: tuple[int, ...],
    limit: int,
    pinned: dict[int, int],
    floor: int,
    sizes,
) -> tuple[tuple[int, ...], int] | None:
    """Place each root at its lowest aligned address left free by its conflicts."""
    placed: list[tuple[int, int, int]] = []
    addresses = [0] * len(values)
    peak = 0
    for index in order:
        size = sizes[values[index]]
        alignment = alignments[index]
        blocked = tuple(item for item in placed if tuple(sorted((index, item[0]))) in interference)
        pins = {pinned[index]} if index in pinned else set()
        lowest = aligned(floor, alignment)
        candidates = sorted(
            pins
            or {
                lowest,
                *(
                    aligned(address + held, alignment)
                    for _, address, held in blocked
                    if aligned(address + held, alignment) >= lowest
                ),
            }
        )
        address = next(
            (
                candidate
                for candidate in candidates
                if all(
                    candidate + size <= other_address or other_address + other_size <= candidate
                    for _, other_address, other_size in blocked
                )
            ),
            None,
        )
        if address is None or address + size > limit:
            return None
        addresses[index] = address
        placed.append((index, address, size))
        peak = max(peak, address + size)
    return tuple(addresses), peak


def satisfies(
    addresses: tuple[int, ...],
    values: tuple[AllocationValue, ...],
    interference: set[tuple[int, int]],
    alignments: tuple[int, ...],
    limit: int,
    pinned: dict[int, int],
    floor: int,
    sizes,
) -> bool:
    """Whether a seed meets every constraint the solver states about addresses."""
    for index, (address, item, alignment) in enumerate(
        zip(addresses, values, alignments, strict=True)
    ):
        if address % alignment or address + sizes[item] > limit:
            return False
        if index in pinned and address != pinned[index]:
            return False
        if index not in pinned and address < floor:
            return False
    return all(
        addresses[left] + sizes[values[left]] <= addresses[right]
        or addresses[right] + sizes[values[right]] <= addresses[left]
        for left, right in interference
    )


def calculate_starts(
    values: tuple[AllocationValue, ...],
    interference: set[tuple[int, int]],
    alignments: tuple[int, ...],
    limit: int,
    *,
    pinned: dict[int, int],
    floor: int,
    sizes,
    starts,
) -> tuple[tuple[int, ...], int] | None:
    """A complete aligned seed for independent values, or None.

    Values are placed first-fit by first definition -- the original
    seed, then again holding persistent values at their pinned addresses -- and
    largest first. Each seed is checked against every address constraint the
    solver states, and the most compact one that meets them all is kept, the
    earlier attempt winning a tie. None when none does: a heuristic that fails
    proves nothing about the model, so it must not bound the search.
    """
    indices = range(len(values))
    by_definition = sorted(indices, key=lambda i: (starts[values[i]], -sizes[values[i]]))
    largest_first = sorted(indices, key=lambda i: (-sizes[values[i]], starts[values[i]]))
    attempts = (
        (by_definition, {}, 0),
        (by_definition, pinned, floor),
        (largest_first, pinned, floor),
    )
    seeds = [
        seed
        for order, pins, lowest in attempts
        if (seed := first_fit(order, values, interference, alignments, limit, pins, lowest, sizes))
        is not None
        and satisfies(seed[0], values, interference, alignments, limit, pinned, floor, sizes)
    ]
    return min(seeds, key=lambda seed: seed[1]) if seeds else None


def solve_allocation(
    summary: AllocationSummaryMetadata,
    ctx: AnalyzeContext,
    options: SolverOptions,
    *,
    memory_level: str,
) -> AllocationResult:
    """Return the first feasible whole-function placement for one level."""
    summary = summary.resolve(ctx)
    groups = defaultdict(list)
    for value in summary.values:
        groups[summary.root(value)].append(value)
    values = tuple(value for value in groups if value.lifetime.memory_level == memory_level)
    sizes = {value: summary.sizes[value] for value in values}
    persistent = {
        value for value in values if any(member.lifetime.persistent for member in groups[value])
    }
    if not values:
        return AllocationResult(0, "optimal")

    alignments = tuple(item.alignment for item in values)
    largest = max(sizes[item] for item in values)
    limit = sum(
        aligned(sizes[item], alignment) for item, alignment in zip(values, alignments, strict=True)
    )
    model = cp_model.CpModel()
    peak = model.new_int_var(largest, limit, f"{memory_level}_peak")
    addresses = tuple(model.new_int_var(0, limit, f"address_{i}") for i in range(len(values)))
    for index, (address, item, alignment) in enumerate(
        zip(addresses, values, alignments, strict=True)
    ):
        model.add(address + sizes[item] <= peak)
        model.add_modulo_equality(0, address, alignment).with_name(f"align_{index}")

    persistent_end = 0
    pinned: dict[int, int] = {}
    for index, (address, item, alignment) in enumerate(
        zip(addresses, values, alignments, strict=True)
    ):
        if item not in persistent:
            continue
        persistent_end = aligned(persistent_end, alignment)
        offset = persistent_end
        persistent_end += sizes[item]
        model.add(address == offset)
        pinned[index] = offset
    for address, item in zip(addresses, values, strict=True):
        if item not in persistent:
            model.add(address >= persistent_end)

    interference = build_interference_graph(values, summary.windows, summary.by_value)
    order_choices: dict[tuple[int, int], tuple[cp_model.IntVar, cp_model.IntVar]] = {}
    for left, right in sorted(interference):
        if values[left] in persistent or values[right] in persistent:
            continue
        left_before = model.new_bool_var(f"before_{left}_{right}")
        right_before = model.new_bool_var(f"before_{right}_{left}")
        order_choices[(left, right)] = (left_before, right_before)
        model.add(addresses[left] + sizes[values[left]] <= addresses[right]).only_enforce_if(
            left_before
        )
        model.add(addresses[right] + sizes[values[right]] <= addresses[left]).only_enforce_if(
            right_before
        )
        model.add_bool_or(left_before, right_before)

    seed = calculate_starts(
        values,
        interference,
        alignments,
        limit,
        pinned=pinned,
        floor=persistent_end,
        sizes=sizes,
        starts={
            value: min(
                window.defined_at
                for window in summary.windows
                if summary.root(window.value) is value
            )
            for value in values
        },
    )
    if seed is not None:
        address_hints, peak_hint = seed
        model.add(peak <= peak_hint)
        for address, suggested in zip(addresses, address_hints, strict=True):
            model.add_hint(address, suggested)
        for (left, right), (left_before, right_before) in order_choices.items():
            left_end = address_hints[left] + sizes[values[left]]
            right_end = address_hints[right] + sizes[values[right]]
            model.add_hint(left_before, int(left_end <= address_hints[right]))
            model.add_hint(right_before, int(right_end <= address_hints[left]))
        model.add_hint(peak, peak_hint)

    solver = cp_model.CpSolver()
    solver.parameters.merge_text_format(
        f"max_time_in_seconds: {float(options.timeout_seconds)!r} "
        f"num_search_workers: {int(options.workers)} "
        f"random_seed: {int(options.random_seed)} "
        "stop_after_first_solution: true search_branching: HINT_SEARCH"
    )
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        status_name = solver.status_name(status).lower()
        raise Error(
            f"allocation: no feasible {memory_level} placement was found "
            f"({status_name}); {len(values)} roots conflict with "
            f"{len(interference)} interference constraints"
        )
    actual_peak = max(
        solver.value(address) + sizes[item] for address, item in zip(addresses, values, strict=True)
    )
    for left, right in interference:
        left_offset, right_offset = solver.value(addresses[left]), solver.value(addresses[right])
        if not (
            left_offset + sizes[values[left]] <= right_offset
            or right_offset + sizes[values[right]] <= left_offset
        ):
            a, b = values[left].lifetime, values[right].lifetime
            raise Error(
                f"allocation: overlapping {memory_level} roots: {expr.value_label(a.value.expr)} [{a.defined_at},{a.last_used_at}] at {left_offset}; {expr.value_label(b.value.expr)} [{b.defined_at},{b.last_used_at}] at {right_offset}"
            )
    return AllocationResult(
        actual_peak,
        "feasible",
        {item: solver.value(address) for item, address in zip(values, addresses, strict=True)},
    )


def covers_result(output_coverage, full_result, outputs, iteration_domain, result_box):
    """A write covers the complete result for every stated iteration."""
    try:
        return output_coverage.is_equal(full_result) or outputs.range().is_equal(
            result_box.intersect_params(iteration_domain.params())
        )
    except isl.Error:
        return False


__all__ = [
    "AllocationConstraintVisitor",
    "AllocationSummaryMetadata",
    "AllocationValue",
    "aligned",
    "covers_result",
    "solve_allocation",
    "storage_bytes",
]
