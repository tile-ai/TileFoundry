"""Internal values and solver results for physical memory placement."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from itertools import combinations
from math import prod
from typing import Protocol

import isl
from ortools.sat.python import cp_model

from tilefoundry.ir.core import (
    Call,
    Constant,
    Expr,
    IRMetadata,
    get_metadata,
    value_label,
)
from tilefoundry.ir.core.expr import SSAValue
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.sharding.mesh_coord import MeshCoord
from tilefoundry.ir.isl_interop import shape_to_isl_set
from tilefoundry.ir.types import (
    ComposedLayout,
    Layout,
    ShardLayout,
    Swizzle,
    TensorType,
    TupleType,
)
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.layout import flatten, resolve_layout
from tilefoundry.ir.types.mesh import levels
from tilefoundry.ir.types.utils import (
    is_literal_shape,
    leaves_of,
    local_type_of,
    static_dim_value,
    tensor_types,
)
from tilefoundry.utils.isl_utils import equates
from tilefoundry.visitor_registry.access_relation import AccessRelation, identity_access
from tilefoundry.visitor_registry.buffer_alias import aliased_operand
from tilefoundry.visitor_registry.call_frame import CallFrame
from tilefoundry.visitor_registry.valuation import ValuationVisitor

from .access import (
    SEMANTIC_KINDS,
    STRUCTURAL_KINDS,
    AliasKind,
    BufferAlias,
    eliminate_parameters,
)
from .errors import AnalysisError
from .footprint import ReachedAddresses
from .iteration_scope import walk_scopes
from .liveness import LivenessMetadata, RegionInterval, result_copies
from .metadata import Footprint, ValueLifetime, shifted
from .precision import AnalysisPrecision
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
    def value(self) -> SSAValue:
        return self.lifetime.value

    @property
    def alignment(self) -> int:
        return alignment_of(self.value.expr)


def identity_of(value: AllocationValue) -> AccessRelation:
    held = local_type_of(leaves_of(value.value.expr.type)[value.value.index])
    box = shape_to_isl_set(tuple(held.shape), {})
    return AccessRelation(isl.map.identity(box.get_space().map_from_set()).intersect_domain(box))


def _edge(value, by_value, *, storage_path=False):
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


def root(value: AllocationValue, by_value, *, storage_path=False) -> AllocationValue:
    """Follow accepted value links without modifying or compressing them."""
    seen = set()
    while True:
        if value in seen:
            raise AnalysisError("allocation: storage links contain a cycle")
        seen.add(value)
        edge = _edge(value, by_value, storage_path=storage_path)
        if edge is None:
            return value
        value = by_value[edge.target]


def address_alias(value, by_value, *, alias=None, precisions=None):
    """Compose accepted address links without merging SSA values or lifetimes."""
    structural = alias is not None
    alias = alias or BufferAlias(value.value, identity_of(value), AliasKind.FORWARD)
    seen = set()
    while alias.target in by_value:
        current = by_value[alias.target]
        edge = _edge(current, by_value)
        if edge is None or (structural and edge.kind not in STRUCTURAL_KINDS):
            return alias
        if precisions is not None:
            precision = precisions.get(current.value, AnalysisPrecision.UNKNOWN)
            if precision is not AnalysisPrecision.EXACT:
                raise AnalysisError(
                    f"non-EXACT address for {current.lifetime.binding}: {edge.relation.relation}"
                )
        if current in seen:
            raise AnalysisError("allocation: storage links contain a cycle")
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
class AllocationSummaryMetadata(IRMetadata):
    """Definition facts interpreted at a leaf call, never an execution context."""

    values: tuple[AllocationValue, ...]
    calls: tuple[tuple[Call, int], ...] = ()
    in_place: tuple[tuple[int, BufferAlias], ...] = ()
    address_decisions: tuple[tuple[SSAValue, BufferAlias, SSAValue], ...] = ()


@dataclass(frozen=True)
class AllocationResult:
    peak_bytes: int
    solver_status: str
    offsets: dict[AllocationValue, int] = field(default_factory=dict)


@dataclass(kw_only=True)
class AllocationModel(AnalyzeContext):
    """One definition's collection; imported values are read-only templates."""

    address_decisions: list[tuple[SSAValue, BufferAlias, SSAValue]] = field(default_factory=list)

    liveness: LivenessMetadata
    quantities: dict[SSAValue, tuple[str, int]]
    persistent: frozenset[SSAValue]
    by_value: dict[SSAValue, AllocationValue] = field(default_factory=dict)
    windows: list[ValueLifetime] = field(default_factory=list)
    candidates: list[tuple[SSAValue, BufferAlias, LoopRegion | None]] = field(default_factory=list)
    locations: dict[int, AnalyzeContext] = field(default_factory=dict)
    instance_groups: list = field(default_factory=list)
    regions: list[RegionInterval] = field(default_factory=list)
    calls: list = field(default_factory=list)
    expanded_calls: dict = field(default_factory=dict)
    _groups: dict | None = field(default=None, init=False, repr=False)
    _occupancy: dict | None = field(default=None, init=False, repr=False)

    def occupancy(self, value):
        """Native lifetime plus translated occupancy, without changing either."""
        if self._occupancy is None:
            self._occupancy = defaultdict(list)
            for window in self.windows:
                self._occupancy[window.value].append(window)
        return tuple(self._occupancy[value.value])

    def span(self, values):
        windows = tuple(window for value in values for window in self.occupancy(value))
        return min(w.defined_at for w in windows), max(w.last_used_at for w in windows)

    def members(self, value):
        if self._groups is None:
            self._groups = defaultdict(list)
            for item in self.by_value.values():
                self._groups[root(item, self.by_value)].append(item)
        return tuple(self._groups[root(value, self.by_value)])

    def roots(self):
        return tuple(dict.fromkeys(root(value, self.by_value) for value in self.by_value.values()))

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


def _bound_relation(model, relation, scope, held):
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
    return bound if precision is AnalysisPrecision.EXACT else None


def _storage_source(value, operand, ctx):
    source = ctx.scope.provenance.lookup(value).narrow
    actual = ctx.scope.provenance.lookup(operand).narrow
    inverse = actual._result(actual.relation.reverse())
    return BufferAlias(operand, source.compose(inverse), AliasKind.VIEW)


def _coordinates(model, value, scope, *, override=None, storage_path=False, seen=()):
    if value in seen:
        return None
    edge = (
        override[1]
        if override is not None and override[0] is value
        else _edge(value, model.by_value, storage_path=storage_path)
    )
    relation = _bound_relation(
        model, identity_of(value) if edge is None else edge.relation, scope, value.value.leaf()
    )
    if relation is None:
        return None
    if edge is None:
        return value, relation
    source = _coordinates(
        model,
        model.by_value[edge.target],
        scope,
        override=override,
        storage_path=storage_path,
        seen=(*seen, value),
    )
    return None if source is None else (source[0], relation.compose(source[1]))


def _visible_at(model, value, producer):
    context = model.locations[id(producer.value.expr)]
    scope = context.scope
    if isinstance(value, (int, Constant, DimVar)):
        return True
    if isinstance(value, Call) and isinstance(value.target, MeshCoord):
        mesh = scope.mesh
        if mesh is None:
            return False
        available = {getattr(held, "name", held) for held in mesh.topologies}
        axis = static_dim_value(value.args[0])
        offset = 0
        for topology, layout in zip(
            value.target.mesh.topologies, levels(value.target.mesh), strict=True
        ):
            width = len(flatten(layout.shape))
            if axis is None or offset <= axis < offset + width:
                if getattr(topology, "name", topology) not in available:
                    return False
            offset += width
        return True
    interval = model.liveness.interval_of(SSAValue(value))
    location = model.locations.get(id(value))
    return (
        location is not None
        and any(location.owner is c.owner for c in context.ancestors())
        and interval is not None
        and interval.defined_at < producer.lifetime.defined_at
    )


def _landing(model, holder, alias, *, window=False):
    try:
        representative = root(holder, model.by_value, storage_path=window)
        scope = model.locations[id(representative.value.expr)].scope
        source = _coordinates(model, holder, scope, storage_path=window)
        target = _bound_relation(model, alias.relation, scope, alias.target.leaf())
        if source is None or target is None:
            return None
        relation = source[1]._result(source[1].relation.reverse()).compose(target)
        if window:
            held = local_type_of(representative.value.leaf())
            expected = shape_to_isl_set(tuple(held.shape), {}).intersect_params(
                relation.relation.params()
            )
            if (
                not relation.relation.domain().is_equal(expected)
                or relation.relation.dim(isl.dim_type.IN) != relation.relation.dim(isl.dim_type.OUT)
                or relation.relation.is_translation() != 1
            ):
                return None
            if not all(
                _visible_at(model, value, representative) for value in relation.values.values()
            ):
                return None
        return representative, replace(alias, relation=relation)
    except (isl.Error, AnalysisError):
        return None


def _mapped_pointwise(model, left, right, start, landing):
    touching = {
        window
        for member in right
        for window in model.occupancy(member)
        if window is not member.lifetime and window.last_used_at == start
    }
    covered, proved = set(), set()
    for output in left:
        context = model.locations.get(id(output.value.expr))
        if context is None or id(output.value.expr) not in model.expanded_calls:
            continue
        written = _coordinates(model, output, context.scope, override=landing)
        for key, actual, windows in in_place_operands(output.value.expr, context):
            if key != output.value:
                continue
            member = model.by_value.get(actual.target)
            read = None if member is None else _coordinates(model, member, context.scope)
            if read is None or written is None or read[0] is not written[0]:
                continue
            reached = actual.relation.compose(read[1]).relation
            if reached.is_equal(written[1].relation):
                covered.update(touching & windows)
                proved.add(member.value)
    return proved if touching <= covered else None


def _pointwise(model, left, right, start, landing, use_events):
    proved = _mapped_pointwise(model, left, right, start, landing)
    if proved is None:
        return False
    read = {
        member.value
        for member in right
        if any(use.at == start for use in use_events.get(member.value, ()))
    }
    for producer in left:
        if producer.lifetime.defined_at != start or not isinstance(producer.value.expr, Call):
            continue
        node = producer.value.expr
        context = model.locations[id(node)]
        scope = context.scope
        relations = _local_relations(node, scope)
        outputs = relations[len(node.args) :]
        if not outputs:
            continue
        positioned = _coordinates(model, producer, scope, override=landing)
        if positioned is None:
            continue
        written = outputs[producer.value.index].compose(positioned[1]).relation
        for holder, alias, _loop in model.candidates:
            if holder == producer.value and alias.kind is AliasKind.INSERT_SLICE:
                for index, operand in enumerate(node.args):
                    if alias.target == SSAValue(operand):
                        written = written.union(relations[index].compose(positioned[1]).relation)
        for member in right:
            candidates = []
            for index, operand in enumerate(node.args):
                for leaf in SSAValue.leaves(operand):
                    value = model.by_value.get(leaf)
                    if value is None:
                        continue
                    reached = _coordinates(model, value, scope)
                    if reached is not None and reached[0] is positioned[0]:
                        candidates.append(relations[index].compose(reached[1]).relation)
            if candidates and all(
                image.is_equal(written.intersect_domain(image.domain())) for image in candidates
            ):
                proved.add(member.value)
    return read <= proved


def tie_analysis(model):
    """All ties read occupancy; global interference keeps individual windows."""
    use_events = defaultdict(list)
    for use in model.liveness.uses:
        use_events[use.value].append(use)
    local = {item.value for item in model.liveness.intervals}
    for holder, alias, loop in sorted(
        model.candidates,
        key=lambda item: (
            item[1].kind not in SEMANTIC_KINDS,
            item[1].kind,
            model.by_value[item[0]].lifetime.defined_at,
        ),
    ):
        value = model.by_value[holder]
        target = model.by_value.get(alias.target)
        if target is None or holder not in local:
            continue
        a = root(value, model.by_value)
        b = root(target, model.by_value)
        if a is b:
            continue
        if a.value not in local:
            continue
        left = model.members(a)
        right = model.members(b)
        semantic = alias.kind in SEMANTIC_KINDS
        if not semantic and (
            model.is_persistent(b)
            or any(
                any(item.value.expr is param for param in model.frame.function.params)
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
        if a.lifetime.memory_level != b.lifetime.memory_level or model.size(a) > model.size(b):
            continue
        start, _ = model.span(left)
        _, end = model.span(right)
        window = alias.kind is AliasKind.WINDOW
        if window:
            node = loop
            destination = model.by_value.get(SSAValue(node.args[0]))
            if destination is None or root(destination, model.by_value) is not b:
                continue
            event = model.liveness.interval_of(SSAValue(node)).defined_at
            if model.is_persistent(a) or end < 0 or model.span(left)[1] > event:
                continue
            if any(
                start < use.at < event for item in right for use in use_events.get(item.value, ())
            ):
                continue
            end = start
        elif alias.kind is AliasKind.CARRY_YIELD:
            phi_event, backedge = model.liveness.loop_windows[id(loop)]
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
                        for window in model.occupancy(item)
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
                for window in model.occupancy(item)
                if phi_event < window.defined_at < backedge
            )
        if not window and not semantic and end > start:
            continue
        landed = _landing(model, value, alias, window=window)
        physical = landed if not window or landed is None else _landing(model, value, alias)
        if physical is None or landed is None:
            continue
        new_root, edge = physical
        if alias.kind is AliasKind.IN_PLACE:
            scope = model.locations[id(new_root.value.expr)].scope
            positioned = _coordinates(model, new_root, scope, override=physical)
            if positioned is None or not is_zero_offset(positioned[1].relation):
                continue
        if (
            not semantic
            and end == start
            and not _pointwise(model, left, right, start, physical, use_events)
        ):
            continue
        new_root.aliases += (edge,)
        model._groups = None
        model.address_decisions.append((holder, alias, new_root.value))
        if value is not new_root and alias not in value.aliases:
            value.aliases += (alias,)
        if window and landed[0] is not new_root:
            landed[0].aliases += (landed[1],)
    for holder, alias, _loop in model.candidates:
        if alias.kind in (AliasKind.CARRY_INIT, AliasKind.CARRY_YIELD, AliasKind.INSERT_SLICE):
            value = model.by_value[holder]
            target = model.by_value.get(alias.target)
            if (
                target is not None
                and root(value, model.by_value) is root(target, model.by_value)
                and alias not in value.aliases
            ):
                value.aliases += (alias,)


def _local_relations(node, scope):
    key = (id(node), "storage", None)
    if key in scope.access.projections:
        return scope.access.projections[key]
    scope.access.projections[key] = ()
    stated = scope.access.relations.get(id(node))
    if stated is None or len(stated[1]) != len(node.args) + 1:
        return ()
    result = []
    for value, relation in zip((*node.args, node), stated[1], strict=True):
        held = local_type_of(value.type)
        relation, precision = eliminate_parameters(
            relation, scope, held, narrow=True, formals=scope.root.owner.params
        )
        if precision is not AnalysisPrecision.EXACT:
            return ()
        if isinstance(held, TensorType) and is_literal_shape(held.shape):
            relation = relation._result(
                relation.relation.intersect_range(shape_to_isl_set(tuple(held.shape), {}))
            )
        result.append(relation)
    scope.access.projections[key] = tuple(result)
    return scope.access.projections[key]


def operand_to_result_relation(node, scope, insert_accesses=None):
    """Read all operation boundaries in the same iteration parameter space."""
    relations = _local_relations(node, scope)
    outputs = relations[len(node.args) :]
    if len(outputs) != 1:
        return ()
    written = outputs[0]
    leaves = SSAValue.leaves(node)
    if len(leaves) != 1 or not is_literal_shape(leaves[0].leaf().shape):
        return ()
    box = shape_to_isl_set(tuple(local_type_of(leaves[0].leaf()).shape), {})
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
        mapped = read._result(read.relation.reverse()).compose(written)
        if mapped.relation.is_single_valued() and mapped.relation.is_injective():
            for value in SSAValue.leaves(operand):
                if value.index == 0:
                    result.append((value, mapped))
    return tuple(result)


def is_zero_offset(relation: isl.map) -> bool:
    """Whether every result coordinate equals the operand coordinate."""
    output_dims = relation.dim(isl.dim_type.OUT)
    loop_dims = relation.dim(isl.dim_type.IN) - output_dims
    if loop_dims < 0:
        return False
    try:
        return all(
            equates(relation, output_axis, loop_dims + output_axis)
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
    return replace(alias, target=SSAValue(argument, alias.target.index), relation=relation)


def in_place_operands(node, ctx):
    frame = ctx.callee_frame(node)
    if frame is None:
        for operand, relation in operand_to_result_relation(node, ctx.scope):
            if is_zero_offset(relation.relation):
                yield (
                    SSAValue(node),
                    BufferAlias(operand, relation, AliasKind.IN_PLACE),
                    frozenset(),
                )
        return
    template = get_metadata(frame.function, AllocationSummaryMetadata)
    windows = ctx.expanded_calls.get(id(node), ((), (), ()))[1]
    for index, capability in template.in_place:
        actual = call_result_address(ctx, capability)
        if actual is not None and is_zero_offset(actual.relation.relation):
            yield (
                SSAValue(node, index),
                actual,
                frozenset(w for w in windows if w.value == actual.target),
            )


def expand(summary, function, origin, boundary, context):
    """Expand sparse definition records at a read boundary without mutating templates."""
    local = {value.value: value for value in summary.values}
    mapped = {root(local[key], local): value for key, value in boundary.items() if key in local}
    values = tuple(value for value in summary.values if root(value, local) not in mapped)
    windows = []
    for item in summary.values:
        target = mapped.get(root(item, local))
        windows.append(
            replace(
                shifted(item.lifetime, origin),
                value=target or item.value,
                persistent=item.lifetime.persistent and target is None,
            )
        )
    liveness = get_metadata(function, LivenessMetadata)
    regions = tuple(shifted(region, origin) for region in liveness.regions)
    for call, start in summary.calls:
        frame = context.callee_frame(call)
        child = get_metadata(frame.function, AllocationSummaryMetadata)
        child_context = replace(context, frame=frame, scope=context.definitions[id(frame.function)])
        formals = tuple(
            p for p in frame.function.params if isinstance(call.target, Function) or not p.is_const
        )
        translated = {
            SSAValue(param, leaf.index): boundary.get(leaf, leaf)
            for param, arg in zip(formals, call.args, strict=True)
            for leaf in SSAValue.leaves(arg)
        }
        translated.update(
            {
                SSAValue(frame.function.body, leaf.index): boundary.get(leaf, leaf)
                for leaf in SSAValue.leaves(call)
            }
        )
        inner_values, inner_windows, inner_regions = expand(
            child, frame.function, origin + start, translated, child_context
        )
        values += inner_values
        windows.extend(inner_windows)
        regions += inner_regions
    return values, tuple(windows), regions


class AllocationConstraintVisitor(ValuationVisitor[None]):
    """Collect native values and interpret called templates at leaf Calls."""

    def __init__(self, *, root_function, context):
        self.locations = {}
        super().__init__(root_function=root_function, contexts=self.locations)

    def add_value(self, lifetime, ctx, aliases=()):
        value = AllocationValue(lifetime, aliases)
        ctx.by_value[value.value] = value
        ctx.windows.append(value.lifetime)
        return value

    def allocate(self, node, ctx):
        if isinstance(node.type, TupleType) and result_copies(node) > 1:
            raise AnalysisError("allocation: staged tuple results are not supported")
        view = aliased_operand(node) if isinstance(node, Call) else None
        for index, leaf in enumerate(leaves_of(node.type)):
            key = SSAValue(node, index)
            cached = ctx.scope.provenance.snapshot.get(key)
            source = None
            if cached is not None and cached.forward is not None:
                source = BufferAlias(
                    cached.forward, identity_access(len(leaf.shape)), AliasKind.FORWARD
                )
            elif view is not None and cached is not None and cached.root != key:
                source = _storage_source(key, SSAValue(node.args[view]), ctx)
            forwarding = source is not None and source.target != key
            target = ctx.by_value.get(source.target) if forwarding else None
            quantity = ctx.quantities.get(key)
            if forwarding:
                if target is None:
                    continue
                level = target.lifetime.memory_level
                amount = target.lifetime.bytes
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
                ctx.liveness.interval_of(key),
                memory_level=level,
                bytes=amount,
                persistent=key in ctx.persistent,
                binding=(value_label(node) or "<value>")
                + (f"[{index}]" if isinstance(node.type, TupleType) else ""),
            )
            self.add_value(lifetime, ctx, (source,) if forwarding else ())
            if getattr(node, "is_const", False):
                held = leaves_of(ctx.local_type_of(node))[index]
                if isinstance(held, TensorType) and is_literal_shape(held.shape):
                    image = shape_to_isl_set(tuple(held.shape), {})
                    ctx.instance_groups.append(
                        (
                            1,
                            (
                                ReachedAddresses(
                                    key, None, held.dtype, image, AnalysisPrecision.EXACT
                                ),
                            ),
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
                SSAValue.leaves(phi),
                SSAValue.leaves(initial),
                SSAValue.leaves(yielded),
                strict=True,
            ):
                if formal not in ctx.by_value:
                    continue
                for holder, operand, kind in (
                    (formal, start, AliasKind.CARRY_INIT),
                    (end, formal, AliasKind.CARRY_YIELD),
                ):
                    if holder in ctx.by_value and operand in ctx.by_value:
                        ctx.candidates.append(
                            (
                                holder,
                                BufferAlias(operand, identity_of(ctx.by_value[holder]), kind),
                                node,
                            )
                        )

    def visit_Call(self, node, ctx):
        self.visit_operands(node, ctx)
        frame = ctx.callee_frame(node)
        if frame is not None:
            self.call_storage(node, frame, ctx)
        else:
            handler = getattr(self, f"visit_{type(node.target).__name__}", None)
            if handler is not None:
                handler(node, ctx)
                return
            self.allocate(node, ctx)
        self.in_place(node, ctx)

    def visit_TupleGetItem(self, node, ctx):
        cached = ctx.scope.provenance.snapshot.get(SSAValue(node))
        if cached is not None and cached.forward is not None:
            self.allocate(node, ctx)

    def visit_InsertSlice(self, node, ctx):
        self.allocate(node, ctx)
        value = ctx.by_value.get(SSAValue(node))
        if value is None:
            return
        destination = ctx.by_value.get(SSAValue(node.args[0]))
        if destination is not None:
            ctx.candidates.append(
                (
                    value.value,
                    BufferAlias(destination.value, identity_of(value), AliasKind.INSERT_SLICE),
                    None,
                )
            )
        update = ctx.by_value.get(SSAValue(node.args[1]))
        relations = ctx.scope.access.projected_relations(node, replace(ctx, topology_level=None))
        if update is not None and len(relations) > len(node.args):
            mapping = (
                relations[1]
                ._result(relations[1].relation.reverse())
                .compose(relations[len(node.args)])
            )
            if mapping.relation.is_single_valued() and mapping.relation.is_injective():
                ctx.candidates.append(
                    (update.value, BufferAlias(value.value, mapping, AliasKind.WINDOW), node)
                )

    def in_place(self, node, ctx):
        for key, operand, _windows in in_place_operands(node, ctx):
            value, target = ctx.by_value.get(key), ctx.by_value.get(operand.target)
            if value is not None and target is not None and not target.lifetime.persistent:
                ctx.candidates.append(
                    (
                        key,
                        replace(operand, relation=identity_of(value), kind=AliasKind.IN_PLACE),
                        None,
                    )
                )

    def call_storage(self, node, callee_frame, ctx):
        template = get_metadata(callee_frame.function, AllocationSummaryMetadata)
        if template is None:
            raise AnalysisError(f"allocation: missing summary for {callee_frame.function.name!r}")
        local = {value.value: value for value in template.values}
        self.allocate(node, ctx)
        for leaf in SSAValue.leaves(node):
            returned = local.get(SSAValue(callee_frame.function.body, leaf.index))
            if returned is not None:
                source = call_result_address(ctx, address_alias(returned, local))
                if source is not None:
                    ctx.candidates.append((leaf, source, None))
        template_liveness = get_metadata(callee_frame.function, LivenessMetadata)
        origin = (
            ctx.liveness.interval_of(SSAValue(node)).defined_at - template_liveness.timeline_end - 1
        )
        ctx.calls.append((node, origin))
        bound = ctx.callee_scope
        boundary = {
            SSAValue(param, leaf.index): leaf
            for param in callee_frame.function.params
            if id(param) in bound.provenance.arguments
            for leaf in SSAValue.leaves(bound.provenance.arguments[id(param)])
        }
        boundary.update(
            {
                SSAValue(callee_frame.function.body, leaf.index): leaf
                for leaf in SSAValue.leaves(node)
            }
        )
        expanded = expand(
            template,
            callee_frame.function,
            origin,
            boundary,
            replace(ctx, frame=callee_frame, scope=bound),
        )
        ctx.expanded_calls[id(node)] = expanded
        ctx.windows.extend(window for window in expanded[1] if window.value in ctx.by_value)


def collect_allocation(function, context, liveness, quantities, persistent):
    model = AllocationModel(
        **vars(context), liveness=liveness, quantities=quantities, persistent=persistent
    )
    visitor = AllocationConstraintVisitor(root_function=function, context=model)
    for param in function.params:
        visitor.visit(param, model)
    visitor.visit_function_body(function, model)
    model.locations.update(visitor.locations)
    model.regions.extend(liveness.regions)
    borrowed = tuple(
        value for value in model.by_value.values() if id(value.value.expr) not in model.locations
    )
    aliases = tuple(value.aliases for value in borrowed)
    for scope in walk_scopes(context.scope):
        for path, (_call, accesses) in scope.access.accesses.get("unit", {}).items():
            if not any(CallFrame.module_slot(site.expr)[0] is not None for site in path[:-1]):
                continue
            for access in accesses:
                if not getattr(access.target.expr, "is_const", False):
                    continue
                held = access.target.leaf(context.local_type_of(access.target.expr))
                image = access.relation.relation.range()
                rank = len(held.shape)
                prefix = image.dim(isl.dim_type.SET) - rank
                if prefix <= 0:
                    continue
                if prefix > 0:
                    image = image.project_out(isl.dim_type.SET, prefix, rank).flat_product(
                        shape_to_isl_set(tuple(held.shape), {})
                    )
                image = image.project_out(isl.dim_type.PARAM, 0, image.dim(isl.dim_type.PARAM))
                model.instance_groups.append(
                    (
                        1,
                        (
                            ReachedAddresses(
                                access.target, None, held.dtype, image, AnalysisPrecision.EXACT
                            ),
                        ),
                    )
                )
    tie_analysis(model)
    assert tuple(value.aliases for value in borrowed) == aliases, (
        "caller ties mutated callee aliases"
    )
    return model


def summary_of(model):
    capabilities = []
    for leaf in SSAValue.leaves(model.frame.function.body):
        value = model.by_value.get(leaf)
        if value is None:
            continue
        for member in model.members(value):
            context = model.locations.get(id(member.value.expr))
            if context is None or not isinstance(member.value.expr, Call):
                continue
            for operand, relation in operand_to_result_relation(member.value.expr, context.scope):
                actual = model.by_value.get(operand)
                if actual is None:
                    continue
                address = address_alias(actual, model.by_value)
                relation = relation.compose(address.relation)
                if (
                    any(address.target.expr is param for param in model.frame.function.params)
                    and is_zero_offset(relation.relation)
                    and model.span(model.members(actual))[1] <= model.span(model.members(value))[0]
                ):
                    capabilities.append(
                        (leaf.index, BufferAlias(address.target, relation, AliasKind.IN_PLACE))
                    )
    return AllocationSummaryMetadata(
        tuple(model.by_value.values()),
        tuple(model.calls),
        tuple(capabilities),
        tuple(model.address_decisions),
    )


def build_interference_graph(values, windows, by_value):
    """Interference uses translated windows, never the local lifetime envelope."""
    intervals = defaultdict(list)
    for window in windows:
        if window.value in by_value:
            intervals[root(by_value[window.value], by_value)].append(window)
    return {
        (i, j)
        for i, j in combinations(range(len(values)), 2)
        if any(
            max(a.defined_at, b.defined_at) <= min(a.last_used_at, b.last_used_at)
            for a in intervals[values[i]]
            for b in intervals[values[j]]
        )
    }


def storage_bytes(held: TensorType, *, topology_level=None, topologies=()) -> int:
    """A unit's safe arena span, including holes and all reachable swizzle phases."""
    if not is_literal_shape(held.shape):
        raise AnalysisError(f"allocation: unresolved physical shape {held.shape!r}")
    if any(extent == 0 for extent in held.shape):
        return 0

    def bounds(layout, low, high, shape=None):
        if layout is None:
            return low, high
        if isinstance(layout, ShardLayout):
            if topology_level is not None:
                shape = local_type_of(
                    replace(held, layout=layout),
                    topology_level=topology_level,
                    topologies=topologies,
                ).shape
            return bounds(layout.layout, low, high, shape)
        if isinstance(layout, ComposedLayout):
            low, high = bounds(layout.outer, low, high, shape)
            offset = static_dim_value(layout.offset)
            if offset is None:
                raise AnalysisError(f"allocation: unresolved physical offset {layout.offset!r}")
            return bounds(layout.inner, low + offset, high + offset)
        if isinstance(layout, Layout):
            flat = flatten(resolve_layout(layout, layout.shape))
            if not all(isinstance(value, int) for value in (*flat.shape, *flat.strides)):
                raise AnalysisError(f"allocation: unresolved physical layout {layout!r}")
            contributions = tuple(
                (extent - 1) * stride
                for extent, stride in zip(
                    flat.shape if shape is None else shape, flat.strides, strict=True
                )
            )
            return sum(min(0, value) for value in contributions), sum(
                max(0, value) for value in contributions
            )
        if isinstance(layout, Swizzle):
            if layout.bits == 0:
                return low, high
            period = 1 << (layout.bits + layout.base + abs(layout.shift))
            return low // period * period, (high // period + 1) * period - 1
        raise AnalysisError(f"allocation: unsupported physical layout {layout!r}")

    low, high = bounds(held.layout, 0, prod(held.shape) - 1)
    if not isinstance(low, int) or not isinstance(high, int) or low < 0:
        raise AnalysisError(f"allocation: invalid physical address range [{low}, {high}]")
    return -(-((high + 1) * held.dtype.bit_width) // 8)


def alignment_of(value: Expr) -> int:
    """Required byte alignment for one allocation."""
    try:
        widths = tuple(
            -(-tensor.dtype.bit_width // 8) for tensor in tensor_types(local_type_of(value.type))
        )
    except (TypeError, ValueError, NotImplementedError):
        widths = ()
    return max((16, *widths))


def aligned(value: int, alignment: int) -> int:
    """Round a byte offset up to its required alignment."""
    return -(-value // alignment) * alignment


def _first_fit(
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
                    for _other, address, held in blocked
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
                    for _other, other_address, other_size in blocked
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


def _satisfies(
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
    pinned: dict[int, int] | None = None,
    floor: int = 0,
    sizes=None,
    starts=None,
) -> tuple[tuple[int, ...], int] | None:
    """A complete aligned seed for independent values, or None.

    Values are placed first-fit by first definition -- the original
    seed, then again holding persistent values at their pinned addresses -- and
    largest first. Each seed is checked against every address constraint the
    solver states, and the most compact one that meets them all is kept, the
    earlier attempt winning a tie. None when none does: a heuristic that fails
    proves nothing about the model, so it must not bound the search.
    """
    pinned = pinned or {}
    sizes = sizes or {v: v.lifetime.bytes for v in values}
    starts = starts or {v: v.lifetime.defined_at for v in values}
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
        if (seed := _first_fit(order, values, interference, alignments, limit, pins, lowest, sizes))
        is not None
        and _satisfies(seed[0], values, interference, alignments, limit, pinned, floor, sizes)
    ]
    return min(seeds, key=lambda seed: seed[1]) if seeds else None


def solve_allocation(
    memory_level: str,
    context: AllocationModel,
    options: SolverOptions,
) -> AllocationResult:
    """Return the first feasible whole-function placement for one level."""
    values = tuple(
        value for value in context.roots() if value.lifetime.memory_level == memory_level
    )
    sizes = {value: context.size(value) for value in values}
    persistent = {value for value in values if context.is_persistent(value)}
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

    interference = build_interference_graph(values, context.windows, context.by_value)
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
        starts={value: context.span(context.members(value))[0] for value in values},
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
    solver.parameters.max_time_in_seconds = options.timeout_seconds
    solver.parameters.num_search_workers = options.workers
    solver.parameters.random_seed = options.random_seed
    solver.parameters.stop_after_first_solution = True
    solver.parameters.search_branching = cp_model.HINT_SEARCH
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        status_name = solver.status_name(status).lower()
        raise AnalysisError(
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
            raise AnalysisError(
                f"allocation: overlapping {memory_level} roots: {a.binding} [{a.defined_at},{a.last_used_at}] at {left_offset}; {b.binding} [{b.defined_at},{b.last_used_at}] at {right_offset}"
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
