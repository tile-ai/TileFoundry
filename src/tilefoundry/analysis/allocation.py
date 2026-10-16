"""Internal values and solver results for physical memory placement."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from itertools import combinations
from typing import ClassVar, Protocol

import isl
from ortools.sat.python import cp_model

from tilefoundry.ir.core import (
    Call,
    Constant,
    Expr,
    IRMetadata,
    get_metadata,
    value_labels,
)
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.sharding.mesh_coord import MeshCoord
from tilefoundry.ir.hir.tensor.insert_slice import InsertSlice
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.isl_interop import shape_to_isl_set
from tilefoundry.ir.types import TensorType, TupleType, Type
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.layout import flatten
from tilefoundry.ir.types.mesh import levels
from tilefoundry.ir.types.utils import (
    is_literal_shape,
    local_type_of,
    static_dim_value,
    tensor_types,
)
from tilefoundry.utils.isl_utils import equates
from tilefoundry.visitor_registry.access_relation import (
    AccessRelation,
    identity_access,
    leaf_span,
    leaves_of,
    relations_of,
)
from tilefoundry.visitor_registry.buffer_alias import aliased_operand
from tilefoundry.visitor_registry.call_frame import ValuationVisitor

from .access import (
    Access,
    AliasKind,
    BufferAlias,
    SSAValue,
    aligned_source,
    apply_source,
    buffer_alias,
    eliminate_parameters,
    forwarded_leaves,
    referenced_values,
    restore_loop_dimensions,
)
from .errors import AnalysisError
from .footprint import ReachedAddresses, _count, _union_loops, qualify_instances, resolve_footprint
from .iteration_scope import IterationScope
from .liveness import Liveness, RegionInterval, result_copies
from .metadata import BindingValuation, Footprint, LoopNestValuation, TimingValuation, ValueLifetime
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


type Storage = AllocationValue | tuple[Storage, ...] | None


def leaves(storage: Storage) -> tuple[AllocationValue, ...]:
    if isinstance(storage, AllocationValue):
        return (storage,)
    return (
        tuple(value for field in storage for value in leaves(field))
        if isinstance(storage, tuple)
        else ()
    )


def storage_like(held: Type, values: Iterator[AllocationValue | None]) -> Storage:
    if isinstance(held, TupleType):
        return tuple(storage_like(field, values) for field in held.fields)
    return next(values) if isinstance(held, TensorType) else None


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
        if edge is None or (structural and edge.kind not in (AliasKind.VIEW, AliasKind.FORWARD)):
            return alias
        if precisions is not None:
            precision = precisions.get(current.value, AnalysisPrecision.UNKNOWN)
            if precision is not AnalysisPrecision.EXACT:
                raise AnalysisError(f"non-EXACT address for {current.lifetime.binding}: {edge.relation.relation}")
        if current in seen:
            raise AnalysisError("allocation: storage links contain a cycle")
        seen.add(current)
        held = local_type_of(leaves_of(current.value.expr.type)[current.value.index])
        values = {**alias.relation.values, **edge.relation.values}
        relation = apply_source(alias.relation.relation, edge, len(held.shape), 0, values)
        alias = replace(
            alias, target=edge.target, relation=AccessRelation(relation, values),
            writes=alias.writes or edge.writes or edge.kind is AliasKind.INSERT_SLICE,
        )
    return alias


@dataclass(frozen=True)
class AllocationSummary(IRMetadata):
    """Definition facts interpreted at a leaf call, never an execution context."""

    values: tuple[AllocationValue, ...]
    windows: tuple[ValueLifetime, ...]
    in_place: tuple[tuple[int, BufferAlias], ...] = ()
    instances: Footprint | None = None
    regions: tuple[RegionInterval, ...] = ()
    views: dict[SSAValue, BufferAlias] = field(default_factory=dict)
    view_precisions: dict[SSAValue, AnalysisPrecision] = field(default_factory=dict)
    requires: ClassVar[tuple[type, ...]] = (BindingValuation, TimingValuation, LoopNestValuation)

    def executable_values(self):
        return {value.value: replace(value, aliases=tuple(
            self.views.get(value.value, alias)
            if value.value in self.views and alias.kind is self.views[value.value].kind else alias
            for alias in value.aliases
        )) for value in self.values}

    def resolve(self, ctx):
        by_value = {value.value: value for value in self.values}
        boundary = {}
        timing = ctx.request_valuation(TimingValuation)
        for formal, argument in ctx.bindings:
            for index in range(len(leaves_of(formal.type))):
                value = by_value.get(SSAValue(formal, index))
                if value is not None:
                    boundary[root(value, by_value)] = SSAValue(argument, index)
        returned = ctx.callee.function.body
        for index in range(len(leaves_of(ctx.call.type))):
            value = by_value.get(SSAValue(returned, index))
            if value is not None:
                boundary.setdefault(root(value, by_value), SSAValue(ctx.call, index))
        translated = []
        for window in self.windows:
            value = by_value.get(window.value)
            mapped = boundary.get(root(value, by_value)) if value is not None else None
            translated.append(
                replace(
                    window,
                    value=mapped or window.value,
                    persistent=window.persistent and mapped is None,
                    defined_at=timing.event(window.defined_at),
                    last_used_at=timing.event(window.last_used_at),
                )
            )
        return replace(
            self,
            values=tuple(value for value in self.values if root(value, by_value) not in boundary),
            windows=tuple(translated),
            instances=resolve_footprint(self.instances, ctx),
            regions=tuple(region.shifted(timing) for region in self.regions),
        )


@dataclass(frozen=True)
class AllocationResult:
    peak_bytes: int
    solver_status: str
    offsets: dict[AllocationValue, int] = field(default_factory=dict)


@dataclass(kw_only=True)
class AllocationModel(AnalyzeContext):
    """One definition's collection; imported values are read-only templates."""

    address_decisions: list[tuple[SSAValue, BufferAlias, SSAValue]] = field(default_factory=list)

    liveness: Liveness
    quantities: dict[SSAValue, tuple[str, int]]
    persistent: frozenset[SSAValue]
    by_value: dict[SSAValue, AllocationValue] = field(default_factory=dict)
    windows: list[ValueLifetime] = field(default_factory=list)
    candidates: list[tuple[SSAValue, BufferAlias, LoopRegion | None]] = field(default_factory=list)
    locations: dict[int, AnalyzeContext] = field(default_factory=dict)
    instance_groups: list = field(default_factory=list)
    regions: list[RegionInterval] = field(default_factory=list)

    def storage_actual(self, formal, index=0):
        argument = next(argument for param, argument in self.bindings if param is formal)
        held = local_type_of(leaves_of(argument.type)[index])
        return BufferAlias(
            SSAValue(argument, index), identity_access(len(held.shape)), AliasKind.FORWARD
        )

    def occupancy(self, value):
        """Native lifetime plus translated occupancy, without changing either."""
        return tuple(window for window in self.windows if window.value == value.value)

    def span(self, values):
        windows = tuple(window for value in values for window in self.occupancy(value))
        return min(w.defined_at for w in windows), max(w.last_used_at for w in windows)

    def members(self, value):
        representative = root(value, self.by_value)
        return tuple(
            item for item in self.by_value.values() if root(item, self.by_value) is representative
        )

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
                    image = _union_loops(reached.reached)
                    totals[reached.target] += _count(replace(reached, reached=image, must=image))[0]
            size = max(size, max(totals.values(), default=0))
        return size

    def is_persistent(self, value):
        return any(item.lifetime.persistent for item in self.members(value))


def _scope_relation(relation, scope, held, formals):
    lifted = relation.relation.insert_dims(isl.dim_type.IN, 0, scope.depth)
    domain = scope.domain.insert_dims(
        isl.dim_type.SET, scope.depth, relation.relation.dim(isl.dim_type.IN)
    )
    lifted = lifted.intersect_domain(domain)
    return eliminate_parameters(
        lifted,
        relation.values,
        scope,
        held,
        narrow=True,
        retained=scope.domain_params,
        formals=formals,
    )


def _bound_relation(model, relation, scope, held):
    lifted, precision = _scope_relation(relation, scope, held, model._binding_formals)
    return lifted if precision is AnalysisPrecision.EXACT else None


def _relation_at(relation, scope):
    return AccessRelation(
        relation,
        referenced_values(relation, scope.domain_params),
    )


def _storage_source(value, operand, ctx):
    """Express a data source in its direct actual operand's coordinates."""
    scope = ctx.current
    source = scope.sources["narrow"][value]
    actual = scope.sources["narrow"][operand]
    maps = []
    for item, key in ((source, value), (actual, operand)):
        rank = len(local_type_of(leaves_of(key.expr.type)[key.index]).shape)
        relation = aligned_source(item, rank, scope.depth, scope.domain_params)
        relation = relation.insert_dims(isl.dim_type.OUT, 0, scope.depth)
        for axis in range(scope.depth):
            relation = relation.equate(isl.dim_type.IN, axis, isl.dim_type.OUT, axis)
        maps.append(relation)
    relation = maps[0].apply_range(maps[1].reverse()).project_out(isl.dim_type.OUT, 0, scope.depth)
    values = {**source.relation.values, **actual.relation.values, **scope.domain_params}
    return BufferAlias(
        operand,
        AccessRelation(
            relation,
            referenced_values(relation, values),
        ),
        source.kind,
    )


def _unprefix(relation, scope):
    values = dict(scope.domain_params)
    for axis, loop in enumerate(scope.context.enclosing_loops()):
        name = f"__tf_alias_loop_{axis}"
        while name in values:
            name += "_"
        relation = relation.set_dim_name(isl.dim_type.IN, axis, name)
        values[name] = loop.induction_var
    relation = relation.move_dims(
        isl.dim_type.PARAM, relation.dim(isl.dim_type.PARAM), isl.dim_type.IN, 0, scope.depth
    )
    varying = relation.gist_params(relation.params())
    for axis in reversed(range(relation.dim(isl.dim_type.PARAM))):
        if not varying.involves_dims(isl.dim_type.PARAM, axis, 1):
            relation = relation.project_out(isl.dim_type.PARAM, axis, 1)
    return AccessRelation(
        relation,
        {
            name: value
            for name, value in values.items()
            if relation.find_dim_by_name(isl.dim_type.PARAM, name) >= 0
        },
    )


def _coordinates(model, value, scope, *, override=None, storage_path=False, seen=()):
    if value in seen:
        return None
    edge = (
        override[1]
        if override is not None and override[0] is value
        else _edge(value, model.by_value, storage_path=storage_path)
    )
    if edge is None:
        relation = _bound_relation(model, identity_of(value), scope, value.value.expr.type)
        return (value, relation) if relation is not None else None
    if edge.kind in (AliasKind.VIEW, AliasKind.FORWARD):
        held = local_type_of(leaves_of(value.value.expr.type)[value.value.index])
        relation = aligned_source(edge, len(held.shape), scope.depth, scope.domain_params)
        relation = relation.intersect_domain(
            scope.domain.insert_dims(
                isl.dim_type.SET, scope.depth, relation.dim(isl.dim_type.IN) - scope.depth
            )
        )
    else:
        relation = _bound_relation(model, edge.relation, scope, edge.target.expr.type)
    source = _coordinates(
        model,
        model.by_value[edge.target],
        scope,
        override=override,
        storage_path=storage_path,
        seen=(*seen, value),
    )
    if relation is None or source is None:
        return None
    representative, source_relation = source
    if relation.dim(isl.dim_type.OUT) != source_relation.dim(isl.dim_type.IN) - scope.depth:
        return None
    return representative, _in_root(relation, source, scope)


def _visible_at(model, value, producer):
    context = model.locations[id(producer.value.expr)]
    scope = context.current
    if isinstance(value, (int, Constant, DimVar)):
        return True
    if isinstance(value, Call) and isinstance(value.target, MeshCoord):
        mesh = scope.context.current_mesh
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
    except AnalysisError:
        return None
    scope = model.locations[id(representative.value.expr)].current
    source = _coordinates(model, holder, scope, storage_path=window)
    target = _bound_relation(model, alias.relation, scope, alias.target.expr.type)
    if source is None or target is None:
        return None
    _representative, relation = source
    with_loops = relation.insert_dims(isl.dim_type.OUT, 0, scope.depth)
    for axis in range(scope.depth):
        with_loops = with_loops.equate(isl.dim_type.IN, axis, isl.dim_type.OUT, axis)
    landed = with_loops.reverse().apply_range(target).coalesce()
    if window:
        held = local_type_of(leaves_of(representative.value.expr.type)[representative.value.index])
        expected = scope.domain.flat_product(shape_to_isl_set(tuple(held.shape), {}))
        if not landed.domain().is_equal(expected.intersect_params(landed.params())):
            return None
    published = _unprefix(landed, scope)
    relation = published.relation
    if window and (
        relation.dim(isl.dim_type.IN) != relation.dim(isl.dim_type.OUT)
        or relation.is_translation() != 1
    ):
        return None
    if window and not all(
        _visible_at(model, value, representative) for value in published.values.values()
    ):
        return None
    return representative, replace(alias, relation=published)


def _in_root(relation, positioned, scope):
    representative, coordinates = positioned
    return apply_source(
        relation,
        BufferAlias(representative.value, _relation_at(coordinates, scope), AliasKind.FORWARD),
        relation.dim(isl.dim_type.OUT),
        scope.depth,
        scope.domain_params,
    )


def _mapped_pointwise(model, left, right, start, landing):
    """Only capability-covered mapped windows may borrow an in-place proof."""
    touching = {
        window
        for member in right
        for window in model.occupancy(member)
        if window is not member.lifetime and window.last_used_at == start
    }
    covered, proved = set(), set()
    for output in left:
        if not isinstance(output.value.expr, Call):
            continue
        context = model.locations[id(output.value.expr)]
        if context.callee is None:
            continue
        written = _coordinates(model, output, context.current, override=landing)
        for key, actual, windows in in_place_operands(output.value.expr, context):
            if key != output.value:
                continue
            member = model.by_value.get(actual.target)
            read = None if member is None else _coordinates(model, member, context.current)
            if read is None or written is None or read[0] is not written[0]:
                continue
            reached = _in_root(
                restore_loop_dimensions(
                    actual.relation.relation,
                    (loop.induction_var for loop in context.current.context.enclosing_loops()),
                ),
                read,
                context.current,
            )
            if not reached.is_equal(written[1]):
                continue
            covered.update(touching & windows)
            for member in right:
                positioned = _coordinates(model, member, context.current)
                if (
                    positioned is not None
                    and positioned[0] is read[0]
                    and positioned[1].is_equal(reached)
                ):
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
    if not read:
        return True
    for producer in left:
        if producer.lifetime.defined_at != start or not isinstance(producer.value.expr, Call):
            continue
        node = producer.value.expr
        scope = _proof_scope(node, model.locations[id(node)])
        output = scope.outputs.get("narrow", {}).get(id(node))
        if output is None or not output[1]:
            continue
        positioned = _coordinates(model, producer, scope, override=landing)
        if positioned is None:
            continue
        output_root, output_map = positioned
        write = output[1][0]
        written = _in_root(write.relation, positioned, scope)
        recorded = scope.accesses.get("narrow", {}).get(id(node))
        inputs = () if recorded is None else recorded[1]
        insert = _insert_accesses(node, scope)
        if insert is not None:
            written = written.union(_in_root(insert[0].relation, positioned, scope))
        for access in inputs:
            readers = {
                value
                for value in read
                if value == access.target
                or (
                    (source := model.current.sources["narrow"].get(value)) is not None
                    and source.target == access.target
                )
            }
            if not readers or access.precision is not AnalysisPrecision.EXACT:
                continue
            member = model.by_value[access.target]
            positioned = _coordinates(model, member, scope)
            if positioned is None or positioned[0] is not output_root:
                continue
            reached = _in_root(access.relation, positioned, scope)
            if reached.is_equal(written.intersect_domain(access.relation.domain())):
                proved.update(readers)
    return read <= proved


def _insert_accesses(node, scope):
    recorded = scope.accesses.get("narrow", {}).get(id(node))
    if not isinstance(node.target, InsertSlice) or recorded is None:
        return None
    accesses = recorded[1]
    destination = next((access for access in accesses if access.input_index == 0), None)
    update = next((access for access in accesses if access.input_index == 1), None)
    return (destination, update) if destination is not None and update is not None else None


def _proof_scope(node, ctx):
    recorded = ctx.current.accesses.get("narrow", {}).get(id(node))
    if recorded is not None and any(
        access.target not in ctx.by_value
        and access.input_index is not None
        and any(
            SSAValue(node.args[access.input_index], index) in ctx.by_value
            for index in range(len(leaves_of(node.args[access.input_index].type)))
        )
        for access in recorded[1]
    ):
        return operand_scope(node, ctx)
    return ctx.current


def tie_analysis(model):
    """All ties read occupancy; global interference keeps individual windows."""
    use_events = defaultdict(list)
    for use in model.liveness.uses:
        use_events[use.value].append(use)
    local = {item.value for item in model.liveness.intervals}
    for holder, alias, loop in sorted(
        model.candidates,
        key=lambda item: (
            item[1].kind not in (AliasKind.INSERT_SLICE, AliasKind.FORWARD),
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
        semantic = alias.kind in (AliasKind.INSERT_SLICE, AliasKind.FORWARD)
        if not semantic and (model.is_persistent(b) or any(
            any(item.value.expr is param for param in model.frame.function.params)
            for item in right
        )):
            continue
        if (alias.kind is AliasKind.INSERT_SLICE or alias.writes) and any(
            not use.synthetic and use.at > value.lifetime.defined_at
            for item in right for use in use_events.get(item.value, ())
        ):
            continue
        if a.lifetime.memory_level != b.lifetime.memory_level or model.size(a) > model.size(b):
            continue
        start, _ = model.span(left)
        _, end = model.span(right)
        window = alias.kind is AliasKind.WINDOW
        if window:
            node = next(
                (
                    candidate.value.expr
                    for candidate in model.members(target)
                    if isinstance(candidate.value.expr, Call)
                    and isinstance(candidate.value.expr.target, InsertSlice)
                ),
                None,
            )
            if node is None:
                continue
            destination = model.by_value.get(SSAValue(model.capture_root(node.args[0])))
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
            scope = model.locations[id(new_root.value.expr)].current
            positioned = _coordinates(model, new_root, scope, override=physical)
            if positioned is None or not is_zero_offset(positioned[1]):
                continue
        if not semantic and end == start and not _pointwise(model, left, right, start, physical, use_events):
            continue
        new_root.aliases += (edge,)
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


def published_aliases(model):
    result = {}
    kinds = (AliasKind.INSERT_SLICE, AliasKind.CARRY_INIT, AliasKind.CARRY_YIELD, AliasKind.FORWARD)
    holders = {(holder, alias.kind) for holder, alias, _loop in model.candidates}
    for value in model.by_value.values():
        if id(value.value.expr) not in model.locations:
            continue
        for kind in kinds:
            if kind is AliasKind.FORWARD:
                if not isinstance(value.value.expr, Call) or model.locations[id(value.value.expr)].callee is None:
                    continue
            elif (value.value, kind) not in holders:
                continue
            alias = next((a for a in value.aliases if a.kind is kind), None)
            if alias is not None:
                result[value.value] = (
                    address_alias(value, model.by_value, alias=alias)
                    if kind in (AliasKind.INSERT_SLICE, AliasKind.FORWARD) else alias
                )
                break
    for holder, alias, _loop in model.candidates:
        if alias.kind is not AliasKind.WINDOW:
            continue
        value = model.by_value[holder]
        if not any(edge.kind is AliasKind.WINDOW for edge in value.aliases):
            continue
        inserted = alias.target.expr
        context = model.locations[id(inserted)]
        destination = SSAValue(context.capture_root(inserted.args[0]))
        landed = _landing(model, value, replace(alias, target=destination), window=True)
        if landed is not None:
            producer, published = landed
            result.setdefault(producer.value, published)
    return result


def covers_result_each_iteration(
    outputs: isl.map,
    iteration_domain: isl.set,
    result_box: isl.set,
) -> bool:
    """Whether every enclosing-loop point writes the complete result value."""
    try:
        domain = outputs.domain()
        call_dims = outputs.dim(isl.dim_type.IN)
        iteration_depth = iteration_domain.dim(isl.dim_type.SET)
        if iteration_depth > call_dims:
            return False
        loop_prefix = (
            isl.map.identity(domain.get_space().map_from_set())
            .intersect_domain(domain)
            .project_out(isl.dim_type.OUT, iteration_depth, call_dims - iteration_depth)
        )
        per_iteration = loop_prefix.reverse().apply_range(outputs).coalesce()
        expected = (
            isl.map.universe(per_iteration.get_space())
            .intersect_domain(iteration_domain)
            .intersect_range(result_box)
            .coalesce()
        )
        return per_iteration.is_equal(expected)
    except isl.Error:
        return False


def covers_result(
    output_coverage: isl.set,
    full_result: isl.set,
    outputs: isl.map,
    iteration_domain: isl.set,
    result_box: isl.set,
) -> bool:
    """Whether a write covers the entire result at every loop point."""
    try:
        return output_coverage.is_equal(full_result) or covers_result_each_iteration(
            outputs,
            iteration_domain,
            result_box,
        )
    except isl.Error:
        return False


def operand_to_result_relation(
    node: Call,
    scope: IterationScope,
    insert_accesses: tuple[Access, Access] | None,
) -> tuple[tuple[SSAValue, isl.map], ...]:
    """Prove exact logical relations between one result and its operands."""

    def coverage(accesses: tuple[Access, ...]) -> isl.set | None:
        if not accesses or any(
            access.precision is not AnalysisPrecision.EXACT for access in accesses
        ):
            return None
        result = accesses[0].relation.domain()
        for access in accesses[1:]:
            result = result.union(access.relation.domain())
        return result.coalesce()

    def access_relation(accesses: tuple[Access, ...]) -> isl.map | None:
        if not accesses or any(
            access.precision is not AnalysisPrecision.EXACT for access in accesses
        ):
            return None
        result = accesses[0].relation
        for access in accesses[1:]:
            result = result.union(access.relation)
        return result.coalesce()

    def value_domain(value: Expr, index: int = 0) -> isl.set | None:
        try:
            held = leaves_of(local_type_of(value.type))[index]
        except (TypeError, ValueError, NotImplementedError):
            return None
        if not isinstance(held, TensorType):
            return None
        if not is_literal_shape(held.shape):
            return None
        return scope.domain.flat_product(shape_to_isl_set(tuple(held.shape), {})).coalesce()

    def value_box(value: Expr) -> isl.set | None:
        try:
            held = local_type_of(value.type)
        except (TypeError, ValueError, NotImplementedError):
            return None
        if not isinstance(held, TensorType) or not is_literal_shape(held.shape):
            return None
        return shape_to_isl_set(tuple(held.shape), {})

    def composed(inputs: isl.map, outputs: isl.map) -> isl.map | None:
        try:
            common = inputs.domain().intersect(outputs.domain()).coalesce()
            if common.is_empty():
                return None
            inputs = inputs.intersect_domain(common)
            outputs = outputs.intersect_domain(common)
            call_dims = inputs.dim(isl.dim_type.IN)
            if call_dims != outputs.dim(isl.dim_type.IN) or scope.depth > call_dims:
                return None
            loop_prefix = isl.map.identity(common.get_space().map_from_set()).project_out(
                isl.dim_type.OUT, scope.depth, call_dims - scope.depth
            )
            operand_with_loops = loop_prefix.flat_range_product(inputs)
            result_with_loops = loop_prefix.flat_range_product(outputs)
            per_iteration = operand_with_loops.reverse().apply_range(result_with_loops).coalesce()
            result = operand_with_loops.reverse().apply_range(outputs).coalesce()
            return (
                result
                if result.is_single_valued()
                and per_iteration.is_single_valued()
                and per_iteration.is_injective()
                else None
            )
        except isl.Error:
            return None

    def complete(
        operand: Expr, index: int, inputs: isl.map | None, outputs: isl.map | None
    ) -> isl.map | None:
        if inputs is None or outputs is None:
            return None
        relation = composed(inputs, outputs)
        expected = value_domain(operand, index)
        if relation is None or expected is None:
            return None
        try:
            return relation if relation.domain().is_equal(expected) else None
        except isl.Error:
            return None

    def identity() -> isl.map | None:
        domain = value_domain(node)
        if domain is None:
            return None
        try:
            return (
                isl.map.identity(domain.get_space().map_from_set())
                .intersect_domain(domain)
                .project_out(isl.dim_type.OUT, 0, scope.depth)
            )
        except isl.Error:
            return None

    recorded_inputs = scope.accesses.get("narrow", {}).get(id(node))
    recorded_outputs = scope.outputs.get("narrow", {}).get(id(node))
    if (
        recorded_inputs is None
        or recorded_inputs[0] is not node
        or recorded_outputs is None
        or recorded_outputs[0] is not node
    ):
        return ()
    inputs = recorded_inputs[1]
    outputs = recorded_outputs[1]
    output_coverage = coverage(outputs)
    output_relation = access_relation(outputs)
    full_result = value_domain(node)
    result_box = value_box(node)
    if (
        output_coverage is None
        or output_relation is None
        or full_result is None
        or result_box is None
    ):
        return ()

    by_buffer: dict[SSAValue, list[Access]] = defaultdict(list)
    operands: dict[SSAValue, Expr] = {}
    for access in inputs:
        operand = access.target.expr
        key = access.target
        by_buffer[key].append(access)
        operands[key] = operand

    result: list[tuple[SSAValue, isl.map]] = []
    seen: set[SSAValue] = set()
    if covers_result(
        output_coverage,
        full_result,
        output_relation,
        scope.domain,
        result_box,
    ):
        for key, operand in operands.items():
            input_relation = access_relation(tuple(by_buffer[key]))
            try:
                pointwise = input_relation is not None and input_relation.is_equal(output_relation)
            except isl.Error:
                pointwise = False
            relation = (
                complete(operand, key.index, input_relation, output_relation) if pointwise else None
            )
            if relation is not None:
                result.append((key, relation))
                seen.add(key)

    if insert_accesses is None:
        return tuple(result)

    destination_access, update_access = insert_accesses
    destination, destination_index = destination_access.target.expr, destination_access.target.index
    update, update_index = update_access.target.expr, update_access.target.index
    written = output_coverage
    destination_coverage = coverage(
        tuple(by_buffer.get(SSAValue(destination, destination_index), ()))
    )
    update_accesses = tuple(by_buffer.get(SSAValue(update, update_index), ()))
    update_coverage = coverage(update_accesses)

    if destination_coverage is not None:
        relation = identity()
        expected_destination = value_domain(destination, destination_index)
        try:
            partitioned = destination_coverage.is_disjoint(written) and (
                destination_coverage.union(written).coalesce().is_equal(full_result)
            )
            complete_identity = (
                relation is not None
                and expected_destination is not None
                and relation.domain().is_equal(expected_destination)
            )
        except isl.Error:
            partitioned = complete_identity = False
        if (
            partitioned
            and complete_identity
            and SSAValue(destination, destination_index) not in seen
        ):
            result.append((SSAValue(destination, destination_index), relation))
            seen.add(SSAValue(destination, destination_index))

    try:
        update_matches_write = update_coverage is not None and update_coverage.is_equal(written)
    except isl.Error:
        update_matches_write = False
    update_relation = (
        complete(update, update_index, access_relation(update_accesses), output_relation)
        if update_matches_write
        else None
    )
    if update_relation is not None and SSAValue(update, update_index) not in seen:
        result.append((SSAValue(update, update_index), update_relation))
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


def operand_scope(node: Call, ctx: AllocationModel) -> IterationScope:
    """State relations on the actual SSA operands, before footprint source tracing.

    An operation on a Slice can be pointwise in that Slice's coordinates even
    when its footprint source names the much larger backing tensor.
    """
    scope = ctx.current
    logical = replace(ctx, topology_level=None)
    relations = scope.projected_relations(node, logical) if id(node) in scope.relations else ()
    inputs, outputs = [], []
    for index, boundary in enumerate(relations):
        is_input = index < len(node.args)
        value = node.args[index] if is_input else node
        output_index = None if is_input else index - len(node.args)
        held = local_type_of(value.type)
        leaf_index = 0
        if isinstance(held, TupleType):
            if is_input:
                continue
            leaf_index, _count = leaf_span(held, output_index)
            held = held.fields[output_index]
        if not isinstance(held, TensorType):
            continue
        relation, precision = _scope_relation(boundary, scope, held, ctx._binding_formals)
        if is_literal_shape(held.shape):
            relation = relation.intersect_range(shape_to_isl_set(tuple(held.shape), {}))
        access = Access(
            index if is_input else None,
            relation,
            SSAValue(value, leaf_index),
            precision,
            output_index,
        )
        (inputs if is_input else outputs).append(access)
    return replace(
        scope,
        accesses={"narrow": {id(node): (node, tuple(inputs))}},
        outputs={"narrow": {id(node): (node, tuple(outputs))}},
    )


def in_place_operands(node, ctx):
    """Exact zero-offset capabilities, before caller lifetime and placement checks."""
    if ctx.callee is None:
        scope = _proof_scope(node, ctx)
        if _insert_accesses(node, scope) is None:
            for operand, relation in operand_to_result_relation(node, scope, None):
                if is_zero_offset(relation):
                    yield SSAValue(node), BufferAlias(
                        operand, _unprefix(relation, scope), AliasKind.IN_PLACE
                    ), frozenset()
        return
    template = get_metadata(ctx.callee.function, AllocationSummary)
    local = {value.value: value for value in template.values}
    resolved = template.resolve(ctx) if template.in_place else None
    for index, capability in template.in_place:
        actual, precision = ctx.project(capability, actual_of=ctx.storage_actual)
        if precision is not AnalysisPrecision.EXACT or not is_zero_offset(actual.relation.relation):
            continue
        formal = local[capability.target]
        windows = frozenset(
            mapped for original, mapped in zip(template.windows, resolved.windows, strict=True)
            if (member := local.get(original.value)) is not None
            and root(member, local) is root(formal, local)
        )
        yield SSAValue(node, index), actual, windows


class AllocationConstraintVisitor(ValuationVisitor[Storage]):
    """Collect native values and interpret called templates at leaf Calls."""

    def __init__(self, *, root_function, context):
        self.locations = {}
        native = [
            life.value for life in context.liveness.intervals if life.value in context.quantities
        ]
        sources = context.current.sources["narrow"]
        ordered = sorted(native, key=lambda key: sources[key].target != key)
        expressions = {id(key.expr): key.expr for key in ordered}
        self.labels = dict(zip(expressions, value_labels(expressions.values()), strict=True))
        super().__init__(root_function=root_function, contexts=self.locations)

    def add_value(self, lifetime, ctx, aliases=()):
        value = AllocationValue(lifetime, aliases)
        ctx.by_value[value.value] = value
        ctx.windows.append(value.lifetime)
        return value

    def allocate(self, node, ctx, sources=None):
        if isinstance(node.type, TupleType) and result_copies(node) > 1:
            raise AnalysisError("allocation: staged tuple results are not supported")
        slots = []
        operands = forwarded_leaves(node, ctx)
        view = aliased_operand(node) if isinstance(node, Call) else None
        for index, leaf in enumerate(leaves_of(node.type)):
            key = SSAValue(node, index)
            cached = ctx.current.sources["narrow"].get(key)
            if sources is not None:
                source = sources.get(key)
            elif operands is not None:
                source = BufferAlias(
                    operands[index],
                    identity_access(len(leaf.shape) if isinstance(leaf, TensorType) else 1),
                    AliasKind.FORWARD,
                )
            elif view is not None and cached is not None and cached.target != key:
                operand = SSAValue(node.args[view])
                data = ctx.by_value.get(cached.target)
                actual = ctx.by_value.get(operand)
                source = (
                    cached
                    if data is not None
                    and actual is not None
                    and root(data, ctx.by_value) is root(actual, ctx.by_value)
                    else _storage_source(key, operand, ctx)
                )
            else:
                source = cached
            forwarding = source is not None and source.target != key
            target = ctx.by_value.get(source.target) if forwarding else None
            quantity = ctx.quantities.get(key)
            if forwarding:
                if target is None:
                    slots.append(None)
                    continue
                level = target.lifetime.memory_level
                amount = target.lifetime.bytes
            elif quantity is not None:
                level, amount = quantity
            else:
                slots.append(None)
                continue
            lifetime = replace(
                ctx.liveness.interval_of(key),
                memory_level=level,
                bytes=amount,
                persistent=key in ctx.persistent,
                binding=self.labels[id(node)]
                + (f"[{index}]" if isinstance(node.type, TupleType) else ""),
            )
            slots.append(self.add_value(lifetime, ctx, (source,) if forwarding else ()))
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
        return storage_like(node.type, iter(slots)) if slots else None

    def visit_Var(self, node, ctx):
        return self.allocate(node, ctx)

    def visit_leaf_Tuple(self, node, operands, ctx):
        return self.allocate(node, ctx)

    def visit_MeshRegion(self, node, ctx):
        self.visit_operands_outside(node, ctx)
        inner = self.region_context(node, ctx)
        for param in node.params:
            self.visit(param, inner)
        self.visit(node.body, inner)
        return self.allocate(node, ctx)

    def visit_LoopRegion(self, node, ctx):
        initials = self.visit_operands_outside(node, ctx)
        inner = self.region_context(node, ctx)
        self.visit(node.induction_var, inner)
        carried = tuple(self.visit(param, inner) for param in node.params[: len(node.yield_values)])
        for param in node.params[len(node.yield_values) :]:
            self.visit(param, inner)
        for result, initial in zip(carried, initials, strict=False):
            for value, operand in zip(leaves(result), leaves(initial), strict=False):
                ctx.candidates.append(
                    (
                        value.value,
                        BufferAlias(operand.value, identity_of(value), AliasKind.CARRY_INIT),
                        node,
                    )
                )
        self.visit(node.body, inner)
        for yielded, phi in zip(node.yield_values, carried, strict=True):
            for value, operand in zip(
                leaves(self.visit(yielded, inner)), leaves(phi), strict=False
            ):
                ctx.candidates.append(
                    (
                        value.value,
                        BufferAlias(operand.value, identity_of(value), AliasKind.CARRY_YIELD),
                        node,
                    )
                )
        return self.allocate(node, ctx)

    def default_visit_leaf(self, node, operands, ctx):
        if (
            isinstance(node, Call)
            and isinstance(node.target, TupleGetItem)
            and not (isinstance(node.args[1], Constant) and type(node.args[1].value) is int)
        ):
            return None
        callee = self.callee_frame(node, ctx.frame) if isinstance(node, Call) else None
        if callee is not None:
            result = self.call_storage(node, callee, ctx)
            self.in_place(node, ctx)
            return result
        source = ctx.current.sources["narrow"].get(SSAValue(node))
        if source is not None and source.target != SSAValue(node):
            return self.allocate(node, ctx)
        result = self.allocate(node, ctx)
        if not isinstance(node, Call) or not isinstance(result, AllocationValue):
            return result
        if isinstance(node.target, InsertSlice):
            destination = ctx.by_value.get(SSAValue(node.args[0]))
            if destination is not None:
                ctx.candidates.append((result.value, BufferAlias(
                    destination.value, identity_of(result), AliasKind.INSERT_SLICE
                ), None))
        local_scope = operand_scope(node, ctx)
        local_insert = _insert_accesses(node, local_scope)
        for operand, relation in operand_to_result_relation(node, local_scope, local_insert):
            if local_insert is not None:
                if operand == local_insert[1].target:
                    update = ctx.by_value.get(SSAValue(node.args[1]))
                    if update is not None:
                        ctx.candidates.append(
                            (
                                update.value,
                                BufferAlias(
                                    result.value, _unprefix(relation, local_scope), AliasKind.WINDOW
                                ),
                                None,
                            )
                        )
        self.in_place(node, ctx)
        return result

    def in_place(self, node, ctx):
        for key, operand, _windows in in_place_operands(node, ctx):
            value, target = ctx.by_value.get(key), ctx.by_value.get(operand.target)
            if value is not None and target is not None and not target.lifetime.persistent:
                ctx.candidates.append((key, replace(
                    operand, relation=identity_of(value), kind=AliasKind.IN_PLACE
                ), None))

    def call_storage(self, node, callee, ctx):
        template = get_metadata(callee.function, AllocationSummary)
        if template is None:
            raise AnalysisError(f"allocation: missing summary for {callee.function.name!r}")
        original = tuple(value.aliases for value in template.values)
        local = {value.value: value for value in template.values}
        sources = {}
        for index in range(len(leaves_of(node.type))):
            returned = local.get(SSAValue(callee.function.body, index))
            if returned is not None and any(
                root(returned, local).value.expr is formal for formal in callee.function.params
            ):
                source, precision = ctx.project(address_alias(returned, local), actual_of=ctx.storage_actual)
                if precision is AnalysisPrecision.EXACT:
                    sources[SSAValue(node, index)] = source
        result = self.allocate(node, ctx, {})
        ctx.candidates.extend((key, alias, None) for key, alias in sources.items())
        resolved = template.resolve(ctx)
        for value in resolved.values:
            ctx.by_value.setdefault(value.value, value)
        ctx.windows.extend(window for window in resolved.windows if window.value in ctx.by_value)
        ctx.regions.extend(resolved.regions)
        instances = qualify_instances(resolved.instances, ctx)
        if instances is not None:
            ctx.instance_groups.extend(instances.groups)
        assert tuple(value.aliases for value in template.values) == original, (
            "resolving a call mutated callee aliases"
        )
        return result


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
    tie_analysis(model)
    assert tuple(value.aliases for value in borrowed) == aliases, (
        "caller ties mutated callee aliases"
    )
    return model


def executable_views(model):
    """Keep runtime address proofs independent of domain-landed estimation.

    Replay accepted decisions only; unavailable proofs retain UNKNOWN precision.
    """
    views, precisions = {}, {}
    for value in model.by_value.values():
        key = value.value
        context = model.locations.get(id(key.expr))
        if context is None:
            continue
        alias = _edge(value, model.by_value)
        if alias is None or alias.kind not in (AliasKind.VIEW, AliasKind.FORWARD):
            continue
        exact, precision = None, AnalysisPrecision.EXACT
        if alias.kind is AliasKind.VIEW:
            exact = buffer_alias(key.expr, context, relations_of(key.expr, context))
            if exact is not None:
                relation, values = exact.relation.relation, {}
                for name, runtime in exact.relation.values.items():
                    unique = f"__tf_address_{id(key.expr)}_{name}"
                    relation = relation.set_dim_name(
                        isl.dim_type.PARAM, relation.find_dim_by_name(isl.dim_type.PARAM, name), unique
                    )
                    values[unique] = runtime
                exact = replace(exact, relation=AccessRelation(relation, values))
        elif (operands := forwarded_leaves(key.expr, context)) is not None:
            exact = BufferAlias(operands[key.index], identity_access(
                len(leaves_of(key.expr.type)[key.index].shape)
            ), AliasKind.FORWARD)
        elif context.callee is not None:
            callee = get_metadata(context.callee.function, AllocationSummary)
            local = callee.executable_values()
            returned = local[SSAValue(context.callee.function.body, key.index)]
            try:
                source = address_alias(returned, local, precisions=callee.view_precisions)
            except AnalysisError:
                precision = AnalysisPrecision.UNKNOWN
            else:
                exact, precision = context.project(source, actual_of=context.storage_actual)
        views[key] = exact or alias
        precisions[key] = precision if exact is not None else AnalysisPrecision.UNKNOWN
    decided = {(holder, decision.kind) for holder, decision, _root in model.address_decisions}
    executable = {key: replace(value, aliases=(views[key],)
                              if key in views and (key, views[key].kind) not in decided else ())
                  for key, value in model.by_value.items()}
    for holder, decision, representative in model.address_decisions:
        if holder in views and views[holder].kind is decision.kind:
            decision = views[holder]
        source = address_alias(executable[holder], executable, precisions=precisions)
        assert source.target == representative
        values = {**source.relation.values, **decision.relation.values}
        held = local_type_of(leaves_of(holder.expr.type)[holder.index])
        relation = apply_source(source.relation.relation.reverse(), decision, len(held.shape), 0, values)
        proof = replace(decision, relation=AccessRelation(relation, values))
        views[representative] = proof
        precisions[representative] = AnalysisPrecision.EXACT
        executable[representative] = replace(executable[representative], aliases=(proof,))
    return views, precisions


def summary_of(model):
    by_value = model.by_value
    capabilities = []
    body = model.frame.function.body
    for index in range(len(leaves_of(body.type))):
        value = by_value.get(SSAValue(body, index))
        source = model.current.sources["narrow"].get(SSAValue(body, index))
        if (
            value is not None
            and source is not None
            and source.relation.relation.dim(isl.dim_type.IN)
            == source.relation.relation.dim(isl.dim_type.OUT)
            and is_zero_offset(source.relation.relation)
        ):
            target_root = root(value, by_value)
            for member in model.members(value):
                node = member.value.expr
                if not isinstance(node, Call) or id(node) not in model.locations:
                    continue
                scope = model.locations[id(node)].current
                for operand, relation in operand_to_result_relation(
                    node, scope, _insert_accesses(node, scope)
                ):
                    formal = by_value.get(operand)
                    if (
                        formal is None
                        or not any(operand.expr is param for param in model.frame.function.params)
                        or not is_zero_offset(relation)
                    ):
                        continue
                    if (
                        model.span(model.members(formal))[1]
                        <= model.span(model.members(target_root))[0]
                    ):
                        capabilities.append((index, BufferAlias(
                            operand, _unprefix(relation, scope), AliasKind.IN_PLACE
                        )))
    return AllocationSummary(
        tuple(by_value.values()),
        tuple(model.windows),
        tuple(capabilities),
        Footprint(tuple(model.instance_groups)),
        tuple(model.regions),
        *executable_views(model),
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
