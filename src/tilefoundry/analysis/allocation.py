"""Internal values and solver results for physical memory placement."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from enum import IntEnum
from itertools import combinations
from typing import Protocol

import isl
from ortools.sat.python import cp_model

from tilefoundry.ir.core import (
    Call,
    Constant,
    Expr,
    Tuple,
    Var,
)
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.sharding.mesh_coord import MeshCoord
from tilefoundry.ir.hir.tensor.insert_slice import InsertSlice
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.isl_interop import shape_to_isl_set
from tilefoundry.ir.types import TensorType, TupleType
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
from tilefoundry.visitor_registry.access_relation import AccessRelation, leaf_span, leaves_of
from tilefoundry.visitor_registry.call_frame import CallFrameVisitor, ValuationVisitor

from .access import Access, BufferAlias, apply_source, buffer_alias, eliminate_parameters
from .errors import AnalysisError
from .iteration_scope import IterationScope
from .liveness import Liveness, UseEvent, result_copies
from .metadata import ValueLifetime
from .precision import AnalysisPrecision
from .visitor import AnalyzeContext


class SolverOptions(Protocol):
    """Placement limits consumed independently of the memory front end."""

    timeout_seconds: float
    workers: int
    random_seed: int


class AliasKind(IntEnum):
    """Structural sharing first, then the specified deterministic tie order."""

    VIEW = 0
    FORWARD = 1
    CARRY_INIT = 2
    CARRY_YIELD = 3
    INSERT_SLICE = 4
    WINDOW = 5
    RETURN_REUSE = 6
    IN_PLACE = 7


@dataclass(eq=False)
class AllocationValue:
    """One SSA result: its source, result index, and own live interval."""

    value: Expr
    index: int
    lifetime: ValueLifetime
    aliases: tuple[Alias, ...] = ()

    def intersects(self, other: AllocationValue) -> bool:
        """Whether two closed structured-SSA intervals share an event."""
        return max(self.lifetime.defined_at, other.lifetime.defined_at) <= min(
            self.lifetime.last_used_at, other.lifetime.last_used_at
        )


type Storage = AllocationValue | tuple[Storage, ...] | None


def leaves(storage: Storage) -> tuple[AllocationValue, ...]:
    """The allocation results a structured storage may use."""
    if isinstance(storage, AllocationValue):
        return (storage,)
    if isinstance(storage, tuple):
        return tuple(value for field in storage for value in leaves(field))
    return ()


@dataclass(frozen=True)
class AllocationResult:
    """The first physical placement found for one memory level."""

    peak_bytes: int
    solver_status: str


@dataclass(kw_only=True)
class AllocationModel(AnalyzeContext):
    """Definition-local relations and their independently computed Buffer partition."""

    liveness: Liveness
    quantities: dict[int, tuple[tuple[str, int] | None, ...]]
    persistent: frozenset[int]
    values: list[AllocationValue] = field(default_factory=list)
    material: set[AllocationValue] = field(default_factory=set)
    models: dict[str, cp_model.CpModel] = field(default_factory=dict)
    parameters: set[AllocationValue] = field(default_factory=set)
    outputs: set[AllocationValue] = field(default_factory=set)
    definitions: dict[int, IterationScope] = field(default_factory=dict)

    def addresses(self) -> None:
        """Create exactly one address variable per physical Buffer."""
        for level in {buffer.lifetime.memory_level for buffer in self.buffers} & {"gmem", "smem"}:
            self.models[level] = cp_model.CpModel()
            limit = sum(
                aligned(buffer.lifetime.bytes, buffer.alignment)
                for buffer in self.buffers
                if buffer.lifetime.memory_level == level
            )
            for index, buffer in enumerate(self.buffers):
                if buffer.lifetime.memory_level == level:
                    self.address[buffer] = self.models[level].new_int_var(
                        0, limit, f"address_{index}"
                    )


def identity_of(value: AllocationValue) -> AccessRelation:
    """Identity on one flattened tensor leaf, without loop coordinates."""
    held = leaves_of(value.value.type)[value.index]
    box = shape_to_isl_set(tuple(held.shape), {})
    return AccessRelation(isl.map.identity(box.get_space().map_from_set()).intersect_domain(box))


def add_alias(value: AllocationValue, target: AllocationValue, kind: AliasKind, relation=None):
    """Record a fact or candidate; collection never merges storage."""
    value.aliases += (
        Alias(target, relation if relation is not None else identity_of(value), kind),
    )


def _source_leaves(source):
    return (
        tuple(leaf for child in source for leaf in _source_leaves(child))
        if isinstance(source, tuple)
        else (source,)
    )


def _edge(model, value, *, storage_path=False):
    """The implemented storage edge, without conflating Buffer membership with coordinates."""
    structural = next((alias for alias in value.aliases if alias.kind <= AliasKind.FORWARD), None)
    if structural is not None:
        return structural
    if not storage_path and value in model.adopted:
        return model.adopted[value]
    accepted = model.accepted.get(value, ())
    kinds = (
        (AliasKind.CARRY_INIT,)
        if storage_path
        else (
            AliasKind.INSERT_SLICE,
            AliasKind.CARRY_INIT,
            AliasKind.CARRY_YIELD,
            AliasKind.RETURN_REUSE,
            AliasKind.IN_PLACE,
        )
    )
    return next(
        (
            alias
            for kind in kinds
            for alias in accepted
            if alias.kind == kind and alias.target is not value
        ),
        None,
    )


def _bound_relation(model, relation, scope, held):
    """Interpret data-coordinate parameters with a common local loop prefix."""
    lifted = relation.relation.insert_dims(isl.dim_type.IN, 0, scope.depth)
    domain = scope.domain.insert_dims(
        isl.dim_type.SET, scope.depth, relation.relation.dim(isl.dim_type.IN)
    )
    lifted = lifted.intersect_domain(domain)
    lifted, precision = eliminate_parameters(
        lifted, relation.values, scope, held, narrow=True, retained=scope.domain_params
    )
    return lifted if precision is AnalysisPrecision.EXACT else None


def _relation_at(relation, scope):
    return AccessRelation(
        relation,
        {
            name: value
            for name, value in scope.domain_params.items()
            if relation.find_dim_by_name(isl.dim_type.PARAM, name) >= 0
        },
    )


def _unprefix(relation, scope):
    """Keep local loop values as expression parameters in a published data map."""
    values = dict(scope.domain_params)
    for axis, loop in enumerate(scope.enclosing_loops()):
        name = f"__tf_alias_loop_{axis}"
        while name in values:
            name += "_"
        relation = relation.set_dim_name(isl.dim_type.IN, axis, name)
        values[name] = loop.induction_var
    relation = relation.move_dims(
        isl.dim_type.PARAM, relation.dim(isl.dim_type.PARAM), isl.dim_type.IN, 0, scope.depth
    )
    return AccessRelation(
        relation,
        {
            name: value
            for name, value in values.items()
            if relation.find_dim_by_name(isl.dim_type.PARAM, name) >= 0
        },
    )


def _coordinates(model, value, scope, *, override=None, storage_path=False, seen=()):
    """Value coordinates -> implemented storage root, including accepted semantic ties."""
    if value in seen:
        return None
    edge = (
        override[1]
        if override is not None and override[0] is value
        else _edge(model, value, storage_path=storage_path)
    )
    if edge is None:
        relation = _bound_relation(model, identity_of(value), scope, value.value.type)
        return (value, relation) if relation is not None else None
    relation = _bound_relation(model, edge.relation, scope, edge.target.value.type)
    source = _coordinates(
        model, edge.target, scope, override=override, storage_path=storage_path, seen=(*seen, value)
    )
    if relation is None or source is None:
        return None
    root, source_relation = source
    target_rank = relation.dim(isl.dim_type.OUT)
    composed = apply_source(
        relation,
        BufferAlias(root.value, _relation_at(source_relation, scope), root.index),
        target_rank,
        scope.depth,
        scope.domain_params,
    )
    return root, composed


def _visible_at(model, value, producer):
    scope = model.definitions[id(producer.value)]
    visible = set()
    cursor = scope
    while cursor is not None:
        visible.add(cursor)
        cursor = cursor.parent
    if isinstance(value, (int, Constant, DimVar)):
        return True
    if isinstance(value, Call) and isinstance(value.target, MeshCoord):
        mesh = scope.execution_mesh()
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
    interval = model.liveness.interval_of(value)
    return (
        model.definitions.get(id(value)) in visible
        and interval is not None
        and interval.defined_at < producer.lifetime.defined_at
    )


def _landing(model, holder, alias, *, window=False):
    """Reverse-compose the holder's path: producer coordinates -> candidate target."""
    root = holder
    seen = set()
    while root not in seen:
        seen.add(root)
        edge = _edge(model, root, storage_path=window)
        if edge is None:
            break
        root = edge.target
    else:
        return None
    if window and root in model.adopted:
        return None
    scope = model.definitions[id(root.value)]
    source = _coordinates(model, holder, scope, storage_path=window)
    target = _bound_relation(model, alias.relation, scope, alias.target.value.type)
    if source is None or target is None:
        return None
    _root, relation = source
    with_loops = relation.insert_dims(isl.dim_type.OUT, 0, scope.depth)
    for axis in range(scope.depth):
        with_loops = with_loops.equate(isl.dim_type.IN, axis, isl.dim_type.OUT, axis)
    landed = with_loops.reverse().apply_range(target).coalesce()
    if window:
        held = local_type_of(leaves_of(root.value.type)[root.index])
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
    return root, Alias(alias.target, published, alias.kind)


def _proved_at(model, left, right, start, landing):
    """Compare every read with the producer's write at its proposed physical landing."""
    read = {
        member for member in right.members if any(use.at == start for use in model.uses[member])
    }
    if not read:
        return False
    proved = set()
    for producer in left.members:
        if producer.lifetime.defined_at != start or producer not in model.writes:
            continue
        scope = model.definitions[id(producer.value)]
        output = model.writes[producer]
        positioned = _coordinates(model, producer, scope, override=landing)
        if positioned is None or output.precision is not AnalysisPrecision.EXACT:
            continue
        output_root, output_map = positioned
        written = apply_source(
            output.relation,
            BufferAlias(
                output_root.value,
                _relation_at(output_map, scope),
                output_root.index,
            ),
            output.relation.dim(isl.dim_type.OUT),
            scope.depth,
            scope.domain_params,
        )
        for member in read:
            positioned = _coordinates(model, member, scope)
            if positioned is None:
                continue
            input_root, input_map = positioned
            if input_root is not output_root:
                continue
            for access in model.reads.get(producer, ()):
                if (
                    access.buffer is not member.value
                    or access.index != member.index
                    or access.precision is not AnalysisPrecision.EXACT
                ):
                    continue
                reached = apply_source(
                    access.relation,
                    BufferAlias(
                        input_root.value,
                        _relation_at(input_map, scope),
                        input_root.index,
                    ),
                    access.relation.dim(isl.dim_type.OUT),
                    scope.depth,
                    scope.domain_params,
                )
                if reached.is_equal(written.intersect_domain(access.relation.domain())):
                    proved.add(member)
    return read <= proved


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
) -> tuple[tuple[Expr, int, isl.map], ...]:
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

    by_buffer: dict[tuple[int, int], list[Access]] = defaultdict(list)
    operands: dict[tuple[int, int], Expr] = {}
    for access in inputs:
        operand = access.buffer
        key = (id(operand), access.index)
        by_buffer[key].append(access)
        operands[key] = operand

    result: list[tuple[Expr, int, isl.map]] = []
    seen: set[tuple[int, int]] = set()
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
                complete(operand, key[1], input_relation, output_relation) if pointwise else None
            )
            if relation is not None:
                result.append((operand, key[1], relation))
                seen.add(key)

    if insert_accesses is None:
        return tuple(result)

    destination_access, update_access = insert_accesses
    destination, destination_index = destination_access.buffer, destination_access.index
    update, update_index = update_access.buffer, update_access.index
    written = output_coverage
    destination_coverage = coverage(tuple(by_buffer.get((id(destination), destination_index), ())))
    update_accesses = tuple(by_buffer.get((id(update), update_index), ()))
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
        if partitioned and complete_identity and (id(destination), destination_index) not in seen:
            result.append((destination, destination_index, relation))
            seen.add((id(destination), destination_index))

    try:
        update_matches_write = update_coverage is not None and update_coverage.is_equal(written)
    except isl.Error:
        update_matches_write = False
    update_relation = (
        complete(update, update_index, access_relation(update_accesses), output_relation)
        if update_matches_write
        else None
    )
    if update_relation is not None and (id(update), update_index) not in seen:
        result.append((update, update_index, update_relation))
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
        if isinstance(held, TupleType):
            if is_input:
                continue
            held = held.fields[output_index]
        if not isinstance(held, TensorType):
            continue
        relation = boundary.relation.insert_dims(isl.dim_type.IN, 0, scope.depth)
        domain = scope.domain.insert_dims(
            isl.dim_type.SET, scope.depth, relation.dim(isl.dim_type.IN) - scope.depth
        )
        relation = relation.intersect_domain(domain)
        relation, precision = eliminate_parameters(
            relation, boundary.values, scope, held, narrow=True, retained=scope.domain_params
        )
        if is_literal_shape(held.shape):
            relation = relation.intersect_range(shape_to_isl_set(tuple(held.shape), {}))
        access = Access(index if is_input else None, relation, value, precision, output_index)
        (inputs if is_input else outputs).append(access)
    return replace(
        scope,
        accesses={"narrow": {id(node): (node, tuple(inputs))}},
        outputs={"narrow": {id(node): (node, tuple(outputs))}},
    )


class AllocationConstraintVisitor(ValuationVisitor[Storage]):
    """Build each SSA value with its own live interval and byte span."""

    def __init__(self, *, root_function, liveness: Liveness) -> None:
        super().__init__(root_function=root_function)
        self.definitions: dict[int, IterationScope] = {}
        projections = {}
        for interval in liveness.intervals:
            node = interval.value
            if (
                isinstance(node, Call)
                and isinstance(node.target, TupleGetItem)
                and isinstance(node.args[1], Constant)
                and type(node.args[1].value) is int
            ):
                begin, count = leaf_span(node.args[0].type, node.args[1].value)
                projections[
                    id(node.args[0]),
                    interval.defined_at,
                ] = range(begin, begin + count)
        self.uses: dict[tuple[int, int], list[UseEvent]] = defaultdict(list)
        for use in liveness.uses:
            indexes = projections.get((id(use.value), use.at)) if not use.synthetic else None
            if indexes is None:
                indexes = range(len(leaves_of(use.value.type)))
            for index in indexes:
                self.uses[id(use.value), index].append(use)

    def dispatch_visit(self, node: Expr, ctx: AllocationModel) -> Storage:
        if id(node) not in self._memo:
            self.definitions[id(node)] = ctx.current
        return super().dispatch_visit(node, ctx)

    def update_alias(
        self, node: Call, update: AllocationValue, relation: isl.map, ctx: AllocationModel
    ) -> BufferAlias | None:
        """Bind an update's window to values visible at its producer."""
        if any(
            relation.get_dim_name(isl.dim_type.PARAM, axis) not in ctx.current.domain_params
            for axis in range(relation.dim(isl.dim_type.PARAM))
        ):
            return None
        held = _unprefix(relation, ctx.current)
        values = {name: ctx.current.capture_root(value) for name, value in held.values.items()}
        destination = ctx.current.capture_root(node.args[0])
        local = replace(ctx, definitions=self.definitions)
        if not _visible_at(local, destination, update) or not all(
            _visible_at(local, value, update) for value in values.values()
        ):
            return None
        return BufferAlias(destination, AccessRelation(held.relation, values))

    def result_interval(self, node: Expr, index: int, ctx: AllocationModel):
        """Use each tuple leaf's own consumers while keeping tensor intervals intact."""
        interval = ctx.liveness.interval_of(node)
        if isinstance(node.type, TupleType):
            interval = replace(
                interval,
                last_used_at=max(
                    (
                        use.at
                        for use in self.uses.get(
                            (
                                id(node),
                                index,
                            ),
                            (),
                        )
                    ),
                    default=interval.defined_at,
                ),
            )
        return interval

    def same_storage(
        self,
        node: Expr,
        storage: Storage,
        ctx: AllocationModel,
        *,
        held=None,
        index: int = 0,
    ) -> Storage:
        """Give forwarded values their own intervals and a structural Alias."""
        if storage is None:
            return None
        if held is None:
            held = node.type
        if isinstance(held, TupleType):
            fields = []
            for field_type, field in zip(held.fields, storage, strict=True):
                fields.append(self.same_storage(node, field, ctx, held=field_type, index=index))
                index += len(leaves_of(field_type))
            return tuple(fields)
        if not isinstance(held, TensorType):
            return None
        operand = storage
        interval = self.result_interval(node, index, ctx)
        lifetime = replace(
            operand.lifetime,
            defined_at=interval.defined_at,
            last_used_at=interval.last_used_at,
            persistent=False,
        )
        result = self.add_value(node, index, lifetime, ctx)
        add_alias(result, operand, AliasKind.FORWARD)
        return result

    def add_value(
        self, node: Expr, index: int, lifetime: ValueLifetime, ctx: AllocationModel
    ) -> AllocationValue:
        """Register one result and, at addressable levels, its address variable."""
        result = AllocationValue(node, index, lifetime)
        ctx.values.append(result)
        return result

    def storage_of(self, node: Expr) -> Storage:
        """Read completed storage in this definition's traversal."""
        return self._memo[id(node)][1]

    def visit_function_body(self, fn: Function, ctx: AllocationModel) -> Storage:
        """Visit function declarations before their body in the one traversal."""
        for parameter in fn.params:
            self.visit(parameter, ctx)
        return super().visit_function_body(fn, ctx)

    def allocate(self, node: Expr, ctx: AllocationModel) -> Storage:
        """Create material results with exactly their SSA live intervals."""
        if isinstance(node.type, TupleType) and result_copies(node) > 1:
            raise AnalysisError("allocation: staged tuple results are not supported")
        key = id(node)
        quantities = ctx.quantities.get(key, ())
        if not quantities:
            return None
        slots: list[Storage] = []
        for index, quantity in enumerate(quantities):
            if quantity is None:
                slots.append(None)
                continue
            level, amount = quantity
            interval = self.result_interval(node, index, ctx)
            lifetime = ValueLifetime(
                binding="",
                memory_level=level,
                bytes=amount,
                persistent=key in ctx.persistent,
                defined_at=interval.defined_at,
                last_used_at=interval.last_used_at,
            )
            result = self.add_value(node, index, lifetime, ctx)
            ctx.material.add(result)
            slots.append(result)
        return tuple(slots) if isinstance(node.type, TupleType) else slots[0]

    def visit_Var(self, node: Var, ctx: AllocationModel) -> Storage:
        source = ctx.capture_root(node)
        if source is not node:
            return self.same_storage(node, self.visit(source, ctx), ctx)
        return self.allocate(node, ctx)

    def visit_leaf_Tuple(
        self, node: Tuple, operands: tuple[Storage, ...], ctx: AllocationModel
    ) -> Storage:
        return self.same_storage(node, operands, ctx)

    def visit_MeshRegion(self, node: MeshRegion, ctx: AllocationModel) -> Storage:
        self.visit_operands_outside(node, ctx)
        inner = self.region_context(node, ctx)
        for parameter in node.params:
            self.visit(parameter, inner)
        storage = self.visit(node.body, inner)
        return self.same_storage(node, storage, ctx)

    def visit_LoopRegion(self, node: LoopRegion, ctx: AllocationModel) -> Storage:
        initials = tuple(self.visit(argument, ctx) for argument in node.args)
        inner = self.region_context(node, ctx)
        self.visit(node.induction_var, inner)
        carried = tuple(
            self.visit(parameter, inner) for parameter in node.params[: len(node.yield_values)]
        )
        for parameter in node.params[len(node.yield_values) :]:
            self.visit(parameter, inner)
        for result, initial in zip(carried, initials, strict=False):
            for value, operand in zip(leaves(result), leaves(initial), strict=True):
                add_alias(value, operand, AliasKind.CARRY_INIT)
        body = self.visit(node.body, inner)
        for yielded, phi in zip(node.yield_values, carried, strict=True):
            for value, operand in zip(leaves(self.visit(yielded, inner)), leaves(phi), strict=True):
                add_alias(value, operand, AliasKind.CARRY_YIELD)
                ctx.carry_loops[value, operand] = node
        storage = carried[0] if len(carried) == 1 else carried if carried else body
        return self.same_storage(node, storage, ctx)

    def visit_Call(self, node: Call, ctx: AllocationModel) -> Storage:
        context = ctx
        callee = None if context is None else CallFrameVisitor.callee_frame(node, context.frame)
        if callee is None:
            return self.default_visit_leaf(node, self.visit_operands(node, ctx), ctx)
        self.visit_operands(node, ctx)
        local = context.definitions[id(callee.function)].allocation
        result = self.allocate(node, ctx)
        results = leaves(result)
        frame = context.call_context(node, event_base=context.call_events[id(node)])
        bindings = []
        for formal, argument in frame.bindings:
            formal_values = sorted(
                (value for value in local.parameters if value.value is formal),
                key=lambda value: value.index,
            )
            for parameter, actual in zip(
                formal_values, leaves(self.storage_of(argument)), strict=True
            ):
                end = frame.event(local.partition[parameter].lifetime.last_used_at)
                actual.lifetime = replace(
                    actual.lifetime, last_used_at=max(actual.lifetime.last_used_at, end)
                )
                self.uses[id(actual.value), actual.index].append(UseEvent(actual.value, end))
                bindings.append((parameter, actual))
        call_reads = defaultdict(set)
        for index, value in enumerate(results):
            returned = local.returned[index]
            start = frame.event(local.partition[returned].lifetime.defined_at)
            value.lifetime = replace(value.lifetime, defined_at=start)
            for parameter in local.returns.get(index, ()):
                actual = next((actual for formal, actual in bindings if formal is parameter), None)
                if actual is not None:
                    source = _source_leaves(context.sources["narrow"][id(actual.value)])[
                        actual.index
                    ]
                    precision = _source_leaves(context.precisions["narrow"][id(actual.value)])[
                        actual.index
                    ]
                    if precision is not AnalysisPrecision.EXACT or not is_zero_offset(
                        source.relation.relation
                    ):
                        continue
                    target = self.source_storage(source.buffer, source.index, node, ctx)
                    if target is not None:
                        add_alias(value, target, AliasKind.RETURN_REUSE)
                        ctx.proofs.setdefault(value, set()).add(target)
                        call_reads[value].add(actual)
        for value in results:
            identity = _bound_relation(ctx, identity_of(value), ctx.current, value.value.type)
            ctx.writes[value] = Access(
                None, identity, node, output_index=value.index, index=value.index
            )
            ctx.reads[value] = tuple(
                Access(None, identity, actual.value, index=actual.index)
                for actual in call_reads[value]
            )
        ctx.call_bindings[id(node)] = tuple(bindings)
        ctx.call_results[id(node)] = results
        return result

    def source_storage(self, operand, index, node, ctx):
        """Locate a source buffer, preserving retained-call boundary storage."""
        if id(operand) in self._memo:
            return next(
                (value for value in leaves(self.storage_of(operand)) if value.index == index), None
            )
        for argument in node.args:
            source = ctx.sources["narrow"].get(id(argument))

            for held, value in zip(
                _source_leaves(source), leaves(self.storage_of(argument)), strict=False
            ):
                if (
                    held is not None
                    and held.buffer is operand
                    and held.index == index
                    and is_zero_offset(held.relation.relation)
                ):
                    return value
        return None

    def default_visit_leaf(
        self, node: Expr, operands: tuple[Storage, ...], ctx: AllocationModel
    ) -> Storage:
        """Create values without recording storage-sharing relations."""
        alias = None
        if isinstance(node, Call):
            if isinstance(node.target, TupleGetItem):
                if operands[0] is None:
                    return None
                index = node.args[1]
                if isinstance(index, Constant) and type(index.value) is int:
                    return self.same_storage(node, operands[0][index.value], ctx)
                held = leaves_of(node.type)
                for field in operands[0]:
                    for slot, operand in enumerate(flatten(field)):
                        if not isinstance(operand, AllocationValue):
                            continue
                        possible = self.same_storage(
                            node, operand, ctx, held=held[slot], index=slot
                        )
                        possible.lifetime = replace(
                            possible.lifetime,
                            last_used_at=ctx.liveness.timeline_end,
                        )
                return None
            logical = replace(ctx, topology_level=None)
            local_relations = (
                ctx.current.projected_relations(node, logical)
                if id(node) in ctx.current.relations
                else ()
            )
            alias = buffer_alias(node, logical, local_relations)
        if alias is not None:
            source = self.storage_of(alias.buffer)
            if isinstance(source, tuple):
                source = leaves(source)[alias.index]
            result = self.same_storage(node, source, ctx)
            if isinstance(result, AllocationValue):
                result.aliases = (Alias(source, alias.relation, AliasKind.VIEW),)
                if is_zero_offset(alias.relation.relation):
                    ctx.proofs.setdefault(result, set()).add(source)
            return result
        result = self.allocate(node, ctx)
        if not isinstance(node, Call) or not isinstance(result, AllocationValue):
            return result
        proof_scope = operand_scope(node, ctx)
        recorded = proof_scope.accesses.get("narrow", {}).get(id(node))
        ctx.reads[result] = recorded[1] if recorded else ()
        output = proof_scope.outputs.get("narrow", {}).get(id(node))
        if output and output[1]:
            ctx.writes[result] = output[1][0]
        insert = None
        if isinstance(node.target, InsertSlice) and recorded is not None:
            accesses = recorded[1]
            destination = next((access for access in accesses if access.input_index == 0), None)
            update = next((access for access in accesses if access.input_index == 1), None)
            if destination is not None and update is not None:
                insert = destination, update
        for operand, index, relation in operand_to_result_relation(node, proof_scope, insert):
            storage = self.storage_of(operand)
            sources = leaves(storage)
            source = next((item for item in sources if item.index == index), None)
            if source is None:
                continue
            if (
                is_zero_offset(relation)
                and insert is not None
                and operand is insert[0].buffer
                and index == insert[0].index
            ):
                add_alias(result, source, AliasKind.INSERT_SLICE)
                output = ctx.writes.get(result)
                if output is not None:
                    ctx.writes[result] = replace(
                        output, relation=output.relation.union(insert[0].relation)
                    )
            if insert is not None and operand is insert[1].buffer and index == insert[1].index:
                published = self.update_alias(node, source, relation, ctx)
                if published is not None:
                    destination_storage = leaves(self.storage_of(insert[0].buffer))
                    destination = next(
                        (item for item in destination_storage if item.index == insert[0].index),
                        None,
                    )
                    if destination is not None:
                        add_alias(source, result, AliasKind.WINDOW, published.relation)
                        ctx.window_candidates[source, result] = result, destination, published
        resolved = ctx.current.accesses.get("narrow", {}).get(id(node))
        resolved_insert = None
        if insert is not None and resolved is not None:
            destination = next((access for access in resolved[1] if access.input_index == 0), None)
            update = next((access for access in resolved[1] if access.input_index == 1), None)
            if destination is not None and update is not None:
                resolved_insert = destination, update
        for operand, index, relation in operand_to_result_relation(
            node, ctx.current, resolved_insert
        ):
            if not is_zero_offset(relation):
                continue
            source = self.source_storage(operand, index, node, ctx)
            if source is None:
                continue
            ctx.proofs.setdefault(result, set()).add(source)
            if (
                insert is None
                and source.lifetime.last_used_at <= result.lifetime.defined_at
                and not source.lifetime.persistent
                and source.lifetime.memory_level == result.lifetime.memory_level
            ):
                add_alias(result, source, AliasKind.IN_PLACE)
        return result


def build_interference_graph(
    values: tuple[Buffer, ...], ctx: AllocationModel
) -> set[tuple[int, int]]:
    """Pairs whose address ranges must not overlap."""
    result: set[tuple[int, int]] = set()
    for left, right in combinations(range(len(values)), 2):
        if values[left].intersects(values[right]):
            result.add((left, right))
    return result


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
    order: list[set[int]],
    values: tuple[Buffer, ...],
    interference: set[tuple[int, int]],
    alignments: tuple[int, ...],
    limit: int,
    pinned: dict[int, int],
    floor: int,
) -> tuple[tuple[int, ...], int] | None:
    """Place alias components in *order*, each at the lowest address its conflicts leave.

    A component holding a pinned value takes that address; every other one starts
    at *floor*, aligned to the component, or above. None when the components
    cannot share a pinned address or one would end past *limit*.
    """
    placed: list[tuple[set[int], int, int]] = []
    addresses = [0] * len(values)
    peak = 0
    for component in order:
        size = max(values[index].lifetime.bytes for index in component)
        alignment = max(alignments[index] for index in component)

        def conflicts(other: set[int]) -> bool:
            return any(
                tuple(sorted((left, right))) in interference
                for left in component
                for right in other
            )

        blocked = tuple(item for item in placed if conflicts(item[0]))
        pins = {pinned[index] for index in component if index in pinned}
        if len(pins) > 1:
            return None
        lowest = aligned(floor, alignment)
        candidates = sorted(
            pins
            or {
                lowest,
                *(
                    aligned(address + held, alignment)
                    for other, address, held in blocked
                    if other and aligned(address + held, alignment) >= lowest
                ),
            }
        )
        address = next(
            (
                candidate
                for candidate in candidates
                if all(
                    candidate + size <= other_address or other_address + other_size <= candidate
                    for other, other_address, other_size in blocked
                    if other
                )
            ),
            None,
        )
        if address is None or address + size > limit:
            return None
        for index in component:
            addresses[index] = address
        placed.append((component, address, size))
        peak = max(peak, address + size)
    return tuple(addresses), peak


def _satisfies(
    addresses: tuple[int, ...],
    values: tuple[Buffer, ...],
    interference: set[tuple[int, int]],
    alignments: tuple[int, ...],
    limit: int,
    pinned: dict[int, int],
    floor: int,
) -> bool:
    """Whether a seed meets every constraint the solver states about addresses."""
    for index, (address, item, alignment) in enumerate(
        zip(addresses, values, alignments, strict=True)
    ):
        if address % alignment or address + item.lifetime.bytes > limit:
            return False
        if index in pinned and address != pinned[index]:
            return False
        if index not in pinned and address < floor:
            return False
    return all(
        addresses[left] + values[left].lifetime.bytes <= addresses[right]
        or addresses[right] + values[right].lifetime.bytes <= addresses[left]
        for left, right in interference
    )


def calculate_starts(
    values: tuple[Buffer, ...],
    interference: set[tuple[int, int]],
    alignments: tuple[int, ...],
    limit: int,
    *,
    pinned: dict[int, int] | None = None,
    floor: int = 0,
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
    components = {index: {index} for index in range(len(values))}

    def size(component: set[int]) -> int:
        return max(values[index].lifetime.bytes for index in component)

    def first(component: set[int]) -> int:
        return min(values[index].lifetime.defined_at for index in component)

    by_definition = sorted(components.values(), key=lambda c: (first(c), -size(c)))
    largest_first = sorted(components.values(), key=lambda c: (-size(c), first(c)))
    attempts = (
        (by_definition, {}, 0),
        (by_definition, pinned, floor),
        (largest_first, pinned, floor),
    )
    seeds = [
        seed
        for order, pins, lowest in attempts
        if (seed := _first_fit(order, values, interference, alignments, limit, pins, lowest))
        is not None
        and _satisfies(seed[0], values, interference, alignments, limit, pinned, floor)
    ]
    return min(seeds, key=lambda seed: seed[1]) if seeds else None


def solve_allocation(
    memory_level: str,
    context: AllocationModel,
    options: SolverOptions,
) -> AllocationResult:
    """Return the first feasible whole-function placement for one level."""
    values = tuple(
        value for value in context.buffers if value.lifetime.memory_level == memory_level
    )
    if not values:
        return AllocationResult(0, "optimal")

    alignments = tuple(item.alignment for item in values)
    largest = max(item.lifetime.bytes for item in values)
    limit = sum(
        aligned(item.lifetime.bytes, alignment)
        for item, alignment in zip(values, alignments, strict=True)
    )
    model = context.models[memory_level]
    peak = model.new_int_var(largest, limit, f"{memory_level}_peak")
    addresses = tuple(context.address[item] for item in values)
    for index, (address, item, alignment) in enumerate(
        zip(addresses, values, alignments, strict=True)
    ):
        model.add(address + item.lifetime.bytes <= peak)
        model.add_modulo_equality(0, address, alignment).with_name(f"align_{index}")

    persistent_end = 0
    pinned: dict[int, int] = {}
    for index, (address, item, alignment) in enumerate(
        zip(addresses, values, alignments, strict=True)
    ):
        if not item.lifetime.persistent:
            continue
        persistent_end = aligned(persistent_end, alignment)
        offset = persistent_end
        persistent_end += item.lifetime.bytes
        model.add(address == offset)
        pinned[index] = offset
    for address, item in zip(addresses, values, strict=True):
        if not item.lifetime.persistent:
            model.add(address >= persistent_end)

    interference = build_interference_graph(values, context)
    order_choices: dict[tuple[int, int], tuple[cp_model.IntVar, cp_model.IntVar]] = {}
    for left, right in sorted(interference):
        if values[left].lifetime.persistent or values[right].lifetime.persistent:
            continue
        left_before = model.new_bool_var(f"before_{left}_{right}")
        right_before = model.new_bool_var(f"before_{right}_{left}")
        order_choices[(left, right)] = (left_before, right_before)
        model.add(
            addresses[left] + values[left].lifetime.bytes <= addresses[right]
        ).only_enforce_if(left_before)
        model.add(
            addresses[right] + values[right].lifetime.bytes <= addresses[left]
        ).only_enforce_if(right_before)
        model.add_bool_or(left_before, right_before)

    seed = calculate_starts(
        values,
        interference,
        alignments,
        limit,
        pinned=pinned,
        floor=persistent_end,
    )
    if seed is not None:
        address_hints, peak_hint = seed
        model.add(peak <= peak_hint)
        hinted = set()
        for address, suggested in zip(addresses, address_hints, strict=True):
            if address.index not in hinted:
                model.add_hint(address, suggested)
                hinted.add(address.index)
        for (left, right), (left_before, right_before) in order_choices.items():
            left_end = address_hints[left] + values[left].lifetime.bytes
            right_end = address_hints[right] + values[right].lifetime.bytes
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
            f"({status_name}); {len(values)} Buffers conflict with "
            f"{len(interference)} interference constraints"
        )
    actual_peak = max(
        solver.value(address) + item.lifetime.bytes
        for address, item in zip(addresses, values, strict=True)
    )
    for left, right in interference:
        left_offset, right_offset = solver.value(addresses[left]), solver.value(addresses[right])
        if not (
            left_offset + values[left].lifetime.bytes <= right_offset
            or right_offset + values[right].lifetime.bytes <= left_offset
        ):
            a, b = values[left].lifetime, values[right].lifetime
            raise AnalysisError(
                f"allocation: overlapping {memory_level} Buffers: {a.binding} [{a.defined_at},{a.last_used_at}] at {left_offset}; {b.binding} [{b.defined_at},{b.last_used_at}] at {right_offset}"
            )
    return AllocationResult(
        actual_peak,
        "feasible",
        {item: solver.value(address) for item, address in zip(values, addresses, strict=True)},
    )


__all__ = [
    "AliasKind",
    "AllocationConstraintVisitor",
    "AllocationModel",
    "AllocationResult",
    "AllocationValue",
    "Storage",
    "solve_allocation",
]
