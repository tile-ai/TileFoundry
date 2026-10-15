"""Internal values and solver results for physical memory placement."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
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
from tilefoundry.ir.visitor import ExprVisitor
from tilefoundry.utils.isl_utils import equates
from tilefoundry.visitor_registry.access_relation import AccessRelation, leaf_span, leaves_of
from tilefoundry.visitor_registry.contexts import CostContext

from .access import Access, BufferAlias, buffer_alias
from .errors import AnalysisError
from .iteration_scope import IterationScope, walk_scopes
from .liveness import Liveness, UseEvent, result_copies
from .metadata import ValueLifetime
from .precision import AnalysisPrecision


class SolverOptions(Protocol):
    """Placement limits consumed independently of the memory front end."""

    timeout_seconds: float
    workers: int
    random_seed: int


@dataclass(eq=False)
class AllocationValue:
    """One SSA result: its source, result index, and own live interval."""

    value: Expr
    index: int
    lifetime: ValueLifetime

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
    offsets: dict[AllocationValue, int] = field(default_factory=dict)


@dataclass
class AllocationModel:
    """All memory levels and independent values collected by one HIR visitor."""

    current: IterationScope
    liveness: Liveness
    local: CostContext
    quantities: dict[int, tuple[tuple[str, int] | None, ...]]
    persistent: frozenset[int]
    limits: dict[str, int]
    values: list[AllocationValue] = field(default_factory=list)
    material: set[AllocationValue] = field(default_factory=set)
    models: dict[str, cp_model.CpModel] = field(default_factory=dict)
    address: dict[AllocationValue, cp_model.IntVar] = field(default_factory=dict)
    resources: dict = field(default_factory=dict)
    parameters: set[AllocationValue] = field(default_factory=set)
    outputs: set[AllocationValue] = field(default_factory=set)

    def addresses(self) -> set[AllocationValue]:
        """Create an independent address variable for every addressable value."""
        limits = {
            level: sum(
                aligned(value.lifetime.bytes, alignment_of(value.value))
                for value in self.values
                if value.lifetime.memory_level == level
            )
            for level in self.models
        }
        for index, value in enumerate(self.values):
            level = value.lifetime.memory_level
            if level in self.models:
                self.address[value] = self.models[level].new_int_var(
                    0, limits[level], f"address_{index}"
                )
        return set()

    def __post_init__(self) -> None:
        for level in self.limits:
            if level in ("gmem", "smem") and level not in self.models:
                self.models[level] = cp_model.CpModel()


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
    destination_coverage = coverage(
        tuple(by_buffer.get((id(destination), destination_index), ()))
    )
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
        if (
            partitioned
            and complete_identity
            and (id(destination), destination_index) not in seen
        ):
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


class AllocationConstraintVisitor(ExprVisitor[Storage]):
    """Build each SSA value with its own live interval and byte span."""

    def __init__(self, *, root_function, liveness: Liveness) -> None:
        super().__init__(root_function=root_function)
        self.bound: dict[int, Storage] = {}
        self.resources = {}
        self.escaped = {}
        self.definitions: dict[int, IterationScope] = {}
        projections = {}
        for interval in liveness.intervals:
            node = interval.value
            if (
                isinstance(node, Call) and isinstance(node.target, TupleGetItem)
                and isinstance(node.args[1], Constant) and type(node.args[1].value) is int
            ):
                begin, count = leaf_span(node.args[0].type, node.args[1].value)
                projections[
                    id(node.args[0]),
                    interval.defined_at,
                ] = range(begin, begin + count)
        self.uses: dict[tuple[int, int], list[UseEvent]] = defaultdict(list)
        for use in liveness.uses:
            indexes = (
                projections.get((id(use.value), use.at))
                if not use.synthetic
                else None
            )
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
        source_scope = self.definitions[id(update.value)]
        visible: set[IterationScope] = set()
        cursor: IterationScope | None = source_scope
        while cursor is not None:
            visible.add(cursor)
            cursor = cursor.parent
        names = tuple(
            relation.get_dim_name(isl.dim_type.PARAM, axis)
            for axis in range(relation.dim(isl.dim_type.PARAM))
        )
        if any(name not in ctx.current.domain_params for name in names):
            return None
        values = {name: ctx.current.domain_params[name] for name in names}
        for axis, loop in enumerate(ctx.current.enclosing_loops()):
            name = f"__tf_alias_loop_{axis}"
            while name in values:
                name += "_"
            relation = relation.set_dim_name(isl.dim_type.IN, axis, name)
            values[name] = loop.induction_var
        relation = relation.move_dims(
            isl.dim_type.PARAM, relation.dim(isl.dim_type.PARAM),
            isl.dim_type.IN, 0, ctx.current.depth,
        )
        values = {name: ctx.current.capture_root(value) for name, value in values.items()}

        def is_visible(value: object) -> bool:
            if isinstance(value, (int, Constant, DimVar)):
                return True
            if isinstance(value, Call) and isinstance(value.target, MeshCoord):
                mesh = value.target.mesh
                axis = static_dim_value(value.args[0])
                offset = 0
                for topology, layout in zip(mesh.topologies, levels(mesh), strict=True):
                    width = len(flatten(layout.shape))
                    if axis is None or offset <= axis < offset + width:
                        name = getattr(topology, "name", topology)
                        if not any(
                            isinstance(scope.owner, MeshRegion)
                            and name in {
                                getattr(held, "name", held)
                                for held in scope.owner.mesh.topologies
                            }
                            for scope in visible
                        ):
                            return False
                    offset += width
                return True
            interval = ctx.liveness.interval_of(value)
            return (
                self.definitions.get(id(value)) in visible
                and interval is not None and interval.defined_at < update.lifetime.defined_at
            )

        destination = ctx.current.capture_root(node.args[0])
        if not is_visible(destination) or not all(is_visible(value) for value in values.values()):
            return None
        return BufferAlias(destination, AccessRelation(relation, values))

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
        self, node: Expr, storage: Storage, ctx: AllocationModel, *, held=None, index: int = 0,
    ) -> Storage:
        """Give forwarded values their own intervals and byte spans."""
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
        )
        result = self.add_value(node, index, lifetime, ctx)
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
        if id(node) in self.bound:
            return self.same_storage(node, self.bound[id(node)], ctx)
        return self.allocate(node, ctx)

    def visit_leaf_Tuple(
        self, node: Tuple, operands: tuple[Storage, ...], ctx: AllocationModel
    ) -> Storage:
        return self.same_storage(node, operands, ctx)

    def visit_MeshRegion(self, node: MeshRegion, ctx: AllocationModel) -> Storage:
        child = next(item for item in walk_scopes(ctx.current) if item.owner is node)
        for parameter, argument in node.captures():
            self.bound[id(parameter)] = self.visit(argument, ctx)
            self.visit(parameter, ctx)
        storage = self.visit(node.body, replace(ctx, current=child))
        return self.same_storage(node, storage, ctx)

    def visit_LoopRegion(self, node: LoopRegion, ctx: AllocationModel) -> Storage:
        child = next(item for item in walk_scopes(ctx.current) if item.owner is node)
        inner = replace(ctx, current=child)
        for argument in node.args:
            self.visit(argument, ctx)
        for parameter, argument in node.captures():
            self.bound[id(parameter)] = self.visit(argument, ctx)
        self.visit(node.induction_var, inner)
        carried = tuple(
            self.visit(parameter, inner) for parameter in node.params[: len(node.yield_values)]
        )
        for parameter in node.params[len(node.yield_values):]:
            self.visit(parameter, inner)
        body = self.visit(node.body, inner)
        for operand in node.yield_values:
            self.visit(operand, inner)
        storage = carried[0] if len(carried) == 1 else carried if carried else body
        return self.same_storage(node, storage, ctx)

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
                        possible = self.same_storage(node, operand, ctx, held=held[slot], index=slot)
                        possible.lifetime = replace(
                            possible.lifetime,
                            last_used_at=ctx.liveness.timeline_end,
                        )
                return None
            logical = replace(ctx.local, topology_level=None)
            local_relations = (
                ctx.current.projected_relations(node, logical)
                if id(node) in ctx.current.relations else ()
            )
            alias = buffer_alias(node, logical, local_relations)
        result = (
            self.same_storage(node, self.storage_of(alias.buffer), ctx)
            if alias is not None else self.allocate(node, ctx)
        )
        return result


def build_interference_graph(
    values: tuple[AllocationValue, ...], ctx: AllocationModel
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
    values: tuple[AllocationValue, ...],
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
    values: tuple[AllocationValue, ...],
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
    values: tuple[AllocationValue, ...],
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
    values = tuple(value for value in context.values if value.lifetime.memory_level == memory_level)
    if not values:
        return AllocationResult(0, "optimal")

    alignments = tuple(alignment_of(item.value) for item in values)
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
            f"({status_name}); {len(values)} values conflict with "
            f"{len(interference)} interference constraints"
        )
    actual_peak = max(
        solver.value(address) + item.lifetime.bytes
        for address, item in zip(addresses, values, strict=True)
    )
    return AllocationResult(
        actual_peak, "feasible",
        {item: solver.value(address) for item, address in zip(values, addresses, strict=True)},
    )


__all__ = [
    "AllocationConstraintVisitor",
    "AllocationModel",
    "AllocationResult",
    "AllocationValue",
    "Storage",
    "solve_allocation",
]
