"""Internal values and solver results for physical memory placement."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from itertools import combinations
from typing import Protocol

import isl
from ortools.sat.python import cp_model

from tilefoundry.ir.core import Call, Constant, Expr, Tuple, Var, value_labels
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.tensor.insert_slice import InsertSlice
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.isl_interop import shape_to_isl_set
from tilefoundry.ir.types import TensorType, TupleType
from tilefoundry.ir.types.utils import is_literal_shape, local_type_of, tensor_types
from tilefoundry.ir.visitor import ExprVisitor
from tilefoundry.utils.isl_utils import equates
from tilefoundry.visitor_registry.contexts import CostContext

from .access import Access, buffer_alias
from .errors import AnalysisError
from .iteration_scope import IterationScope, walk_scopes
from .liveness import Liveness, UseEvent, result_copies
from .precision import AnalysisPrecision


class SolverOptions(Protocol):
    """Placement limits consumed without depending on the memory front end."""

    timeout_seconds: float
    workers: int
    random_seed: int


@dataclass(eq=False)
class Box:
    """One storage the placement solver addresses."""

    producer: Expr
    leaf: int
    memory_level: str
    bytes: int
    persistent: bool
    defined_at: int
    last_used_at: int
    uses: list[UseEvent] = field(default_factory=list)
    offset: int | None = None

    def intersects(self, other: Box) -> bool:
        """Whether two closed structured-SSA intervals share an event."""
        return max(self.defined_at, other.defined_at) <= min(
            self.last_used_at, other.last_used_at
        )


@dataclass(frozen=True)
class OneOf:
    """Every possible leaf of a dynamically selected tuple field."""

    boxes: tuple[Box, ...]


type Storage = Box | OneOf | tuple[Storage, ...] | None


def leaves(storage: Storage) -> tuple[Box, ...]:
    """The boxes a structured storage may use."""
    if isinstance(storage, Box):
        return (storage,)
    if isinstance(storage, OneOf):
        return storage.boxes
    if isinstance(storage, tuple):
        return tuple(box for field in storage for box in leaves(field))
    return ()


def box_labels(boxes: tuple[Box, ...]) -> tuple[str, ...]:
    """Name each producer once, distinguishing tuple result leaves."""
    producers = {id(box.producer): box.producer for box in boxes}
    labels = dict(zip(producers, value_labels(producers.values()), strict=True))
    return tuple(
        labels[id(box.producer)] + (
            f"[{box.leaf}]" if isinstance(box.producer.type, TupleType) else ""
        )
        for box in boxes
    )


class AllocationGroups:
    """Required co-location components, merged as constraints are adopted."""

    def __init__(self) -> None:
        self.parent: dict[Box, Box] = {}

    def add(self, box: Box) -> None:
        self.parent[box] = box

    def find(self, box: Box) -> Box:
        parent = self.parent[box]
        if parent is not box:
            self.parent[box] = self.find(parent)
        return self.parent[box]

    def union(self, left: Box, right: Box) -> None:
        first, second = self.find(left), self.find(right)
        if first is second:
            return
        if first.defined_at > second.defined_at:
            first, second = second, first
        self.parent[second] = first

    def components(self, boxes: tuple[Box, ...]) -> tuple[int, ...]:
        """One stable component index for each box in definition order."""
        indices: dict[Box, int] = {}
        return tuple(indices.setdefault(self.find(box), i) for i, box in enumerate(boxes))


@dataclass(frozen=True)
class AllocationResult:
    """The first physical placement found for one memory level."""

    peak_bytes: int
    solver_status: str


@dataclass
class AllocationModel:
    """All memory levels while one HIR visitor applies placement relations."""

    current: IterationScope
    liveness: Liveness
    local: CostContext
    quantities: dict[int, tuple[tuple[str, int] | None, ...]]
    persistent: frozenset[int]
    limits: dict[str, int]
    boxes: list[Box] = field(default_factory=list)
    models: dict[str, cp_model.CpModel] = field(default_factory=dict)
    address: dict[Box, cp_model.IntVar] = field(default_factory=dict)
    groups: AllocationGroups = field(default_factory=AllocationGroups)

    def __post_init__(self) -> None:
        for level in self.limits:
            if level in ("gmem", "smem") and level not in self.models:
                self.models[level] = cp_model.CpModel()


def covers_result_each_iteration(
    outputs: isl.map,
    iteration_domain: isl.set,
    result_box: isl.set,
) -> bool:
    """Whether every enclosing-loop point writes the complete result box."""
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
    producer_of: Callable[[Expr], Expr],
) -> tuple[tuple[Expr, isl.map], ...]:
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

    def value_domain(value: Expr) -> isl.set | None:
        try:
            held = local_type_of(value.type)
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

    def complete(operand: Expr, inputs: isl.map | None, outputs: isl.map | None) -> isl.map | None:
        if inputs is None or outputs is None:
            return None
        relation = composed(inputs, outputs)
        expected = value_domain(operand)
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

    by_buffer: dict[int, list[Access]] = defaultdict(list)
    operands: dict[int, Expr] = {}
    for access in inputs:
        operand = access.buffer
        key = id(operand)
        by_buffer[key].append(access)
        operands[key] = operand

    result: list[tuple[Expr, isl.map]] = []
    seen: set[int] = set()
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
            relation = complete(operand, input_relation, output_relation) if pointwise else None
            if relation is not None:
                result.append((operand, relation))
                seen.add(key)

    if not isinstance(node.target, InsertSlice):
        return tuple(result)

    destination = producer_of(node.args[0])
    update = producer_of(node.args[1])
    written = output_coverage
    destination_coverage = coverage(tuple(by_buffer.get(id(destination), ())))
    update_accesses = tuple(by_buffer.get(id(update), ()))
    update_coverage = coverage(update_accesses)

    if destination_coverage is not None:
        relation = identity()
        expected_destination = value_domain(destination)
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
            and id(destination) not in seen
        ):
            result.append((destination, relation))
            seen.add(id(destination))

    try:
        update_matches_write = update_coverage is not None and update_coverage.is_equal(written)
    except isl.Error:
        update_matches_write = False
    update_relation = (
        complete(update, access_relation(update_accesses), output_relation)
        if update_matches_write
        else None
    )
    if (
        update_relation is not None
        and id(update) not in seen
    ):
        result.append((update, update_relation))
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
    """Return storage and apply each node/operand placement relation once."""

    def __init__(self, *, root_function, liveness: Liveness) -> None:
        super().__init__(root_function=root_function)
        self.bound: dict[int, Storage] = {}
        self.members: set[int] = set()
        self.pending: dict[int, list[tuple[Box, int, isl.map]]] = defaultdict(list)
        self.uses: dict[int, list[UseEvent]] = defaultdict(list)
        for use in liveness.uses:
            self.uses[id(use.value)].append(use)
        self.static_projections = {
            (id(interval.value.args[0]), interval.defined_at)
            for interval in liveness.intervals
            if isinstance(interval.value, Call)
            and isinstance(interval.value.target, TupleGetItem)
            and isinstance(interval.value.args[1], Constant)
            and type(interval.value.args[1].value) is int
        }

    def share(self, node: Expr, storage: Storage, ctx: AllocationModel) -> Storage:
        """Extend storage by member intervals, or actual uses of a tuple."""
        if id(node) in self.members:
            return storage
        self.members.add(id(node))
        interval = ctx.liveness.interval_of(node)
        if interval is not None:
            uses = self.uses.get(id(node), ())
            if isinstance(node.type, TupleType):
                uses = [
                    use for use in uses
                    if use.synthetic or (id(node), use.at) not in self.static_projections
                ]
                last_used_at = max((use.at for use in uses), default=-1)
            else:
                last_used_at = interval.last_used_at
            for box in leaves(storage):
                box.last_used_at = max(box.last_used_at, last_used_at)
                box.uses.extend(uses)
        return storage

    def storage_of(self, node: Expr) -> Storage:
        """Read completed storage; an unvisited value violates traversal order."""
        return self._memo[id(node)][1]

    def visit_function_body(self, fn: Function, ctx: AllocationModel) -> Storage:
        """Visit function declarations before their body in the one traversal."""
        for parameter in fn.params:
            self.visit(parameter, ctx)
        return super().visit_function_body(fn, ctx)

    def box(self, node: Expr, ctx: AllocationModel) -> Storage:
        if isinstance(node.type, TupleType) and result_copies(node) > 1:
            raise AnalysisError("allocation: staged tuple results are not supported")
        quantities = ctx.quantities.get(id(node), ())
        if not quantities:
            return None
        interval = ctx.liveness.interval_of(node)
        slots: list[Storage] = []
        for leaf, quantity in enumerate(quantities):
            if quantity is None:
                slots.append(None)
                continue
            level, amount = quantity
            box = Box(
                producer=node,
                leaf=leaf,
                memory_level=level,
                bytes=amount,
                persistent=id(node) in ctx.persistent,
                defined_at=interval.defined_at,
                last_used_at=(
                    ctx.liveness.timeline_end if id(node) in ctx.persistent
                    else interval.defined_at if isinstance(node.type, TupleType)
                    else interval.last_used_at
                ),
            )
            ctx.boxes.append(box)
            ctx.groups.add(box)
            if level in ctx.models:
                model = ctx.models[level]
                ctx.address[box] = model.new_int_var(
                    0, ctx.limits[level] - box.bytes, f"address_{len(ctx.address)}"
                )
            slots.append(box)
        storage = tuple(slots) if isinstance(node.type, TupleType) else slots[0]
        return self.share(node, storage, ctx)

    def visit_Var(self, node: Var, ctx: AllocationModel) -> Storage:
        storage = self.bound[id(node)] if id(node) in self.bound else self.box(node, ctx)
        return self.share(node, storage, ctx)

    def visit_leaf_Tuple(
        self, node: Tuple, operands: tuple[Storage, ...], ctx: AllocationModel
    ) -> Storage:
        return self.share(node, operands, ctx)

    def visit_MeshRegion(self, node: MeshRegion, ctx: AllocationModel) -> Storage:
        child = next(item for item in walk_scopes(ctx.current) if item.owner is node)
        for parameter, argument in node.captures():
            self.bound[id(parameter)] = self.visit(argument, ctx)
            self.visit(parameter, ctx)
        storage = self.visit(node.body, replace(ctx, current=child))
        return self.share(node, storage, ctx)

    def visit_LoopRegion(self, node: LoopRegion, ctx: AllocationModel) -> Storage:
        child = next(item for item in walk_scopes(ctx.current) if item.owner is node)
        inner = replace(ctx, current=child)
        initials = tuple(self.visit(argument, ctx) for argument in node.args)
        for parameter, argument in node.captures():
            self.bound[id(parameter)] = self.visit(argument, ctx)
        self.visit(node.induction_var, inner)
        carried = tuple(
            self.visit(parameter, inner) for parameter in node.params[: len(node.yield_values)]
        )
        for parameter in node.params[len(node.yield_values):]:
            self.visit(parameter, inner)
        for result, operand in zip(carried, initials, strict=False):
            if isinstance(result, Box) and isinstance(operand, Box):
                self.tie(result, operand, inner)
        body = self.visit(node.body, inner)
        yields = tuple(self.visit(operand, inner) for operand in node.yield_values)
        for source, slot, relation in self.pending.pop(id(node), ()):
            if yields[slot] is source:
                self.alias(source, carried[slot], relation, inner)
        for result, operand in zip(yields, carried, strict=True):
            if isinstance(result, Box) and isinstance(operand, Box):
                self.tie(result, operand, inner)
        storage = carried[0] if len(carried) == 1 else carried if carried else body
        return self.share(node, storage, ctx)

    def tie(self, result: Box, operand: Box, ctx: AllocationModel) -> None:
        """Require the single backing buffer stated by one loop-carried slot."""
        if result is operand or result.memory_level != operand.memory_level:
            return
        if operand.persistent or result.bytes != operand.bytes:
            return
        ctx.groups.union(result, operand)
        if result.memory_level in ctx.models:
            ctx.models[result.memory_level].add(ctx.address[result] == ctx.address[operand])

    def alias(
        self, result: Box, operand: Box, relation: isl.map, ctx: AllocationModel
    ) -> None:
        """Require one proven logical alias in the physical placement."""
        if result is operand or result.memory_level != operand.memory_level or operand.persistent:
            return
        if is_zero_offset(relation):
            if result.bytes > operand.bytes:
                return
            if result.memory_level in ctx.models:
                ctx.models[result.memory_level].add(ctx.address[result] == ctx.address[operand])
        else:
            if operand.bytes > result.bytes:
                return
            if result.memory_level in ctx.models:
                model = ctx.models[result.memory_level]
                model.add(ctx.address[operand] >= ctx.address[result])
                model.add(
                    ctx.address[operand] + operand.bytes <= ctx.address[result] + result.bytes
                )
        ctx.groups.union(result, operand)

    def default_visit_leaf(
        self, node: Expr, operands: tuple[Storage, ...], ctx: AllocationModel
    ) -> Storage:
        """Apply ordinary aliases now and defer carry aliases until loop yields."""
        if isinstance(node, Call):
            if isinstance(node.target, TupleGetItem):
                index = node.args[1]
                storage = (
                    operands[0][index.value]
                    if isinstance(index, Constant) and type(index.value) is int
                    else OneOf(leaves(operands[0]))
                )
                return self.share(node, storage, ctx)
            logical = replace(ctx.local, topology_level=None)
            local_relations = (
                ctx.current.projected_relations(node, logical)
                if id(node) in ctx.current.relations else ()
            )
            alias = buffer_alias(node, logical, local_relations)
            if alias is not None:
                return self.share(node, self.storage_of(alias.buffer), ctx)
        result = self.box(node, ctx)
        if isinstance(node, Call) and isinstance(result, Box):
            def producer_of(value: Expr) -> Expr:
                storage = self.storage_of(value)
                return storage.producer if isinstance(storage, Box) else value

            for operand, relation in operand_to_result_relation(node, ctx.current, producer_of):
                storage = self.storage_of(operand)
                if not isinstance(storage, Box):
                    continue
                if any(not use.synthetic and use.at > result.defined_at for use in storage.uses):
                    continue
                if storage.last_used_at <= result.defined_at:
                    self.alias(result, storage, relation, ctx)
                    continue
                cursor: IterationScope | None = ctx.current
                while cursor is not None:
                    loop = cursor.owner
                    if isinstance(loop, LoopRegion):
                        for slot, phi in enumerate(loop.params[: len(loop.yield_values)]):
                            if self.storage_of(phi) is storage:
                                self.pending[id(loop)].append((result, slot, relation))
                    cursor = cursor.parent
        return result


def build_interference_graph(
    boxes: tuple[Box, ...], ctx: AllocationModel
) -> set[tuple[int, int]]:
    """Pairs whose address ranges must not overlap."""
    result: set[tuple[int, int]] = set()
    for left, right in combinations(range(len(boxes)), 2):
        if ctx.groups.find(boxes[left]) is ctx.groups.find(boxes[right]):
            continue
        if boxes[left].intersects(boxes[right]):
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
    values: tuple[Box, ...],
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
        size = max(values[index].bytes for index in component)
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
    values: tuple[Box, ...],
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
        if address % alignment or address + item.bytes > limit:
            return False
        if index in pinned and address != pinned[index]:
            return False
        if index not in pinned and address < floor:
            return False
    return all(
        addresses[left] + values[left].bytes <= addresses[right]
        or addresses[right] + values[right].bytes <= addresses[left]
        for left, right in interference
    )


def calculate_starts(
    values: tuple[Box, ...],
    groups: AllocationGroups,
    interference: set[tuple[int, int]],
    alignments: tuple[int, ...],
    limit: int,
    *,
    pinned: dict[int, int] | None = None,
    floor: int = 0,
) -> tuple[tuple[int, ...], int] | None:
    """A complete aligned seed with every required alias merged, or None.

    Alias components are placed first-fit by first definition -- the original
    seed, then again holding persistent values at their pinned addresses -- and
    largest first. Each seed is checked against every address constraint the
    solver states, and the most compact one that meets them all is kept, the
    earlier attempt winning a tie. None when none does: a heuristic that fails
    proves nothing about the model, so it must not bound the search.
    """
    pinned = pinned or {}
    components_by_index = groups.components(values)
    components: dict[int, set[int]] = defaultdict(set)
    for index in range(len(values)):
        components[components_by_index[index]].add(index)

    def size(component: set[int]) -> int:
        return max(values[index].bytes for index in component)

    def first(component: set[int]) -> int:
        return min(values[index].defined_at for index in component)

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
    values = tuple(box for box in context.boxes if box.memory_level == memory_level)
    if not values:
        return AllocationResult(0, "optimal")

    alignments = tuple(alignment_of(item.producer) for item in values)
    largest = max(item.bytes for item in values)
    limit = sum(
        aligned(item.bytes, alignment)
        for item, alignment in zip(values, alignments, strict=True)
    )
    model = context.models[memory_level]
    peak = model.new_int_var(largest, limit, f"{memory_level}_peak")
    addresses = tuple(context.address[item] for item in values)
    for index, (address, item, alignment) in enumerate(
        zip(addresses, values, alignments, strict=True)
    ):
        model.add(address + item.bytes <= peak)
        model.add_modulo_equality(0, address, alignment).with_name(f"align_{index}")

    persistent_end = 0
    pinned: dict[int, int] = {}
    for index, (address, item, alignment) in enumerate(
        zip(addresses, values, alignments, strict=True)
    ):
        if not item.persistent:
            continue
        persistent_end = aligned(persistent_end, alignment)
        model.add(address == persistent_end)
        pinned[index] = persistent_end
        persistent_end += item.bytes
    for address, item in zip(addresses, values, strict=True):
        if not item.persistent:
            model.add(address >= persistent_end)

    interference = build_interference_graph(values, context)
    order_choices: dict[tuple[int, int], tuple[cp_model.IntVar, cp_model.IntVar]] = {}
    for left, right in sorted(interference):
        if values[left].persistent or values[right].persistent:
            continue
        left_before = model.new_bool_var(f"before_{left}_{right}")
        right_before = model.new_bool_var(f"before_{right}_{left}")
        order_choices[(left, right)] = (left_before, right_before)
        model.add(
            addresses[left] + values[left].bytes <= addresses[right]
        ).only_enforce_if(left_before)
        model.add(
            addresses[right] + values[right].bytes <= addresses[left]
        ).only_enforce_if(right_before)
        model.add_bool_or(left_before, right_before)

    seed = calculate_starts(
        values,
        context.groups,
        interference,
        alignments,
        limit,
        pinned=pinned,
        floor=persistent_end,
    )
    if seed is not None:
        address_hints, peak_hint = seed
        model.add(peak <= peak_hint)
        for address, suggested in zip(addresses, address_hints, strict=True):
            model.add_hint(address, suggested)
        for (left, right), (left_before, right_before) in order_choices.items():
            left_end = address_hints[left] + values[left].bytes
            right_end = address_hints[right] + values[right].bytes
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
        labels = box_labels(values)
        aliases = ", ".join(
            f"{labels[left]}={labels[right]}"
            for left, right in combinations(range(len(values)), 2)
            if context.groups.find(values[left]) is context.groups.find(values[right])
        )
        raise AnalysisError(
            f"allocation: no feasible {memory_level} placement was found "
            f"({status_name}); required aliases [{aliases}] conflict with "
            f"{len(interference)} interference constraints"
        )
    actual_peak = max(
        solver.value(address) + item.bytes
        for address, item in zip(addresses, values, strict=True)
    )
    for box, address in zip(values, addresses, strict=True):
        box.offset = solver.value(address)
    return AllocationResult(actual_peak, "feasible")


__all__ = [
    "AllocationConstraintVisitor",
    "AllocationModel",
    "AllocationResult",
    "Box",
    "OneOf",
    "Storage",
    "solve_allocation",
]
