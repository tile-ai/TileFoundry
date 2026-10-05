"""Internal values and solver results for physical memory placement."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from itertools import combinations
from typing import Protocol

import isl
from ortools.sat.python import cp_model

from tilefoundry.ir.core import Call, Expr
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.tensor.insert_slice import InsertSlice
from tilefoundry.ir.isl_interop import shape_to_isl_set
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.utils import is_literal_shape, local_type_of, tensor_types
from tilefoundry.ir.visitor import ExprVisitor
from tilefoundry.utils.isl_utils import equates
from tilefoundry.visitor_registry.access_relation import renaming_relation
from tilefoundry.visitor_registry.buffer_alias import aliased_operand
from tilefoundry.visitor_registry.contexts import TypeInferContext

from .access import Access
from .errors import AnalysisError
from .iteration_scope import IterationScope, walk_scopes
from .liveness import Liveness, storage_source
from .metadata import ValueLifetime
from .precision import AnalysisPrecision


class SolverOptions(Protocol):
    """Placement limits consumed without depending on the memory front end."""

    timeout_seconds: float
    workers: int
    random_seed: int


@dataclass(frozen=True)
class AllocationValue:
    """Connect one logical expression to its target-aware lifetime."""

    value: Expr
    lifetime: ValueLifetime

    def intersects(self, other: AllocationValue) -> bool:
        """Whether two closed structured-SSA intervals share an event."""
        return max(self.lifetime.defined_at, other.lifetime.defined_at) <= min(
            self.lifetime.last_used_at, other.lifetime.last_used_at
        )


@dataclass(frozen=True)
class AllocationResult:
    """The first physical placement found for one memory level."""

    peak_bytes: int
    solver_status: str
    offsets: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class AliasConstraint:
    """An exact logical relation from one material operand into a result."""

    operand: Expr
    relation: isl.map


@dataclass
class AllocationModel:
    """One memory-level model while the HIR visitor applies logical relations."""

    current: IterationScope
    liveness: Liveness
    values: tuple[AllocationValue, ...]
    boxes_by_expr: dict[int, int]
    model: cp_model.CpModel
    addresses: tuple[cp_model.IntVar, ...]
    owners: dict[int, Expr]
    aliased: set[tuple[int, int]] = field(default_factory=set)


def storage_owners(root: IterationScope, liveness: Liveness) -> dict[int, Expr]:
    """Resolve storage once per value and prove each registered alias once."""
    declarations = {key: scope for scope in walk_scopes(root) for key in scope.relations}
    owners: dict[int, Expr] = {}

    def resolve(value: Expr) -> Expr:
        key = id(value)
        if key in owners:
            return owners[key]
        following = storage_source(value, liveness.bindings)
        if isinstance(value, Call) and (position := aliased_operand(value)) is not None:
            operand = value.args[position]
            ctx = TypeInferContext()
            relation = renaming_relation(
                value, ctx, declarations[key].projected_relations(value, ctx)
            ).relation
            box = (
                shape_to_isl_set(tuple(operand.type.shape), {})
                if is_literal_shape(operand.type.shape)
                else None
            )
            if (
                box is None
                or not relation.is_single_valued()
                or not relation.is_injective()
                or not relation.range().is_subset(box)
            ):
                param = tuple(
                    param for param in type(value.target)._op_schema.signature
                    if param.kind == "input"
                )[position]
                raise AnalysisError(
                    f"{type(value.target).__name__} declares its result is {param.name}'s bytes, "
                    f"but its access relation {relation} is not single-valued, injective, "
                    "and within the operand"
                )
        owners[key] = value if following is None else resolve(following)
        return owners[key]

    for interval in liveness.intervals:
        resolve(interval.value)
    return owners


def is_non_conflicting(
    source: Expr,
    operand: Expr,
    scope: IterationScope,
    liveness: Liveness,
    owners: dict[int, Expr],
) -> bool:
    """Prove that reusing *operand* cannot clobber a later ordinary use."""
    source_interval = liveness.interval_of(source)
    operand_interval = liveness.interval_of(operand)
    if source_interval is None or operand_interval is None:
        return False
    if any(
        not use.synthetic
        and use.at > source_interval.defined_at
        and use.value is operand
        for use in liveness.uses
    ):
        return False
    if operand_interval.last_used_at <= source_interval.defined_at:
        return True

    operand_owner = owners[id(operand)]
    source_owner = owners[id(source)]
    cursor: IterationScope | None = scope
    while cursor is not None:
        loop = cursor.owner
        if isinstance(loop, LoopRegion):
            for slot, carried in enumerate(loop.params[: len(loop.yield_values)]):
                if (
                    owners[id(carried)] is operand_owner
                    and slot < len(loop.yield_values)
                    and owners[id(loop.yield_values[slot])] is source_owner
                ):
                    return True
        cursor = cursor.parent
    return False


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
    liveness: Liveness,
    owners: dict[int, Expr],
) -> tuple[AliasConstraint, ...]:
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

    result: list[AliasConstraint] = []
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
            if relation is not None and is_non_conflicting(
                node, operand, scope, liveness, owners
            ):
                result.append(AliasConstraint(operand, relation))
                seen.add(key)

    if not isinstance(node.target, InsertSlice):
        return tuple(result)

    destination = owners[id(node.args[0])]
    update = owners[id(node.args[1])]
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
            and is_non_conflicting(node, destination, scope, liveness, owners)
        ):
            result.append(AliasConstraint(destination, relation))
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
        and is_non_conflicting(node, update, scope, liveness, owners)
    ):
        result.append(AliasConstraint(update, update_relation))
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


class AllocationConstraintVisitor(ExprVisitor[None]):
    """Visit the HIR DAG once and apply each node/operand placement relation."""

    def visit_MeshRegion(self, node: MeshRegion, ctx: AllocationModel) -> None:
        child = next(item for item in ctx.current.children if item.owner is node)
        for argument in node.args:
            self.visit(argument, ctx)
        self.visit(node.body, replace(ctx, current=child))

    def visit_LoopRegion(self, node: LoopRegion, ctx: AllocationModel) -> None:
        child = next(item for item in ctx.current.children if item.owner is node)
        inner = replace(ctx, current=child)
        for operand in node.args:
            self.visit(operand, ctx)
        self.visit(node.body, inner)
        for operand in node.yield_values:
            self.visit(operand, inner)
        for initial, carried, yielded in zip(
            node.args[: len(node.yield_values)],
            node.params[: len(node.yield_values)],
            node.yield_values,
            strict=True,
        ):
            self.tie(carried, initial, ctx)
            self.tie(yielded, carried, inner)
        if len(node.yield_values) == 1:
            self.tie(node, node.yield_values[0], inner)

    def tie(self, result_value: Expr, operand_value: Expr, ctx: AllocationModel) -> None:
        """Require the single backing buffer stated by one loop-carried slot."""
        result_index = allocation_index(result_value, ctx)
        operand_index = allocation_index(operand_value, ctx)
        if result_index is None or operand_index is None or result_index == operand_index:
            return
        result = ctx.values[result_index]
        operand = ctx.values[operand_index]
        if operand.lifetime.persistent or result.lifetime.bytes != operand.lifetime.bytes:
            return
        ctx.model.add(ctx.addresses[result_index] == ctx.addresses[operand_index]).with_name(
            f"carry_{result_index}_{operand_index}"
        )
        ctx.aliased.add(tuple(sorted((result_index, operand_index))))

    def alias(
        self,
        result_value: Expr,
        operand_value: Expr,
        relation: isl.map,
        ctx: AllocationModel,
    ) -> None:
        """Require one proven logical alias in the physical placement."""
        result_index = allocation_index(result_value, ctx)
        operand_index = allocation_index(operand_value, ctx)
        if result_index is None or operand_index is None or operand_index == result_index:
            return
        result = ctx.values[result_index]
        operand = ctx.values[operand_index]
        if operand.lifetime.persistent:
            return
        if is_zero_offset(relation):
            if result.lifetime.bytes > operand.lifetime.bytes:
                return
            ctx.model.add(ctx.addresses[result_index] == ctx.addresses[operand_index]).with_name(
                f"alias_{result_index}_{operand_index}"
            )
        else:
            if operand.lifetime.bytes > result.lifetime.bytes:
                return
            ctx.model.add(ctx.addresses[operand_index] >= ctx.addresses[result_index]).with_name(
                f"alias_start_{result_index}_{operand_index}"
            )
            ctx.model.add(
                ctx.addresses[operand_index] + operand.lifetime.bytes
                <= ctx.addresses[result_index] + result.lifetime.bytes
            ).with_name(f"alias_end_{result_index}_{operand_index}")
        ctx.aliased.add(tuple(sorted((result_index, operand_index))))

    def default_visit_leaf(
        self, node: Expr, operands: tuple[None, ...], ctx: AllocationModel
    ) -> None:
        del operands
        result_index = allocation_index(node, ctx)
        if (
            not isinstance(node, Call)
            or aliased_operand(node) is not None
            or id(node) not in ctx.current.accesses.get("narrow", {})
            or result_index is None
        ):
            return
        for constraint in operand_to_result_relation(
            node, ctx.current, ctx.liveness, ctx.owners
        ):
            self.alias(node, constraint.operand, constraint.relation, ctx)


def allocation_index(value: Expr, ctx: AllocationModel) -> int | None:
    """Find the allocation belonging to this value's proven storage owner."""
    return ctx.boxes_by_expr.get(id(ctx.owners[id(value)]))


def alias_components(count: int, aliased: set[tuple[int, int]]) -> tuple[int, ...]:
    """Transitive alias component for every allocation index."""
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left, right in sorted(aliased):
        left_root = find(left)
        right_root = find(right)
        parent[right_root] = left_root
    return tuple(find(index) for index in range(count))


def build_interference_graph(
    values: tuple[AllocationValue, ...], ctx: AllocationModel
) -> set[tuple[int, int]]:
    """Pairs whose address ranges must not overlap."""
    result: set[tuple[int, int]] = set()
    components = alias_components(len(values), ctx.aliased)
    for left, right in combinations(range(len(values)), 2):
        pair = (left, right)
        if components[left] == components[right]:
            continue
        if values[left].intersects(values[right]):
            result.add(pair)
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
    aliased: set[tuple[int, int]],
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
    roots = alias_components(len(values), aliased)
    components: dict[int, set[int]] = defaultdict(set)
    for index in range(len(values)):
        components[roots[index]].add(index)

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


def find_aliases(
    values: tuple[AllocationValue, ...],
    liveness: Liveness,
    root: IterationScope,
    owners: dict[int, Expr],
) -> set[tuple[int, int]]:
    """Required physical aliases, without asking the address solver to place them."""
    limit = max(1, sum(item.lifetime.bytes for item in values))
    model = cp_model.CpModel()
    context = AllocationModel(
        current=root,
        liveness=liveness,
        values=values,
        boxes_by_expr={id(item.value): index for index, item in enumerate(values)},
        model=model,
        addresses=tuple(
            model.new_int_var(0, limit, f"alias_address_{index}") for index in range(len(values))
        ),
        owners=owners,
    )
    AllocationConstraintVisitor(root_function=root.owner).visit_function_body(root.owner, context)
    return context.aliased


def solve_allocation(
    memory_level: str,
    values: tuple[AllocationValue, ...],
    liveness: Liveness,
    root: IterationScope,
    *,
    owners: dict[int, Expr],
    options: SolverOptions,
) -> AllocationResult:
    """Return the first feasible whole-function placement for one level."""
    if any(item.lifetime.memory_level != memory_level for item in values):
        raise ValueError("allocation values must all belong to the requested memory level")
    if not values:
        return AllocationResult(0, "optimal")

    alignments = tuple(alignment_of(item.value) for item in values)
    largest = max(item.lifetime.bytes for item in values)
    limit = sum(
        aligned(item.lifetime.bytes, alignment)
        for item, alignment in zip(values, alignments, strict=True)
    )
    model = cp_model.CpModel()
    peak = model.new_int_var(largest, limit, f"{memory_level}_peak")
    addresses = tuple(
        model.new_int_var(0, limit - item.lifetime.bytes, f"address_{index}")
        for index, item in enumerate(values)
    )
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
        model.add(address == persistent_end)
        pinned[index] = persistent_end
        persistent_end += item.lifetime.bytes
    for address, item in zip(addresses, values, strict=True):
        if not item.lifetime.persistent:
            model.add(address >= persistent_end)

    context = AllocationModel(
        current=root,
        liveness=liveness,
        values=values,
        boxes_by_expr={id(item.value): index for index, item in enumerate(values)},
        model=model,
        addresses=addresses,
        owners=owners,
    )
    AllocationConstraintVisitor(root_function=root.owner).visit_function_body(root.owner, context)
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
        context.aliased,
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
        aliases = ", ".join(
            f"{values[left].lifetime.binding}={values[right].lifetime.binding}"
            for left, right in sorted(context.aliased)
        )
        raise AnalysisError(
            f"allocation: no feasible {memory_level} placement was found "
            f"({status_name}); required aliases [{aliases}] conflict with "
            f"{len(interference)} interference constraints"
        )
    solved_offsets = tuple(
        (id(item.value), solver.value(address))
        for address, item in zip(addresses, values, strict=True)
    )
    actual_peak = max(
        solver.value(address) + item.lifetime.bytes
        for address, item in zip(addresses, values, strict=True)
    )
    return AllocationResult(actual_peak, "feasible", solved_offsets)


__all__ = ["AllocationResult", "AllocationValue", "solve_allocation"]
