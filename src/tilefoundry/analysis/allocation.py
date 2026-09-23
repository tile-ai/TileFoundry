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
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.isl_interop import index_set
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.utils import local_type_of
from tilefoundry.ir.visitor import ExprVisitor
from tilefoundry.utils.isl_utils import equates

from .access import Access, AccessPrecision
from .errors import AnalysisError
from .iteration_scope import IterationScope
from .liveness import Liveness
from .metadata import ValueLifetime


class _MemoryOptions(Protocol):
    timeout_seconds: float
    workers: int
    random_seed: int


@dataclass(frozen=True)
class AllocationValue:
    """Connect one logical expression to its target-aware lifetime."""

    value: Expr
    lifetime: ValueLifetime


@dataclass(frozen=True)
class AllocationResult:
    """The first physical placement found for one memory level."""

    peak_bytes: int
    solver_status: str


@dataclass(frozen=True)
class _OperandConstraint:
    """An exact logical relation from one material operand into a result."""

    operand: Expr
    relation: isl.map


@dataclass
class _ConstraintContext:
    """One memory-level model while the HIR visitor applies logical relations."""

    current: IterationScope
    liveness: Liveness
    values: tuple[AllocationValue, ...]
    boxes_by_expr: dict[int, int]
    model: cp_model.CpModel
    addresses: tuple[cp_model.IntVar, ...]
    selected_by_pair: dict[tuple[int, int], list[cp_model.IntVar]] = field(
        default_factory=lambda: defaultdict(list)
    )
    applied: list[tuple[cp_model.IntVar, int, int]] = field(default_factory=list)


def _base_value(value: Expr) -> Expr:
    """Return the material allocation below non-material tensor views."""
    while isinstance(value, Call) and isinstance(value.target, (Slice, Reshape)):
        value = value.args[0]
    return value


def _is_view_of(value: Expr, source: Expr) -> bool:
    """Whether ``value`` reaches ``source`` through only non-material views."""
    while True:
        if value is source:
            return True
        if not isinstance(value, Call) or not isinstance(value.target, (Slice, Reshape)):
            return False
        value = value.args[0]


def _coverage(accesses: tuple[Access, ...]) -> isl.set | None:
    """Union the call coordinates on which exact accesses reach one buffer."""
    if not accesses or any(
        access.precision is not AccessPrecision.EXACT for access in accesses
    ):
        return None
    result = accesses[0].relation.domain()
    for access in accesses[1:]:
        result = result.union(access.relation.domain())
    return result.coalesce()


def _access_relation(accesses: tuple[Access, ...]) -> isl.map | None:
    """Union exact accesses to one buffer without discarding their maps."""
    if not accesses or any(
        access.precision is not AccessPrecision.EXACT for access in accesses
    ):
        return None
    result = accesses[0].relation
    for access in accesses[1:]:
        result = result.union(access.relation)
    return result.coalesce()


def _value_domain(value: Expr, scope: IterationScope) -> isl.set | None:
    """The complete loop-aware coordinate domain of one material value."""
    try:
        held = local_type_of(value.type)
    except (TypeError, ValueError, NotImplementedError):
        return None
    if not isinstance(held, TensorType):
        return None
    box = index_set(held.shape)
    if box is None:
        return None
    return scope.domain.flat_product(box).coalesce()


def _operand_to_result_relation(
    inputs: isl.map, outputs: isl.map, loop_depth: int
) -> isl.map | None:
    """Compose call accesses into ``[loops..., operand] -> [result]``."""
    try:
        common = inputs.domain().intersect(outputs.domain()).coalesce()
        if common.is_empty():
            return None
        inputs = inputs.intersect_domain(common)
        outputs = outputs.intersect_domain(common)
        call_dims = inputs.dim(isl.dim_type.IN)
        if call_dims != outputs.dim(isl.dim_type.IN) or loop_depth > call_dims:
            return None
        loop_prefix = isl.map.identity(common.get_space().map_from_set()).project_out(
            isl.dim_type.OUT, loop_depth, call_dims - loop_depth
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


def _complete_operand_relation(
    operand: Expr,
    inputs: isl.map | None,
    outputs: isl.map | None,
    scope: IterationScope,
) -> isl.map | None:
    """Return a composed relation only when it covers the whole operand."""
    if inputs is None or outputs is None:
        return None
    relation = _operand_to_result_relation(inputs, outputs, scope.depth)
    expected = _value_domain(operand, scope)
    if relation is None or expected is None:
        return None
    try:
        return relation if relation.domain().is_equal(expected) else None
    except isl.Error:
        return None


def _identity_result_relation(node: Call, scope: IterationScope) -> isl.map | None:
    """Map a same-shaped operand to the result while retaining loop axes."""
    domain = _value_domain(node, scope)
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


def _intervals_by_expr(liveness: Liveness) -> dict[int, tuple[int, int]]:
    """Index the immutable liveness answer without traversing the HIR again."""
    return {
        id(interval.value): (interval.defined_at, interval.last_used_at)
        for interval in liveness.intervals
    }


def _corresponding_carry(
    source: Expr, operand: Expr, scope: IterationScope
) -> LoopRegion | None:
    """Find the carry whose own yield is ``source``, without walking its body."""
    cursor: IterationScope | None = scope
    while cursor is not None:
        loop = cursor.owner
        if isinstance(loop, LoopRegion):
            for slot, carried in enumerate(loop.carried_args):
                if carried is operand and slot < len(loop.yield_values):
                    return loop if _is_view_of(loop.yield_values[slot], source) else None
        cursor = cursor.parent
    return None


def _tie_is_live(
    source: Expr, operand: Expr, scope: IterationScope, liveness: Liveness
) -> bool:
    """Prove that reusing ``operand`` cannot clobber a later ordinary use."""
    intervals = _intervals_by_expr(liveness)
    source_interval = intervals.get(id(source))
    operand_interval = intervals.get(id(operand))
    if source_interval is None or operand_interval is None:
        return False
    if any(
        not use.synthetic
        and use.at > source_interval[0]
        and _base_value(use.value) is operand
        and not _is_view_of(use.value, source)
        for use in liveness.uses
    ):
        return False
    if operand_interval[1] <= source_interval[0]:
        return True

    return _corresponding_carry(source, operand, scope) is not None


def _analyze_operand_constraints(
    node: Call, scope: IterationScope, liveness: Liveness
) -> tuple[_OperandConstraint, ...]:
    """Prove exact logical relations between one result and its operands."""
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
    output_coverage = _coverage(outputs)
    output_relation = _access_relation(outputs)
    full_result = _value_domain(node, scope)
    if output_coverage is None or output_relation is None or full_result is None:
        return ()

    by_buffer: dict[int, list[Access]] = defaultdict(list)
    operands: dict[int, Expr] = {}
    for access in inputs:
        key = id(access.buffer)
        by_buffer[key].append(access)
        operands[key] = access.buffer

    result: list[_OperandConstraint] = []
    seen: set[int] = set()

    try:
        covers_result = output_coverage.is_equal(full_result)
    except isl.Error:
        covers_result = False
    if covers_result:
        for key, operand in operands.items():
            input_relation = _access_relation(tuple(by_buffer[key]))
            try:
                pointwise = input_relation is not None and input_relation.is_equal(output_relation)
            except isl.Error:
                pointwise = False
            relation = (
                _complete_operand_relation(operand, input_relation, output_relation, scope)
                if pointwise
                else None
            )
            if relation is not None and _tie_is_live(node, operand, scope, liveness):
                result.append(_OperandConstraint(operand, relation))
                seen.add(key)

    if not isinstance(node.target, InsertSlice):
        return tuple(result)

    dst = _base_value(node.args[0])
    update = _base_value(node.args[1])
    written = output_coverage
    dst_coverage = _coverage(tuple(by_buffer.get(id(dst), ())))
    update_accesses = tuple(by_buffer.get(id(update), ()))
    update_coverage = _coverage(update_accesses)

    if dst_coverage is not None:
        relation = _identity_result_relation(node, scope)
        expected_dst = _value_domain(dst, scope)
        try:
            partitioned = dst_coverage.is_disjoint(written) and dst_coverage.union(
                written
            ).coalesce().is_equal(full_result)
            complete_identity = (
                relation is not None
                and expected_dst is not None
                and relation.domain().is_equal(expected_dst)
            )
        except isl.Error:
            partitioned = complete_identity = False
        if (
            partitioned
            and complete_identity
            and id(dst) not in seen
            and _tie_is_live(node, dst, scope, liveness)
        ):
            result.append(_OperandConstraint(dst, relation))
            seen.add(id(dst))

    try:
        update_matches_write = update_coverage is not None and update_coverage.is_equal(written)
    except isl.Error:
        update_matches_write = False
    update_relation = (
        _complete_operand_relation(
            update,
            _access_relation(update_accesses),
            output_relation,
            scope,
        )
        if update_matches_write
        else None
    )
    if (
        update_relation is not None
        and id(update) not in seen
        and _tie_is_live(node, update, scope, liveness)
    ):
        result.append(_OperandConstraint(update, update_relation))
    return tuple(result)


def _proves_zero_offset(relation: isl.map) -> bool:
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


def _apply_constraints(
    node: Call,
    constraints: tuple[_OperandConstraint, ...],
    ctx: _ConstraintContext,
) -> None:
    """Compile logical relations into this memory level's CP model."""
    result_index = ctx.boxes_by_expr.get(id(node))
    if result_index is None:
        return
    result = ctx.values[result_index]
    for constraint in constraints:
        operand_index = ctx.boxes_by_expr.get(id(constraint.operand))
        if operand_index is None or operand_index == result_index:
            continue
        operand = ctx.values[operand_index]
        if operand.lifetime.persistent or not _lifetimes_overlap(result, operand):
            continue
        if _proves_zero_offset(constraint.relation):
            if operand.lifetime.bytes != result.lifetime.bytes:
                continue
            selected = ctx.model.new_bool_var(f"embedded_{len(ctx.applied)}")
            ctx.model.add(
                ctx.addresses[operand_index] == ctx.addresses[result_index]
            ).only_enforce_if(selected)
        else:
            if operand.lifetime.bytes > result.lifetime.bytes:
                continue
            selected = ctx.model.new_bool_var(f"embedded_{len(ctx.applied)}")
            ctx.model.add(
                ctx.addresses[operand_index] >= ctx.addresses[result_index]
            ).only_enforce_if(selected)
            ctx.model.add(
                ctx.addresses[operand_index] + operand.lifetime.bytes
                <= ctx.addresses[result_index] + result.lifetime.bytes
            ).only_enforce_if(selected)
        pair = tuple(sorted((result_index, operand_index)))
        ctx.selected_by_pair[pair].append(selected)
        ctx.applied.append((selected, result_index, operand_index))


class AllocationConstraintVisitor(ExprVisitor[None]):
    """Visit the HIR DAG once and apply each node/operand placement relation."""

    def visit_MeshRegion(self, node: MeshRegion, ctx: _ConstraintContext) -> None:
        child = next(item for item in ctx.current.children if item.owner is node)
        for arg in node.args:
            self.visit(arg, ctx)
        self.visit(node.body, replace(ctx, current=child))

    def visit_LoopRegion(self, node: LoopRegion, ctx: _ConstraintContext) -> None:
        child = next(item for item in ctx.current.children if item.owner is node)
        inner = replace(ctx, current=child)
        for operand in node.init_args:
            self.visit(operand, ctx)
        self.visit(node.body, inner)
        for operand in node.yield_values:
            self.visit(operand, inner)

    def default_visit_leaf(
        self, node: Expr, _operands: tuple[None, ...], ctx: _ConstraintContext
    ) -> None:
        if (
            not isinstance(node, Call)
            or id(node) not in ctx.current.accesses.get("narrow", {})
            or id(node) not in ctx.boxes_by_expr
        ):
            return
        constraints = _analyze_operand_constraints(node, ctx.current, ctx.liveness)
        _apply_constraints(node, constraints, ctx)


def _lifetimes_overlap(left: AllocationValue, right: AllocationValue) -> bool:
    """Whether two closed structured-SSA intervals share an event."""
    return max(left.lifetime.defined_at, right.lifetime.defined_at) <= min(
        left.lifetime.last_used_at, right.lifetime.last_used_at
    )


def _construct_feasible_seed(
    values: tuple[AllocationValue, ...],
    applied: list[tuple[cp_model.IntVar, int, int]],
    limit: int,
) -> tuple[tuple[bool, ...], tuple[int, ...], int]:
    """Construct one complete feasible seed from already-applied CP edges.

    Components exist only while building the hint.  The CP model still contains
    every logical box and independently validates or rejects every suggested
    overlap.
    """
    parent = list(range(len(values)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def members(root: int) -> set[int]:
        return {index for index in range(len(values)) if find(index) == root}

    allowed_pairs: set[tuple[int, int]] = set()
    selected: list[bool] = []
    for _choice, left, right in applied:
        pair = tuple(sorted((left, right)))
        roots = {find(left), find(right)}
        merged = set().union(*(members(root) for root in roots))
        permitted = allowed_pairs | {pair}
        compatible = all(
            not _lifetimes_overlap(values[left], values[right]) or (left, right) in permitted
            for left, right in combinations(sorted(merged), 2)
        )
        selected.append(compatible)
        if not compatible:
            continue
        root = min(roots)
        for other in roots:
            parent[find(other)] = root
        allowed_pairs.add(pair)

    components: dict[int, set[int]] = defaultdict(set)
    for index in range(len(values)):
        components[find(index)].add(index)
    ordered = sorted(
        components.values(),
        key=lambda component: (
            min(values[index].lifetime.defined_at for index in component),
            -max(values[index].lifetime.bytes for index in component),
        ),
    )
    placed: list[tuple[set[int], int, int]] = []
    addresses = [0] * len(values)
    peak = 0
    for component in ordered:
        size = max(values[index].lifetime.bytes for index in component)

        def conflicts(other: set[int]) -> bool:
            return any(
                _lifetimes_overlap(values[left], values[right])
                for left in component
                for right in other
            )

        blocked = tuple(item for item in placed if conflicts(item[0]))
        candidates = sorted({0, *(address + held for _other, address, held in blocked)})
        address = next(
            candidate
            for candidate in candidates
            if all(
                candidate + size <= other_address or other_address + other_size <= candidate
                for _other, other_address, other_size in blocked
            )
        )
        if address + size > limit:
            raise AnalysisError("allocation: failed to construct a bounded feasible seed")
        for index in component:
            addresses[index] = address
        placed.append((component, address, size))
        peak = max(peak, address + size)
    return tuple(selected), tuple(addresses), peak


def solve_allocation(
    memory_level: str,
    values: tuple[AllocationValue, ...],
    liveness: Liveness,
    root: IterationScope,
    *,
    options: _MemoryOptions,
) -> AllocationResult:
    """Return the first feasible whole-function placement for one level."""
    if any(item.lifetime.memory_level != memory_level for item in values):
        raise ValueError("allocation values must all belong to the requested memory level")
    if not values:
        return AllocationResult(0, "optimal")

    largest = max(item.lifetime.bytes for item in values)
    total = sum(item.lifetime.bytes for item in values)
    limit = total

    model = cp_model.CpModel()
    peak = model.new_int_var(largest, limit, f"{memory_level}_peak")
    addresses = tuple(
        model.new_int_var(0, limit - item.lifetime.bytes, f"address_{index}")
        for index, item in enumerate(values)
    )
    for address, item in zip(addresses, values, strict=True):
        model.add(address + item.lifetime.bytes <= peak)

    persistent_end = 0
    for address, item in zip(addresses, values, strict=True):
        if not item.lifetime.persistent:
            continue
        model.add(address == persistent_end)
        persistent_end += item.lifetime.bytes
    for address, item in zip(addresses, values, strict=True):
        if not item.lifetime.persistent:
            model.add(address >= persistent_end)

    context = _ConstraintContext(
        current=root,
        liveness=liveness,
        values=values,
        boxes_by_expr={id(item.value): index for index, item in enumerate(values)},
        model=model,
        addresses=addresses,
    )
    AllocationConstraintVisitor(root_function=root.owner).visit_function_body(root.owner, context)

    order_choices: dict[tuple[int, int], tuple[cp_model.IntVar, cp_model.IntVar]] = {}
    for left, right in combinations(range(len(values)), 2):
        if not _lifetimes_overlap(values[left], values[right]):
            continue
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
        model.add_bool_or(
            left_before,
            right_before,
            *context.selected_by_pair.get((left, right), ()),
        )

    selected_hints, address_hints, peak_hint = _construct_feasible_seed(
        values, context.applied, limit
    )
    model.add(peak <= peak_hint)
    for (selected, _result, _operand), suggested in zip(
        context.applied, selected_hints, strict=True
    ):
        model.add_hint(selected, int(suggested))
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
        raise AnalysisError(
            f"allocation: no feasible {memory_level} placement was found ({status_name})"
        )
    actual_peak = max(
        solver.value(address) + item.lifetime.bytes
        for address, item in zip(addresses, values, strict=True)
    )
    return AllocationResult(actual_peak, "feasible")


__all__ = ["AllocationResult", "AllocationValue", "solve_allocation"]
