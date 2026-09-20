"""Internal values and solver results for physical memory placement."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Protocol

import isl
from ortools.sat.python import cp_model

from tilefoundry.ir.core import Call, Expr
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.tensor.insert_slice import InsertSlice
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.hir.tensor.slice import Slice

from .errors import AnalysisError
from .metadata import ValueLifetime
from .scope import Access, Scope, walk_scopes


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


def _base_value(value: Expr) -> Expr:
    """Return the material allocation below non-material tensor views."""
    while isinstance(value, Call) and isinstance(value.target, (Slice, Reshape)):
        value = value.args[0]
    return value


def _coverage(accesses: tuple[Access, ...]) -> isl.set | None:
    """Union the call coordinates on which exact accesses reach one buffer."""
    if not accesses or any(not access.exact for access in accesses):
        return None
    result = accesses[0].relation.domain()
    for access in accesses[1:]:
        result = result.union(access.relation.domain())
    return result.coalesce()


def _access_relation(accesses: tuple[Access, ...]) -> isl.map | None:
    """Union exact accesses to one buffer without discarding their maps."""
    if not accesses or any(not access.exact for access in accesses):
        return None
    result = accesses[0].relation
    for access in accesses[1:]:
        result = result.union(access.relation)
    return result.coalesce()


def _proved_overlap_groups(
    values: tuple[AllocationValue, ...], root: Scope
) -> tuple[tuple[int, ...], ...]:
    """Prove exact pointwise ties and dynamic-update embedded views.

    This is scheme B: the exact per-iteration offset remains in ISL.  The CP
    model only receives the weaker fact that a group can occupy one
    result-sized allocation.
    """
    by_expr = {id(item.value): index for index, item in enumerate(values)}
    carried_ids = {
        id(carried)
        for scope in walk_scopes(root)
        if isinstance(scope.owner, LoopRegion)
        for carried in scope.owner.carried_args
    }
    groups: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for scope in walk_scopes(root):
        for call_id, (call, inputs) in scope.accesses.get("narrow", {}).items():
            if call_id not in by_expr:
                continue
            recorded_output = scope.outputs.get("narrow", {}).get(call_id)
            if recorded_output is None or recorded_output[0] is not call:
                continue
            outputs = recorded_output[1]
            if not outputs or any(not access.exact for access in outputs):
                continue

            result_index = by_expr[call_id]
            result = values[result_index]
            output_coverage = _coverage(outputs)
            output_relation = _access_relation(outputs)
            by_buffer: dict[int, list[Access]] = defaultdict(list)
            for access in inputs:
                by_buffer[id(access.buffer)].append(access)

            if output_coverage is not None and output_relation is not None:
                for buffer_id, accesses in by_buffer.items():
                    input_index = by_expr.get(buffer_id)
                    if input_index is None or input_index == result_index:
                        continue
                    source = values[input_index]
                    if source.lifetime.persistent:
                        continue
                    if source.lifetime.bytes != result.lifetime.bytes:
                        continue
                    if source.lifetime.last_used_at > result.lifetime.defined_at:
                        continue
                    input_relation = _access_relation(tuple(accesses))
                    try:
                        pointwise = input_relation is not None and input_relation.is_equal(
                            output_relation
                        )
                    except isl.Error:
                        pointwise = False
                    group = (result_index, input_index)
                    if pointwise and group not in seen:
                        seen.add(group)
                        groups.append(group)

            if not isinstance(call.target, InsertSlice):
                continue

            dst = _base_value(call.args[0])
            update = _base_value(call.args[1])
            member_ids = (id(call), id(dst), id(update))
            if any(member_id not in by_expr for member_id in member_ids):
                continue
            indices = tuple(dict.fromkeys(by_expr[member_id] for member_id in member_ids))
            if len(indices) != 3:
                continue
            result_index, dst_index, update_index = indices
            result = values[result_index]
            destination = values[dst_index]
            patch = values[update_index]
            if destination.lifetime.persistent or patch.lifetime.persistent:
                continue
            if (
                destination.lifetime.last_used_at > result.lifetime.defined_at
                and id(destination.value) not in carried_ids
            ):
                continue
            if destination.lifetime.bytes != result.lifetime.bytes:
                continue
            if patch.lifetime.bytes > result.lifetime.bytes:
                continue
            if patch.lifetime.last_used_at > result.lifetime.defined_at:
                continue

            dst_coverage = _coverage(tuple(by_buffer.get(id(dst), ())))
            update_coverage = _coverage(tuple(by_buffer.get(id(update), ())))
            written_coverage = _coverage(outputs)
            if any(
                coverage is None for coverage in (dst_coverage, update_coverage, written_coverage)
            ):
                continue
            try:
                partitioned = update_coverage.is_equal(
                    written_coverage
                ) and dst_coverage.is_disjoint(written_coverage)
            except isl.Error:
                partitioned = False
            if partitioned and indices not in seen:
                seen.add(indices)
                groups.append(indices)
    return tuple(groups)


def _lifetimes_overlap(left: AllocationValue, right: AllocationValue) -> bool:
    """Whether two closed structured-SSA intervals share an event."""
    return max(left.lifetime.defined_at, right.lifetime.defined_at) <= min(
        left.lifetime.last_used_at, right.lifetime.last_used_at
    )


def _placement_hint(
    values: tuple[AllocationValue, ...],
    groups: tuple[tuple[int, ...], ...],
    limit: int,
) -> tuple[tuple[bool, ...], tuple[int, ...], int] | None:
    """Construct one complete feasible suggestion without deciding the model.

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
    for group in groups:
        roots = {find(index) for index in group}
        merged = set().union(*(members(root) for root in roots))
        group_pairs = {
            tuple(sorted((left, right)))
            for left, right in combinations(group, 2)
            if _lifetimes_overlap(values[left], values[right])
        }
        permitted = allowed_pairs | group_pairs
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
        allowed_pairs.update(group_pairs)

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
            return None
        for index in component:
            addresses[index] = address
        placed.append((component, address, size))
        peak = max(peak, address + size)
    return tuple(selected), tuple(addresses), peak


def solve_allocation(
    memory_level: str,
    values: tuple[AllocationValue, ...],
    root: Scope,
    *,
    capacity_bytes: int | None,
    options: _MemoryOptions,
) -> AllocationResult:
    """Return the first feasible whole-function placement for one level."""
    if any(item.lifetime.memory_level != memory_level for item in values):
        raise ValueError("allocation values must all belong to the requested memory level")
    if not values:
        return AllocationResult(0, "optimal")

    largest = max(item.lifetime.bytes for item in values)
    total = sum(item.lifetime.bytes for item in values)
    limit = total if capacity_bytes is None else capacity_bytes
    if largest > limit:
        raise AnalysisError(
            f"allocation: a value needs {largest} B in {memory_level}, "
            f"which exceeds its {limit} B placement limit"
        )

    model = cp_model.CpModel()
    peak = model.new_int_var(largest, limit, f"{memory_level}_peak")
    addresses = [
        model.new_int_var(0, limit - item.lifetime.bytes, f"address_{index}")
        for index, item in enumerate(values)
    ]
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

    groups = _proved_overlap_groups(values, root)
    choices_by_pair: dict[tuple[int, int], list[cp_model.IntVar]] = defaultdict(list)
    choices: list[tuple[cp_model.IntVar, tuple[int, ...]]] = []
    for group_index, members in enumerate(groups):
        selected = model.new_bool_var(f"embedded_{group_index}")
        choices.append((selected, members))
        container = members[0]
        for member in members[1:]:
            model.add(addresses[member] >= addresses[container]).only_enforce_if(selected)
            model.add(
                addresses[member] + values[member].lifetime.bytes
                <= addresses[container] + values[container].lifetime.bytes
            ).only_enforce_if(selected)
        for left, right in combinations(members, 2):
            if _lifetimes_overlap(values[left], values[right]):
                choices_by_pair[tuple(sorted((left, right)))].append(selected)

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
            *choices_by_pair.get((left, right), ()),
        )

    hint = _placement_hint(values, groups, limit)
    if hint is not None:
        selected_hints, address_hints, peak_hint = hint
        model.add(peak <= peak_hint)
        for (selected, _members), suggested in zip(choices, selected_hints, strict=True):
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
