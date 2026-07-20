"""Private physical allocation grouping shared by CP-SAT and debug output."""

from __future__ import annotations

from .planner import PlanningProblem


def _allocation_groups(problem: PlanningProblem) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Return conservative physical groups for all possible bucket selections.

    Carry facts are unconditional in-place aliases. View aliases are selected
    candidate facts, so they are intentionally kept as singleton groups in the
    pre-solve capacity model. This can overestimate HBM usage, but cannot merge
    an unselected view path and undercount resident bytes.
    """
    bucket_ids = tuple(sorted(problem.buckets))
    parent = {bucket_id: bucket_id for bucket_id in bucket_ids}

    def find(bucket_id: int) -> int:
        while parent[bucket_id] != bucket_id:
            parent[bucket_id] = parent[parent[bucket_id]]
            bucket_id = parent[bucket_id]
        return bucket_id

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for region in problem.regions.values():
        for carry in region.carry_infos:
            carry_values = (
                carry.init_value_id,
                carry.carried_value_id,
                carry.yield_value_id,
                carry.result_value_id,
            )
            for left_value, right_value in zip(carry_values, carry_values[1:]):
                left_by_type = {
                    bucket.type_id: bucket_id
                    for bucket_id, bucket in problem.buckets.items()
                    if bucket.value_id == left_value
                }
                right_by_type = {
                    bucket.type_id: bucket_id
                    for bucket_id, bucket in problem.buckets.items()
                    if bucket.value_id == right_value
                }
                for type_id in left_by_type.keys() & right_by_type.keys():
                    union(left_by_type[type_id], right_by_type[type_id])

    groups: dict[int, list[int]] = {}
    for bucket_id in bucket_ids:
        groups.setdefault(find(bucket_id), []).append(bucket_id)
    return tuple(
        (group_id, tuple(sorted(group_bucket_ids)))
        for group_id, group_bucket_ids in sorted(groups.items())
    )


__all__ = ["_allocation_groups"]
