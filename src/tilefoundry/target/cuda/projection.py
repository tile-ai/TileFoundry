"""Private projections from a selected CTA planning graph."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.hir.sharding.reshard import Reshard

from .planner import PlanningProblem


@dataclass(frozen=True)
class _PhysicalFusionEdge:
    producer_candidate_id: int | None
    consumer_candidate_id: int
    input_index: int
    bucket_id: int
    relation: str


@dataclass(frozen=True)
class _ReshardCut:
    candidate_id: int
    input_bucket_id: int
    output_bucket_id: int


def _selected_producer(problem: PlanningProblem, selected: set[int], bucket_id: int) -> int | None:
    producers = [
        candidate_id for candidate_id in selected
        if bucket_id in problem.candidates[candidate_id].output_bucket_ids
    ]
    return min(producers) if producers else None


def project_physical_fusion(
    problem: PlanningProblem,
    solution: "PlanningSolution",
) -> tuple[tuple[_PhysicalFusionEdge, ...], tuple[_ReshardCut, ...]]:
    """Project direct physical-fusion edges and selected Reshard cuts."""
    selected_candidates = set(solution.selected_candidate_ids)
    selected_buckets = set(solution.selected_bucket_ids)
    opportunities: list[_PhysicalFusionEdge] = []
    cuts: list[_ReshardCut] = []
    for candidate_id in sorted(selected_candidates):
        candidate = problem.candidates[candidate_id]
        if isinstance(candidate.op, Reshard):
            cuts.extend(
                _ReshardCut(candidate_id, input_bucket, output_bucket)
                for input_bucket, output_bucket in zip(
                    candidate.input_bucket_ids, candidate.output_bucket_ids
                )
            )
    for dependency in sorted(
        problem.dependencies,
        key=lambda item: (item.parent_candidate_id, item.input_index, item.child_bucket_id),
    ):
        if dependency.parent_candidate_id not in selected_candidates:
            continue
        if dependency.child_bucket_id not in selected_buckets or dependency.placement_relation is None:
            continue
        producer = _selected_producer(problem, selected_candidates, dependency.child_bucket_id)
        if producer is not None and isinstance(problem.candidates[producer].op, Reshard):
            continue
        opportunities.append(
            _PhysicalFusionEdge(
                producer_candidate_id=producer,
                consumer_candidate_id=dependency.parent_candidate_id,
                input_index=dependency.input_index,
                bucket_id=dependency.child_bucket_id,
                relation=dependency.placement_relation,
            )
        )
    return tuple(opportunities), tuple(cuts)


__all__ = ["project_physical_fusion"]
