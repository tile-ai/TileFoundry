from __future__ import annotations

from itertools import product

from ..solution import EdgeAssignment, NodeAssignment, ScheduleSolution
from ..solver import SolveOptions, SolveProblem
from ..space import EdgeOption, NodeOption


class CudaCtaSolver:
    """Deterministic exact enumeration for the small static CTA MVP."""

    def solve(self, problem: SolveProblem, options: SolveOptions) -> ScheduleSolution:
        graph = problem.graph
        space = problem.space
        costs = problem.costs
        node_ids = tuple(graph.root.nodes)
        option_lists = tuple(space.options_for_node(node_id) for node_id in node_ids)
        if any(not candidates for candidates in option_lists):
            raise ValueError("CUDA CTA schedule has a node with no legal options")
        edge_lists = tuple(space.options_for_use(use.id) for use in graph.uses)
        if any(not candidates for candidates in edge_lists):
            raise ValueError("CUDA CTA schedule has a use with no legal edge options")

        best = None
        best_key = None
        for selected_node_options in product(*option_lists):
            selected_nodes = dict(zip(node_ids, selected_node_options))
            for selected_edge_options in product(*edge_lists):
                selected_edges = dict(zip((use.id for use in graph.uses), selected_edge_options))
                if not self._compatible(graph, selected_nodes, selected_edges):
                    continue
                for order in self._topological_orders(graph, selected_edges):
                    node_assignments = self._schedule_order(
                        graph,
                        costs,
                        selected_nodes,
                        selected_edges,
                        order,
                    )
                    edge_assignments = self._edge_assignments(
                        graph, costs, selected_edges, node_assignments
                    )
                    makespan = max(
                        (item.end_time for item in node_assignments),
                        default=0.0,
                    )
                    makespan = max(
                        makespan,
                        max((item.end_time for item in edge_assignments), default=0.0),
                    )
                    key = (
                        makespan,
                        tuple(option.id for option in selected_node_options),
                        tuple(option.id for option in selected_edge_options),
                        tuple(item.start_time for item in node_assignments),
                    )
                    if best_key is None or key < best_key:
                        best_key = key
                        best = ScheduleSolution(
                            node_assignments=tuple(node_assignments),
                            edge_assignments=tuple(edge_assignments),
                            makespan=makespan,
                            problem_fingerprint=problem.problem_fingerprint,
                        )
        if best is None:
            raise ValueError("CUDA CTA schedule is infeasible for the finite option space")
        return best

    @staticmethod
    def _compatible(graph, selected_nodes, selected_edges) -> bool:
        for use in graph.uses:
            edge = selected_edges[use.id]
            consumer = selected_nodes[use.consumer]
            if consumer.input_representations[use.operand_index] != edge.destination_representation:
                return False
            producer = graph.producer_of(use.value)
            if producer is None:
                continue
            producer_option = selected_nodes[producer.id]
            if edge.source_representation not in producer_option.output_representations:
                return False
            if edge.same_placement_required and (
                producer_option.placement.axis_starts != consumer.placement.axis_starts
                or producer_option.placement.axis_extents != consumer.placement.axis_extents
            ):
                return False
        return True

    @staticmethod
    def _topological_orders(graph, selected_edges):
        dependencies: dict[int, set[int]] = {node_id: set() for node_id in graph.root.nodes}
        for use in graph.uses:
            producer = graph.producer_of(use.value)
            if producer is not None:
                dependencies[use.consumer].add(producer.id)

        def visit(done: tuple[int, ...]):
            if len(done) == len(dependencies):
                yield done
                return
            done_set = set(done)
            ready = sorted(
                node_id
                for node_id, deps in dependencies.items()
                if node_id not in done_set and deps <= done_set
            )
            for node_id in ready:
                yield from visit((*done, node_id))

        return tuple(visit(()))

    @staticmethod
    def _placements_overlap(left, right) -> bool:
        left_start = left.axis_starts[0]
        left_end = left_start + left.axis_extents[0]
        right_start = right.axis_starts[0]
        right_end = right_start + right.axis_extents[0]
        return left_start < right_end and right_start < left_end

    def _schedule_order(
        self,
        graph,
        costs,
        selected_nodes: dict[int, NodeOption],
        selected_edges: dict[int, EdgeOption],
        order: tuple[int, ...],
    ) -> list[NodeAssignment]:
        assignments: dict[int, NodeAssignment] = {}
        for node_id in order:
            option = selected_nodes[node_id]
            start = 0.0
            for use in graph.uses:
                if use.consumer != node_id:
                    continue
                producer = graph.producer_of(use.value)
                if producer is None:
                    continue
                predecessor = assignments.get(producer.id)
                if predecessor is None:
                    raise ValueError("topological order omitted a dependency")
                start = max(start, predecessor.end_time + costs.edge(selected_edges[use.id].id).duration)
            for prior_id in order[: order.index(node_id)]:
                prior = assignments[prior_id]
                if self._placements_overlap(option.placement, prior.placement):
                    start = max(start, prior.end_time)
            duration = costs.node(option.id).duration
            assignments[node_id] = NodeAssignment(
                node=node_id,
                option=option.id,
                placement=option.placement,
                start_time=start,
                end_time=start + duration,
            )
        return [assignments[node_id] for node_id in graph.root.nodes]

    @staticmethod
    def _edge_assignments(graph, costs, selected_edges, node_assignments):
        by_node = {item.node: item for item in node_assignments}
        result = []
        for use in graph.uses:
            edge = selected_edges[use.id]
            producer = graph.producer_of(use.value)
            start = 0.0 if producer is None else by_node[producer.id].end_time
            duration = costs.edge(edge.id).duration
            result.append(
                EdgeAssignment(
                    use=use.id,
                    option=edge.id,
                    kind=edge.kind,
                    start_time=start,
                    end_time=start + duration,
                )
            )
        return result


__all__ = ["CudaCtaSolver"]
