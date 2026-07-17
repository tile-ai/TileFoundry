from __future__ import annotations

import dataclasses

from tilefoundry.ir.core import Call, Constant, Expr, Tuple, Var
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.types import TensorType, TupleType
from tilefoundry.ir.types.shard import Broadcast, Layout, Mesh, Partial, ShardLayout, Split

from .candidate import DistributionState, LayoutState
from .graph import GraphOp
from .solution import NodeAssignment, ScheduleSolution
from .solver import SolveProblem


class _Materializer:
    def __init__(self, problem: SolveProblem, solution: ScheduleSolution, context) -> None:
        self.problem = problem
        self.solution = solution
        self.context = context
        self.graph = problem.graph
        self.parent_mesh = context.mesh
        self.assignment_by_op = {
            assignment.node: assignment for assignment in solution.node_assignments
        }
        self.node_option_by_id = {
            option.id: option for option in problem.space.node_options
        }
        self.value_assignment_by_ref = {
            assignment.value: assignment for assignment in solution.value_assignments
        }
        self.value_ref_by_key = {
            (value.ref.call_path, id(value.ir_value)): value.ref
            for value in self.graph.values
        }
        self.use_assignment_by_edge = {
            assignment.edge_id: assignment for assignment in solution.use_assignments
        }
        self.op_by_key = {
            (op.call_path, id(op.ir_expr)): op for op in self.graph.ops
        }
        self.edge_by_key = {}
        for edge in self.graph.edges:
            for assignment in solution.edge_assignments:
                if assignment.use == edge.id:
                    self.edge_by_key[edge.id] = assignment
                    break
        self.function_cache: dict[tuple[int, tuple[int, ...]], Function] = {}
        self.active: set[tuple[int, tuple[int, ...]]] = set()
        self.materialized_functions: list[Function] = []

    def materialize(self) -> Module:
        entry = self.graph.module.entry_function()
        self.visit_function(entry, ())
        for function in self.graph.module.functions:
            if not any(key[0] == id(function) for key in self.function_cache):
                paths = [
                    region.call_path
                    for region in self.graph.regions
                    if region.function is function
                ]
                if paths:
                    self.visit_function(function, paths[0])
        functions = [self.function_cache[(id(entry), ())]]
        functions.extend(
            function for function in self.materialized_functions if function is not functions[0]
        )
        result = Module(
            name=self.problem.graph.module.name,
            functions=tuple(functions),
            entry=self.problem.graph.module.entry,
            topologies=self.problem.graph.module.topologies,
            metadata=dict(self.problem.graph.module.metadata),
        )
        return result

    def _region_for(self, function: Function, call_path: tuple[int, ...]):
        for region in self.graph.regions:
            if region.function is function and region.call_path == call_path:
                return region
        raise KeyError((function.name, call_path))

    def visit_function(self, function: Function, call_path: tuple[int, ...]) -> Function:
        key = (id(function), call_path)
        cached = self.function_cache.get(key)
        if cached is not None:
            return cached
        if key in self.active:
            raise ValueError(f"recursive materialization of {function.name!r} is unsupported")
        if function.body is None:
            result = Function.build(
                name=function.name,
                params=tuple(self._strip_expr(param) for param in function.params),
                body=None,
                return_type=function.return_type,
                topologies=function.topologies,
                specializations=function.specializations,
                target=function.target,
            )
            self.function_cache[key] = result
            self.materialized_functions.append(result)
            return result
        self.active.add(key)
        region = self._region_for(function, call_path)
        parameter_map = {
            id(old): self._parameter(old, region.inputs[index])
            for index, old in enumerate(function.params)
        }
        body = self.visit_expr(function.body, parameter_map, {}, call_path)
        name = function.name
        if call_path:
            name += "__schedule_" + "_".join(str(item) for item in call_path)
        result = Function.build(
            name=name,
            params=tuple(parameter_map[id(param)] for param in function.params),
            body=body,
            return_type=body.type,
            topologies=function.topologies,
            specializations=function.specializations,
            target=function.target,
        )
        self.active.remove(key)
        self.function_cache[key] = result
        self.materialized_functions.append(result)
        return result

    def _parameter(self, param: Var, value_ref) -> Var:
        ty = param.type
        assignment = self.value_assignment_by_ref.get(value_ref)
        if assignment is not None and isinstance(ty, TensorType):
            mesh = self._mesh_for_submesh(assignment.placement)
            ty = dataclasses.replace(
                ty,
                layout=self._layout_for_state(ty, assignment.state, mesh),
            )
        return Var(type=ty, name=param.name, loc=param.loc, metadata=())

    def _materialize_value_type(self, expr: Expr, call_path: tuple[int, ...]) -> Expr:
        value_ref = self.value_ref_by_key.get((call_path, id(expr)))
        assignment = (
            None
            if value_ref is None
            else self.value_assignment_by_ref.get(value_ref)
        )
        if assignment is None or not isinstance(expr.type, TensorType):
            return expr
        layout = self._layout_for_state(
            expr.type,
            assignment.state,
            self._mesh_for_submesh(assignment.placement),
        )
        return dataclasses.replace(expr, type=dataclasses.replace(expr.type, layout=layout))

    @staticmethod
    def _strip_expr(expr: Expr) -> Expr:
        return dataclasses.replace(expr, metadata=())

    @staticmethod
    def _broadcast_state(ty: TensorType) -> DistributionState:
        return DistributionState(LayoutState(len(ty.shape)), 1)

    def _mesh_for_submesh(self, submesh) -> Mesh:
        start = submesh.offsets[0]
        extent = submesh.extents[0]
        parent_extent = self.parent_mesh.shape[0]
        if start == 0 and extent == parent_extent:
            return self.parent_mesh
        return self.parent_mesh[start : start + extent]

    def _mesh_for_assignment(self, assignment: NodeAssignment) -> Mesh:
        return self._mesh_for_submesh(assignment.placement.submesh)

    def _layout_for_state(self, ty: TensorType, state: DistributionState, mesh: Mesh) -> ShardLayout:
        if state.partial is not None:
            attrs = (Partial(state.partial.reduction),)
        elif state.layout.split_axis is not None:
            attrs = (Split(state.layout.split_axis),)
        else:
            attrs = (Broadcast(),)
        return ShardLayout(Layout(tuple(ty.shape), None), attrs, mesh)

    def _reshard(self, value: Expr, layout: ShardLayout) -> Expr:
        if not isinstance(value.type, TensorType):
            return value
        storage = value.type.storage
        target = Reshard(layout=layout, storage=storage)
        result_type = TensorType(
            shape=value.type.shape,
            dtype=value.type.dtype,
            layout=layout,
            storage=storage,
        )
        return Call(type=result_type, target=target, args=(value,), metadata=())

    def _reshard_to_parent(self, value: Expr) -> Expr:
        if not isinstance(value.type, TensorType):
            return value
        state = self._broadcast_state(value.type)
        return self._reshard(value, self._layout_for_state(value.type, state, self.parent_mesh))

    def _assignment_for_expr(
        self, expr: Expr, call_path: tuple[int, ...]
    ) -> NodeAssignment | None:
        op = self.op_by_key.get((call_path, id(expr)))
        return None if op is None else self.assignment_by_op.get(op.id)

    def _edge_for_operand(self, op: GraphOp, index: int):
        for edge in self.graph.edges:
            if (
                edge.kind in {"data", "call_arg"}
                and edge.op_id == op.id
                and edge.operand_index == index
            ):
                return edge
        return None

    def _materialize_output(
        self, original: Expr, rebuilt: Expr, op: GraphOp | None
    ) -> Expr:
        assignment = None if op is None else self.assignment_by_op.get(op.id)
        if assignment is None or not isinstance(rebuilt.type, TensorType):
            return dataclasses.replace(rebuilt, metadata=())
        selected = self.node_option_by_id[assignment.option]
        output_state = selected.candidate.output_states[0]
        mesh = self._mesh_for_assignment(assignment)
        layout = self._layout_for_state(rebuilt.type, output_state, mesh)
        concrete_type = dataclasses.replace(rebuilt.type, layout=layout)
        result = dataclasses.replace(rebuilt, type=concrete_type, metadata=())
        if op.call_instance is not None:
            result_edge = next(
                (
                    edge
                    for edge in self.graph.edges
                    if edge.kind == "call_result" and edge.op_id == op.id
                ),
                None,
            )
            if result_edge is not None:
                use = self.use_assignment_by_edge.get(result_edge.id)
                destination = self.value_assignment_by_ref.get(result_edge.destination)
                if (
                    use is not None
                    and use.kind.value == "reshard"
                    and destination is not None
                ):
                    result = self._reshard(
                        result,
                        self._layout_for_state(
                            result.type,
                            use.destination_state,
                            self._mesh_for_submesh(destination.placement),
                        ),
                    )
        return result

    def visit_expr(
        self,
        expr: Expr,
        substitutions: dict[int, Expr],
        memo: dict[int, Expr],
        call_path: tuple[int, ...],
    ) -> Expr:
        cached = memo.get(id(expr))
        if cached is not None:
            return cached
        if isinstance(expr, Var):
            result = substitutions.get(id(expr), self._strip_expr(expr))
            memo[id(expr)] = result
            return result
        if isinstance(expr, Constant):
            result = self._strip_expr(self._materialize_value_type(expr, call_path))
            memo[id(expr)] = result
            return result
        if isinstance(expr, Tuple):
            elements = tuple(
                self.visit_expr(element, substitutions, memo, call_path)
                for element in expr.elements
            )
            result = dataclasses.replace(
                expr,
                elements=elements,
                type=TupleType(tuple(element.type for element in elements)),
                metadata=(),
            )
            memo[id(expr)] = result
            return result
        if not isinstance(expr, Call):
            result = self._strip_expr(expr)
            memo[id(expr)] = result
            return result

        op = self.op_by_key.get((call_path, id(expr)))
        args = []
        for index, argument in enumerate(expr.args):
            value = self.visit_expr(argument, substitutions, memo, call_path)
            edge = None if op is None else self._edge_for_operand(op, index)
            assignment = None if edge is None else self.edge_by_key.get(edge.id)
            if (
                assignment is not None
                and assignment.kind.value == "reshard"
                and op is not None
                and isinstance(value.type, TensorType)
            ):
                if op is not None and edge.kind == "call_arg":
                    destination = self.value_assignment_by_ref.get(edge.destination)
                    if destination is not None:
                        value = self._reshard(
                            value,
                            self._layout_for_state(
                                value.type,
                                self.use_assignment_by_edge[edge.id].destination_state,
                                self._mesh_for_submesh(destination.placement),
                            ),
                        )
                elif op is not None:
                    consumer_assignment = self.assignment_by_op.get(op.id)
                    if consumer_assignment is not None:
                        selected = next(
                            node_option
                            for node_option in self.problem.space.node_options
                            if node_option.id == consumer_assignment.option
                        )
                        state = selected.candidate.input_states[index]
                        value = self._reshard(
                            value,
                            self._layout_for_state(
                                value.type,
                                state,
                                self._mesh_for_assignment(consumer_assignment),
                            ),
                        )
            args.append(value)
        target = expr.target
        if isinstance(target, Function):
            child_path = (*call_path, op.call_instance) if op and op.call_instance is not None else call_path
            target = self.visit_function(target, child_path)
        rebuilt = Call(type=expr.type, target=target, args=tuple(args), loc=expr.loc, metadata=())
        result = self._materialize_output(expr, rebuilt, op)
        memo[id(expr)] = result
        return result


def materialize_schedule(problem: SolveProblem, solution: ScheduleSolution, context) -> Module:
    if solution.problem_fingerprint != problem.problem_fingerprint:
        raise ValueError("cannot materialize a solution for a different problem")
    return _Materializer(problem, solution, context).materialize()


__all__ = ["materialize_schedule"]
