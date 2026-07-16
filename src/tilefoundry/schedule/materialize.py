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
        self.op_by_expr = {id(op.ir_expr): op for op in self.graph.ops}
        self.edge_by_key = {}
        for edge in self.graph.edges:
            for assignment in solution.edge_assignments:
                if assignment.use == edge.id:
                    self.edge_by_key[edge.id] = assignment
                    break
        self.function_cache: dict[int, Function] = {}
        self.active: set[int] = set()

    def materialize(self) -> Module:
        functions = tuple(self.visit_function(function) for function in self.problem.graph.module.functions)
        result = Module(
            name=self.problem.graph.module.name,
            functions=functions,
            entry=self.problem.graph.module.entry,
            topologies=self.problem.graph.module.topologies,
            metadata=dict(self.problem.graph.module.metadata),
        )
        return result

    def visit_function(self, function: Function) -> Function:
        cached = self.function_cache.get(id(function))
        if cached is not None:
            return cached
        if id(function) in self.active:
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
            self.function_cache[id(function)] = result
            return result
        self.active.add(id(function))
        is_entry = function is self.graph.module.entry_function()
        parameter_map = {
            id(old): self._parameter(old, concrete=is_entry) for old in function.params
        }
        body = self.visit_expr(function.body, parameter_map, {})
        if function is self.graph.module.entry_function():
            body = self._reshard_to_parent(body)
        result = Function.build(
            name=function.name,
            params=tuple(parameter_map[id(param)] for param in function.params),
            body=body,
            return_type=body.type,
            topologies=function.topologies,
            specializations=function.specializations,
            target=function.target,
        )
        self.active.remove(id(function))
        self.function_cache[id(function)] = result
        return result

    def _parameter(self, param: Var, *, concrete: bool) -> Var:
        ty = param.type
        if concrete and isinstance(ty, TensorType):
            ty = dataclasses.replace(ty, layout=self._layout_for_state(ty, self._broadcast_state(ty), self.parent_mesh))
        return Var(type=ty, name=param.name, loc=param.loc, metadata=())

    @staticmethod
    def _strip_expr(expr: Expr) -> Expr:
        return dataclasses.replace(expr, metadata=())

    @staticmethod
    def _broadcast_state(ty: TensorType) -> DistributionState:
        return DistributionState(LayoutState(len(ty.shape)), 1)

    def _mesh_for_assignment(self, assignment: NodeAssignment) -> Mesh:
        start = assignment.placement.submesh.offsets[0]
        extent = assignment.placement.submesh.extents[0]
        parent_extent = self.parent_mesh.shape[0]
        if start == 0 and extent == parent_extent:
            return self.parent_mesh
        return self.parent_mesh[start : start + extent]

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
        if value.type.layout == layout and storage is not None:
            return value
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

    def _assignment_for_expr(self, expr: Expr) -> NodeAssignment | None:
        op = self.op_by_expr.get(id(expr))
        return None if op is None else self.assignment_by_op.get(op.id)

    def _edge_for_operand(self, op: GraphOp, index: int):
        for edge in self.graph.edges:
            if edge.kind == "data" and edge.op_id == op.id and edge.operand_index == index:
                return edge
        return None

    def _materialize_output(self, original: Expr, rebuilt: Expr) -> Expr:
        assignment = self._assignment_for_expr(original)
        if assignment is None or not isinstance(rebuilt.type, TensorType):
            return dataclasses.replace(rebuilt, metadata=())
        op = self.op_by_expr[id(original)]
        if op.call_instance is None:
            return dataclasses.replace(rebuilt, metadata=())
        selected = next(
            node_option
            for node_option in self.problem.space.node_options
            if node_option.id == assignment.option
        )
        output_state = selected.candidate.output_states[0]
        mesh = self._mesh_for_assignment(assignment)
        return self._reshard(rebuilt, self._layout_for_state(rebuilt.type, output_state, mesh))

    def visit_expr(
        self,
        expr: Expr,
        substitutions: dict[int, Expr],
        memo: dict[int, Expr],
    ) -> Expr:
        cached = memo.get(id(expr))
        if cached is not None:
            return cached
        if isinstance(expr, Var):
            result = substitutions.get(id(expr), self._strip_expr(expr))
            memo[id(expr)] = result
            return result
        if isinstance(expr, Constant):
            result = self._strip_expr(expr)
            memo[id(expr)] = result
            return result
        if isinstance(expr, Tuple):
            elements = tuple(self.visit_expr(element, substitutions, memo) for element in expr.elements)
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

        op = self.op_by_expr.get(id(expr))
        args = []
        for index, argument in enumerate(expr.args):
            value = self.visit_expr(argument, substitutions, memo)
            edge = None if op is None else self._edge_for_operand(op, index)
            assignment = None if edge is None else self.edge_by_key.get(edge.id)
            if (
                assignment is not None
                and assignment.kind.value == "reshard"
                and op is not None
                and (op.call_instance is not None or op.call_path == ())
                and isinstance(value.type, TensorType)
            ):
                if op is not None:
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
            target = self.visit_function(target)
            for index, argument in enumerate(args):
                if index >= len(target.params):
                    break
                parameter_type = target.params[index].type
                if isinstance(argument.type, TensorType) and isinstance(parameter_type, TensorType):
                    desired_layout = parameter_type.layout
                    if desired_layout is None and argument.type.layout is not None:
                        desired_layout = self._layout_for_state(
                            argument.type,
                            self._broadcast_state(argument.type),
                            self.parent_mesh,
                        )
                    if argument.type.layout != desired_layout and desired_layout is not None:
                        args[index] = self._reshard(argument, desired_layout)
        rebuilt = Call(type=expr.type, target=target, args=tuple(args), loc=expr.loc, metadata=())
        result = self._materialize_output(expr, rebuilt)
        memo[id(expr)] = result
        return result


def materialize_schedule(problem: SolveProblem, solution: ScheduleSolution, context) -> Module:
    if solution.problem_fingerprint != problem.problem_fingerprint:
        raise ValueError("cannot materialize a solution for a different problem")
    return _Materializer(problem, solution, context).materialize()


__all__ = ["materialize_schedule"]
