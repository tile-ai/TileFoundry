from __future__ import annotations

import dataclasses

from tilefoundry.ir.core import Call, Expr, Tuple, TypeInferContext, Var
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard import Broadcast, Layout, S, ShardLayout

from ..solution import ScheduleSolution
from ..solver import SolveProblem
from ..space import EdgeKind, PhysicalRepresentation


def _type_call(target, args: tuple[Expr, ...], loc: str | None = None) -> Call:
    placeholder = Call(
        type=TensorType.scalar(DType.f32),
        target=target,
        args=args,
        loc=loc,
    )
    context = TypeInferContext()
    inferred = context.type_of(placeholder)
    return dataclasses.replace(placeholder, type=inferred)


def _layout_for_value(value_type: TensorType, mesh, *, split: bool) -> ShardLayout:
    if split and value_type.shape:
        extent = mesh.shape[0]
        axes = [
            axis
            for axis, dim in enumerate(value_type.shape)
            if type(dim) is int and dim >= extent and dim % extent == 0
        ]
        if not axes:
            raise ValueError(
                "schedule materialization needs a tensor axis divisible by "
                f"the CTA slice extent {extent}"
            )
        attrs = (S(axes[-1]),)
    else:
        attrs = (Broadcast(),)
    return ShardLayout(
        layout=Layout(shape=tuple(value_type.shape), strides=None),
        attrs=attrs,
        mesh=mesh,
    )


def _mesh_for_placement(parent, placement):
    start = placement.axis_starts[0]
    extent = placement.axis_extents[0]
    parent_extent = parent.layout.shape[0]
    if start == 0 and extent == parent_extent:
        return parent
    return parent[start : start + extent]


def _reshard(value: Expr, representation: PhysicalRepresentation, parent_mesh) -> Expr:
    if not isinstance(value.type, TensorType):
        raise ValueError("schedule materialization can only reshard tensor values")
    target_layout = _layout_for_value(value.type, parent_mesh, split=False)
    return _type_call(
        Reshard(layout=target_layout, storage=representation.storage),
        (value,),
        loc="schedule_reshard",
    )


def materialize_cuda_schedule(
    problem: SolveProblem,
    solution: ScheduleSolution,
    context,
) -> Function:
    if solution.problem_fingerprint != problem.problem_fingerprint:
        raise ValueError("cannot materialize a solution for a different problem")
    graph = problem.graph
    node_by_call = {id(node.ir_call): node for node in graph.nodes}
    use_by_operand = {
        (use.consumer, use.operand_index): use for use in graph.uses
    }
    edge_by_use = {item.use: item for item in solution.edge_assignments}
    option_by_id = {option.id: option for option in problem.space.node_options}
    edge_option_by_id = {option.id: option for option in problem.space.edge_options}
    representation_by_id = {
        representation.id: representation
        for representation in problem.space.representations
    }
    memo: dict[int, Expr] = {}

    def visit(expr: Expr) -> Expr:
        cached = memo.get(id(expr))
        if cached is not None:
            return cached
        if isinstance(expr, Var):
            result = expr
        elif isinstance(expr, Tuple):
            result = dataclasses.replace(
                expr,
                elements=tuple(visit(element) for element in expr.elements),
            )
        elif isinstance(expr, Call) and isinstance(expr.target, Function):
            node = node_by_call[id(expr)]
            arguments = []
            for index, argument in enumerate(expr.args):
                value = visit(argument)
                use = use_by_operand[(node.id, index)]
                edge_assignment = edge_by_use[use.id]
                edge_option = edge_option_by_id[edge_assignment.option]
                if edge_option.kind is EdgeKind.RESHARD:
                    destination = representation_by_id[edge_option.destination_representation]
                    value = _reshard(value, destination, context.mesh)
                arguments.append(value)
            rebuilt = _type_call(expr.target, tuple(arguments), loc=expr.loc)
            selected_option = option_by_id[solution.assignment_for(node.id).option]
            output_representation = representation_by_id[
                selected_option.output_representations[0]
            ]
            placement_mesh = _mesh_for_placement(
                context.mesh,
                solution.assignment_for(node.id).placement,
            )
            if not isinstance(rebuilt.type, TensorType):
                raise ValueError("scheduled function calls must return tensors")
            target_layout = _layout_for_value(
                rebuilt.type,
                placement_mesh,
                split=placement_mesh is not context.mesh,
            )
            if (
                rebuilt.type.storage != output_representation.storage
                or rebuilt.type.layout != target_layout
            ):
                rebuilt = _type_call(
                    Reshard(
                        layout=target_layout,
                        storage=output_representation.storage,
                    ),
                    (rebuilt,),
                    loc="schedule_placement",
                )
            result = rebuilt
        elif isinstance(expr, Call):
            result = _type_call(
                expr.target,
                tuple(visit(argument) for argument in expr.args),
                loc=expr.loc,
            )
        else:
            result = expr
        memo[id(expr)] = result
        return result

    body = visit(graph.function.body)
    if not isinstance(body, Expr):
        raise ValueError("scheduled function body did not materialize to an expression")
    return Function.build(
        name=graph.function.name,
        params=graph.function.params,
        body=body,
        return_type=body.type,
        topologies=graph.function.topologies,
        target=graph.function.target,
    )


__all__ = ["materialize_cuda_schedule"]
