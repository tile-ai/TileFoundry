from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.core import Call, Expr, Tuple, Var
from tilefoundry.ir.hir.function import Function

from ..graph import (
    GraphStorageConstraint,
    ScheduleGraph,
    ScheduleNode,
    ScheduleRegion,
    ScheduleUse,
    ScheduleValue,
)
from ..input import ScheduleInput


class FunctionCallGraphError(ValueError):
    pass


@dataclass
class _BuilderState:
    function: Function
    value_by_expr: dict[int, int]
    values: list[ScheduleValue]
    nodes: list[ScheduleNode]
    uses: list[ScheduleUse]
    consumers: dict[int, list[int]]
    producer_by_value: dict[int, int]
    node_by_id: dict[int, ScheduleNode]


class FunctionCallGraphBuilder:
    """Lift only the entry function's composed function calls."""

    def build(self, schedule_input: ScheduleInput) -> ScheduleGraph:
        function = schedule_input.function
        if function.body is None:
            raise FunctionCallGraphError("schedule entry function must have a body")
        state = _BuilderState(
            function=function,
            value_by_expr={},
            values=[],
            nodes=[],
            uses=[],
            consumers={},
            producer_by_value={},
            node_by_id={},
        )

        input_ids = tuple(self._value(state, param) for param in function.params)
        output_ids = self._visit_expr(state, function.body)
        if not isinstance(output_ids, tuple):
            output_ids = (output_ids,)

        constraints: list[GraphStorageConstraint] = []
        for authored in schedule_input.constraints:
            value_id = state.value_by_expr.get(id(authored.target))
            if value_id is None:
                raise FunctionCallGraphError(
                    f"storage constraint {authored.id} targets an expression outside "
                    "the admitted entry graph"
                )
            consumer_ids = state.consumers.get(value_id, [])
            if len(consumer_ids) != 1:
                raise FunctionCallGraphError(
                    f"storage constraint {authored.id} target value {value_id} "
                    f"has {len(consumer_ids)} consumers; exactly one is required"
                )
            constraints.append(
                GraphStorageConstraint(
                    id=authored.id,
                    target=value_id,
                    storage=authored.storage,
                    source_loc=authored.source_loc,
                    provenance=authored.provenance,
                    authored=authored,
                )
            )

        values = tuple(
            ScheduleValue(
                id=value.id,
                ir_value=value.ir_value,
                producer=value.producer,
                consumers=tuple(state.consumers.get(value.id, ())),
            )
            for value in state.values
        )
        root = ScheduleRegion(
            id=0,
            inputs=input_ids,
            outputs=output_ids,
            nodes=tuple(node.id for node in state.nodes),
        )
        graph = ScheduleGraph(
            function=function,
            root=root,
            nodes=tuple(state.nodes),
            values=values,
            uses=tuple(state.uses),
            constraints=tuple(constraints),
        )
        return graph

    def _value(self, state: _BuilderState, expr: Expr, producer: int | None = None) -> int:
        existing = state.value_by_expr.get(id(expr))
        if existing is not None:
            if producer is not None and existing in state.producer_by_value:
                if state.producer_by_value[existing] != producer:
                    raise FunctionCallGraphError(
                        f"HIR expression {id(expr)} has multiple graph producers"
                    )
            return existing
        value_id = len(state.values)
        state.value_by_expr[id(expr)] = value_id
        state.values.append(
            ScheduleValue(id=value_id, ir_value=expr, producer=producer, consumers=())
        )
        if producer is not None:
            state.producer_by_value[value_id] = producer
        return value_id

    def _visit_expr(self, state: _BuilderState, expr: Expr) -> int | tuple[int, ...]:
        if isinstance(expr, Call) and isinstance(expr.target, Function):
            existing = state.value_by_expr.get(id(expr))
            if existing is not None:
                return existing
            input_ids: list[int] = []
            for arg in expr.args:
                value = self._visit_expr(state, arg)
                if isinstance(value, tuple):
                    raise FunctionCallGraphError(
                        f"function call {expr.target.name!r} operand is a tuple"
                    )
                input_ids.append(value)
            node_id = len(state.nodes)
            for index, value in enumerate(input_ids):
                use_id = len(state.uses)
                state.uses.append(
                    ScheduleUse(
                        id=use_id,
                        value=value,
                        consumer=node_id,
                        operand_index=index,
                    )
                )
                state.consumers.setdefault(value, []).append(node_id)
            output_id = self._value(state, expr, producer=node_id)
            node = ScheduleNode(
                id=node_id,
                ir_call=expr,
                callee=expr.target,
                inputs=tuple(input_ids),
                outputs=(output_id,),
            )
            state.nodes.append(node)
            state.node_by_id[node_id] = node
            return output_id
        if isinstance(expr, Tuple):
            return tuple(self._visit_expr(state, item) for item in expr.elements)
        if isinstance(expr, Var):
            return self._value(state, expr)
        if isinstance(expr, Call):
            for arg in expr.args:
                self._visit_expr(state, arg)
            return self._value(state, expr)
        return self._value(state, expr)


__all__ = ["FunctionCallGraphBuilder", "FunctionCallGraphError"]
