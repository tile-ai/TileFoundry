from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.core import Call, Constant, Expr, Tuple, Var
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.schedule.constraints import AgentConstraint, constraint_metadata

from .fingerprint import logical_fingerprint


class ScheduleGraphError(ValueError):
    """Raised when a Module cannot be admitted to the v1 schedule graph."""


@dataclass(frozen=True, slots=True)
class GraphValueRef:
    call_path: tuple[int, ...]
    function_id: int
    local_value_id: int


@dataclass(frozen=True, slots=True)
class FunctionRegion:
    function_id: int
    call_path: tuple[int, ...]
    function: Function
    inputs: tuple[GraphValueRef, ...]
    outputs: tuple[GraphValueRef, ...]
    ops: tuple[int, ...]
    calls: tuple[int, ...]

    @property
    def nodes(self) -> tuple[int, ...]:
        return self.calls


@dataclass(frozen=True, slots=True)
class GraphValue:
    ref: GraphValueRef
    ir_value: Expr
    producer: int | None
    consumers: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GraphOp:
    id: int
    function_id: int
    call_path: tuple[int, ...]
    ir_expr: Call
    target: object
    inputs: tuple[GraphValueRef, ...]
    output: GraphValueRef
    call_instance: int | None = None


@dataclass(frozen=True, slots=True)
class CallInstance:
    id: int
    call_path: tuple[int, ...]
    caller_function_id: int
    callee_function_id: int
    ir_call: Call
    arguments: tuple[GraphValueRef, ...]
    result: GraphValueRef
    callee_inputs: tuple[GraphValueRef, ...]
    callee_outputs: tuple[GraphValueRef, ...]


@dataclass(frozen=True, slots=True)
class GraphEdge:
    id: int
    source: GraphValueRef
    destination: GraphValueRef
    kind: str
    op_id: int | None = None
    operand_index: int | None = None


@dataclass(frozen=True, slots=True)
class GraphConstraint:
    id: int
    target: GraphValueRef
    constraint: AgentConstraint
    function_id: int
    call_path: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ProgramScheduleGraph:
    module: Module
    entry_function_id: int
    regions: tuple[FunctionRegion, ...]
    values: tuple[GraphValue, ...]
    ops: tuple[GraphOp, ...]
    calls: tuple[CallInstance, ...]
    edges: tuple[GraphEdge, ...]
    constraints: tuple[GraphConstraint, ...]
    logical_fingerprint: str

    @property
    def root(self) -> FunctionRegion:
        return next(region for region in self.regions if region.call_path == ())

    @property
    def nodes(self) -> tuple[GraphOp, ...]:
        return self.ops

    def value(self, ref: GraphValueRef) -> GraphValue:
        for value in self.values:
            if value.ref == ref:
                return value
        raise KeyError(ref)


@dataclass
class _RegionState:
    function_id: int
    call_path: tuple[int, ...]
    function: Function
    value_by_expr: dict[int, GraphValueRef]
    values: list[GraphValue]
    consumers: dict[GraphValueRef, list[int]]
    producer_by_value: dict[GraphValueRef, int]
    ops: list[int]
    calls: list[int]
    inputs: tuple[GraphValueRef, ...] = ()
    outputs: tuple[GraphValueRef, ...] = ()


class ProgramScheduleGraphBuilder:
    """Expand the complete HIR call graph without descending through opaque op targets."""

    def __init__(self) -> None:
        self._function_ids: dict[int, int] = {}
        self._functions: dict[int, Function] = {}
        self._regions: list[FunctionRegion] = []
        self._states: dict[tuple[int, tuple[int, ...]], _RegionState] = {}
        self._ops: list[GraphOp] = []
        self._calls: list[CallInstance] = []
        self._next_op_id = 0
        self._next_call_id = 0
        self._edges: list[GraphEdge] = []
        self._constraints: list[GraphConstraint] = []
        self._active: list[int] = []

    def build(self, module: Module) -> ProgramScheduleGraph:
        if not isinstance(module, Module):
            raise TypeError("ProgramScheduleGraphBuilder expects a Module")
        entry = module.entry_function()
        if not isinstance(entry, Function):
            raise ScheduleGraphError("schedule graph currently supports HIR functions only")
        entry_id = self._function_id(entry)
        self._expand_region(entry, (), entry_id)
        regions = tuple(self._regions)
        values = tuple(value for state in self._states.values() for value in state.values)
        return ProgramScheduleGraph(
            module=module,
            entry_function_id=entry_id,
            regions=regions,
            values=values,
            ops=tuple(self._ops),
            calls=tuple(self._calls),
            edges=tuple(self._edges),
            constraints=tuple(self._constraints),
            logical_fingerprint=logical_fingerprint(module),
        )

    def _function_id(self, function: Function) -> int:
        existing = self._function_ids.get(id(function))
        if existing is not None:
            return existing
        function_id = len(self._function_ids)
        self._function_ids[id(function)] = function_id
        self._functions[function_id] = function
        return function_id

    def _expand_region(
        self, function: Function, call_path: tuple[int, ...], function_id: int
    ) -> _RegionState:
        key = (function_id, call_path)
        existing = self._states.get(key)
        if existing is not None:
            return existing
        if function_id in self._active:
            raise ScheduleGraphError(
                f"recursive HIR call is unsupported: {function.name!r} at {call_path}"
            )
        if function.body is None:
            raise ScheduleGraphError(f"function {function.name!r} has no body")
        state = _RegionState(
            function_id=function_id,
            call_path=call_path,
            function=function,
            value_by_expr={},
            values=[],
            consumers={},
            producer_by_value={},
            ops=[],
            calls=[],
        )
        self._states[key] = state
        self._active.append(function_id)
        state.inputs = tuple(self._value(state, param) for param in function.params)
        state.outputs = (self._visit_expr(state, function.body),)
        self._active.pop()
        state_values = tuple(
            GraphValue(
                ref=value.ref,
                ir_value=value.ir_value,
                producer=value.producer,
                consumers=tuple(state.consumers.get(value.ref, ())),
            )
            for value in state.values
        )
        state.values[:] = list(state_values)
        self._regions.append(
            FunctionRegion(
                function_id=function_id,
                call_path=call_path,
                function=function,
                inputs=state.inputs,
                outputs=state.outputs,
                ops=tuple(state.ops),
                calls=tuple(state.calls),
            )
        )
        return state

    def _value(self, state: _RegionState, expr: Expr, producer: int | None = None) -> GraphValueRef:
        existing = state.value_by_expr.get(id(expr))
        if existing is not None:
            return existing
        ref = GraphValueRef(state.call_path, state.function_id, len(state.values))
        state.value_by_expr[id(expr)] = ref
        state.values.append(GraphValue(ref, expr, producer, ()))
        if producer is not None:
            state.producer_by_value[ref] = producer
        self._record_constraints(state, expr, ref)
        return ref

    def _record_constraints(self, state: _RegionState, expr: Expr, ref: GraphValueRef) -> None:
        metadata = constraint_metadata(expr)
        if metadata is None:
            return
        for constraint in metadata.constraints:
            self._constraints.append(
                GraphConstraint(
                    id=len(self._constraints),
                    target=ref,
                    constraint=constraint,
                    function_id=state.function_id,
                    call_path=state.call_path,
                )
            )

    def _visit_expr(self, state: _RegionState, expr: Expr) -> GraphValueRef:
        existing = state.value_by_expr.get(id(expr))
        if existing is not None:
            return existing
        if isinstance(expr, Tuple):
            for element in expr.elements:
                self._visit_expr(state, element)
            return self._value(state, expr)
        if isinstance(expr, (Var, Constant)):
            return self._value(state, expr)
        if not isinstance(expr, Call):
            return self._value(state, expr)

        inputs = tuple(self._visit_expr(state, arg) for arg in expr.args)
        op_id = self._next_op_id
        self._next_op_id += 1
        output = self._value(state, expr, producer=op_id)
        call_instance_id: int | None = None
        if isinstance(expr.target, Function):
            callee_id = self._function_id(expr.target)
            call_instance_id = self._next_call_id
            self._next_call_id += 1
            child_path = (*state.call_path, call_instance_id)
            if callee_id in self._active:
                raise ScheduleGraphError(
                    f"recursive HIR call {expr.target.name!r} at {child_path}"
                )
            child = self._expand_region(expr.target, child_path, callee_id)
            call = CallInstance(
                id=call_instance_id,
                call_path=child_path,
                caller_function_id=state.function_id,
                callee_function_id=callee_id,
                ir_call=expr,
                arguments=inputs,
                result=output,
                callee_inputs=child.inputs,
                callee_outputs=child.outputs,
            )
            self._calls.append(call)
            state.calls.append(call_instance_id)
            for index, (source, destination) in enumerate(zip(inputs, child.inputs)):
                self._edges.append(
                    GraphEdge(len(self._edges), source, destination, "call_arg", op_id, index)
                )
            for source in child.outputs:
                self._edges.append(GraphEdge(len(self._edges), source, output, "call_result", op_id))
        else:
            state.ops.append(op_id)
        if isinstance(expr.target, Function):
            state.ops.append(op_id)
        graph_op = GraphOp(
            id=op_id,
            function_id=state.function_id,
            call_path=state.call_path,
            ir_expr=expr,
            target=expr.target,
            inputs=inputs,
            output=output,
            call_instance=call_instance_id,
        )
        self._ops.append(graph_op)
        for index, source in enumerate(inputs):
            state.consumers.setdefault(source, []).append(op_id)
            self._edges.append(
                GraphEdge(len(self._edges), source, output, "data", op_id, index)
            )
        return output


def build_program_schedule_graph(module: Module) -> ProgramScheduleGraph:
    return ProgramScheduleGraphBuilder().build(module)


__all__ = [
    "CallInstance",
    "FunctionRegion",
    "GraphConstraint",
    "GraphEdge",
    "GraphOp",
    "GraphValue",
    "GraphValueRef",
    "ProgramScheduleGraph",
    "ProgramScheduleGraphBuilder",
    "ScheduleGraphError",
    "build_program_schedule_graph",
]
