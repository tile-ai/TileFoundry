"""Shared structural facts collected from one normalized HIR graph."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, fields, replace

import isl

from tilefoundry.ir.core import Call, Constant, Var, value_label
from tilefoundry.ir.core.metadata import ValuationProvider
from tilefoundry.ir.core.module import Module, ModuleList
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.types import TensorType, TupleType
from tilefoundry.ir.types.mesh import Mesh
from tilefoundry.ir.types.utils import local_type_of
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.target import Target
from tilefoundry.visitor_registry.access_relation import AccessRelation, identity_access, leaves_of
from tilefoundry.visitor_registry.call_frame import CallFrame, CallFrameVisitor

from .access import BufferAlias, apply_source, bind_parameters
from .errors import AnalysisError
from .metadata import (
    BindingValuation,
    InstanceValuation,
    LoopNestValuation,
    TimelineMetadata,
    TimingValuation,
    VarianceValuation,
)
from .precision import AnalysisPrecision


class _FrameContext(ValuationProvider):
    def request_valuation[V](self, valuation: type[V]) -> V:
        if isinstance(self, valuation):
            return self
        raise AnalysisError(f"{type(self).__name__} does not provide {valuation.__name__}")


@dataclass
class RootContext(_FrameContext, TimingValuation, LoopNestValuation, InstanceValuation):
    """The root has no caller bindings, surrounding loops, or instance prefix."""

    def event(self, local: int) -> int:
        return local

    def timeline(self, local: TimelineMetadata) -> TimelineMetadata:
        return local

    def repeats(self) -> Repeats:
        from .iteration_scope import Repeats  # noqa: PLC0415

        return Repeats(1, 1, 1, AnalysisPrecision.EXACT)

    def domain(self) -> isl.set:
        return isl.set("{ [] }")

    def path(self) -> tuple[str, ...]:
        return ()


@dataclass
class CallContext(
    _FrameContext,
    TimingValuation,
    LoopNestValuation,
    VarianceValuation,
    BindingValuation,
    InstanceValuation,
):
    """One call's interpretation relative to its direct caller."""

    call: Call
    scope: IterationScope
    bindings: tuple[tuple[Var, Expr], ...]
    sources: dict[int, BufferAlias | tuple]
    topology_level: str | None
    event_base: int = 0
    timeline_base: int = 0
    instance: tuple[str, ...] = ()
    view: str = "narrow"
    implicit: tuple[Var, ...] = ()

    def event(self, local: int) -> int:
        return self.event_base + local

    def timeline(self, local: TimelineMetadata) -> TimelineMetadata:
        return replace(
            local,
            start_ns=self.timeline_base + local.start_ns,
            end_ns=self.timeline_base + local.end_ns,
        )

    def repeats(self) -> Repeats:
        """Scale outer loops; executing positions belong to the checked definition mesh."""
        return replace(self.scope.repeats_of(self.call, self.topology_level), mesh_positions=1)

    def domain(self) -> isl.set:
        return self.scope.domain

    def varies(self, formal: Var) -> frozenset[int]:
        argument = next((actual for param, actual in self.bindings if param is formal), None)
        if argument is None:
            if any(formal is weight for weight in self.implicit):
                index = self.call.target.args[1] if isinstance(self.call.target, Call) else None
                return self.scope.variance.get(id(index), frozenset())
            raise AnalysisError(f"call has no binding for parameter {formal.name!r}")
        return self.scope.variance.get(id(argument), frozenset())

    def trips(self, formals: tuple[Var, ...]) -> tuple[int, AnalysisPrecision]:
        changing = set().union(*(self.varies(formal) for formal in formals))
        product = 1
        precision = AnalysisPrecision.EXACT
        cursor = self.scope
        while cursor is not None:
            if id(cursor) in changing:
                product *= cursor.trips()
                precision = precision.join(cursor.trips_precision)
            cursor = cursor.parent
        return product, precision

    def actual(self, formal: Var, index: int = 0) -> BufferAlias:
        argument = next((actual for param, actual in self.bindings if param is formal), None)
        if argument is None:
            raise AnalysisError(f"call has no binding for parameter {formal.name!r}")

        def leaves(source):
            if isinstance(source, tuple):
                return tuple(leaf for child in source for leaf in leaves(child))
            return (source,)

        return leaves(self.sources[id(argument)])[index]

    def result(self) -> BufferAlias:
        source = self.sources[id(self.call)]
        if isinstance(source, tuple):
            relation = identity_access(1).relation.insert_dims(isl.dim_type.IN, 0, self.scope.depth)
            relation = relation.intersect_domain(
                self.scope.domain.insert_dims(isl.dim_type.SET, self.scope.depth, 1)
            )
            return BufferAlias(self.call, AccessRelation(relation, self.scope.domain_params))
        return source, precision

    def project(self, source: BufferAlias) -> tuple[BufferAlias, AnalysisPrecision]:
        """Interpret a definition-local source in this direct caller."""
        return VisitContext((self,)).project(source)

    def path(self) -> tuple[str, ...]:
        return self.instance


@dataclass
class VisitContext(
    _FrameContext,
    TimingValuation,
    LoopNestValuation,
    VarianceValuation,
    BindingValuation,
    InstanceValuation,
):
    """Compose one layer per entered call, from the root to the current definition."""

    stack: tuple[CallContext, ...]

    def event(self, local: int) -> int:
        for frame in reversed(self.stack):
            local = frame.event(local)
        return local

    def timeline(self, local: TimelineMetadata) -> TimelineMetadata:
        for frame in reversed(self.stack):
            local = frame.timeline(local)
        return local

    def repeats(self) -> Repeats:
        from .iteration_scope import Repeats  # noqa: PLC0415

        varying = trips = positions = 1
        precision = AnalysisPrecision.EXACT
        for frame in self.stack:
            held = frame.repeats()
            varying *= held.varying_loop_trips
            trips *= held.loop_trips
            positions *= held.mesh_positions
            precision = precision.join(held.trips_precision)
        return Repeats(varying, trips, positions, precision)

    def domain(self) -> isl.set:
        domain = isl.set("{ [] }")
        for frame in self.stack:
            domain = domain.flat_product(frame.domain())
        return domain

    def varies(self, formal: Var) -> frozenset[int]:
        changing = set(self.stack[-1].varies(formal))
        for frame in reversed(self.stack[:-1]):
            for param in (*(param for param, _argument in frame.bindings), *frame.implicit):
                if id(param) in changing:
                    changing.remove(id(param))
                    changing.update(frame.varies(param))
        return frozenset(changing)

    def trips(self, formals: tuple[Var, ...]) -> tuple[int, AnalysisPrecision]:
        changing = set().union(*(self.varies(formal) for formal in formals))
        product = 1
        precision = AnalysisPrecision.EXACT
        seen = set()
        for frame in self.stack:
            cursor = frame.scope
            while cursor is not None:
                if id(cursor) in changing and id(cursor) not in seen:
                    seen.add(id(cursor))
                    product *= cursor.trips()
                    precision = precision.join(cursor.trips_precision)
                cursor = cursor.parent
        return product, precision

    def _resolve_source(
        self, source: BufferAlias, frames: tuple[CallContext, ...]
    ) -> tuple[BufferAlias, AnalysisPrecision]:
        precision = AnalysisPrecision.EXACT
        for frame in reversed(frames):
            depth = frame.scope.depth
            relation = source.relation.relation.insert_dims(isl.dim_type.IN, 0, depth)
            domain = frame.scope.domain.insert_dims(
                isl.dim_type.SET, depth, relation.dim(isl.dim_type.IN) - depth
            )
            relation = relation.intersect_domain(domain)
            relation, bounded = bind_parameters(
                relation,
                source.relation.values,
                frame,
                tuple(param for param, _argument in frame.bindings),
                frame.scope,
                source.buffer.type,
                narrow=frame.view == "narrow",
            )
            precision = precision.join(bounded)
            if any(source.buffer is formal for formal, _argument in frame.bindings):
                actual = frame.actual(source.buffer, source.index)
                held = source.buffer.type
                if isinstance(held, TupleType):
                    held = leaves_of(held)[source.index]
                rank = len(held.shape) if isinstance(held, TensorType) else 1
                relation = apply_source(relation, actual, rank, depth, frame.scope.domain_params)
                values = {
                    **frame.scope.domain_params,
                    **source.relation.values,
                    **actual.relation.values,
                }
                values = {
                    name: value
                    for name, value in values.items()
                    if relation.find_dim_by_name(isl.dim_type.PARAM, name) >= 0
                }
                source = BufferAlias(actual.buffer, AccessRelation(relation, values), actual.index)
            else:
                values = {**frame.scope.domain_params, **source.relation.values}
                values = {
                    name: value
                    for name, value in values.items()
                    if relation.find_dim_by_name(isl.dim_type.PARAM, name) >= 0
                }
                source = replace(source, relation=AccessRelation(relation, values))
        return source, precision

    def project(self, source: BufferAlias) -> tuple[BufferAlias, AnalysisPrecision]:
        """Compose a local source with every call boundary on the root stack."""
        return self._resolve_source(source, self.stack)

    def actual(self, formal: Var, index: int = 0) -> BufferAlias:
        return self._resolve_source(self.stack[-1].actual(formal, index), self.stack[:-1])[0]

    def result(self) -> BufferAlias:
        return self._resolve_source(self.stack[-1].result(), self.stack[:-1])[0]

    def path(self) -> tuple[str, ...]:
        return tuple(part for frame in self.stack for part in frame.path())


@dataclass
class AnalyzeContext(ValuationProvider):
    """Per-call inputs and the current shared lexical scope.

    Analysis reads the program ``check_program`` typed for it, so the type of a
    value is the one stored on that IR. ``root`` and ``current`` are bound once
    the scope tree is built; reading types and relations does not need them.
    ``current_mesh`` is the execution mesh a relation is asked under; analysis
    asks outside any, as type inference does at a Function's top level.
    """

    module: Module
    target: Target
    topology_level: str | None
    options: object | None
    root: "IterationScope | None" = None
    current: "IterationScope | None" = None
    current_mesh: "Mesh | None" = None
    frame: CallFrame | None = None
    valuation: ValuationProvider = field(default_factory=RootContext)
    definitions: dict[int, AnalyzeContext] = field(default_factory=dict)
    sources: dict[str, dict[int, BufferAlias | tuple]] = field(default_factory=dict)
    precisions: dict[str, dict[int, AnalysisPrecision | tuple]] = field(default_factory=dict)
    access_summary: dict[str, tuple[tuple[Access, bool], ...]] = field(default_factory=dict)
    access_parameters: dict[str, object] = field(default_factory=dict)
    node_scopes: dict[int, IterationScope] = field(default_factory=dict)
    definition_runs: set[str] = field(default_factory=set)
    liveness: Liveness | None = None
    call_events: dict[int, int] = field(default_factory=dict)
    call_times: dict[int, int] = field(default_factory=dict)
    allocation: AllocationModel | None = None

    def run_definitions(
        self, selector: str, run: Callable[[Function, AnalyzeContext], None]
    ) -> None:
        """Run a family once per definition, with callees preceding callers."""
        for local in self.definitions.values():
            if selector in local.definition_runs:
                raise AnalysisError(
                    f"{selector}: metadata for definition {local.frame.function.name!r} "
                    "is computed more than once"
                )
            local.definition_runs.add(selector)
            run(local.frame.function, local)

    def walk_visits(self, *, view: str = "narrow"):
        """Yield local definitions with one composed frame for each root call path."""

        def walk(local, stack):
            yield local, VisitContext(stack) if stack else RootContext()
            for node in collect_exprs(local.frame.function.body):
                if not isinstance(node, Call):
                    continue
                callee = CallFrameVisitor.callee_frame(node, local.frame)
                if callee is None:
                    continue
                frame = local.call_context(
                    node,
                    view=view,
                    event_base=local.call_events.get(id(node), 0),
                    timeline_base=local.call_times.get(id(node), 0),
                )
                yield from walk(self.definitions[id(callee.function)], (*stack, frame))

        yield from walk(self, ())

    def family[C](self, kind: type[C], **changes) -> C:
        """Create family state while retaining the current definition's common context."""
        values = {item.name: getattr(self, item.name) for item in fields(AnalyzeContext)}
        values.update(changes)
        return kind(**values)

    def request_valuation[V](self, valuation: type[V]) -> V:
        """Ask the current visit frame for its interpretation."""
        return self.valuation.request_valuation(valuation)

    def call_context(
        self, call: Call, *, view: str = "narrow", event_base: int = 0, timeline_base: int = 0
    ) -> CallContext:
        """Bind one leaf call without entering the callee's local scope tree."""
        callee = CallFrameVisitor.callee_frame(call, self.frame)
        if callee is None:
            raise AnalysisError("an op has no callee valuation")
        formals = tuple(
            param
            for param in callee.function.params
            if isinstance(call.target, Function) or not param.is_const
        )
        target = call.target
        slot = (
            target
            if isinstance(target, ModuleList)
            else (
                target.args[0]
                if isinstance(target, Call) and isinstance(target.args[0], ModuleList)
                else None
            )
        )
        instance = ()
        if slot is not None:
            index = target.args[1] if isinstance(target, Call) else 0
            if isinstance(index, Constant):
                index = index.value
            instance = (slot.key, str(index) if isinstance(index, int) else value_label(index))
        return CallContext(
            call,
            self.node_scopes[id(call)],
            tuple(zip(formals, call.args, strict=True)),
            self.sources[view],
            self.topology_level,
            event_base,
            timeline_base,
            instance,
            view,
            tuple(
                param
                for param in callee.function.params
                if all(param is not formal for formal in formals)
            ),
        )

    @property
    def topologies(self):
        return (
            self.frame.topologies if self.frame is not None else self.module.effective_topologies()
        )

    def type_of(self, expr):
        """The type the analyzed IR holds for *expr*."""
        return expr.type

    def local_type_of(self, expr):
        """*expr*'s type as one unit of ``topology_level`` holds it, or whole without one."""
        if self.topology_level is None:
            return expr.type
        return local_type_of(
            expr.type, topology_level=self.topology_level, topologies=self.topologies
        )


__all__ = ["AnalyzeContext", "RootContext", "CallContext", "VisitContext"]
