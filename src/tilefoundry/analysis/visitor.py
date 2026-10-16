"""The common inputs and lexical position used by analysis families."""

from __future__ import annotations

from dataclasses import dataclass, replace

import isl

from tilefoundry.ir.core import Call, Var, get_metadata, value_label
from tilefoundry.ir.core.module import ModuleList
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.types import TensorType
from tilefoundry.visitor_registry.access_relation import AccessRelation, leaves_of
from tilefoundry.visitor_registry.call_frame import CallFrameVisitor
from tilefoundry.visitor_registry.contexts import CostContext

from .access import (
    AliasKind,
    BufferAlias,
    SSAValue,
    apply_source,
    bind_parameters,
    name_loop_parameters,
)
from .errors import AnalysisError
from .metadata import (
    BindingValuation,
    InstanceValuation,
    LoopNestValuation,
    PerformanceMetadata,
    TimelineMetadata,
    TimingValuation,
    VarianceValuation,
)
from .precision import AnalysisPrecision


@dataclass
class AnalyzeContext(
    CostContext,
    BindingValuation,
    VarianceValuation,
    LoopNestValuation,
    InstanceValuation,
    TimingValuation,
):
    """Read checked IR in one definition's local iteration scope."""

    options: object | None = None
    entry: Function | None = None
    access_view: str = "narrow"
    root: IterationScope | None = None
    current: IterationScope | None = None

    @property
    def module(self):
        return self.frame.module

    @property
    def target(self):
        return self.resolve_target()

    def for_region(self, region):
        return replace(
            super().for_region(region),
            current=(
                next(child for child in self.current.children if child.owner is region)
                if self.current is not None
                else None
            ),
        )

    @property
    def callee(self):
        return None if self.call is None else CallFrameVisitor.callee_frame(self.call, self.frame)

    @property
    def bindings(self):
        frame = self.callee
        if frame is None:
            return ()
        formals = tuple(
            param
            for param in frame.function.params
            if isinstance(self.call.target, Function) or not param.is_const
        )
        return tuple(zip(formals, self.call.args, strict=True))

    def actual(self, formal: Var, index: int = 0) -> BufferAlias:
        argument = next((actual for param, actual in self.bindings if param is formal), None)
        if argument is None:
            raise AnalysisError(f"binding: {formal.name!r} is not a parameter of the current call")
        return self.current.sources[self.access_view][SSAValue(argument, index)]

    def result(self) -> BufferAlias:
        return self.current.sources[self.access_view][SSAValue(self.call)]

    @property
    def _binding_formals(self):
        return () if self.frame.function is self.entry else self.frame.function.params

    def project(self, source: BufferAlias, actual_of=None) -> tuple[BufferAlias, AnalysisPrecision]:
        actual_of = self.actual if actual_of is None else actual_of
        depth = self.current.depth
        relation = source.relation.relation.insert_dims(isl.dim_type.IN, 0, depth)
        relation = relation.intersect_domain(
            self.current.domain.insert_dims(
                isl.dim_type.SET, depth, relation.dim(isl.dim_type.IN) - depth
            )
        )
        held = leaves_of(source.target.expr.type)
        held = held[source.target.index] if held else source.target.expr.type
        relation, precision = bind_parameters(
            relation,
            source.relation.values,
            self,
            tuple(param for param, _ in self.bindings),
            self.current,
            held,
            narrow=self.access_view == "narrow",
            retained_formals=self._binding_formals,
        )
        values = {**source.relation.values, **self.current.domain_params}
        target = source.target
        kind = source.kind
        if any(target.expr is formal for formal, _actual in self.bindings):
            actual = actual_of(target.expr, target.index)
            rank = len(held.shape) if isinstance(held, TensorType) else 1
            relation = apply_source(relation, actual, rank, depth, self.current.domain_params)
            argument = next(argument for formal, argument in self.bindings if formal is target.expr)
            precision = precision.join(
                self.current.precisions[self.access_view][SSAValue(argument, target.index)]
            )
            target = actual.target
            values.update(actual.relation.values)
            if actual.kind is AliasKind.VIEW:
                kind = AliasKind.VIEW
        relation, loop_values = name_loop_parameters(relation, (loop[0] for loop in self.loops))
        values.update(loop_values)
        values = {
            name: value
            for name, value in values.items()
            if relation.find_dim_by_name(isl.dim_type.PARAM, name) >= 0
        }
        return BufferAlias(target, AccessRelation(relation, values), kind), precision

    def varies(self, formal: Var) -> frozenset[int]:
        for parameter, argument in self.bindings:
            if formal is parameter:
                return self.current.variance.get(id(argument), frozenset())
        for local in self.current.context.ancestors():
            if getattr(local.owner, "induction_var", None) is formal:
                return frozenset((id(local.current),))
        if self.callee is not None and any(
            formal is p and p.is_const for p in self.callee.function.params
        ):
            index = self.call.target.args[1] if isinstance(self.call.target, Call) else None
            return self.current.variance.get(id(index), frozenset())
        return self.current.variance.get(id(formal), frozenset())

    def trips(self, formals: tuple[Var, ...]) -> tuple[int, AnalysisPrecision]:
        changing = set().union(*(self.varies(formal) for formal in formals))
        product = 1
        precision = AnalysisPrecision.EXACT
        current = self.current
        while current is not None:
            if id(current) in changing:
                product *= current.trips()
                precision = precision.join(current.trips_precision)
            current = current.parent
        return product, precision

    def dependencies(self, expr):
        changing = self.current.variance.get(id(expr), frozenset())
        formals = tuple(param for param in self.root.owner.params if id(param) in changing)
        loops = tuple(
            local.owner.induction_var
            for local in self.current.context.ancestors()
            if id(local.current) in changing and hasattr(local.owner, "induction_var")
        )
        return (*formals, *loops)

    def repeats(self):
        return replace(self.current.repeats_of(self.call, self.topology_level), mesh_positions=1)

    def domain(self) -> isl.set:
        return self.current.domain

    def path(self) -> tuple[str, ...]:
        if self.call is None:
            return ()
        target = self.call.target
        slot = target.args[0] if isinstance(target, Call) else target
        if not isinstance(slot, ModuleList):
            return ()
        label = value_label(target.args[1]) if isinstance(target, Call) else "0"
        return (f"{slot.key}[{label}]",)

    def event(self, local: int) -> int:
        from .liveness import Liveness  # noqa: PLC0415

        if self.call is None:
            return local
        own = get_metadata(self.frame.function, Liveness)
        template = get_metadata(self.callee.function, Liveness)
        interval = own.interval_of(SSAValue(self.call))
        return interval.defined_at - template.timeline_end - 1 + local

    def timeline(self, local: TimelineMetadata) -> TimelineMetadata:
        own = None if self.call is None else get_metadata(self.call, PerformanceMetadata)
        start = 0 if own is None else own.timeline.start_ns
        return replace(local, start_ns=start + local.start_ns, end_ns=start + local.end_ns)

    def error(self, node, message):
        raise AnalysisError(message)


__all__ = ["AnalyzeContext"]
