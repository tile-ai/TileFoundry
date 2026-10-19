"""The shared authored-program gate for analysis.

An analysis reads inferred types and assumes the authored program holds
together. Both conditions are established once per public call rather than
per family, so no family can be the one that forgot.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from tilefoundry.ir.core import (
    Call,
    Expr,
    RangeMetadata,
    attach_metadata,
    value_labels,
)
from tilefoundry.ir.core.module import Module, ModuleList
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.specialize import (
    InstantiateDimensionCloner,
    SpecializationError,
    _DimVarCollector,
    dim_vars_reached,
)
from tilefoundry.ir.types import Topology, callable_type_for
from tilefoundry.ir.types.substitute import dim_vars_by_name, substitute_shape_dim
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.target import UnsupportedCapabilityError
from tilefoundry.target.facts import TopologyFacts
from tilefoundry.visitor_registry.contexts import TypeInferContext, ValuationContext
from tilefoundry.visitor_registry.typeinfer import TypeInferVisitor
from tilefoundry.visitor_registry.valuation import ValuationVisitor

from .errors import AnalysisError
from .facts import PerformanceServiceFacts
from .metadata import (
    ComputeCostMetadata,
    ExecutionMeshMetadata,
    MemoryMetadata,
    PerformanceMetadata,
    PerformanceSummaryMetadata,
    RegionMemoryMetadata,
    RooflineMetadata,
)

_NODE_BUDGET = 10_000
_DERIVED_METADATA = {
    ComputeCostMetadata,
    ExecutionMeshMetadata,
    MemoryMetadata,
    PerformanceMetadata,
    PerformanceSummaryMetadata,
    RangeMetadata,
    RegionMemoryMetadata,
    RooflineMetadata,
}


@dataclass
class AnalysisCheckContext(ValuationContext):
    """What every input check reads: the program, the machine, and the topology level.

    The context is bound to the derived Function the analyses will read, so a
    check answers about the same program they do. Nothing a check reads is a
    derived analysis record.
    """

    topology_level: str | None = None

    @property
    def selected_topology(self) -> Topology:
        """The topology level the analyses were asked about."""
        if self.topology_level is None:
            raise AnalysisError(
                "no topology level was selected, so results cannot carry an execution placement"
            )
        return self.module.resolve_topology(self.topology_level)


def analysis_check_context(
    module: Module, function: Function, topology_level: str | None
) -> AnalysisCheckContext:
    """Bind one context to the derived Function the analyses will read."""
    return AnalysisCheckContext(
        module=module,
        function=function,
        target=module.resolve_target(),
        topology_level=topology_level,
    )


class PerformanceChecker:
    """What `performance` needs before anything measures the program.

    Rates it can put on a clock, and nothing about storage: where the buffers go
    is not this question, and nothing here decides it. Where an occurrence runs
    is a fact the program states, so there is nothing left to establish
    occurrence by occurrence and this walks nothing.
    """

    def check_target_facts(self, ctx: AnalysisCheckContext) -> None:
        """Require a machine whose stated capacity and rates fit the question."""
        try:
            topologies = ctx.target.get_facts(TopologyFacts)
        except UnsupportedCapabilityError as error:
            raise AnalysisError(f"performance: {error}") from None
        if ctx.topology_level is None:
            raise AnalysisError(
                "performance: no topology level was selected, so there is no "
                "unit for a rate to be stated per"
            )
        level = topologies.level(ctx.topology_level)
        if level is None or level.max_physical_units is None:
            raise AnalysisError(
                "performance: the target states no physical units for topology "
                f"level {ctx.topology_level!r}"
            )
        units = level.max_physical_units
        try:
            services = ctx.target.get_facts(PerformanceServiceFacts, ctx.topology_level)
        except UnsupportedCapabilityError as error:
            raise AnalysisError(f"performance: {error}") from None
        if isinstance(units, bool) or not isinstance(units, int) or units <= 0:
            raise AnalysisError(
                "performance: the target must publish a positive parallel-unit "
                f"capacity, got {units!r}"
            )
        if services.unit != ctx.topology_level:
            raise AnalysisError(
                f"performance: selected topology level {ctx.topology_level!r}, but "
                f"the target's one-unit throughputs are stated for {services.unit!r}"
            )

    def visit(self, expr: Expr, ctx: AnalysisCheckContext) -> None:
        """Ask nothing of any single occurrence."""


def program_dim_vars(module: Module, function: Function) -> dict[str, object]:
    """Dimension declarations in reachable values and effective execution geometry."""
    return dim_vars_reached(function, _required_frame(module, function))


def _required_frame(module: Module, function: Function):
    """Read a function directly from its supplied owning definition."""
    if module.owns(function, derived=True):
        return ValuationContext(module, function)
    raise AnalysisError(
        f"function {function.name!r} is owned by no single node of module "
        f"{module.name!r}; supply its owning module directly"
    )


def _prepare_function(function):
    """Clear derived facts through the shared binding-aware traversal."""
    class Prepare(ValuationVisitor[None]):
        def default_visit_leaf(self, expr, operands, ctx):
            if not isinstance(expr, ModuleList):
                expr.metadata = tuple(
                    item for item in expr.metadata if type(item) not in _DERIVED_METADATA
                )

    visitor = Prepare()
    for expr in (*function.params, function):
        visitor.visit(expr)


class _CheckCalls(ValuationVisitor[None]):
    def __init__(self, derive, label, function):
        super().__init__()
        self.derive, self.label = derive, label
        calls = tuple(node for node in collect_exprs(function.body) if isinstance(node, Call))
        self.labels = dict(zip(calls, value_labels(calls), strict=True))

    def visit_Call(self, call, ctx):
        self.visit_operands(call, ctx)
        inner = ctx._callee_context(call)
        if inner is None:
            return
        callee = inner.function
        implicit = not isinstance(call.target, Function)
        given = iter(call.args)
        types = tuple(
            param.type if implicit and param.is_const else next(given).type
            for param in callee.params
        )
        self.derive(inner, types, ctx.current_mesh, f"{self.label}/{self.labels[call]}")


def check_program(
    module: Module,
    function: Function,
    *,
    topology_level: str | None = None,
    budget: int = _NODE_BUDGET,
    analyzers: tuple[object, ...] = (),
    dims: Mapping[str, int] | None = None,
) -> tuple[Module, Function]:
    """Prove one program holds together, then return the view analysis will read.

    The returned Module retains calls and has one typed derived body per definition.
    LoopRegion survives unchanged. The authored Module and Function are untouched.

    Each analysis states what it needs of the program through its own checker,
    and every one of them answers before any of them writes. A checker that
    needs the program walks it itself.
    """
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise AnalysisError(
            f"checking {function.name!r} needs a non-negative integer node budget, got {budget!r}"
        )
    if dims is not None:
        if not isinstance(dims, Mapping) or not dims:
            raise SpecializationError(
                f"specialising {function.name!r} needs a non-empty mapping of dimension names to extents, got {dims!r}"
            )
        for name, extent in dims.items():
            if not isinstance(name, str) or not name:
                raise SpecializationError(
                    f"specialising {function.name!r}: {name!r} is not a dimension name"
                )
            if isinstance(extent, bool) or not isinstance(extent, int):
                raise SpecializationError(
                    f"specialising {function.name!r}: dimension {name!r} takes an integer extent, got {extent!r}"
                )
    supplied = {} if dims is None else dict(dims)
    original_frame = _required_frame(module, function)
    declared = {}
    collector = _DimVarCollector(declared, supplied)
    collector.visit_function(function, original_frame)
    if collector.cycles:
        raise AnalysisError(f"recursive call path through {collector.cycles[0]!r}")
    if collector.missing_bodies:
        name, path = collector.missing_bodies[0]
        raise AnalysisError(f"cannot analyze Function {name!r} without a body at {path!r}")
    unknown = sorted(set(supplied) - declared.keys())
    if unknown:
        raise SpecializationError(
            f"{function.name!r} has no dimension named {unknown!r}; it states {sorted(declared)}"
        )
    for name, extent in supplied.items():
        substitute_shape_dim(declared[name], {name: extent})
    cloner = InstantiateDimensionCloner(dims)
    for reached, _frame in collector.functions:
        cloner.function(reached)
    bound_root = cloner.function(function)
    bound_module = cloner.module(module)
    initial = ValuationContext(bound_module, bound_root)
    residual = set(dim_vars_reached(bound_root, initial))
    for owner in cloner.modules.values():
        residual.update(dim_vars_by_name(owner.topologies))
    if residual:
        raise SpecializationError(
            f"{function.name!r} still states {sorted(residual)} as ranges after binding {sorted(supplied)}; every dimension has to be given an extent"
        )
    signatures = {}

    def derive(frame, types, mesh, label):
        definition = frame.function
        identity = id(definition)
        signature = (types, mesh)
        if identity in signatures:
            previous, path = signatures[identity]
            if previous != signature:
                raise AnalysisError(
                    f"callee {definition.name!r} has incompatible signatures at {path!r} and "
                    f"{label!r}: {previous!r} versus {signature!r}"
                )
            return
        signatures[identity] = signature, label
        for param, type_ in zip(definition.params, types, strict=True):
            param.type = type_
        _prepare_function(definition)
        attach_metadata(definition, ExecutionMeshMetadata(mesh))
        context = TypeInferContext(module=frame.module, function=frame.function, target=frame.target, topologies=frame.topologies, owner=definition, current_mesh=mesh)
        TypeInferVisitor(ranges=True).visit(definition, context)
        definition.return_type = definition.body.type
        definition.type = callable_type_for(definition.params, definition.return_type)
        _CheckCalls(derive, label, definition).visit(definition.body, context)

    root, module = bound_root, bound_module
    derive(initial, tuple(p.type for p in root.params), None, function.name)

    frames = tuple(ValuationVisitor.definitions(ValuationContext(module, root)))
    functions = tuple(frame.function for frame in frames)
    size = sum(len(collect_exprs(body.body)) for body in functions)
    if size > budget:
        raise AnalysisError(
            f"analysis program {function.name!r} has {size} derived body nodes, exceeding the node budget {budget}"
        )
    target = module.resolve_target()
    for topology in module.effective_topologies():
        try:
            target.validate_program_topology(topology)
        except ValueError as error:
            raise AnalysisError(
                f"program topology level {topology.name!r} with extent "
                f"{topology.size!r} is invalid: {error}"
            ) from None
    if topology_level is not None:
        try:
            module.resolve_topology(topology_level)
        except ValueError as error:
            raise AnalysisError(
                f"program topology level {topology_level!r} is invalid: {error}"
            ) from None

    for frame in frames:
        for call in collect_exprs(frame.function.body):
            if not isinstance(call, Call) or (inner := frame._callee_context(call) if frame is not None else None) is None:
                continue
            for field, noun in (("target", "target"), ("topologies", "topology hierarchy")):
                expected, actual = getattr(frame, field), getattr(inner, field)
                if expected != actual:
                    suffix = (
                        "one kernel invocation runs one target"
                        if field == "target"
                        else (
                            "one kernel invocation runs one hierarchy -- declare none on the child and inherit"
                        )
                    )
                    raise AnalysisError(
                        f"{frame.module.name!r} calls {inner.module.name!r}, which resolves "
                        f"a different {noun} {actual!r} against {expected!r}; {suffix}"
                    )
    check_analyses(module, root, analyzers, topology_level=topology_level)
    return module, root


def check_analyses(
    module: Module,
    function: Function,
    analyzers: tuple[object, ...],
    *,
    topology_level: str | None = None,
) -> None:
    """Check requested families against a built program before any writes."""
    functions = tuple(frame.function for frame in ValuationVisitor.definitions(ValuationContext(module, function)))
    ctx = analysis_check_context(module, function, topology_level)
    for analyzer in analyzers:
        checker = analyzer.get_checker()
        if checker is None:
            continue
        checker.check_target_facts(ctx)
        for body in functions:
            checker.visit(body.body, replace(ctx, function=body))
