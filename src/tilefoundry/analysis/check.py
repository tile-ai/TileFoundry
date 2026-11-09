"""The shared authored-program gate for analysis.

An analysis reads inferred types and assumes the authored program holds
together. Both conditions are established once per public call rather than
per family, so no family can be the one that forgot.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from tilefoundry import target
from tilefoundry.interfaces import typeinfer, valuation, verify
from tilefoundry.ir import core, dim, expr, types

from .errors import Error
from .metadata import (
    ComputeCostMetadata,
    ExecutionMeshMetadata,
    MemoryMetadata,
    PerformanceMetadata,
    PerformanceSummaryMetadata,
    RegionMemoryMetadata,
    RooflineMetadata,
)

NODE_BUDGET = 10_000
DERIVED_METADATA = frozenset(
    {
        ComputeCostMetadata,
        ExecutionMeshMetadata,
        MemoryMetadata,
        PerformanceMetadata,
        PerformanceSummaryMetadata,
        expr.RangeMetadata,
        RegionMemoryMetadata,
        RooflineMetadata,
    }
)


@dataclass
class CheckContext(valuation.Context):
    """What every input check reads: the program, the machine, and the topology level.

    The context is bound to the derived Function the analyses will read, so a
    check answers about the same program they do. Nothing a check reads is a
    derived analysis record.
    """

    topology_level: str | None = None

    @property
    def selected_topology(self) -> types.Topology:
        """The topology level the analyses were asked about."""
        if self.topology_level is None:
            raise Error(
                "no topology level was selected, so results cannot carry an execution placement"
            )
        return self.module.resolve_topology(self.topology_level)


def check_context(
    module: core.Module, function: core.Function, topology_level: str | None
) -> CheckContext:
    """Bind one context to the derived Function the analyses will read."""
    return CheckContext(
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

    def check_target_facts(self, ctx: CheckContext) -> None:
        """Require a machine whose stated capacity and rates fit the question."""
        try:
            topologies = ctx.target.get_facts(target.TopologyFacts)
        except target.UnsupportedCapabilityError as error:
            raise Error(f"performance: {error}") from None
        if ctx.topology_level is None:
            raise Error(
                "performance: no topology level was selected, so there is no "
                "unit for a rate to be stated per"
            )
        level = topologies.level(ctx.topology_level)
        if level is None or level.max_physical_units is None:
            raise Error(
                "performance: the target states no physical units for topology "
                f"level {ctx.topology_level!r}"
            )
        units = level.max_physical_units
        try:
            services = ctx.target.get_facts(target.PerformanceServiceFacts, ctx.topology_level)
        except target.UnsupportedCapabilityError as error:
            raise Error(f"performance: {error}") from None
        if isinstance(units, bool) or not isinstance(units, int) or units <= 0:
            raise Error(
                "performance: the target must publish a positive parallel-unit "
                f"capacity, got {units!r}"
            )
        if services.unit != ctx.topology_level:
            raise Error(
                f"performance: selected topology level {ctx.topology_level!r}, but "
                f"the target's one-unit throughputs are stated for {services.unit!r}"
            )

    def visit(self, node: expr.Expr, ctx: CheckContext) -> None:
        """Ask nothing of any single occurrence."""


def program_dim_vars(module: core.Module, function: core.Function) -> dict[str, object]:
    """Dimension declarations in reachable values and effective execution geometry."""
    return valuation.dim_vars_reached(function, required_frame(module, function))


def required_frame(module: core.Module, function: core.Function):
    """Read a function directly from its supplied owning definition."""
    if module.owns(function, derived=True):
        return valuation.Context(module, function)
    raise Error(
        f"function {function.name!r} is owned by no single node of module "
        f"{module.name!r}; supply its owning module directly"
    )


class ClearDerived(valuation.Visitor[None]):
    """Drop the derived records from every value it reaches."""

    def default_visit_leaf(self, node, operands, ctx):
        if not isinstance(node, core.ModuleList):
            for record_type in DERIVED_METADATA:
                node.detach_metadata(record_type)


def prepare_function(function):
    """Clear derived facts through the shared binding-aware traversal."""
    visitor = ClearDerived()
    for node in (*function.params, function):
        visitor.visit(node)


class CheckCalls(valuation.Visitor[None]):
    def __init__(self, derive, label, function):
        super().__init__()
        self.derive, self.label = derive, label
        calls = tuple(node for node in core.collect_exprs(function.body) if isinstance(node, expr.Call))
        self.labels = dict(zip(calls, expr.value_labels(calls), strict=True))

    def visit_Call(self, call, ctx):
        self.visit_operands(call, ctx)
        inner = ctx.callee_context(call)
        if inner is None:
            return
        callee = inner.function
        implicit = not isinstance(call.target, core.Function)
        given = iter(call.args)
        type_list = tuple(
            param.type if implicit and param.is_const else next(given).type
            for param in callee.params
        )
        self.derive(inner, type_list, ctx.current_mesh, f"{self.label}/{self.labels[call]}")


def check_program(
    module: core.Module,
    function: core.Function,
    *,
    topology_level: str | None = None,
    budget: int = NODE_BUDGET,
    analyzers: tuple[object, ...] = (),
    dims: Mapping[str, int] | None = None,
) -> tuple[core.Module, core.Function]:
    """Prove one program holds together, then return the view analysis will read.

    The returned Module retains calls and has one typed derived body per definition.
    LoopRegion survives unchanged. The authored Module and Function are untouched.

    Each analysis states what it needs of the program through its own checker,
    and every one of them answers before any of them writes. A checker that
    needs the program walks it itself.
    """
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise Error(
            f"checking {function.name!r} needs a non-negative integer node budget, got {budget!r}"
        )
    if dims is not None:
        if not isinstance(dims, Mapping) or not dims:
            raise valuation.SpecializationError(
                f"specialising {function.name!r} needs a non-empty mapping of dimension names to extents, got {dims!r}"
            )
        for name, extent in dims.items():
            if not isinstance(name, str) or not name:
                raise valuation.SpecializationError(
                    f"specialising {function.name!r}: {name!r} is not a dimension name"
                )
            if isinstance(extent, bool) or not isinstance(extent, int):
                raise valuation.SpecializationError(
                    f"specialising {function.name!r}: dimension {name!r} takes an integer extent, got {extent!r}"
                )
    required_frame(module, function)
    bound_module, bound_root = valuation.instantiate(module, function, dims)
    initial = valuation.Context(bound_module, bound_root)
    residual = set(valuation.dim_vars_reached(bound_root, initial))
    owners = [bound_module]
    for owner in owners:
        residual.update(dim.vars_by_name(owner.topologies))
        owners.extend(child for slot in owner.children.values() for child in slot.elements)
    if residual:
        raise valuation.SpecializationError(
            f"{function.name!r} still states {sorted(residual)} as ranges after binding {sorted(dims or ())}; every dimension has to be given an extent"
        )
    signatures = {}

    def derive(frame, type_list, mesh, label):
        definition = frame.function
        identity = id(definition)
        signature = (type_list, mesh)
        if identity in signatures:
            previous, path = signatures[identity]
            if previous != signature:
                raise Error(
                    f"callee {definition.name!r} has incompatible signatures at {path!r} and "
                    f"{label!r}: {previous!r} versus {signature!r}"
                )
            return
        signatures[identity] = signature, label
        bindings = {
            id(param): (param, type_) for param, type_ in zip(definition.params, type_list, strict=True)
        }
        prepare_function(definition)
        definition.attach_metadata(ExecutionMeshMetadata(mesh))
        context = typeinfer.TypeInferContext(
            module=frame.module,
            function=frame.function,
            target=frame.target,
            topologies=frame.topologies,
            owner=definition,
            current_mesh=mesh,
            memo=bindings,
        )
        typeinfer.infer_function(definition, context)
        for value in (*definition.params, definition.body):
            verify.require_layouts(value)
        CheckCalls(derive, label, definition).visit(definition.body, context)

    root, module = bound_root, bound_module
    derive(
        initial,
        tuple(types.bind_annotation(p.type) for p in root.params),
        None,
        function.name,
    )

    frames = tuple(valuation.definitions(valuation.Context(module, root)))
    functions = tuple(frame.function for frame in frames)
    size = sum(len(core.collect_exprs(body.body)) for body in functions)
    if size > budget:
        raise Error(
            f"analysis program {function.name!r} has {size} derived body nodes, exceeding the node budget {budget}"
        )
    tgt = module.resolve_target()
    for topology in module.effective_topologies():
        try:
            tgt.validate_program_topology(topology)
        except ValueError as error:
            raise Error(
                f"program topology level {topology.name!r} with extent "
                f"{topology.size!r} is invalid: {error}"
            ) from None
    if topology_level is not None:
        try:
            module.resolve_topology(topology_level)
        except ValueError as error:
            raise Error(f"program topology level {topology_level!r} is invalid: {error}") from None

    for frame in frames:
        for call in core.collect_exprs(frame.function.body):
            if not isinstance(call, expr.Call) or (inner := frame.callee_context(call) if frame is not None else None) is None:
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
                    raise Error(
                        f"{frame.module.name!r} calls {inner.module.name!r}, which resolves "
                        f"a different {noun} {actual!r} against {expected!r}; {suffix}"
                    )
    check_analyses(module, root, analyzers, topology_level=topology_level)
    return module, root


def check_analyses(
    module: core.Module,
    function: core.Function,
    analyzers: tuple[object, ...],
    *,
    topology_level: str | None = None,
) -> None:
    """Check requested families against a built program before any writes."""
    functions = tuple(
        frame.function for frame in valuation.definitions(valuation.Context(module, function))
    )
    ctx = check_context(module, function, topology_level)
    for analyzer in analyzers:
        checker = analyzer.get_checker()
        if checker is None:
            continue
        checker.check_target_facts(ctx)
        for body in functions:
            checker.visit(body.body, replace(ctx, function=body))


__all__ = [
    "CheckContext",
    "PerformanceChecker",
    "check_analyses",
    "check_program",
    "program_dim_vars",
]
