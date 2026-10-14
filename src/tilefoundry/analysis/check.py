"""The shared authored-program gate for analysis.

An analysis reads inferred types and assumes the authored program holds
together. Both conditions are established once per public call rather than
per family, so no family can be the one that forgot.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from tilefoundry.ir.core import (
    BindingMetadata,
    Call,
    Expr,
    RangeMetadata,
    Var,
)
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.specialize import dim_vars_reached
from tilefoundry.ir.types import Topology, callable_type_for
from tilefoundry.ir.visitor import BindingSubstitutionCloner, collect_exprs
from tilefoundry.target import UnsupportedCapabilityError
from tilefoundry.target.facts import TopologyFacts
from tilefoundry.visitor_registry.contexts import CallFrame

from .errors import AnalysisError
from .facts import PerformanceServiceFacts
from .metadata import (
    ComputeCostMetadata,
    MemoryMetadata,
    PerformanceMetadata,
    PerformanceSummaryMetadata,
    RegionMemoryMetadata,
    RooflineMetadata,
)
from .program import Program, build_program

_INLINE_NODES = 10_000
_DERIVED_METADATA = {
    ComputeCostMetadata,
    MemoryMetadata,
    PerformanceMetadata,
    PerformanceSummaryMetadata,
    RangeMetadata,
    RegionMemoryMetadata,
    RooflineMetadata,
}
_ResourceKey = tuple[str, str]


@dataclass
class AnalysisCheckContext:
    """What every input check reads: the program, the machine, and the topology level.

    The context is bound to the derived Function the analyses will read, so a
    check answers about the same program they do. Nothing a check reads is a
    derived analysis record.
    """

    module: Module
    function: Function
    target: object
    topology_level: str | None

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
        return CallFrame(module, function)
    raise AnalysisError(
        f"function {function.name!r} is owned by no single node of module "
        f"{module.name!r}; supply its owning module directly"
    )


def _resource_parameters(program):
    """ConstTensor declarations at the retained program's constant call paths."""
    needed = {}

    def visit(body, path):
        for expr in collect_exprs(body.body):
            site = program.sites.get(id(expr))
            if site is None:
                continue
            inner = path
            if site.key is not None:
                inner = (*path, site.key)
                if site.index is not None:
                    if not isinstance(site.index, int):
                        raise AnalysisError(f"cannot inline variable ModuleList index for slot {site.key!r}")
                    inner = (*inner, str(site.index))
                for param in site.weights:
                    key = (".".join(inner), param.name)
                    previous = needed.get(key)
                    if previous is not None and previous.type != param.type:
                        raise AnalysisError(f"reachable ConstTensor {key!r} has conflicting types")
                    needed.setdefault(key, param)
            visit(site.body, inner)

    visit(program.root, ())
    return tuple(needed.items())


def _view_metadata(expr: Expr) -> tuple:
    """Authored metadata carried into a fresh analysis view."""
    return tuple(
        item
        for item in expr.metadata
        if type(item) not in _DERIVED_METADATA and type(item) is not BindingMetadata
    )


class _AnalysisViewCloner(BindingSubstitutionCloner):
    """Clone one inline call site after substituting its parameter bindings.

    ``BindingSubstitutionCloner`` preserves sharing within this call site by
    memoizing source expressions by identity. ``_Inliner`` creates a fresh
    instance for every call site, so two calls to the same Function still get
    independent cloned bodies and derived metadata.
    """

    def __init__(self, owner, function, path, active) -> None:
        super().__init__(metadata=_view_metadata)
        self.owner = owner
        self.function = function
        self.path = path
        self.active = active

    def visit_Call(self, expr: Call, ctx: Mapping[int, Expr]) -> Expr:
        site = self.owner.program.sites.get(id(expr))
        new_args = tuple(self.visit(arg, ctx) for arg in expr.args)
        if site is not None:
            prefix = self.path
            if site.key is not None:
                prefix = (*prefix, site.key)
                if site.index is not None:
                    prefix = (*prefix, str(site.index))
            supplied = iter(new_args)
            callee_env = {
                id(param): self.owner.resources[(".".join(prefix), param.name)]
                if site.key is not None and param.is_const else next(supplied)
                for param in site.body.params
            }
            return self.owner.function_body(site.body, callee_env, prefix, self.active)
        metadata = (*_view_metadata(expr), BindingMetadata(self.owner._binding()))
        new_target = self.visit(expr.target, ctx) if isinstance(expr.target, Call) else expr.target
        return replace(expr, target=new_target, args=new_args, metadata=metadata)

    def visit_MeshRegion(self, expr: MeshRegion, ctx: Mapping[int, Expr]) -> MeshRegion:
        """Inline through a HIR execution region while preserving its boundary."""
        params = tuple(replace(param, metadata=_view_metadata(param)) for param in expr.params)
        body_ctx = {**ctx, **{id(old): new for old, new in zip(expr.params, params)}}
        rebuilt = replace(
            expr,
            params=params,
            args=tuple(self.visit(arg, ctx) for arg in expr.args),
            body=self.visit(expr.body, body_ctx),
            metadata=_view_metadata(expr),
        )
        from tilefoundry.visitor_registry.verify import verify_region_isolated  # noqa: PLC0415

        verify_region_isolated(rebuilt)
        return rebuilt

    def default_visit(self, expr: Expr, ctx: Mapping[int, Expr]) -> Expr:
        raise AnalysisError(f"cannot inline unsupported HIR node {type(expr).__name__}")


class _Inliner:
    def __init__(
        self,
        program: Program,
        resources: Mapping[_ResourceKey, Var],
        used_names: set[str],
    ) -> None:
        self.program = program
        self.resources = resources
        self.used_names = used_names
        self.next_binding = 0
        self.function_call_counters: list[int] = []

    def _binding(self) -> str:
        while True:
            name = f"v{self.next_binding}"
            self.next_binding += 1
            if name not in self.used_names:
                self.used_names.add(name)
                return name

    def function_body(
        self,
        function: Function,
        env: Mapping[int, Expr],
        path: tuple[str, ...],
        active: frozenset[int],
    ) -> Expr:
        identity = id(function)
        if identity in active:
            raise AnalysisError(
                f"cannot inline recursive Function call path through {function.name!r}"
            )
        if function.body is None:
            raise AnalysisError(f"cannot inline Function {function.name!r} without a body")
        self.function_call_counters.append(0)
        try:
            return self.expr(
                function.body,
                env,
                function,
                path,
                active | {identity},
                    )
        finally:
            self.function_call_counters.pop()

    def expr(
        self,
        expr: Expr,
        env: Mapping[int, Expr],
        function: Function,
        path: tuple[str, ...],
        active: frozenset[int],
    ) -> Expr:
        return _AnalysisViewCloner(self, function, path, active).visit(expr, env)


def _inline_view(module: Module, function: Function, budget: int) -> Function:
    program = build_program(module, function)
    function = program.root
    declared_resources = _resource_parameters(program)
    params = tuple(replace(param, metadata=_view_metadata(param)) for param in function.params)
    resources: dict[_ResourceKey, Var] = {}
    appended: list[Var] = []
    for key, declaration in declared_resources:
        prefix, name = key
        qualified = f"{prefix}.{name}" if prefix else name
        promoted = replace(
            declaration,
            name=qualified,
            metadata=_view_metadata(declaration),
        )
        resources[key] = promoted
        appended.append(promoted)

    view_params = (*params, *appended)
    env = {id(old): new for old, new in zip(function.params, params)}
    inliner = _Inliner(program, resources, {param.name for param in view_params})
    body = inliner.function_body(
        function, env, (), frozenset()
    )
    size = len(collect_exprs(body))
    if size > budget:
        raise AnalysisError(
            f"inlining {function.name!r} produces {size} body nodes, exceeding "
            f"the node budget {budget}"
        )
    view = replace(
        function,
        type=callable_type_for(view_params, body.type),
        metadata=_view_metadata(function),
        params=view_params,
        body=body,
        return_type=body.type,
        variants=(),
        converters=(),
    )
    view._specialized_from = function
    return view


class InlineCloner:
    """Clone and inline one authored Function into an analysis view."""

    def __init__(self, module: Module, function: Function, budget: int = _INLINE_NODES) -> None:
        self.module = module
        self.function = function
        self.budget = budget

    def clone(self) -> Function:
        """Return a fresh single-function view with every call site inlined."""
        return _inline_view(self.module, self.function, self.budget)


def check_program(
    module: Module,
    function: Function,
    *,
    topology_level: str | None = None,
    budget: int = _INLINE_NODES,
    analyzers: tuple[object, ...] = (),
    dims: Mapping[str, int] | None = None,
) -> "Program":
    """Prove one program holds together, then return the view analysis will read.

    The Program retains calls and has one typed derived body per definition.
    LoopRegion survives unchanged. The authored Module and Function are untouched.

    Each analysis states what it needs of the program through its own checker,
    and every one of them answers before any of them writes. A checker that
    needs the program walks it itself.
    """
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise AnalysisError(
            f"inlining {function.name!r} needs a non-negative integer node budget, got {budget!r}"
        )
    program = build_program(module, function, dims)
    module = program.module
    size = sum(len(collect_exprs(body.body)) for body in program.bodies())
    if size > budget:
        raise AnalysisError(
            f"analysis program {function.name!r} has {size} derived body nodes, exceeding the node budget {budget}"
        )
    derived = program.root
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

    validate_call_context(module, program)
    ctx = analysis_check_context(module, derived, topology_level)
    for analyzer in analyzers:
        checker = analyzer.get_checker()
        if checker is None:
            continue
        checker.check_target_facts(ctx)
        for body in program.bodies():
            checker.visit(body.body, replace(ctx, function=body))
    return program


def validate_call_context(module: Module, program: Program) -> None:
    """Reject a reached call whose two ends do not share one execution context.

    Checked over what the selected query reaches, not at construction: an
    attached child no call reaches has no edge to validate. Inheritance is the
    canonical spelling, so a child declaring the caller's hierarchy explicitly
    passes and any other value is a different context, not a nested launch.
    """
    for body in program.bodies():
        frame = program.frames[id(body)]
        for expr in collect_exprs(body.body):
            site = program.sites.get(id(expr))
            if site is None:
                continue
            inner = site.frame
            if frame.target != inner.target:
                raise AnalysisError(
                    f"{frame.module.name!r} calls {inner.module.name!r}, which resolves "
                    f"a different target {inner.target!r} against {frame.target!r}; "
                    "one kernel invocation runs one target"
                )
            if frame.topologies != inner.topologies:
                raise AnalysisError(
                    f"{frame.module.name!r} calls {inner.module.name!r}, which "
                    f"resolves a different topology hierarchy {inner.topologies} against "
                    f"{frame.topologies}; one kernel invocation runs one hierarchy "
                    "-- declare none on the child and inherit"
                )
