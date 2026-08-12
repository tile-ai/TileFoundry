"""The shared authored-program gate for analysis and scheduling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from tilefoundry.ir.core import BindingMetadata, Call, Constant, Expr, Tuple, Var
from tilefoundry.ir.core.module import Module, owning_module, subtree
from tilefoundry.ir.core.pattern import DimVarRangePat
from tilefoundry.ir.hir._call_binding import binding_for
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.hir.specialize import (
    PROVENANCE,
    SpecializationError,
    _record_complete_bindings,
    dim_vars_reached,
    is_concrete,
    origin_of,
    specialize_concretely,
)
from tilefoundry.ir.types import Type, callable_type_for
from tilefoundry.ir.types.shape_helpers import static_dim_value
from tilefoundry.ir.types.shard import ComposedLayout, Layout, Mesh, ShardLayout, Topology
from tilefoundry.ir.types.shard.layout_algebra import (
    NotProjectable,
    apply,
    image,
    is_inverse_projectable,
    project,
    size,
)
from tilefoundry.ir.types.substitute import (
    DimSubstitutionError,
    dim_vars_by_name,
    substitute_shape_dim,
    substitute_topology_dims,
)
from tilefoundry.visitor_registry.contexts import CostContext, FunctionScope, TypeInferContext
from tilefoundry.visitor_registry.visitors import CostEvaluator

from .compute_cost import _call_cost_record, _is_structural_occurrence
from .errors import AnalysisError
from .facts import ThroughputFacts
from .metadata import (
    ComputeCostMetadata,
    MemoryMetadata,
    OccurrenceProvenance,
    RooflineMetadata,
    TimelineMetadata,
    TimelineSummaryMetadata,
)
from .preflight import infer_authored_types, validate_call_context
from .walk import describe, postorder, reachable_functions, tensor_types

_INLINE_NODES = 10_000
_DERIVED_METADATA = {
    ComputeCostMetadata,
    MemoryMetadata,
    RooflineMetadata,
    TimelineMetadata,
    TimelineSummaryMetadata,
}
_ResourceKey = tuple[str, str]
Placement = frozenset[int]


def _layout_shards(layout: object) -> tuple[ShardLayout, ...]:
    """Every ShardLayout nested in one tensor layout, outside first."""
    if isinstance(layout, ShardLayout):
        return (layout, *_layout_shards(layout.layout))
    if isinstance(layout, ComposedLayout):
        return (*_layout_shards(layout.inner), *_layout_shards(layout.outer))
    return ()


def _mesh_image(mesh: Mesh, selected: Topology) -> Placement:
    """The exact selected-topology positions named by one Mesh."""
    actual = tuple(topology.name for topology in mesh.topologies)
    if len(mesh.topologies) != 1:
        raise AnalysisError(
            f"one placement Mesh names topology levels {actual}; selected level "
            f"{selected.name!r} requires one level whose positions can be projected"
        )
    (mesh_topology,) = mesh.topologies
    if mesh_topology.name != selected.name:
        raise AnalysisError(
            f"selected topology level {selected.name!r}, but the result Mesh is "
            f"placed at level {mesh_topology.name!r}"
        )
    selected_size = static_dim_value(selected.size)
    mesh_size = static_dim_value(mesh_topology.size)
    if selected_size is None or selected_size <= 0:
        raise AnalysisError(
            f"selected topology level {selected.name!r} has unresolved or invalid "
            f"extent {selected.size!r}"
        )
    if mesh_size != selected_size:
        raise AnalysisError(
            f"the result Mesh declares {selected.name!r} extent "
            f"{mesh_topology.size!r}, but the selected topology has extent "
            f"{selected.size!r}"
        )

    layout = mesh.layout
    try:
        if isinstance(layout, Layout):
            count = size(layout)
            if not is_inverse_projectable(layout):
                raise NotProjectable("plain layout is not inverse-projectable")
            positions = tuple(apply(layout, coord) for coord in range(count))
        elif isinstance(layout, ComposedLayout):
            if not isinstance(layout.outer, Layout):
                raise NotProjectable("sliced layout has no resolved plain outer layout")
            count = size(layout.outer)
            positions = tuple(image(layout, coord) for coord in range(count))
            if any(project(layout, position) is None for position in positions):
                raise NotProjectable("sliced layout does not round-trip")
        else:  # pragma: no cover - Mesh's annotation excludes this shape
            raise NotProjectable(f"unsupported mesh layout {type(layout).__name__}")
    except (ArithmeticError, NotProjectable, TypeError, ValueError) as error:
        raise AnalysisError(
            f"the {selected.name!r} result Mesh is not projectable: {error}"
        ) from None

    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or any(
            isinstance(position, bool) or not isinstance(position, int) for position in positions
        )
    ):
        raise AnalysisError(
            f"the {selected.name!r} result Mesh needs a positive static position image"
        )
    if len(set(positions)) != len(positions):
        raise AnalysisError(
            f"the {selected.name!r} result Mesh maps multiple coordinates to the "
            f"same position: {positions}"
        )
    outside = sorted(position for position in positions if not 0 <= position < selected_size)
    if outside:
        raise AnalysisError(
            f"the {selected.name!r} result Mesh positions {outside} fall outside "
            f"the selected domain [0, {selected_size})"
        )
    placement = frozenset(positions)
    if isinstance(layout, Layout) and placement != frozenset(range(selected_size)):
        raise AnalysisError(
            f"an unsliced {selected.name!r} Mesh must describe the full selected "
            "domain; use a sliced Mesh so a strict subdomain retains its offset"
        )
    return placement


def _result_placement(type_: Type, selected: Topology) -> Placement:
    """The unique execution placement carried by every result tensor leaf."""
    leaves = tensor_types(type_)
    if not leaves:
        raise AnalysisError("the result has no tensor leaf that can carry placement")

    placements: list[tuple[Placement, Mesh]] = []
    missing: list[str] = []
    wrong_levels: set[str] = set()
    for leaf in leaves:
        shards = _layout_shards(leaf.layout)
        selected_shards = []
        leaf_levels: set[str] = set()
        for shard in shards:
            names = tuple(topology.name for topology in shard.mesh.topologies)
            if len(names) != 1:
                raise AnalysisError(
                    f"one result Mesh names topology levels {names}; selected level "
                    f"{selected.name!r} requires one projectable level"
                )
            if names[0] == selected.name:
                selected_shards.append(shard)
            else:
                leaf_levels.add(names[0])
        if not selected_shards:
            wrong_levels.update(leaf_levels)
            missing.append(type(leaf.layout).__name__ if leaf.layout is not None else "no layout")
            continue
        leaf_placements = [
            (_mesh_image(shard.mesh, selected), shard.mesh) for shard in selected_shards
        ]
        unique = {placement for placement, _mesh in leaf_placements}
        if len(unique) != 1:
            raise AnalysisError(
                f"one result tensor carries conflicting {selected.name!r} "
                f"placements {sorted(tuple(sorted(item)) for item in unique)}"
            )
        placements.append(leaf_placements[0])

    if missing:
        if wrong_levels:
            actual = ", ".join(repr(name) for name in sorted(wrong_levels))
            raise AnalysisError(
                f"selected topology level {selected.name!r}, but the result is "
                f"placed at level(s) {actual}"
            )
        carried = ", ".join(dict.fromkeys(missing))
        raise AnalysisError(
            f"has no {selected.name} placement; its result type carries {carried}, "
            "not a ShardLayout on the selected level. Reshard it onto a Mesh of "
            "the selected level, or analyse a family that does not require placement"
        )
    unique = {placement for placement, _mesh in placements}
    if len(unique) != 1:
        described = [f"{mesh!r} -> {sorted(placement)}" for placement, mesh in placements]
        raise AnalysisError(
            "tuple result leaves carry different execution placements: " + "; ".join(described)
        )
    return next(iter(unique))


def _timeline_placements(
    module: Module,
    function: Function,
    level: str,
    facts: ThroughputFacts,
) -> dict[int, Placement]:
    """Validate and prepare every primitive Call placement for timeline."""
    selected = module.resolve_topology(level)
    scope = FunctionScope(module, function)
    whole = CostEvaluator(CostContext(scope=scope))
    local = CostEvaluator(
        CostContext(
            scope=scope,
            level=level,
            topologies=module.effective_topologies(),
        )
    )
    result: dict[int, Placement] = {}
    for expr in postorder(function.body):
        if not isinstance(expr, Call) or isinstance(expr.target, Function):
            continue
        try:
            result[id(expr)] = _result_placement(expr.type, selected)
        except AnalysisError as error:
            cost = _call_cost_record(expr, whole, local)
            if _is_structural_occurrence(cost, facts):
                result[id(expr)] = frozenset()
                continue
            raise AnalysisError(f"timeline: {describe(expr)}: {error}") from None
    return result


def _program_dim_vars(module: Module, function: Function) -> dict[str, object]:
    """Dimension declarations reached through program values and execution geometry."""
    found: dict[str, object] = dict(dim_vars_reached(function))
    for owner in _reached_owners(module, function):
        found.update(dim_vars_by_name(owner.effective_topologies()))
    return found


def _resolve_program_geometry(
    module: Module,
    function: Function,
    dims: Mapping[str, int] | None,
    ctx: TypeInferContext | None = None,
) -> tuple[Module, Function]:
    """Resolve one call's Function, Mesh, and Module topology dimensions."""
    if dims is None:
        _require_concrete_function_geometry(function, error_type=SpecializationError)
        return module, function
    if not isinstance(dims, Mapping) or not dims:
        raise SpecializationError(
            f"specialising {function.name!r} needs a non-empty mapping of dimension "
            f"names to extents, got {dims!r}"
        )
    for name, extent in dims.items():
        if not isinstance(name, str) or not name:
            raise SpecializationError(
                f"specialising {function.name!r}: {name!r} is not a dimension name"
            )
        if isinstance(extent, bool) or not isinstance(extent, int):
            raise SpecializationError(
                f"specialising {function.name!r}: dimension {name!r} takes an "
                f"integer extent, got {extent!r}"
            )

    declared = _program_dim_vars(module, function)
    function_names = _function_dimension_names(function)
    geometry_only = set(declared) - function_names
    try:
        for name in geometry_only & dims.keys():
            variable = declared[name]
            if name in dims:
                substitute_shape_dim(variable, dims)
    except DimSubstitutionError as error:
        raise SpecializationError(str(error)) from None

    function_bindings = {name: extent for name, extent in dims.items() if name not in geometry_only}
    try:
        if function_bindings:
            function = specialize_concretely(function, function_bindings, ctx)
        function = _record_complete_bindings(function, dims)
        execution_module = _substitute_module_tree(module, dims)
    except DimSubstitutionError as error:
        raise SpecializationError(str(error)) from None
    _require_concrete_geometry(
        execution_module, function, error_type=SpecializationError
    )
    return execution_module, function


def _function_dimension_names(function: Function) -> set[str]:
    return set(dim_vars_reached(function)) | _pattern_dimension_names(function)


def _pattern_dimension_names(function: Function) -> set[str]:
    found: set[str] = set()
    seen: set[int] = set()

    def visit(fn: Function) -> None:
        if id(fn) in seen:
            return
        seen.add(id(fn))
        for pattern in fn.specializations:
            if isinstance(pattern, DimVarRangePat):
                found.add(pattern.dim_var)
        for variant in fn.variants:
            visit(variant)
        for callee in reachable_functions(fn)[1:]:
            visit(callee)

    visit(function)
    return found


def _reached_owners(module: Module, function: Function) -> tuple[Module, ...]:
    owners: list[Module] = [module]
    seen = {id(module)}
    for reached in reachable_functions(function):
        owner = owning_module(module, reached)
        if owner is not None and id(owner) not in seen:
            owners.append(owner)
            seen.add(id(owner))
    return tuple(owners)


def _substitute_module_tree(module: Module, dims: Mapping[str, int]) -> Module:
    effective = tuple(
        substitute_topology_dims(topology, dims)
        for topology in module.effective_topologies()
    )

    def declared(node: Module) -> Module:
        children = tuple(declared(child) for child in node.modules)
        topologies = (
            None
            if node.topologies is None
            else tuple(substitute_topology_dims(topology, dims) for topology in node.topologies)
        )
        if children == node.modules and topologies == node.topologies:
            return node
        return replace(node, modules=children, topologies=topologies)

    children = tuple(declared(child) for child in module.modules)
    if children == module.modules and effective == module.effective_topologies():
        return module
    try:
        target = module.resolve_target()
    except ValueError:
        target = module.target
    return replace(module, modules=children, target=target, topologies=effective)


def _require_concrete_geometry(
    module: Module,
    function: Function,
    *,
    error_type: type[ValueError],
) -> None:
    for owner in _reached_owners(module, function):
        for topology in owner.effective_topologies():
            if static_dim_value(topology.size) is None:
                raise error_type(
                    f"program topology level {topology.name!r} still states "
                    f"symbolic extent {topology.size!r}; bind every dimension "
                    "before analysis or scheduling"
                )
    _require_concrete_function_geometry(function, error_type=error_type)


def _require_concrete_function_geometry(
    function: Function,
    *,
    error_type: type[ValueError],
) -> None:
    if not is_concrete(function):
        raise error_type(
            f"{function.name!r} still states symbolic dimensions in its reachable "
            "Function or Mesh geometry; bind every dimension before analysis or scheduling"
        )


def _authored_call(function: Function, call: Call) -> Call:
    """Follow Function rebuilds to the authored Call at the same SSA position."""
    current_function = function
    current_call = call
    seen: set[int] = set()
    while id(current_function) not in seen:
        seen.add(id(current_function))
        origin = origin_of(current_function)
        if origin is None:
            return current_call
        current_calls = tuple(
            expr for expr in postorder(current_function.body) if isinstance(expr, Call)
        )
        origin_calls = tuple(
            expr for expr in postorder(origin.body) if isinstance(expr, Call)
        )
        if len(current_calls) != len(origin_calls):
            raise AnalysisError(
                f"cannot trace Call provenance through rebuilt Function "
                f"{current_function.name!r}: body shape changed"
            )
        try:
            index = next(
                index for index, candidate in enumerate(current_calls) if candidate is current_call
            )
        except StopIteration:
            raise AnalysisError(
                f"cannot trace a Call outside rebuilt Function {current_function.name!r}"
            ) from None
        current_function = origin
        current_call = origin_calls[index]
    raise AnalysisError(f"cyclic Function provenance on {current_function.name!r}")


def _module_paths(module: Module) -> dict[int, str]:
    paths = {id(module): ""}

    def visit(owner: Module, prefix: str) -> None:
        for child in owner.modules:
            path = f"{prefix}.{child.name}" if prefix else child.name
            paths[id(child)] = path
            visit(child, path)

    visit(module, "")
    return paths


def _call_reading(module: Module, caller: Function, call: Call) -> Module | None:
    binding = binding_for(
        call.target,
        call,
        TypeInferContext(scope=FunctionScope(module, caller)),
    )
    if not binding.from_reading:
        return None
    owner = owning_module(module, call.target)
    if owner is None:
        raise AnalysisError(
            f"Function call {call.target.name!r} has no unique owner in Module "
            f"{module.name!r}"
        )
    return owner


def _resource_parameters(
    module: Module, function: Function
) -> tuple[tuple[_ResourceKey, Var], ...]:
    """ConstTensor declarations needed by reached Module readings."""
    paths = _module_paths(module)
    needed: dict[_ResourceKey, Var] = {}
    for caller in reachable_functions(function):
        for expr in postorder(caller.body):
            if not isinstance(expr, Call) or not isinstance(expr.target, Function):
                continue
            owner = _call_reading(module, caller, expr)
            if owner is None:
                continue
            for param in expr.target.params:
                if not param.is_const:
                    continue
                key = (paths[id(owner)], param.name)
                previous = needed.get(key)
                if previous is not None and previous.type != param.type:
                    raise AnalysisError(
                        f"reachable ConstTensor {param.name!r} in Module "
                        f"{owner._owner_path()!r} has conflicting types "
                        f"{previous.type!r} and {param.type!r}"
                    )
                needed.setdefault(key, param)

    ordered: list[tuple[_ResourceKey, Var]] = []
    emitted: set[_ResourceKey] = set()
    for owner in subtree(module):
        for owned in owner.functions:
            if not isinstance(owned, Function):
                continue
            for param in owned.params:
                if not param.is_const:
                    continue
                key = (paths[id(owner)], param.name)
                declaration = needed.get(key)
                if declaration is not None and key not in emitted:
                    ordered.append((key, declaration))
                    emitted.add(key)
    return tuple(ordered)


def _view_metadata(expr: Expr) -> tuple:
    """Authored metadata carried into a fresh analysis view."""
    return tuple(
        item
        for item in expr.metadata
        if type(item) not in _DERIVED_METADATA
        and type(item) not in {BindingMetadata, OccurrenceProvenance}
    )


class _Inliner:
    def __init__(
        self,
        module: Module,
        resources: Mapping[_ResourceKey, Var],
        module_paths: Mapping[int, str],
        used_names: set[str],
    ) -> None:
        self.module = module
        self.resources = resources
        self.module_paths = module_paths
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
                {},
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
        memo: dict[int, Expr],
        function: Function,
        path: tuple[str, ...],
        active: frozenset[int],
    ) -> Expr:
        bound = env.get(id(expr))
        if bound is not None:
            return bound
        cached = memo.get(id(expr))
        if cached is not None:
            return cached

        match expr:
            case Var():
                rebuilt = replace(expr, metadata=_view_metadata(expr))
            case Constant():
                rebuilt = replace(expr, metadata=_view_metadata(expr))
            case Tuple(elements=elements):
                rebuilt = replace(
                    expr,
                    elements=tuple(
                        self.expr(item, env, memo, function, path, active)
                        for item in elements
                    ),
                    metadata=_view_metadata(expr),
                )
            case GridRegionExpr():
                rebuilt = self._grid(expr, env, memo, function, path, active)
            case Call(target=target, args=args) if isinstance(target, Function):
                call_index = self.function_call_counters[-1]
                self.function_call_counters[-1] += 1
                new_args = tuple(
                    self.expr(arg, env, memo, function, path, active) for arg in args
                )
                reading = _call_reading(self.module, function, expr)
                supplied = iter(new_args)
                callee_env: dict[int, Expr] = {}
                for param in target.params:
                    if reading is not None and param.is_const:
                        callee_env[id(param)] = self.resources[
                            (self.module_paths[id(reading)], param.name)
                        ]
                    else:
                        callee_env[id(param)] = next(supplied)
                rebuilt = self.function_body(
                    target,
                    callee_env,
                    (*path, target.name, str(call_index)),
                    active,
                )
            case Call(args=args):
                new_args = tuple(
                    self.expr(arg, env, memo, function, path, active) for arg in args
                )
                metadata = (
                    *_view_metadata(expr),
                    BindingMetadata(self._binding()),
                    OccurrenceProvenance(
                        source_call=id(_authored_call(function, expr)), call_path=path
                    ),
                )
                rebuilt = replace(expr, args=new_args, metadata=metadata)
            case _:
                raise AnalysisError(
                    f"cannot inline unsupported HIR node {type(expr).__name__}"
                )
        memo[id(expr)] = rebuilt
        return rebuilt

    def _grid(
        self,
        grid: GridRegionExpr,
        env: Mapping[int, Expr],
        memo: dict[int, Expr],
        function: Function,
        path: tuple[str, ...],
        active: frozenset[int],
    ) -> GridRegionExpr:
        init_args = tuple(
            self.expr(item, env, memo, function, path, active)
            for item in grid.init_args
        )
        induction_var = replace(
            grid.induction_var, metadata=_view_metadata(grid.induction_var)
        )
        carried_args = tuple(
            replace(item, metadata=_view_metadata(item)) for item in grid.carried_args
        )
        inner_env = dict(env)
        inner_env[id(grid.induction_var)] = induction_var
        inner_env.update(
            (id(old), new) for old, new in zip(grid.carried_args, carried_args)
        )
        body = self.expr(grid.body, inner_env, memo, function, path, active)
        yield_values = tuple(
            self.expr(item, inner_env, memo, function, path, active)
            for item in grid.yield_values
        )
        return replace(
            grid,
            induction_var=induction_var,
            carried_args=carried_args,
            init_args=init_args,
            body=body,
            yield_values=yield_values,
            metadata=_view_metadata(grid),
        )


def _inline_view(module: Module, function: Function, budget: int) -> Function:
    declared_resources = _resource_parameters(module, function)
    paths = _module_paths(module)
    params = tuple(
        replace(param, metadata=_view_metadata(param)) for param in function.params
    )
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
    inliner = _Inliner(
        module, resources, paths, {param.name for param in view_params}
    )
    body = inliner.function_body(function, env, (function.name,), frozenset())
    size = len(postorder(body))
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
    object.__setattr__(view, PROVENANCE, function)
    return view


def check_program(
    module: Module,
    function: Function,
    *,
    level: str | None = None,
    budget: int = _INLINE_NODES,
) -> Function:
    """Prove one program holds together, then return the view analysis will read.

    The returned Function has no Function-call wrapper. GridRegionExpr survives
    unchanged: a loop stays a loop. The authored Module and Function are untouched.
    """
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise AnalysisError(
            f"inlining {function.name!r} needs a non-negative integer node budget, "
            f"got {budget!r}"
        )
    _require_concrete_geometry(module, function, error_type=AnalysisError)
    target = module.resolve_target()
    for topology in module.effective_topologies():
        try:
            target.validate_program_topology(topology)
        except ValueError as error:
            raise AnalysisError(
                f"program topology level {topology.name!r} with extent "
                f"{topology.size!r} is invalid: {error}"
            ) from None
    if level is not None:
        try:
            module.resolve_topology(level)
        except ValueError as error:
            raise AnalysisError(f"program topology level {level!r} is invalid: {error}") from None

    functions = reachable_functions(function)
    infer_authored_types(functions, module)
    validate_call_context(module, functions)
    return _inline_view(module, function, budget)


__all__ = ["check_program"]
