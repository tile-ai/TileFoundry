"""The shared authored-program gate for analysis and scheduling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from tilefoundry.ir.core.module import Module, owning_module
from tilefoundry.ir.core.pattern import DimVarRangePat
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.specialize import (
    SpecializationError,
    _record_complete_bindings,
    dim_vars_reached,
    is_concrete,
    specialize_concretely,
)
from tilefoundry.ir.types.shape_helpers import static_dim_value
from tilefoundry.ir.types.substitute import (
    DimSubstitutionError,
    dim_vars_by_name,
    substitute_shape_dim,
    substitute_topology_dims,
)
from tilefoundry.visitor_registry.contexts import TypeInferContext

from .errors import AnalysisError
from .preflight import infer_authored_types, validate_call_context
from .walk import reachable_functions


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


def check_program(
    module: Module,
    function: Function,
    *,
    level: str | None = None,
) -> None:
    """Prove one authored program's shared invariants before an algorithm runs."""
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


__all__ = ["check_program"]
