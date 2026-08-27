"""Choose a specialization variant and bind its symbolic extents.

Derived functions record their origin and sorted bindings because signatures
need not expose every bound dimension. Display labels remain outside equality,
hashing, and canonical printing. Dispatch and shape binding stay separate so
their failures remain distinguishable.

See [hir §2](docs/spec/hir.md#2-function-specialization-api).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

from tilefoundry.ir.core import Call, Constant, Expr, Op, Tuple, Var
from tilefoundry.ir.core.pattern import DimVarRangePat, Pattern
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.types.dim import is_dim_expr
from tilefoundry.ir.types.substitute import (
    dim_vars_by_name,
    has_symbolic_dims,
    substitute_dims,
    substitute_shape_dim,
)
from tilefoundry.ir.types.tensor_type import TensorType, Type
from tilefoundry.ir.visitor import ExprCloner, ExprVisitor, ExprWalker
from tilefoundry.visitor_registry.contexts import TypeInferContext
from tilefoundry.visitor_registry.visitors import TypeInferVisitor

from .function import Function


def canonical_specialization_signature(
    specializations: tuple[Pattern, ...],
) -> str:
    """Deterministic identity string for a Function's specialization tuple."""
    parts: list[str] = []
    for pat in specializations:
        if isinstance(pat, DimVarRangePat):
            parts.append(f"{pat.dim_var}${pat.lo}_{pat.hi}")
        else:
            parts.append(repr(pat))
    return ";".join(parts)


class SpecializationError(ValueError):
    """A function cannot be specialised as asked."""


PROVENANCE = "_specialized_from"

BOUND_DIMS = "_specialized_dims"

DISPLAY_NAME = "_display_name"


def display_name(fn: Function) -> str | None:
    """This variant's label, or ``None`` where its author gave none."""
    return getattr(fn, DISPLAY_NAME, None)


def _record_provenance(
    derived: Function, origin: Function, dims: Mapping[str, int] | None
) -> None:
    """Note that *derived* is *origin*, at *dims* when a size was chosen.

    These fields are declared on Function with ``compare=False`` because they
    do not participate in equality, so two functions specialised from different
    origins are still equal when they are the same program.

    Extents are stored sorted by name, so one binding set has one representation.
    A rebuild that chose none records none rather than an empty set, which would
    compare equal to another such rebuild's.
    """
    derived._specialized_from = origin
    if dims is not None:
        derived._specialized_dims = tuple(sorted(dims.items()))


def _record_complete_bindings(
    function: Function, dims: Mapping[str, int]
) -> Function:
    """Record a public call's complete program bindings on a derived Function."""
    if bound_dims_of(function) is None:
        derived = dataclasses.replace(function)
        _record_provenance(derived, function, dims)
        return derived
    function._specialized_dims = tuple(sorted(dims.items()))
    return function


def origin_of(function: object) -> Function | None:
    """The function *function* was rebuilt from, if it was."""
    return getattr(function, PROVENANCE, None)


def bound_dims_of(function: object) -> tuple[tuple[str, int], ...] | None:
    """The extents *function* was rebuilt at, if any were chosen, sorted by name."""
    return getattr(function, BOUND_DIMS, None)


def variant_for(fn: Function, dims: Mapping[str, int]) -> Function:
    """The one implementation of *fn* that covers *dims*.

    A function with no variants is its own implementation. Otherwise exactly
    one variant must claim the stated extents: none means the source never
    covered this size, and more than one means the source contradicts itself.
    Neither is something to resolve by picking an order.
    """
    if not fn.variants:
        return fn

    matching = [variant for variant in fn.variants if _covers(fn, variant, dims)]
    if len(matching) == 1:
        return matching[0]
    stated = ", ".join(f"{name}={value}" for name, value in sorted(dims.items()))
    if not matching:
        raise SpecializationError(
            f"{fn.name!r} declares no variant covering {stated or 'anything'}; "
            f"its variants cover {_coverage(fn)}"
        )
    raise SpecializationError(
        f"{fn.name!r} has {len(matching)} variants covering {stated}; "
        f"they cover {_coverage(fn)} and a size may belong to only one"
    )


def _covers(fn: Function, variant: Function, dims: Mapping[str, int]) -> bool:
    """Whether *variant*'s every stated range admits the chosen extents.

    A pattern the caller said nothing about is refused rather than skipped: a
    variant is selected by the dimensions it names, so an unstated one means
    the caller does not yet know which implementation they are asking for.
    """
    for pattern in variant.specializations:
        if not isinstance(pattern, DimVarRangePat):
            continue
        if pattern.dim_var not in dims:
            raise SpecializationError(
                f"{fn.name!r} selects a variant on {pattern.dim_var!r}, which "
                f"was not given a size; state it to choose an implementation"
            )
        if not pattern.lo <= dims[pattern.dim_var] < pattern.hi:
            return False
    return True


def _coverage(fn: Function) -> str:
    return "; ".join(
        ", ".join(
            f"{pattern.dim_var} in [{pattern.lo}, {pattern.hi})"
            for pattern in variant.specializations
            if isinstance(pattern, DimVarRangePat)
        )
        or "everything"
        for variant in fn.variants
    )


def specialize_function(
    fn: Function,
    dims: Mapping[str, int],
    *,
    ctx: TypeInferContext | None = None,
) -> Function:
    """*fn* at the stated extents: one implementation, its ranges resolved.

    The dimensions are checked against what the function actually has before
    anything is rebuilt. A name nothing uses is a caller who believes they
    specialised something, and quietly substituting nothing would let them
    carry on believing it.
    """
    if not dims:
        raise SpecializationError(f"specialising {fn.name!r} needs at least one dimension to bind")
    chosen = variant_for(fn, dims)
    if chosen.body is None:
        raise SpecializationError(
            f"{fn.name!r} resolved to a variant with no body; a dispatch "
            "prototype states which implementation to use, not what it does"
        )

    present = set(residual_dims(chosen))
    for pattern in chosen.specializations:
        if isinstance(pattern, DimVarRangePat):
            present.add(pattern.dim_var)
    unknown = sorted(set(dims) - present)
    if unknown:
        raise SpecializationError(
            f"{fn.name!r} has no dimension named {unknown!r}; it states {sorted(present)}"
        )

    if not set(dims) & set(residual_dims(chosen)):
        return chosen
    bound = tuple(substitute_dims(param.type, dims) for param in chosen.params)
    return instantiate_dimensions(
        chosen,
        bound,
        ctx if ctx is not None else TypeInferContext(),
        dims,
    )


class DimensionInstantiator(ExprCloner):
    """Clone a Function body with symbolic dimensions replaced by values.

    DAG memoization and identity pinning come from ``ExprCloner``. The visit
    context carries the substitutions and shared type-inference visitor for
    this one instantiation.
    """

    def visit_Var(self, var: Var, ctx: InstantiateContext) -> Expr:
        return ctx.subst.get(id(var), var)

    def visit_Constant(self, const: Constant, ctx: InstantiateContext) -> Expr:
        return const

    def visit_Call(self, call: Call, ctx: InstantiateContext) -> Expr:
        new_args = tuple(self.visit(arg, ctx) for arg in call.args)
        new_target = call.target
        if isinstance(new_target, Function):
            new_target = _specialize_callee(
                new_target, ctx.dims, ctx.type_ctx, call
            )
        new_target = _substitute_op_dims(new_target, ctx.dims)
        new_metadata = _substitute_authored_dims(call.metadata, ctx.dims)
        if (
            all(new is old for new, old in zip(new_args, call.args))
            and new_target is call.target
            and new_metadata is call.metadata
        ):
            return call
        rebuilt = dataclasses.replace(
            call,
            args=new_args,
            target=new_target,
            metadata=new_metadata,
        )
        return self._retyped(rebuilt, ctx)

    def visit_GridRegionExpr(
        self, grid: GridRegionExpr, ctx: InstantiateContext
    ) -> Expr:
        """Rebuild loop bindings and shape fields excluded by generic cloning."""
        new_inits = tuple(self.visit(arg, ctx) for arg in grid.init_args)
        new_phis = tuple(
            old_phi
            if new_init.type == old_phi.type
            else Var(type=new_init.type, name=old_phi.name)
            for old_phi, new_init in zip(grid.carried_args, new_inits)
        )
        for old_phi, new_phi in zip(grid.carried_args, new_phis):
            if new_phi is not old_phi:
                ctx.subst[id(old_phi)] = new_phi
        new_body = self.visit(grid.body, ctx)
        new_yields = tuple(self.visit(value, ctx) for value in grid.yield_values)
        bounds = (grid.extent, grid.step, grid.start)
        new_bounds = tuple(substitute_shape_dim(bound, ctx.dims) for bound in bounds)
        if (
            all(new is old for new, old in zip(new_inits, grid.init_args))
            and all(new is old for new, old in zip(new_phis, grid.carried_args))
            and new_body is grid.body
            and all(new is old for new, old in zip(new_yields, grid.yield_values))
            and new_bounds == bounds
        ):
            return grid
        rebuilt = dataclasses.replace(
            grid,
            carried_args=new_phis,
            init_args=new_inits,
            body=new_body,
            yield_values=new_yields,
            extent=new_bounds[0],
            step=new_bounds[1],
            start=new_bounds[2],
        )
        return self._retyped(rebuilt, ctx)

    def default_visit(self, expr: Expr, ctx: InstantiateContext) -> Expr:
        rebuilt = super().default_visit(expr, ctx)
        return rebuilt if rebuilt is expr else self._retyped(rebuilt, ctx)

    def _retyped(self, rebuilt: Expr, ctx: InstantiateContext) -> Expr:
        return dataclasses.replace(
            rebuilt, type=ctx.type_visitor.visit(rebuilt, ctx.type_ctx)
        )


@dataclasses.dataclass
class InstantiateContext:
    """Mutable state shared by one dimension-instantiation traversal."""

    subst: dict[int, Var]
    dims: Mapping[str, int]
    type_ctx: TypeInferContext
    type_visitor: TypeInferVisitor


def instantiate_dimensions(
    chosen: Function,
    bound_param_types: tuple[Type, ...],
    ctx: TypeInferContext,
    dims: Mapping[str, int],
) -> Function:
    """Rebuild *chosen* at concrete dimension bindings.

    The cloner memo preserves SSA DAG identity while one shared type visitor
    retypes every rebuilt node.
    """
    new_params = tuple(
        Var(type=type_, name=param.name, is_const=param.is_const)
        for type_, param in zip(bound_param_types, chosen.params)
    )
    scope = None if ctx.scope is None else dataclasses.replace(ctx.scope, function=chosen)
    body_ctx = dataclasses.replace(ctx, scope=scope)
    instantiate_ctx = InstantiateContext(
        subst={id(old): new for old, new in zip(chosen.params, new_params)},
        dims=dims,
        type_ctx=body_ctx,
        type_visitor=TypeInferVisitor(),
    )
    new_body = DimensionInstantiator().visit(chosen.body, instantiate_ctx)
    derived = Function.build(
        name=chosen.name,
        params=new_params,
        body=new_body,
        return_type=new_body.type,
        specializations=chosen.specializations,
    )
    _record_provenance(derived, chosen, dims)
    return derived


def _specialize_callee(
    callee: Function,
    dims: Mapping[str, int],
    ctx: TypeInferContext,
    call: Call,
) -> Function:
    """Rebuild a nested callee at the dimensions its caller was given."""
    if callee.variants:
        raise ValueError(
            f"specialising through {call and callee.name!r}: the callee "
            "dispatches on its own variants, which this rebuild does not choose"
        )
    if callee.body is None:
        return callee
    bound = tuple(substitute_dims(param.type, dims) for param in callee.params)
    if all(new is param.type for new, param in zip(bound, callee.params)):
        return callee
    return instantiate_dimensions(callee, bound, ctx, dims)


def _substitute_authored_dims(
    metadata: tuple, dims: Mapping[str, int]
) -> tuple:
    """Substitute dimension bindings in authored execution-domain metadata."""
    if not metadata:
        return metadata
    from tilefoundry.ir.core.metadata import ExecutionDomainMetadata  # noqa: PLC0415
    from tilefoundry.ir.types.substitute import substitute_mesh_dims  # noqa: PLC0415

    rebuilt = tuple(
        dataclasses.replace(
            item,
            scopes=tuple(substitute_mesh_dims(mesh, dims) for mesh in item.scopes),
        )
        if isinstance(item, ExecutionDomainMetadata)
        else item
        for item in metadata
    )
    if all(new is old for new, old in zip(rebuilt, metadata)):
        return metadata
    return rebuilt


def _substitute_op_dims(target: object, dims: Mapping[str, int]) -> object:
    """Substitute bindings in an operation's shape-valued attributes."""
    if isinstance(target, Function) or not isinstance(target, Op):
        return target
    from tilefoundry.ir.types.shard.layout import LayoutBase  # noqa: PLC0415
    from tilefoundry.ir.types.shard.mesh import Mesh  # noqa: PLC0415
    from tilefoundry.ir.types.substitute import (  # noqa: PLC0415
        substitute_layout_dims,
        substitute_mesh_dims,
    )

    changed: dict[str, object] = {}
    for param in type(target).params():
        if param.kind != "attribute":
            continue
        value = getattr(target, param.name, None)
        if isinstance(value, TensorType):
            rebuilt = substitute_dims(value, dims)
        elif isinstance(value, LayoutBase):
            rebuilt = substitute_layout_dims(value, dims)
        elif isinstance(value, Mesh):
            rebuilt = substitute_mesh_dims(value, dims)
        elif is_dim_expr(value):
            rebuilt = substitute_shape_dim(value, dims)
        elif isinstance(value, tuple) and value and all(is_dim_expr(item) for item in value):
            rebuilt = tuple(substitute_shape_dim(item, dims) for item in value)
        else:
            continue
        if rebuilt != value:
            changed[param.name] = rebuilt
    if not changed:
        return target
    attributes = {
        param.name: getattr(target, param.name)
        for param in type(target).params()
        if param.kind == "attribute" and hasattr(target, param.name)
    }
    attributes.update(changed)
    return type(target)(**attributes)


def specialize_concretely(
    fn: Function, dims: Mapping[str, int], ctx: TypeInferContext | None = None
) -> Function:
    """*fn* at the stated extents, with nothing left as a range.

    The stricter half of `specialize_function`, for callers that go on to run
    something over the result. Partial binding is useful when the choices are
    still being made one at a time; it is useless to anything that has to count
    elements, so a dimension left unbound is refused here rather than surfacing
    later as an extent that is not a number.
    """
    if not isinstance(dims, Mapping) or not dims:
        raise SpecializationError(
            f"specialising {fn.name!r} needs a non-empty mapping of dimension "
            f"names to extents, got {dims!r}"
        )
    for name, extent in dims.items():
        if not isinstance(name, str) or not name:
            raise SpecializationError(f"specialising {fn.name!r}: {name!r} is not a dimension name")
        if isinstance(extent, bool) or not isinstance(extent, int):
            raise SpecializationError(
                f"specialising {fn.name!r}: dimension {name!r} takes an integer "
                f"extent, got {extent!r}"
            )
    concrete = specialize_function(fn, dims, ctx=ctx)
    residual = residual_dims(concrete)
    if residual:
        raise SpecializationError(
            f"{fn.name!r} still states {list(residual)} as ranges after binding "
            f"{sorted(dims)}; every dimension has to be given an extent"
        )
    return concrete


def residual_dims(fn: Function) -> tuple[str, ...]:
    """Every dimension still stated as a range anywhere *fn* reaches.

    Signature, body, the shape-valued attributes its operations carry, the
    bounds of its loops, and the same again for every function it calls. A
    dimension can be introduced deep inside a callee -- a reshape that states a
    block length the caller never mentions -- and a scan that stopped at the
    signature would call such a function concrete while it still holds a range.
    """
    return tuple(dim_vars_reached(fn))


def dim_vars_reached(fn: Function) -> dict[str, object]:
    """The declarations behind `residual_dims`, by name.

    Same traversal, keeping the `DimVar` rather than only its name, for a caller
    that has to restate the bounds a dimension was declared with.
    """
    found: dict[str, object] = {}
    _DimVarCollector(found).visit_function(fn)
    return found


class _DimVarCollector(ExprWalker[None]):
    """Collect dimensions from Expr values plus Function/Op side fields."""

    def __init__(self, found: dict[str, object]) -> None:
        super().__init__()
        self.found = found
        self.seen_functions: set[int] = set()

    def _collect_expr(self, expr: Expr) -> None:
        from tilefoundry.ir.types.substitute import _collect  # noqa: PLC0415

        _collect(expr.type, self.found)

    def _collect_target(self, expr: Call) -> None:
        target = expr.target
        if isinstance(target, Function):
            self.visit_function(target)
            return
        for attribute in getattr(type(target), "params", lambda: ())():
            if attribute.kind == "attribute":
                _collect_entries(getattr(target, attribute.name, None), self.found)

    def _collect_bounds(self, expr: Expr) -> None:
        for bound in ("extent", "step", "start"):
            _collect_entries(getattr(expr, bound, None), self.found)

    def _visit_children(self, expr: Expr, ctx=None) -> None:
        for child in (
            getattr(expr, "args", ())
            + getattr(expr, "elements", ())
            + getattr(expr, "init_args", ())
            + getattr(expr, "yield_values", ())
            + getattr(expr, "carried_args", ())
        ):
            self.visit(child, ctx)
        body = getattr(expr, "body", None)
        if body is not None:
            self.visit(body, ctx)

    def visit_function(self, fn: Function, ctx=None) -> None:
        if id(fn) in self.seen_functions:
            return
        self.seen_functions.add(id(fn))
        for param in fn.params:
            self.found.update(dim_vars_by_name(param.type))
        self.found.update(dim_vars_by_name(fn.return_type))
        for variant in fn.variants:
            self.visit_function(variant, ctx)
        if fn.body is not None:
            self.visit(fn.body, ctx)

    def visit_Call(self, expr: Call, ctx=None) -> None:
        self._collect_expr(expr)
        self._collect_target(expr)
        self._collect_bounds(expr)
        self._visit_children(expr, ctx)

    def visit_Tuple(self, expr: Tuple, ctx=None) -> None:
        self._collect_expr(expr)
        self._visit_children(expr, ctx)

    def visit_GridRegionExpr(self, expr: GridRegionExpr, ctx=None) -> None:
        self._collect_expr(expr)
        self._collect_bounds(expr)
        self._visit_children(expr, ctx)

    def visit_Function(self, expr: Function, ctx=None) -> None:
        self.visit_function(expr, ctx)

    def visit_Var(self, expr: Var, ctx=None) -> None:
        self._collect_expr(expr)

    def visit_Constant(self, expr: Constant, ctx=None) -> None:
        self._collect_expr(expr)

    def visit_SymbolRef(self, expr: Expr, ctx=None) -> None:
        self._collect_expr(expr)

    def visit_ShapeOf(self, expr: Expr, ctx=None) -> None:
        self._collect_expr(expr)


def _collect_entries(value: object, found: dict[str, object]) -> None:
    from tilefoundry.ir.types.substitute import _collect  # noqa: PLC0415

    if isinstance(value, tuple):
        for entry in value:
            _collect(entry, found)
        return
    _collect(value, found)


def is_concrete(fn: Function) -> bool:
    """Whether *fn* states extents everywhere a size is required."""
    return not _SymbolicDimVisitor().visit_function(fn)


class _SymbolicDimVisitor(ExprVisitor[bool]):
    """Detect symbolic dimensions across Expr values and Function side fields."""

    def __init__(self) -> None:
        super().__init__()
        self.seen_functions: set[int] = set()

    def _expr_has_symbolic(self, expr: Expr) -> bool:
        if has_symbolic_dims(expr.type):
            return True
        for bound in ("extent", "step", "start"):
            if has_symbolic_dims(getattr(expr, bound, None)):
                return True
        return False

    def _target_has_symbolic(self, target, ctx=None) -> bool:
        if isinstance(target, Function):
            return self.visit_function(target, ctx)
        return any(
            attribute.kind == "attribute"
            and has_symbolic_dims(getattr(target, attribute.name, None))
            for attribute in getattr(type(target), "params", lambda: ())()
        )

    def _children_have_symbolic(self, expr: Expr, ctx=None) -> bool:
        children = (
            getattr(expr, "args", ())
            + getattr(expr, "elements", ())
            + getattr(expr, "init_args", ())
            + getattr(expr, "yield_values", ())
            + getattr(expr, "carried_args", ())
        )
        body = getattr(expr, "body", None)
        return any(self.visit(child, ctx) for child in children) or (
            body is not None and self.visit(body, ctx)
        )

    def visit_function(self, fn: Function, ctx=None) -> bool:
        if id(fn) in self.seen_functions:
            return False
        self.seen_functions.add(id(fn))
        if any(has_symbolic_dims(param.type) for param in fn.params):
            return True
        if has_symbolic_dims(fn.return_type):
            return True
        if any(self.visit_function(variant, ctx) for variant in fn.variants):
            return True
        return fn.body is not None and self.visit(fn.body, ctx)

    def visit_Call(self, expr: Call, ctx=None) -> bool:
        return (
            self._expr_has_symbolic(expr)
            or self._target_has_symbolic(expr.target, ctx)
            or self._children_have_symbolic(expr, ctx)
        )

    def visit_Tuple(self, expr: Tuple, ctx=None) -> bool:
        return self._expr_has_symbolic(expr) or self._children_have_symbolic(expr, ctx)

    def visit_GridRegionExpr(self, expr: GridRegionExpr, ctx=None) -> bool:
        return self._expr_has_symbolic(expr) or self._children_have_symbolic(expr, ctx)

    def visit_Function(self, expr: Function, ctx=None) -> bool:
        return self.visit_function(expr, ctx)

    def visit_Var(self, expr: Var, ctx=None) -> bool:
        return self._expr_has_symbolic(expr)

    def visit_Constant(self, expr: Constant, ctx=None) -> bool:
        return self._expr_has_symbolic(expr)

    def visit_SymbolRef(self, expr: Expr, ctx=None) -> bool:
        return self._expr_has_symbolic(expr)

    def visit_ShapeOf(self, expr: Expr, ctx=None) -> bool:
        return self._expr_has_symbolic(expr)


__all__ = [
    "BOUND_DIMS",
    "DimensionInstantiator",
    "InstantiateContext",
    "PROVENANCE",
    "SpecializationError",
    "bound_dims_of",
    "canonical_specialization_signature",
    "dim_vars_reached",
    "is_concrete",
    "instantiate_dimensions",
    "origin_of",
    "residual_dims",
    "specialize_concretely",
    "specialize_function",
    "variant_for",
]
