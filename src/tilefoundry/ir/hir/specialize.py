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

from tilefoundry.ir.core import Call, Constant, Expr, Op, Tuple, Var, VerifyError
from tilefoundry.ir.core.module import Module, ModuleList
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.pattern import Pattern, RangePattern
from tilefoundry.ir.types import callable_type_for
from tilefoundry.ir.types.dim import is_dim_expr
from tilefoundry.ir.types.substitute import (
    dim_vars_by_name,
    substitute_dims,
    substitute_mesh_dims,
    substitute_shape_dim,
    substitute_topology_dims,
)
from tilefoundry.ir.types.tensor_type import TensorType
from tilefoundry.ir.visitor import ExprCloner, ExprVisitor
from tilefoundry.visitor_registry.contexts import ValuationContext

from .function import Function


def canonical_specialization_signature(
    specializations: tuple[Pattern, ...],
) -> str:
    """Deterministic identity string for a Function's specialization tuple."""
    parts: list[str] = []
    for pat in specializations:
        if isinstance(pat, RangePattern):
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


def _record_provenance(derived: Function, origin: Function, dims: Mapping[str, int] | None) -> None:
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
        if not isinstance(pattern, RangePattern):
            continue
        if pattern.dim_var not in dims:
            raise SpecializationError(
                f"{fn.name!r} selects a variant on {pattern.dim_var!r}, which "
                f"was not given a size; state it to choose an implementation"
            )
        if not pattern.lo <= dims[pattern.dim_var] <= pattern.hi:
            return False
    return True


def _coverage(fn: Function) -> str:
    return "; ".join(
        ", ".join(
            f"{pattern.dim_var} in [{pattern.lo}, {pattern.hi}]"
            for pattern in variant.specializations
            if isinstance(pattern, RangePattern)
        )
        or "everything"
        for variant in fn.variants
    )


class InstantiateDimensionCloner(ExprCloner):
    """Bind dimensions without inference, preserving definition and SSA sharing."""

    def __init__(self, dims):
        super().__init__()
        self.dims = {} if dims is None else dict(dims)
        self.functions = {}
        self.modules = {}

    def function(self, fn: Function) -> Function:
        """Clone the chosen implementation and its same-module callees once."""
        if id(fn) in self.functions:
            return self.functions[id(fn)]
        chosen = variant_for(fn, self.dims)
        if id(chosen) in self.functions:
            self.functions[id(fn)] = self.functions[id(chosen)]
            return self.functions[id(chosen)]
        params = tuple(self.visit(param) for param in chosen.params)
        body = None if chosen.body is None else self.visit(chosen.body)
        result_type = substitute_dims(chosen.return_type, self.dims)
        derived = dataclasses.replace(
            chosen,
            name=fn.name,
            params=params,
            body=body,
            return_type=result_type,
            type=callable_type_for(params, result_type),
            variants=(),
            converters=(),
        )
        _record_provenance(derived, chosen, self.dims or None)
        self.functions[id(fn)] = self.functions[id(chosen)] = derived
        return derived

    def module(self, node):
        """Rebuild children first and replace the functions reached by this clone."""
        if id(node) in self.modules:
            return self.modules[id(node)]
        children = {}
        for key, slot in node.children.items():
            elements = tuple(self.module(child) for child in slot.elements)
            children[key] = ModuleList(elements if slot.indexed else elements[0])
        topologies = (
            None
            if node.topologies is None
            else tuple(
                substitute_topology_dims(topology, self.dims) for topology in node.topologies
            )
        )
        functions = tuple(self.functions.get(id(fn), fn) for fn in node.functions)
        cloned = dataclasses.replace(
            node, children=children, functions=functions, topologies=topologies
        )
        self.modules[id(node)] = cloned
        return cloned

    def visit_Var(self, value, ctx=None):
        return dataclasses.replace(
            value,
            type=substitute_dims(value.type, self.dims),
            annotation=substitute_dims(value.annotation, self.dims),
        )

    def visit_Constant(self, value, ctx=None):
        return dataclasses.replace(value, type=substitute_dims(value.type, self.dims))

    def visit_ModuleList(self, slot, ctx=None):
        return slot

    def visit_Call(self, call, ctx=None):
        target = call.target
        if isinstance(target, Function):
            target = self.function(target)
        elif isinstance(target, Call):
            target = self.visit(target)
        else:
            target = _substitute_op_dims(target, self.dims)
        args = tuple(self.visit(arg) for arg in call.args)
        return dataclasses.replace(
            call, target=target, args=args, type=substitute_dims(call.type, self.dims)
        )

    def visit_MeshRegion(self, region, ctx=None):
        return dataclasses.replace(
            region,
            args=tuple(self.visit(arg) for arg in region.args),
            params=tuple(self.visit(param) for param in region.params),
            body=self.visit(region.body),
            mesh=substitute_mesh_dims(region.mesh, self.dims),
            type=substitute_dims(region.type, self.dims),
        )

    def visit_LoopRegion(self, region, ctx=None):
        def bound(value):
            if is_dim_expr(value):
                return substitute_shape_dim(value, self.dims)
            return self.visit(value) if isinstance(value, Expr) else value

        return dataclasses.replace(
            region,
            args=tuple(self.visit(arg) for arg in region.args),
            params=tuple(self.visit(param) for param in region.params),
            induction_var=self.visit(region.induction_var),
            body=self.visit(region.body),
            yield_values=tuple(self.visit(value) for value in region.yield_values),
            start=bound(region.start),
            extent=bound(region.extent),
            step=bound(region.step),
            type=substitute_dims(region.type, self.dims),
        )

    def default_visit(self, value, ctx=None):
        rebuilt = super().default_visit(value, ctx)
        return dataclasses.replace(rebuilt, type=substitute_dims(value.type, self.dims))


def instantiate(
    module: Module, function: Function, dims: Mapping[str, int] | None
) -> tuple[Module, Function]:
    """Bind *dims* wherever *function* reaches them, in a clone of *module*.

    A dimension may be declared by any reachable function, not only by the
    signature of *function*; a name that nothing reachable declares is refused.
    The authored Module and Function are untouched.
    """
    supplied = {} if dims is None else dict(dims)
    declared = {}
    collector = _DimVarCollector(declared, supplied)
    collector.visit_function(function, ValuationContext(module, function))
    if collector.cycles:
        raise VerifyError(f"recursive call path through {collector.cycles[0]!r}")
    if collector.missing_bodies:
        name, path = collector.missing_bodies[0]
        raise VerifyError(f"cannot analyze Function {name!r} without a body at {path!r}")
    unknown = sorted(set(supplied) - declared.keys())
    if unknown:
        if declared:
            reached = ", ".join(
                f"{name} ({collector.declared_by.get(name, function.name)})"
                for name in sorted(declared)
            )
            detail = f"the dimensions it reaches are {reached}"
        else:
            detail = "no function it reaches declares a dimension"
        raise SpecializationError(
            f"{function.name!r} has no dimension named {unknown!r}; {detail}"
        )
    for name, extent in supplied.items():
        substitute_shape_dim(declared[name], {name: extent})
    cloner = InstantiateDimensionCloner(dims)
    for reached, _frame in collector.functions:
        cloner.function(reached)
    root = cloner.function(function)
    return cloner.module(module), root


def _substitute_op_dims(target: object, dims: Mapping[str, int]) -> object:
    """Substitute bindings in an operation's shape-valued attributes."""
    if isinstance(target, Function) or not isinstance(target, Op):
        return target
    from tilefoundry.ir.types.layout import LayoutBase  # noqa: PLC0415
    from tilefoundry.ir.types.mesh import Mesh  # noqa: PLC0415
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


def residual_dims(fn: Function, frame=None) -> tuple[str, ...]:
    """Every dimension still stated as a range anywhere *fn* reaches.

    Signature, body, the shape-valued attributes its operations carry, the
    bounds of its loops, and the same again for every function it calls. A
    dimension can be introduced deep inside a callee -- a reshape that states a
    block length the caller never mentions -- and a scan that stopped at the
    signature would call such a function concrete while it still holds a range.
    """
    return tuple(dim_vars_reached(fn, frame))


def dim_vars_reached(fn: Function, frame=None) -> dict[str, object]:
    """The declarations behind `residual_dims`, by name.

    Same traversal, keeping the `DimVar` rather than only its name, for a caller
    that has to restate the bounds a dimension was declared with.
    """
    found: dict[str, object] = {}
    _DimVarCollector(found).visit_function(fn, frame)
    return found


class _DimVarCollector(ExprVisitor[None]):
    """Collect dimensions from Expr values plus Function/Op side fields."""

    def __init__(self, found, dims=None):
        super().__init__()
        self.found = found
        self.dims = dims
        self.declared_by: dict[str, str] = {}
        self.seen_functions = set()
        self.active_functions = set()
        self.functions = []
        self.cycles = []
        self.missing_bodies = []
        self.path = []

    def default_visit_leaf(self, expr, operands, ctx):
        return None

    def _record(self, found: dict[str, object]) -> None:
        """Note declarations, remembering which function stated each name first."""
        owner = self.path[-1] if self.path else None
        for name in found:
            if name not in self.found and owner is not None:
                self.declared_by[name] = owner
        self.found.update(found)

    def _collect_expr(self, expr: Expr, ctx=None) -> None:
        from tilefoundry.ir.types.substitute import _collect  # noqa: PLC0415

        current = (
            ctx.module.children[expr.key]
            if isinstance(expr, ModuleList) and ctx is not None
            else expr
        )
        staging: dict[str, object] = {}
        _collect(current.type, staging)
        self._record(staging)

    def _collect_bounds(self, expr: Expr) -> None:
        staging: dict[str, object] = {}
        for bound in ("extent", "step", "start"):
            _collect_entries(getattr(expr, bound, None), staging)
        self._record(staging)

    def _visit_children(self, expr: Expr, ctx=None) -> None:
        if isinstance(expr, Call) and isinstance(expr.target, Call):
            self.visit(expr.target, ctx)
        for child in (
            getattr(expr, "args", ())
            + (expr.elements if isinstance(expr, Tuple) else ())
            + getattr(expr, "yield_values", ())
            + getattr(expr, "params", ())
        ):
            self.visit(child, ctx)
        body = getattr(expr, "body", None)
        if body is not None:
            self.visit(body, ctx)

    def visit_function(self, fn: Function, frame=None) -> None:
        if id(fn) in self.active_functions:
            self.cycles.append("/".join((*self.path, fn.name)))
            return
        key = (id(fn), id(frame.module) if frame is not None else 0)
        if key in self.seen_functions:
            return
        self.seen_functions.add(key)
        self.active_functions.add(id(fn))
        self.functions.append((fn, frame))
        self.path.append(fn.name)
        previous_memo, self._memo = self._memo, {}
        try:
            if frame is not None:
                self._record(dim_vars_by_name(frame.topologies))
            for param in fn.params:
                self._record(dim_vars_by_name(param.type))
            self._record(dim_vars_by_name(fn.return_type))
            if self.dims is None:
                for variant in fn.variants:
                    self.visit_function(variant, frame)
                chosen = fn
            else:
                chosen = variant_for(fn, self.dims)
            if chosen.body is not None:
                self.visit(chosen.body, frame)
            elif not chosen.variants:
                self.missing_bodies.append((fn.name, "/".join(self.path)))
        finally:
            self._memo = previous_memo
            self.path.pop()
            self.active_functions.remove(id(fn))

    def visit_Call(self, expr: Call, frame=None) -> None:
        self._collect_expr(expr, frame)
        self._collect_bounds(expr)
        self._visit_children(expr, frame)
        target = expr.target
        inner = None if frame is None else frame._callee_context(expr)
        callee = inner.function if inner is not None else target
        if isinstance(callee, Function):
            self.visit_function(callee, inner)
        else:
            staging: dict[str, object] = {}
            for attribute in getattr(type(target), "params", lambda: ())():
                if attribute.kind == "attribute":
                    _collect_entries(getattr(target, attribute.name, None), staging)
            self._record(staging)

    def visit_Tuple(self, expr: Tuple, ctx=None) -> None:
        self._collect_expr(expr, ctx)
        self._visit_children(expr, ctx)

    def visit_LoopRegion(self, expr: LoopRegion, ctx=None) -> None:
        self._collect_expr(expr, ctx)
        self._collect_bounds(expr)
        self._visit_children(expr, ctx)

    def visit_Function(self, expr: Function, ctx=None) -> None:
        self.visit_function(expr, ctx)

    def visit_Var(self, expr: Var, ctx=None) -> None:
        self._collect_expr(expr, ctx)

    def visit_Constant(self, expr: Constant, ctx=None) -> None:
        self._collect_expr(expr, ctx)

    def visit_SymbolRef(self, expr: Expr, ctx=None) -> None:
        self._collect_expr(expr, ctx)

    def visit_ModuleList(self, expr, ctx=None) -> None:
        self._collect_expr(expr, ctx)

    def visit_ShapeOf(self, expr: Expr, ctx=None) -> None:
        self._collect_expr(expr, ctx)


def _collect_entries(value: object, found: dict[str, object]) -> None:
    from tilefoundry.ir.types.substitute import _collect  # noqa: PLC0415

    if isinstance(value, tuple):
        for entry in value:
            _collect(entry, found)
        return
    _collect(value, found)


__all__ = [
    "InstantiateDimensionCloner",
    "BOUND_DIMS",
    "PROVENANCE",
    "SpecializationError",
    "bound_dims_of",
    "canonical_specialization_signature",
    "dim_vars_reached",
    "origin_of",
    "residual_dims",
    "variant_for",
]
