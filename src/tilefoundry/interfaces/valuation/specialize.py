"""Choose a specialization variant and bind its symbolic extents.

Derived functions record their origin and sorted bindings because signatures
need not expose every bound dimension. Display labels remain outside equality,
hashing, and canonical printing. Dispatch and shape binding stay separate so
their failures remain distinguishable.

See [hir §2](docs/spec/hir.md#2-function-specialization-api).
"""

from __future__ import annotations

import dataclasses
import typing
from collections.abc import Mapping

from tilefoundry.ir import core, dim, expr, pattern, types

from .context import Context


def canonical_specialization_signature(
    specializations: tuple[pattern.Pattern, ...],
) -> str:
    """Deterministic identity string for a Function's specialization tuple."""
    parts: list[str] = []
    for pat in specializations:
        if isinstance(pat, pattern.Range):
            parts.append(f"{pat.dim_var}${pat.lo}_{pat.hi}")
        else:
            parts.append(repr(pat))
    return ";".join(parts)


class SpecializationError(ValueError):
    """A function cannot be specialised as asked."""


PROVENANCE = "specialized_from"

BOUND_DIMS = "specialized_dims"

DISPLAY_NAME = "display_name"


def display_name(fn: core.Function) -> str | None:
    """This variant's label, or ``None`` where its author gave none."""
    return getattr(fn, DISPLAY_NAME, None)


def origin_of(function: object) -> core.Function | None:
    """The function *function* was rebuilt from, if it was."""
    return getattr(function, PROVENANCE, None)


def bound_dims_of(function: object) -> tuple[tuple[str, int], ...] | None:
    """The extents *function* was rebuilt at, if any were chosen, sorted by name."""
    return getattr(function, BOUND_DIMS, None)


def variant_for(fn: core.Function, dims: Mapping[str, int]) -> core.Function:
    """The one implementation of *fn* that covers *dims*.

    A function with no variants is its own implementation. Otherwise exactly
    one variant must claim the stated extents: none means the source never
    covered this size, and more than one means the source contradicts itself.
    Neither is something to resolve by picking an order.
    """
    if not fn.variants:
        return fn

    matching = [variant for variant in fn.variants if covers(fn, variant, dims)]
    if len(matching) == 1:
        return matching[0]
    stated = ", ".join(f"{name}={value}" for name, value in sorted(dims.items()))
    if not matching:
        raise SpecializationError(
            f"{fn.name!r} declares no variant covering {stated or 'anything'}; "
            f"its variants cover {coverage(fn)}"
        )
    raise SpecializationError(
        f"{fn.name!r} has {len(matching)} variants covering {stated}; "
        f"they cover {coverage(fn)} and a size may belong to only one"
    )


def covers(fn: core.Function, variant: core.Function, dims: Mapping[str, int]) -> bool:
    """Whether *variant*'s every stated range admits the chosen extents.

    A pattern the caller said nothing about is refused rather than skipped: a
    variant is selected by the dimensions it names, so an unstated one means
    the caller does not yet know which implementation they are asking for.
    """
    for declared in variant.specializations:
        if not isinstance(declared, pattern.Range):
            continue
        if declared.dim_var not in dims:
            raise SpecializationError(
                f"{fn.name!r} selects a variant on {declared.dim_var!r}, which "
                f"was not given a size; state it to choose an implementation"
            )
        if not declared.lo <= dims[declared.dim_var] <= declared.hi:
            return False
    return True


def coverage(fn: core.Function) -> str:
    return "; ".join(
        ", ".join(
            f"{declared.dim_var} in [{declared.lo}, {declared.hi}]"
            for declared in variant.specializations
            if isinstance(declared, pattern.Range)
        )
        or "everything"
        for variant in fn.variants
    )


class InstantiateDimensionCloner(core.ExprCloner):
    """Bind dimensions without inference, preserving definition and SSA sharing."""

    def __init__(self, dims):
        super().__init__()
        self.dims = {} if dims is None else dict(dims)
        self.functions = {}
        self.modules = {}

    def function(self, fn: core.Function) -> core.Function:
        """Clone the chosen implementation and its same-module callees once.

        The clone records its origin and the extents it was rebuilt at. Neither takes part in
        equality, so two functions specialised from different origins are equal when they are
        the same program. Extents are sorted by name, so one binding set has one representation,
        and a rebuild that chose none records none.
        """
        if id(fn) in self.functions:
            return self.functions[id(fn)]
        chosen = variant_for(fn, self.dims)
        if id(chosen) in self.functions:
            self.functions[id(fn)] = self.functions[id(chosen)]
            return self.functions[id(chosen)]
        params = tuple(self.visit(param) for param in chosen.params)
        body = None if chosen.body is None else self.visit(chosen.body)
        result_type = dim.substitute_dims(chosen.return_type, self.dims)
        provenance = {PROVENANCE: chosen}
        if self.dims:
            provenance[BOUND_DIMS] = tuple(sorted(self.dims.items()))
        derived = dataclasses.replace(
            chosen,
            name=fn.name,
            params=params,
            body=body,
            return_type=result_type,
            return_annotation=dim.substitute_dims(chosen.return_annotation, self.dims),
            type=types.callable_type_for(params, result_type),
            variants=(),
            converters=(),
            **provenance,
        )
        self.functions[id(fn)] = self.functions[id(chosen)] = derived
        return derived

    def module(self, node):
        """Rebuild children first and replace the functions reached by this clone."""
        if id(node) in self.modules:
            return self.modules[id(node)]
        children = {}
        for key, slot in node.children.items():
            elements = tuple(self.module(child) for child in slot.elements)
            children[key] = core.ModuleList(elements if slot.indexed else elements[0])
        topologies = (
            None
            if node.topologies is None
            else tuple(
                dim.substitute_topology_dims(topology, self.dims) for topology in node.topologies
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
            type=dim.substitute_dims(value.type, self.dims),
            annotation=dim.substitute_dims(value.annotation, self.dims),
        )

    def visit_Constant(self, value, ctx=None):
        return dataclasses.replace(value, type=dim.substitute_dims(value.type, self.dims))

    def visit_ModuleList(self, slot, ctx=None):
        return slot

    def visit_Call(self, call, ctx=None):
        target = call.target
        if isinstance(target, core.Function):
            target = self.function(target)
        elif isinstance(target, expr.Call):
            target = self.visit(target)
        else:
            target = substitute_op_dims(target, self.dims)
        args = tuple(self.visit(arg) for arg in call.args)
        return dataclasses.replace(
            call, target=target, args=args, type=dim.substitute_dims(call.type, self.dims)
        )

    def visit_MeshRegion(self, region, ctx=None):
        return dataclasses.replace(
            region,
            args=tuple(self.visit(arg) for arg in region.args),
            params=tuple(self.visit(param) for param in region.params),
            body=self.visit(region.body),
            mesh=dim.substitute_mesh_dims(region.mesh, self.dims),
            type=dim.substitute_dims(region.type, self.dims),
        )

    def bound(self, value):
        """One loop bound with the dimensions bound in it."""
        if dim.is_dimension(value):
            return dim.substitute_shape_entry(value, self.dims)
        return self.visit(value) if isinstance(value, expr.Expr) else value

    def visit_LoopRegion(self, region, ctx=None):
        return dataclasses.replace(
            region,
            args=tuple(self.visit(arg) for arg in region.args),
            params=tuple(self.visit(param) for param in region.params),
            induction_var=self.visit(region.induction_var),
            body=self.visit(region.body),
            yield_values=tuple(self.visit(value) for value in region.yield_values),
            start=self.bound(region.start),
            extent=self.bound(region.extent),
            step=self.bound(region.step),
            type=dim.substitute_dims(region.type, self.dims),
        )

    @typing.override
    def default_visit(self, value, ctx=None):
        rebuilt = super().default_visit(value, ctx)
        return dataclasses.replace(rebuilt, type=dim.substitute_dims(value.type, self.dims))


def instantiate(
    module: core.Module, function: core.Function, dims: Mapping[str, int] | None
) -> tuple[core.Module, core.Function]:
    """Bind *dims* wherever *function* reaches them, in a clone of *module*.

    A dimension may be declared by any reachable function, not only by the
    signature of *function*; a name that nothing reachable declares is refused.
    The authored Module and Function are untouched.
    """
    supplied = {} if dims is None else dict(dims)
    declared = {}
    collector = DimVarCollector(declared, supplied)
    collector.visit_function(function, Context(module, function))
    if collector.cycles:
        raise expr.VerifyError(f"recursive call path through {collector.cycles[0]!r}")
    if collector.missing_bodies:
        name, path = collector.missing_bodies[0]
        raise expr.VerifyError(f"cannot analyze Function {name!r} without a body at {path!r}")
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
        dim.substitute_shape_entry(declared[name], {name: extent})
    cloner = InstantiateDimensionCloner(dims)
    for reached, _ in collector.functions:
        cloner.function(reached)
    root = cloner.function(function)
    return cloner.module(module), root


def substitute_op_dims(target: object, dims: Mapping[str, int]) -> object:
    """Substitute bindings in an operation's shape-valued attributes."""
    if isinstance(target, core.Function) or not isinstance(target, expr.Op):
        return target

    changed: dict[str, object] = {}
    for param in expr.parameters_of(type(target)):
        if param.kind != "attribute":
            continue
        value = getattr(target, param.name, None)
        if isinstance(value, types.TensorType):
            rebuilt = dim.substitute_dims(value, dims)
        elif isinstance(value, types.LayoutBase):
            rebuilt = dim.substitute_layout_dims(value, dims)
        elif isinstance(value, types.Mesh):
            rebuilt = dim.substitute_mesh_dims(value, dims)
        elif dim.is_dimension(value):
            rebuilt = dim.substitute_shape_entry(value, dims)
        elif isinstance(value, tuple) and value and all(dim.is_dimension(item) for item in value):
            rebuilt = tuple(dim.substitute_shape_entry(item, dims) for item in value)
        else:
            continue
        if rebuilt != value:
            changed[param.name] = rebuilt
    if not changed:
        return target
    attributes = {
        param.name: getattr(target, param.name)
        for param in expr.parameters_of(type(target))
        if param.kind == "attribute" and hasattr(target, param.name)
    }
    attributes.update(changed)
    return type(target)(**attributes)


def residual_dims(fn: core.Function, frame=None) -> tuple[str, ...]:
    """Every dimension still stated as a range anywhere *fn* reaches.

    Signature, body, the shape-valued attributes its operations carry, the
    bounds of its loops, and the same again for every function it calls. A
    dimension can be introduced deep inside a callee -- a reshape that states a
    block length the caller never mentions -- and a scan that stopped at the
    signature would call such a function concrete while it still holds a range.
    """
    return tuple(dim_vars_reached(fn, frame))


def dim_vars_reached(fn: core.Function, frame=None) -> dict[str, object]:
    """The declarations behind `residual_dims`, by name.

    Same traversal, keeping the `DimVar` rather than only its name, for a caller
    that has to restate the bounds a dimension was declared with.
    """
    found: dict[str, object] = {}
    DimVarCollector(found).visit_function(fn, frame)
    return found


class DimVarCollector(core.ExprVisitor[None]):
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

    @typing.override
    def default_visit_leaf(self, node, operands, ctx):
        return None

    def record(self, found: dict[str, object]) -> None:
        """Note declarations, remembering which function stated each name first."""
        owner = self.path[-1] if self.path else None
        for name in found:
            if name not in self.found and owner is not None:
                self.declared_by[name] = owner
        self.found.update(found)

    def collect_expr(self, node: expr.Expr, ctx=None) -> None:
        current = (
            ctx.module.children[node.key]
            if isinstance(node, core.ModuleList) and ctx is not None
            else node
        )
        staging: dict[str, object] = {}
        dim.collect_vars(current.type, staging)
        self.record(staging)

    def collect_bounds(self, node: expr.Expr) -> None:
        staging: dict[str, object] = {}
        for bound in ("extent", "step", "start"):
            collect_entries(getattr(node, bound, None), staging)
        self.record(staging)

    def visit_children(self, node: expr.Expr, ctx=None) -> None:
        if isinstance(node, expr.Call) and isinstance(node.target, expr.Call):
            self.visit(node.target, ctx)
        for child in (
            getattr(node, "args", ())
            + (node.elements if isinstance(node, expr.Tuple) else ())
            + getattr(node, "yield_values", ())
            + getattr(node, "params", ())
        ):
            self.visit(child, ctx)
        body = getattr(node, "body", None)
        if body is not None:
            self.visit(body, ctx)

    def visit_function(self, fn: core.Function, frame=None) -> None:
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
        previous_memo, self.memo = self.memo, {}
        try:
            if frame is not None:
                self.record(dim.vars_by_name(frame.topologies))
            for param in fn.params:
                self.record(dim.vars_by_name(param.type))
            self.record(dim.vars_by_name(fn.return_type))
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
            self.memo = previous_memo
            self.path.pop()
            self.active_functions.remove(id(fn))

    def visit_Call(self, node: expr.Call, frame=None) -> None:
        self.collect_expr(node, frame)
        self.collect_bounds(node)
        self.visit_children(node, frame)
        target = node.target
        inner = None if frame is None else frame.callee_context(node)
        callee = inner.function if inner is not None else target
        if isinstance(callee, core.Function):
            self.visit_function(callee, inner)
        else:
            staging: dict[str, object] = {}
            for attribute in getattr(type(target), "params", lambda: ())():
                if attribute.kind == "attribute":
                    collect_entries(getattr(target, attribute.name, None), staging)
            self.record(staging)

    def visit_Tuple(self, node: expr.Tuple, ctx=None) -> None:
        self.collect_expr(node, ctx)
        self.visit_children(node, ctx)

    def visit_LoopRegion(self, node: core.LoopRegion, ctx=None) -> None:
        self.collect_expr(node, ctx)
        self.collect_bounds(node)
        self.visit_children(node, ctx)

    def visit_Function(self, node: core.Function, ctx=None) -> None:
        self.visit_function(node, ctx)

    def visit_Var(self, node: expr.Var, ctx=None) -> None:
        self.collect_expr(node, ctx)

    def visit_Constant(self, node: expr.Constant, ctx=None) -> None:
        self.collect_expr(node, ctx)

    def visit_FunctionRef(self, node: expr.Expr, ctx=None) -> None:
        self.collect_expr(node, ctx)

    def visit_ModuleList(self, node, ctx=None) -> None:
        self.collect_expr(node, ctx)

    def visit_ShapeOf(self, node: expr.Expr, ctx=None) -> None:
        self.collect_expr(node, ctx)


def collect_entries(value: object, found: dict[str, object]) -> None:
    if isinstance(value, tuple):
        for entry in value:
            dim.collect_vars(entry, found)
        return
    dim.collect_vars(value, found)


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
    "instantiate",
    "display_name",
    "DISPLAY_NAME",
]
