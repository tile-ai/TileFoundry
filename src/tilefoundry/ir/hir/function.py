from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field

from tilefoundry.ir.core import Expr, Op, Var
from tilefoundry.ir.core.expr import Call, Constant
from tilefoundry.ir.core.pattern import DimVarRangePat, Pattern
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.types import TensorType, Type, callable_type_for
from tilefoundry.ir.types.dim import is_dim_expr
from tilefoundry.ir.types.substitute import substitute_shape_dim
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.contexts import FunctionScope, TypeInferContext


@dataclass(frozen=True)
class Function(Expr):
    """Contain a pure-SSA HIR body with its callable value type.

    ``body=None`` marks a dispatch prototype. Variants and weight converters
    use tuples so frozen function equality and hashing remain stable. Execution
    context belongs to the owning Module.

    See [hir §1.1](docs/spec/hir.md#11-function).
    """

    name: str
    params: tuple[Var, ...]
    body: Expr | None
    return_type: Type
    specializations: tuple[Pattern, ...] = field(default_factory=tuple)
    variants: tuple["Function", ...] = field(default_factory=tuple)
    converters: tuple[tuple[str, "Function"], ...] = field(default_factory=tuple)

    @classmethod
    def build(
        cls,
        *,
        name: str,
        params: tuple[Var, ...],
        body: Expr | None,
        return_type: Type,
        specializations: tuple[Pattern, ...] = (),
        variants: tuple["Function", ...] = (),
        converters: tuple[tuple[str, "Function"], ...] = (),
    ) -> "Function":
        """Construct a Function with the canonical CallableType."""
        return cls(
            name=name,
            params=params,
            body=body,
            return_type=return_type,
            specializations=tuple(specializations),
            variants=tuple(variants),
            converters=tuple(converters),
            type=callable_type_for(params, return_type),
        )

    def add_variant(self, variant: "Function") -> None:
        """Append a specialization ``variant`` during authoring.

        ``variants`` participates in eq/hash, so accumulation uses controlled
        authoring-phase mutation (``object.__setattr__``); a sealed base
        rejects further variants.
        """
        if getattr(self, "_sealed", False):
            raise RuntimeError(
                f"hir Function {self.name!r}: cannot add a specialization "
                f"variant after the function has entered a Module (sealed)"
            )
        object.__setattr__(self, "variants", (*self.variants, variant))

    def add_converter(self, weight_name: str, fn: "Function") -> None:
        """Register a per-weight offline converter ``fn`` for ``weight_name``.

        Mirrors ``add_variant``: authoring-phase mutation via
        ``object.__setattr__`` (``converters`` participates in eq/hash), a
        sealed base rejects further registration, and a repeated
        ``weight_name`` raises.
        """
        if getattr(self, "_sealed", False):
            raise RuntimeError(
                f"hir Function {self.name!r}: cannot add a converter after "
                f"the function has entered a Module (sealed)"
            )
        if any(existing == weight_name for existing, _ in self.converters):
            raise ValueError(
                f"hir Function {self.name!r}: a converter for weight "
                f"{weight_name!r} is already registered"
            )
        object.__setattr__(self, "converters", (*self.converters, (weight_name, fn)))

    def seal(self) -> None:
        """Freeze authoring mutation: ``add_variant`` raises afterwards.

        Called by ``Module`` construction on each function it contains.
        Idempotent. Variants and converters are sealed alongside their base.
        """
        object.__setattr__(self, "_sealed", True)
        for v in self.variants:
            v.seal()
        for _, conv in self.converters:
            conv.seal()


from tilefoundry.ir.hir._call_binding import binding_for  # noqa: E402
from tilefoundry.ir.visitor import ExprMutator  # noqa: E402


def canonical_specialization_signature(
    specializations: tuple[Pattern, ...],
) -> str:
    """Deterministic identity string for a Function's specialization tuple.

    Same-name Functions are distinguished by this signature. For v0 the
    only allowed pattern is ``DimVarRangePat``, so the signature is
    ``"<dim_var>$<lo>_<hi>"`` joined by ``;`` in declared order.
    """
    parts: list[str] = []
    for pat in specializations:
        if isinstance(pat, DimVarRangePat):
            parts.append(f"{pat.dim_var}${pat.lo}_{pat.hi}")
        else:
            parts.append(repr(pat))
    return ";".join(parts)


def _scope_for(ctx: TypeInferContext, callee: "Function") -> "FunctionScope | None":
    """*ctx*'s scope moved to *callee*, whose body is about to be read."""
    scope = ctx.scope
    return None if scope is None else dataclasses.replace(scope, function=callee)


def _bind_param_type(
    ctx,
    callee: "Function",
    i: int,
    param: Var,
    arg_ty: Type,
    call: Call | None = None,
) -> Type:
    """Bind one parameter's elaborated type from the caller's argument type.

    A ``layout is None`` tensor parameter accepts the argument's full layout
    after shape and dtype match. Other parameter types must match exactly.
    ``call`` anchors errors at the call site when provided.

    See [hir §1.1](docs/spec/hir.md#11-function).
    """
    error_node = call if call is not None else callee
    p = param.type
    if isinstance(p, TensorType) and isinstance(arg_ty, TensorType) and p.layout is None:
        if arg_ty.shape != p.shape or arg_ty.dtype != p.dtype:
            ctx.error(
                error_node,
                f"hir Function call {callee.name!r}: arg {i} shape/dtype "
                f"mismatch — callee param {param.name!r} expects logical "
                f"{p.shape} {p.dtype}, got {arg_ty.shape} {arg_ty.dtype}",
            )
        return arg_ty
    if arg_ty != p:
        ctx.error(
            error_node,
            f"hir Function call {callee.name!r}: arg {i} type mismatch — "
            f"callee param {param.name!r} expects {p!r}, got {arg_ty!r}",
        )
    return p


def elaborate(
    callee: "Function",
    arg_types: tuple[Type, ...],
    ctx: TypeInferContext | None = None,
    call: Call | None = None,
) -> "Function":
    """Construct a concrete callee for one call site's argument types.

    Dispatch prototypes and already-equal bindings return unchanged. Other
    templates rebuild per distinct type tuple and reuse the construction
    session's elaboration cache. ``call`` anchors binding errors when present,
    and is what says how the call binds: one whose callee's constants come from
    a Module reading binds the non-constant parameters alone.

    See [hir §1.1](docs/spec/hir.md#11-function).
    """
    if ctx is None:
        ctx = TypeInferContext()
    binding = binding_for(callee, call, ctx)
    expected = len(binding.params)
    got = len(arg_types)
    if got != expected:
        kind = "activation(s)" if binding.from_reading else "parameter(s)"
        ctx.error(
            call if call is not None else callee,
            f"hir Function call {callee.name!r}: arity mismatch — "
            f"callee declares {expected} {kind}, call passed {got}",
        )
    given = iter(enumerate(arg_types))
    bound_types = []
    for param in callee.params:
        if binding.from_reading and param.is_const:
            bound_types.append(param.type)
            continue
        index, arg_ty = next(given)
        bound_types.append(_bind_param_type(ctx, callee, index, param, arg_ty, call))
    if callee.variants or callee.body is None:
        return callee
    if all(bt == p.type for bt, p in zip(bound_types, callee.params)):
        return callee

    cache_key = (id(callee), arg_types, binding.from_reading)
    cached = ctx.elaboration_cache.get(cache_key)
    if cached is not None:
        return cached

    instance = _elaborate_from_bound_types(
        callee, bound_types, ctx, scope=binding.scope
    )
    ctx.elaboration_cache[cache_key] = instance
    return instance


def _elaborate_from_bound_types(
    callee: "Function",
    bound_types: "list[Type] | tuple[Type, ...]",
    ctx: TypeInferContext,
    *,
    dims: "Mapping[str, int] | None" = None,
    scope: "FunctionScope | None" = None,
) -> "Function":
    """Rebuild ``callee`` with parameters at *bound_types*.

    Calls and specializations share this path. Optional *dims* also substitute
    loop bounds and shape-valued operation attributes that are not reachable
    through expression types. The rebuild records ``callee`` as its origin, so
    the instance a `Call` targets stays connected to the function a Module owns
    without matching on the name they share; a call site chooses no extent and
    so records none.
    """
    new_params = tuple(
        Var(type=bt, name=p.name, is_const=p.is_const) for bt, p in zip(bound_types, callee.params)
    )
    subst = {id(old): new for old, new in zip(callee.params, new_params)}

    pinned: list[object] = []

    class _Elaborator(ExprMutator):
        """Represent Elaborator.

        Rebuild ``callee.body`` under ``subst`` (memoized by node
        identity so SSA-as-DAG sharing survives), re-stamping every
        changed node's type through the shared typeinfer visitor.
        """

        def __init__(self, body_ctx: TypeInferContext) -> None:
            self.body_ctx = body_ctx
            self._memo: dict[int, Expr] = {}

        def visit(self, expr: Expr) -> Expr:
            cached = self._memo.get(id(expr))
            if cached is not None:
                return cached
            new = super().visit(expr)
            self._memo[id(expr)] = new
            return new

        def visit_Var(self, var: Var) -> Expr:
            return subst.get(id(var), var)

        def visit_Constant(self, c: Constant) -> Expr:
            return c

        def visit_Call(self, call_expr: Call) -> Expr:
            """Rebuild arguments and re-elaborate nested HIR callees.

            Updating the target as well as the call type exposes the concrete
            body to viewers and printers.

            See [hir §1.1](docs/spec/hir.md#11-function).
            """
            new_args = tuple(self.visit(a) for a in call_expr.args)
            args_changed = any(na is not oa for na, oa in zip(new_args, call_expr.args))
            new_target = call_expr.target
            if isinstance(call_expr.target, Function):
                if dims is None:
                    new_target = elaborate(
                        call_expr.target,
                        tuple(a.type for a in new_args),
                        self.body_ctx,
                        call=call_expr,
                    )
                else:
                    new_target = _specialize_callee(
                        call_expr.target, dims, self.body_ctx, call_expr
                    )

                    pinned.append(new_target)
            if dims is not None:
                new_target = _substitute_op_dims(new_target, dims)
            if not args_changed and new_target is call_expr.target:
                return call_expr
            rebuilt = dataclasses.replace(call_expr, args=new_args, target=new_target)
            return self._retyped(rebuilt)

        def visit_GridRegionExpr(self, grid: GridRegionExpr) -> Expr:
            """Visit GridRegionExpr.

            Rebuild loop phis from rewritten ``init_args``, then substitute the
            fresh values through the body and yields.

            See [hir §1.2](docs/spec/hir.md#12-gridregionexpr).
            """
            new_init_args = tuple(self.visit(a) for a in grid.init_args)
            new_phis = tuple(
                old_phi
                if new_init.type == old_phi.type
                else Var(type=new_init.type, name=old_phi.name)
                for old_phi, new_init in zip(grid.carried_args, new_init_args)
            )
            for old_phi, new_phi in zip(grid.carried_args, new_phis):
                if new_phi is not old_phi:
                    subst[id(old_phi)] = new_phi
            new_body = self.visit(grid.body)
            new_yields = tuple(self.visit(y) for y in grid.yield_values)

            bounds = (grid.extent, grid.step, grid.start)
            if dims is None:
                new_bounds = bounds
            else:
                new_bounds = tuple(substitute_shape_dim(b, dims) for b in bounds)
            unchanged = (
                all(ni is oi for ni, oi in zip(new_init_args, grid.init_args))
                and all(np_ is op for np_, op in zip(new_phis, grid.carried_args))
                and new_body is grid.body
                and all(ny is oy for ny, oy in zip(new_yields, grid.yield_values))
                and new_bounds == bounds
            )
            if unchanged:
                return grid
            rebuilt = dataclasses.replace(
                grid,
                carried_args=new_phis,
                init_args=new_init_args,
                body=new_body,
                yield_values=new_yields,
                extent=new_bounds[0],
                step=new_bounds[1],
                start=new_bounds[2],
            )
            return self._retyped(rebuilt)

        def generic_visit(self, expr: Expr) -> Expr:
            rebuilt = super().generic_visit(expr)
            if rebuilt is expr:
                return expr
            return self._retyped(rebuilt)

        def _retyped(self, rebuilt: Expr) -> Expr:
            """*rebuilt* carrying the type its new children give it.

            The node asked about is then thrown away -- the returned node is a
            further copy of it -- while the type cache it just populated is
            keyed on its identity and lives on for the whole rebuild. Holding a
            reference is what stops that identity being handed to the next
            node, which would otherwise read a type belonging to something
            else. This is only reachable at all when many nodes change at once,
            which is why substituting a dimension found it.
            """
            pinned.append(rebuilt)
            return dataclasses.replace(rebuilt, type=self.body_ctx.type_of(rebuilt))

    inner = scope if scope is not None else _scope_for(ctx, callee)
    if dims is None:
        body_ctx = TypeInferContext(
            scope=inner, elaboration_cache=ctx.elaboration_cache
        )
    else:
        body_ctx = TypeInferContext(scope=inner)
    new_body = _Elaborator(body_ctx).visit(callee.body)
    derived = Function.build(
        name=callee.name,
        params=new_params,
        body=new_body,
        return_type=new_body.type,
        specializations=callee.specializations,
    )
    from .specialize import _record_provenance  # noqa: PLC0415 — avoid import cycle

    _record_provenance(derived, callee, dims)
    return derived


def _specialize_callee(
    callee: "Function",
    dims: "Mapping[str, int]",
    ctx: TypeInferContext,
    call: Call,
) -> "Function":
    """*callee* rebuilt at the same extents its caller was given.

    A nested dispatch prototype is refused rather than guessed at: choosing
    among its variants is a decision the caller has not stated, and picking
    one here would bury that choice inside a rebuild.
    """
    from tilefoundry.ir.types.substitute import substitute_dims  # noqa: PLC0415

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
    return _elaborate_from_bound_types(callee, bound, ctx, dims=dims)


def _substitute_op_dims(target: object, dims: "Mapping[str, int]") -> object:
    """*target* with any shape-valued attribute rebuilt at the bound extents.

    Which attributes those are is read off the operation rather than listed:
    an attribute whose entries are all dimension expressions is a shape, and
    an entry that is an ordinary integer substitutes to itself, so an
    attribute that merely looks like one -- a permutation, a set of axes --
    passes through untouched.
    """
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

        if isinstance(value, LayoutBase):
            rebuilt_layout = substitute_layout_dims(value, dims)
            if rebuilt_layout is not value:
                changed[param.name] = rebuilt_layout
            continue
        if isinstance(value, Mesh):
            rebuilt_mesh = substitute_mesh_dims(value, dims)
            if rebuilt_mesh is not value:
                changed[param.name] = rebuilt_mesh
            continue
        if not isinstance(value, tuple) or not value:
            continue
        if not all(is_dim_expr(entry) for entry in value):
            continue
        rebuilt = tuple(substitute_shape_dim(entry, dims) for entry in value)
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


@register_typeinfer(Function)
def _typeinfer_hir_function_call(call: Call, ctx) -> Type:
    """Derive a function call's type by elaborating its callee.

    The type is re-derived from the elaborated body's type rather than a stale
    ``return_type`` field. Dispatch prototypes retain their declared type.

    See [hir §1.1](docs/spec/hir.md#11-function).
    """
    callee: Function = call.target  # type: ignore[assignment]
    arg_types = tuple(ctx.type_of(a) for a in call.args)
    binding = binding_for(callee, call, ctx)
    instance = elaborate(callee, arg_types, ctx, call=call)
    if instance.body is None:
        return instance.return_type
    inner = binding.scope
    body_ctx = TypeInferContext(
        scope=None if inner is None else dataclasses.replace(inner, function=instance)
    )
    return body_ctx.type_of(instance.body)


__all__ = [
    "Function",
    "canonical_specialization_signature",
    "elaborate",
]
