from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from tilefoundry.ir.core import Expr, Var
from tilefoundry.ir.core.expr import Call, Constant
from tilefoundry.ir.core.pattern import DimVarRangePat, Pattern
from tilefoundry.ir.core.registry import register_typeinfer
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.types import CallableType, TensorType, Type, callable_type_for
from tilefoundry.ir.types.shard.mesh import Topology
from tilefoundry.target import Target
from tilefoundry.visitor_registry.contexts import TypeInferContext


def _callable_type_for(params: tuple[Var, ...], return_type: Type) -> CallableType:
    """Project ``Function.params`` + ``return_type`` into the IR-level
    ``CallableType``.
    """
    return callable_type_for(params, return_type)


@dataclass(frozen=True)
class Function(Expr):
    """HIR function container: a pure-SSA ``Expr`` whose value type is its callable signature."""
    name: str
    params: tuple[Var, ...]
    body: Expr | None                       # None for a dispatch prototype (DSL ``pass``)
    return_type: Type
    topologies: tuple[Topology, ...] = field(default_factory=tuple)
    specializations: tuple[Pattern, ...] = field(default_factory=tuple)
    variants: tuple["Function", ...] = field(default_factory=tuple)
    target: Target | None = None

    @classmethod
    def build(
        cls,
        *,
        name: str,
        params: tuple[Var, ...],
        body: Expr | None,
        return_type: Type,
        topologies: tuple[Topology, ...] = (),
        specializations: tuple[Pattern, ...] = (),
        variants: tuple["Function", ...] = (),
        target: Target | None = None,
        loc: str | None = None,
    ) -> "Function":
        """Construct a Function with the canonical CallableType."""
        return cls(
            name=name,
            params=params,
            body=body,
            return_type=return_type,
            topologies=tuple(topologies),
            specializations=tuple(specializations),
            variants=tuple(variants),
            target=target,
            type=_callable_type_for(params, return_type),
            loc=loc,
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

    def seal(self) -> None:
        """Freeze authoring mutation: ``add_variant`` raises afterwards.

        Called by ``Module`` construction on each function it contains.
        Idempotent. Variants are sealed alongside their base.
        """
        object.__setattr__(self, "_sealed", True)
        for v in self.variants:
            v.seal()


# ir.visitor imports Function from this module at module level; this
# module-level import is positioned after Function is defined, so
# whichever of the two modules loads first, the other's back-reference
# finds an already-bound name instead of hitting a partially-initialized
# module.
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
            # Fall back to repr for forward-compat; v0 verifier rejects
            # non-DimVarRangePat patterns elsewhere.
            parts.append(repr(pat))
    return ";".join(parts)


def _bind_param_type(
    ctx, callee: "Function", i: int, param: Var, arg_ty: Type,
    call: Call | None = None,
) -> Type:
    """Bind one parameter's elaborated type from the caller's argument type.

    A ``layout is None`` ``TensorType`` parameter is a template wildcard —
    the bound type is the argument's own full type (including any
    ``ShardLayout``), once its logical shape/dtype match. Any other
    parameter type is an explicit contract: the argument MUST match it
    exactly (hir.md §1.1). ``call``, when given, anchors a bind error's
    location instead of the callee's own (always ``None``) ``.loc``.
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
    callee: "Function", arg_types: tuple[Type, ...], ctx: TypeInferContext | None = None,
    call: Call | None = None,
) -> "Function":
    """Construct the concrete callee instance for one call site's argument
    types (hir.md §1.1). The template lives at the Python-source level;
    every differently-typed call gets its own IR construction here.

    Returns ``callee`` unchanged for a dispatch prototype (``variants !=
    ()``/``body is None`` — no body to elaborate; shape dispatch stays
    envelope-matched, untouched by this function) and whenever every bound
    parameter type already equals the callee's current parameter type
    (dedup — an allowed optimization, not a semantic). ``call``, when
    given, anchors an arity/bind error's location. Within one construction
    session (``ctx.elaboration_cache``), repeated (callee, arg_types) call
    sites reuse the same rebuilt instance.
    """
    if ctx is None:
        ctx = TypeInferContext()
    expected = len(callee.params)
    got = len(arg_types)
    if got != expected:
        ctx.error(
            call if call is not None else callee,
            f"hir Function call {callee.name!r}: arity mismatch — "
            f"callee declares {expected} parameter(s), call passed {got}",
        )
    bound_types = [
        _bind_param_type(ctx, callee, i, param, arg_ty, call)
        for i, (param, arg_ty) in enumerate(zip(callee.params, arg_types))
    ]
    if callee.variants or callee.body is None:
        return callee
    if all(bt == p.type for bt, p in zip(bound_types, callee.params)):
        return callee

    cache_key = (id(callee), arg_types)
    cached = ctx.elaboration_cache.get(cache_key)
    if cached is not None:
        return cached

    new_params = tuple(
        Var(type=bt, name=p.name) for bt, p in zip(bound_types, callee.params)
    )
    subst = {id(old): new for old, new in zip(callee.params, new_params)}

    class _Elaborator(ExprMutator):
        """Rebuild ``callee.body`` under ``subst`` (memoized by node
        identity so SSA-as-DAG sharing survives), re-stamping every
        changed node's type through the shared typeinfer visitor."""

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
            """Rebuild args as usual; additionally, a Call whose target is
            a hir Function is re-elaborated against the rewritten arg
            types so ``.target`` (not just ``.type``) reflects the fresh
            instance — required per hir.md §1.1 for a viewer/printer read
            of ``call.target.body`` under a wildcard chain."""
            new_args = tuple(self.visit(a) for a in call_expr.args)
            args_changed = any(na is not oa for na, oa in zip(new_args, call_expr.args))
            new_target = call_expr.target
            if isinstance(call_expr.target, Function):
                new_target = elaborate(
                    call_expr.target, tuple(a.type for a in new_args), self.body_ctx,
                    call=call_expr,
                )
            if not args_changed and new_target is call_expr.target:
                return call_expr
            rebuilt = dataclasses.replace(call_expr, args=new_args, target=new_target)
            return dataclasses.replace(rebuilt, type=self.body_ctx.type_of(rebuilt))

        def visit_GridRegionExpr(self, grid: GridRegionExpr) -> Expr:
            """Re-stamp the loop-phi ``carried_args`` from the rewritten
            ``init_args`` (hir.md §1.2: "the first-iteration value of each
            carried_args phi is its init_args entry"), the same rule the
            parser applies when constructing the node, then substitute the
            fresh phi into the body/yield_values before rebuilding them."""
            new_init_args = tuple(self.visit(a) for a in grid.init_args)
            new_phis = tuple(
                old_phi if new_init.type == old_phi.type
                else Var(type=new_init.type, name=old_phi.name)
                for old_phi, new_init in zip(grid.carried_args, new_init_args)
            )
            for old_phi, new_phi in zip(grid.carried_args, new_phis):
                if new_phi is not old_phi:
                    subst[id(old_phi)] = new_phi
            new_body = self.visit(grid.body)
            new_yields = tuple(self.visit(y) for y in grid.yield_values)
            unchanged = (
                all(ni is oi for ni, oi in zip(new_init_args, grid.init_args))
                and all(np_ is op for np_, op in zip(new_phis, grid.carried_args))
                and new_body is grid.body
                and all(ny is oy for ny, oy in zip(new_yields, grid.yield_values))
            )
            if unchanged:
                return grid
            rebuilt = dataclasses.replace(
                grid, carried_args=new_phis, init_args=new_init_args,
                body=new_body, yield_values=new_yields,
            )
            return dataclasses.replace(rebuilt, type=self.body_ctx.type_of(rebuilt))

        def generic_visit(self, expr: Expr) -> Expr:
            rebuilt = super().generic_visit(expr)
            if rebuilt is expr:
                return expr
            return dataclasses.replace(rebuilt, type=self.body_ctx.type_of(rebuilt))

    body_ctx = TypeInferContext(module=ctx.module, elaboration_cache=ctx.elaboration_cache)
    new_body = _Elaborator(body_ctx).visit(callee.body)
    instance = Function.build(
        name=callee.name,
        params=new_params,
        body=new_body,
        return_type=new_body.type,
        topologies=callee.topologies,
        specializations=callee.specializations,
        target=callee.target,
    )
    ctx.elaboration_cache[cache_key] = instance
    return instance


@register_typeinfer(Function)
def _typeinfer_hir_function_call(call: Call, ctx) -> Type:
    """Typeinfer handler for ``Call(target=hir.Function, args=...)``:
    derive the type by elaboration (hir.md §1.1). The Call's type is always
    the freshly re-derived type of the (possibly deduped) instance's body —
    never a possibly-stale ``Function.return_type`` field — except for a
    dispatch prototype, whose ``None`` body is never inspected."""
    callee: Function = call.target  # type: ignore[assignment]
    arg_types = tuple(ctx.type_of(a) for a in call.args)
    instance = elaborate(callee, arg_types, ctx, call=call)
    if instance.body is None:
        return instance.return_type
    return TypeInferContext(module=ctx.module).type_of(instance.body)


__all__ = [
    "Function",
    "_callable_type_for",
    "canonical_specialization_signature",
    "elaborate",
]
