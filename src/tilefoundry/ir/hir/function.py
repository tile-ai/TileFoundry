from __future__ import annotations

from dataclasses import dataclass, field

from tilefoundry.ir.core import Expr, Var
from tilefoundry.ir.core.expr import Call
from tilefoundry.ir.core.pattern import DimVarRangePat, Pattern
from tilefoundry.ir.core.registry import register_typeinfer
from tilefoundry.ir.target import CudaTarget, Target
from tilefoundry.ir.types import CallableType, TensorType, Type, callable_type_for
from tilefoundry.ir.types.shard.mesh import Topology
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
    target: Target = field(default_factory=CudaTarget)

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
            target=target if target is not None else CudaTarget(),
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


def _check_arg_against_param(call, ctx, callee, i, param, arg_ty: Type) -> None:
    """Validate one call argument against a callee parameter."""
    p = param.type
    if isinstance(p, TensorType) and isinstance(arg_ty, TensorType) and p.layout is None:
        if arg_ty.shape != p.shape or arg_ty.dtype != p.dtype:
            ctx.error(
                call,
                f"hir Function call {callee.name!r}: arg {i} shape/dtype "
                f"mismatch — callee param {param.name!r} expects logical "
                f"{p.shape} {p.dtype}, got {arg_ty.shape} {arg_ty.dtype}",
            )
        return
    if arg_ty != p:
        ctx.error(
            call,
            f"hir Function call {callee.name!r}: arg {i} type mismatch — "
            f"callee param {param.name!r} expects {p!r}, got {arg_ty!r}",
        )


@register_typeinfer(Function)
def _typeinfer_hir_function_call(call: Call, ctx) -> Type:
    """Typeinfer handler for ``Call(target=hir.Function, args=...)``."""
    callee: Function = call.target  # type: ignore[assignment]
    expected = len(callee.params)
    got = len(call.args)
    if got != expected:
        ctx.error(
            call,
            f"hir Function call {callee.name!r}: arity mismatch — "
            f"callee declares {expected} parameter(s), call passed {got}",
        )
    if callee.variants:
        # Dispatch prototype: validate args against the shared signature and
        # return the declared return type — never typeinfer the None body.
        for i, (param, arg) in enumerate(zip(callee.params, call.args)):
            _check_arg_against_param(call, ctx, callee, i, param, ctx.type_of(arg))
        return callee.return_type
    sub = TypeInferContext(module=ctx.module)
    for i, (param, arg) in enumerate(zip(callee.params, call.args)):
        arg_ty = ctx.type_of(arg)
        _check_arg_against_param(call, ctx, callee, i, param, arg_ty)
        sub.cache[param] = arg_ty
    return sub.type_of(callee.body)


__all__ = [
    "Function",
    "_callable_type_for",
    "canonical_specialization_signature",
]
