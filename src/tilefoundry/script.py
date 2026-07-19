"""`@tilefoundry.func` / `@tilefoundry.prim_func` decorator entry (spec 011 §1).

Wraps the parser in `tilefoundry.parser` and verifies the resulting IR. The
decorator *evaluates to the IR node*: `@func` to a `hir.Function`, `@prim_func`
to a `tir.PrimFunction`. The decorated name binds to that IR node, not to the
original Python function.

Shape specialization is authored with `Function.specialize`: a base function is
defined with `@func` (its body is `pass`, declaring a dispatch prototype) and
each variant is added by decorating a throwaway `def` with
`@base.specialize(pattern)`:

    @func
    def f(x: Tensor[(S,), "f32"]) -> Tensor[(S,), "f32"]:
        pass

    @f.specialize(DimVarRangePat("S", 1, 4))
    def _(x: Tensor[(S,), "f32"]) -> Tensor[(S,), "f32"]:
        return small_impl(x)
"""

from __future__ import annotations

import sys
from typing import Any

from tilefoundry.ir.core.pattern import DimVarRangePat, Pattern
from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.hir.verify import verify_function
from tilefoundry.ir.tir.intrinsic import intrinsic as _intrinsic
from tilefoundry.ir.tir.verify import verify_prim_function
from tilefoundry.parser import parse_func, parse_prim_func
from tilefoundry.target import resolve_target


def _validate_one_pattern(pattern: Any) -> Pattern:
    if not isinstance(pattern, Pattern):
        raise TypeError(
            f"tilefoundry.specialize: pattern must be a Pattern instance, got "
            f"{type(pattern).__name__}"
        )
    if not isinstance(pattern, DimVarRangePat):
        raise TypeError(
            f"tilefoundry.specialize: only DimVarRangePat is supported for v0, "
            f"got {type(pattern).__name__}"
        )
    return pattern


def _definition_namespace() -> dict[str, Any]:
    """Sibling IR functions visible where this decorator is applied.

    Walks to the first frame outside this module (the ``@module`` class body or
    the enclosing scope) and returns only its bindings that are already parsed
    IR functions — i.e. ``@func`` / ``@prim_func`` siblings defined above this
    one. Restricting to IR functions keeps the merge additive (it cannot shadow
    an unrelated global) and only enables callee-before-caller sibling calls;
    forward references stay unresolved."""
    from tilefoundry.ir.tir.prim_function import PrimFunction  # noqa: PLC0415

    frame = sys._getframe(1)
    here = __file__
    while frame is not None and frame.f_code.co_filename == here:
        frame = frame.f_back
    if frame is None:
        return {}
    return {
        name: value
        for name, value in frame.f_locals.items()
        if isinstance(value, (HirFunction, PrimFunction))
    }


def func(fn=None, *, topologies=(), target=None):
    """Decorator: parse an ``@func``-decorated function into a ``hir.Function``.

    The decorated name binds to the resulting ``hir.Function``. ``topologies``
    declares the topology namespace for this function, enabling
    ``with Mesh(topology="cta", ...)`` string-name resolution. ``target``
    selects the function's compilation target (a string reflected to a target
    object, or a target object); an omitted target remains unresolved until a
    normal compile entry resolves its backend default.

    A ``pass`` body declares a dispatch prototype (``Function.body is None``);
    its implementations are registered via :meth:`Function.specialize`.
    """
    resolved_target = resolve_target(target) if target is not None else None

    def _wrap(fn_inner):
        # Sibling @func / @prim_func bindings defined above this one in the
        # definition frame, so a composed kernel can call them as nested targets.
        extra_closure = _definition_namespace()
        ir = parse_func(
            fn_inner, topologies=topologies, target=resolved_target,
            extra_closure=extra_closure,
        )
        verify_function(ir)
        return ir

    if fn is not None:
        return _wrap(fn)
    return _wrap


def _specialize(self: HirFunction, pattern: Any):
    """``@base.specialize(DimVarRangePat(...))`` — register a shape variant.

    Returns a decorator that parses the decorated ``def`` into a variant
    ``hir.Function`` (same signature as the base, carrying the one pattern) and
    appends it to ``base.variants``. The decorated name is a throwaway — ``def
    _`` is reusable across variants because the base is the persistent handle.
    Legal only during authoring, before the base enters a ``Module`` (the base
    seals on Module entry; a later ``specialize`` raises).
    """
    pat = _validate_one_pattern(pattern)

    def _wrap_variant(fn_inner):
        extra_closure = _definition_namespace()
        ir = parse_func(
            fn_inner, topologies=self.topologies, specializations=(pat,),
            target=self.target, extra_closure=extra_closure,
        )
        if ir.body is None:
            raise TypeError(
                "tilefoundry.specialize: a variant must have a real body, not "
                "`pass` (only the base prototype declares a `pass` body)"
            )
        # The decorated def name (`_`) is a throwaway; the variant carries the
        # base's name (authoring mutation, before the base is sealed).
        object.__setattr__(ir, "name", self.name)
        verify_function(ir)
        self.add_variant(ir)
        return ir

    return _wrap_variant


# `f.specialize(...)` is the authoring surface for shape dispatch. It lives here
# (next to `func` and the definition-frame walk) so sibling resolution sees the
# same frame chain; `hir.Function` stays free of any parser/decorator import.
HirFunction.specialize = _specialize


def prim_func(fn=None, *, target=None):
    """Decorator: parse a ``@prim_func`` function into a ``tir.PrimFunction``.

    The decorated name binds to the resulting ``tir.PrimFunction``. ``target``
    selects the function's compilation target (string reflected to a target
    object, or a target object); an omitted target uses the normal compile-entry
    default.
    """
    resolved_target = resolve_target(target) if target is not None else None

    def _wrap(fn_inner):
        # Sibling @func / @prim_func bindings defined above this one in the
        # definition frame (e.g. a @module class body), so a host entry can
        # resolve the device function it launches.
        extra_closure = _definition_namespace()
        ir = parse_prim_func(fn_inner, target=resolved_target, extra_closure=extra_closure)
        verify_prim_function(ir)
        return ir

    if fn is not None:
        return _wrap(fn)
    return _wrap


intrinsic = _intrinsic

__all__ = ["func", "prim_func", "intrinsic"]
