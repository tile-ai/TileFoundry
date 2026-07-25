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


def _validate_converter_weight_name(base: HirFunction, weight_name: str) -> None:
    for p in base.params:
        if p.name == weight_name:
            if not p.is_const:
                raise TypeError(
                    f"tilefoundry.converter: {base.name!r} param {weight_name!r} "
                    f"is not a ConstTensor; a converter target must be declared "
                    f"ConstTensor[...]"
                )
            return
    raise TypeError(
        f"tilefoundry.converter: {base.name!r} has no ConstTensor param named "
        f"{weight_name!r}"
    )


def _definition_namespace() -> dict[str, Any]:
    """Names visible where this decorator is applied.

    Walks to the first frame outside this module (the ``@module`` class body or
    the enclosing scope) and returns its bindings. ``_collect_closure`` merges
    them *below* the function's own globals and freevars, so they can only add
    names, never shadow one. Two things need this: ``@func`` / ``@prim_func``
    siblings defined above this one (callee-before-caller sibling calls; a
    forward reference stays unresolved), and a value a factory holds in a local
    — notably a config object referenced only inside a type annotation, which
    never becomes a closure freevar because ``from __future__ import
    annotations`` leaves annotations unevaluated, so the compiler emits no
    load for it."""
    frame = sys._getframe(1)
    here = __file__
    while frame is not None and frame.f_code.co_filename == here:
        frame = frame.f_back
    # Walk the enclosing scopes outward: a @func inside a factory's @module
    # class body sees the class body first, then the factory's own locals (where
    # a config object lives). Stop at the module frame — past it is only import
    # / test-runner machinery. An inner scope wins over an outer one.
    ns: dict[str, Any] = {}
    while frame is not None:
        for name, value in frame.f_locals.items():
            ns.setdefault(name, value)
        if frame.f_code.co_name == "<module>":
            break
        frame = frame.f_back
    return ns


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


def _converter(self: HirFunction, weight_name: str):
    """``@base.converter(weight_name)`` — register a per-weight offline
    converter (docs/spec/runtime.md §1.1.2).

    Mirrors ``.specialize``: returns a decorator that parses a throwaway
    ``def`` (params annotated with the raw checkpoint names/types, exactly
    like a ``@func``) into a ``hir.Function``, registers it on
    ``base.converters``, and returns the parsed IR so ``@module`` can
    recognise and skip it. ``weight_name`` must name a ``ConstTensor`` param
    of ``base``. Legal only before ``base`` enters a ``Module`` (sealed).
    """
    _validate_converter_weight_name(self, weight_name)

    def _wrap_converter(fn_inner):
        extra_closure = _definition_namespace()
        ir = parse_func(
            fn_inner, topologies=self.topologies, target=self.target,
            extra_closure=extra_closure,
        )
        if ir.body is None:
            raise TypeError(
                "tilefoundry.converter: a converter must have a real body, "
                "not `pass`"
            )
        # The decorated def name (`_`) is a throwaway; the converter carries a
        # traceable name derived from the base + weight (authoring mutation,
        # before the base is sealed).
        object.__setattr__(ir, "name", f"{self.name}.converter[{weight_name}]")
        verify_function(ir)
        self.add_converter(weight_name, ir)
        return ir

    return _wrap_converter


# `f.converter(weight_name)` is the authoring surface for a per-weight offline
# converter — same rationale as `f.specialize` above.
HirFunction.converter = _converter


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
