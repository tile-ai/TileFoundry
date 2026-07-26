"""`@tilefoundry.func` / `@tilefoundry.prim_func` decorator entry (spec 011 §1).
Wraps the parser in `tilefoundry.parser` and verifies the resulting IR; the
decorator evaluates to the parsed IR node, not the original function."""

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
    """Locals visible where this decorator is applied: walks to the first
    frame outside this module and collects its (and outer scopes') locals,
    inner scope winning over outer."""
    frame = sys._getframe(1)
    here = __file__
    while frame is not None and frame.f_code.co_filename == here:
        frame = frame.f_back
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

    The decorated name binds to the resulting IR node, not the original
    function. ``topologies`` declares the topology namespace (enabling
    ``with Mesh(topology="cta", ...)``); ``target`` (a string or target object)
    selects the compilation target, left unresolved by default until a normal
    compile entry picks a backend. A ``pass`` body declares a dispatch
    prototype; implementations are registered via :meth:`Function.specialize`."""
    resolved_target = resolve_target(target) if target is not None else None

    def _wrap(fn_inner):
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

    Parses the decorated ``def`` into a variant ``hir.Function`` and appends
    it to ``base.variants``; the decorated name is a throwaway. Legal only
    before ``base`` enters a ``Module`` (a later call raises)."""
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
        # throwaway def name; give it the base's name instead.
        object.__setattr__(ir, "name", self.name)
        verify_function(ir)
        self.add_variant(ir)
        return ir

    return _wrap_variant


# Monkeypatch: keeps `hir.Function` free of a parser import.
HirFunction.specialize = _specialize


def _converter(self: HirFunction, weight_name: str):
    """``@base.converter(weight_name)`` — register a per-weight offline
    converter. See docs/spec/runtime.md §1.1.2."""
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
        # throwaway def name; give it a traceable base+weight name.
        object.__setattr__(ir, "name", f"{self.name}.converter[{weight_name}]")
        verify_function(ir)
        self.add_converter(weight_name, ir)
        return ir

    return _wrap_converter


HirFunction.converter = _converter


def prim_func(fn=None, *, target=None):
    """Decorator: parse a ``@prim_func`` function into a ``tir.PrimFunction``.

    The decorated name binds to the resulting IR node. ``target`` (a string or
    target object) selects the compilation target; omitted, it uses the
    normal compile-entry default."""
    resolved_target = resolve_target(target) if target is not None else None

    def _wrap(fn_inner):
        extra_closure = _definition_namespace()
        ir = parse_prim_func(fn_inner, target=resolved_target, extra_closure=extra_closure)
        verify_prim_function(ir)
        return ir

    if fn is not None:
        return _wrap(fn)
    return _wrap


intrinsic = _intrinsic

__all__ = ["func", "prim_func", "intrinsic"]
