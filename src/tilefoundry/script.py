"""Define parser-backed ``@func`` and ``@prim_func`` decorators.

The surface follows [parser §1](docs/spec/parser.md#1-dsl-syntax). A decorator
returns the parsed and verified IR node, not the original Python function.
"""

from __future__ import annotations

import sys
from typing import Any

from tilefoundry.ir.core.module import Module
from tilefoundry.ir.core.pattern import DimVarRangePat, Pattern
from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.hir.specialize import DISPLAY_NAME
from tilefoundry.ir.hir.verify import verify_function
from tilefoundry.ir.tir.intrinsic import intrinsic as _intrinsic
from tilefoundry.ir.tir.verify import verify_prim_function
from tilefoundry.module import _DECLARING, UNDECLARED
from tilefoundry.parser import parse_func, parse_prim_func
from tilefoundry.target.base import target_instance


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
    """Locals visible where this decorator is applied.

    Locals visible where this decorator is applied: walks to the first
    frame outside this module and collects its (and outer scopes') locals,
    inner scope winning over outer.
    """
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


def _enclosing_topologies() -> tuple | None:
    """Find the declaration belonging to the enclosing ``@module`` class body."""
    frame = sys._getframe(1)
    here = __file__
    while frame is not None and frame.f_code.co_filename == here:
        frame = frame.f_back
    while frame is not None:
        if "__qualname__" in frame.f_locals:
            for entry in reversed(_DECLARING):
                if entry.frame is frame.f_back:
                    if entry.topologies is not None:
                        return entry.topologies
                    break
        elif frame.f_code.co_name == "<module>":
            break
        frame = frame.f_back
    return None


def func(fn=None, *, topologies=UNDECLARED, target=None):
    """Decorator: parse an ``@func``-decorated function into HIR.

    Plain ``@func`` inherits its owning module's topology. Supplying a target or
    topology makes an implicit single-function module with its own execution
    domain. A ``pass`` body declares a dispatch prototype whose implementations
    are registered through :meth:`Function.specialize`.
    """
    if target is not None:
        target_instance(target)
    resolved_target = target
    declares_context = resolved_target is not None or topologies is not UNDECLARED
    declared_topologies = None if topologies is UNDECLARED else tuple(topologies)

    def _wrap(fn_inner):
        extra_closure = _definition_namespace()
        parse_topologies = declared_topologies
        if parse_topologies is None:
            parse_topologies = _enclosing_topologies()
        ir = parse_func(
            fn_inner, topologies=parse_topologies or (),
            extra_closure=extra_closure,
        )
        verify_function(ir)
        if not declares_context:
            return ir
        return Module(
            name=ir.name,
            functions=(ir,),
            entry=ir.name,
            target=resolved_target,
            topologies=declared_topologies,
        )

    if fn is not None:
        return _wrap(fn)
    return _wrap


def _specialize(self: HirFunction, pattern: Any):
    """``@base.specialize(DimVarRangePat(...))`` — register a shape variant.

    Parses the decorated ``def`` into a variant ``hir.Function`` and appends it to
    ``base.variants``. The identifier becomes the variant's display label, or
    nothing when it is ``_``; the variant's ``name`` is the base's either way.
    Legal only before ``base`` enters a ``Module`` (a later call raises).
    """
    pat = _validate_one_pattern(pattern)

    def _wrap_variant(fn_inner):
        extra_closure = _definition_namespace()
        ir = parse_func(
            fn_inner, topologies=_enclosing_topologies() or (),
            specializations=(pat,), extra_closure=extra_closure,
        )
        if ir.body is None:
            raise TypeError(
                "tilefoundry.specialize: a variant must have a real body, not "
                "`pass` (only the base prototype declares a `pass` body)"
            )

        if fn_inner.__name__ != "_":
            object.__setattr__(ir, DISPLAY_NAME, fn_inner.__name__)
        object.__setattr__(ir, "name", self.name)
        verify_function(ir)
        self.add_variant(ir)
        return ir

    return _wrap_variant



HirFunction.specialize = _specialize


def _converter(self: HirFunction, weight_name: str):
    """``@base.converter(weight_name)`` — register a per-weight offline converter.

    ``@base.converter(weight_name)`` — register a per-weight offline
    converter. See [runtime §1.1.2](docs/spec/runtime.md#112-weight-converter-and-prepare--forward).
    """
    _validate_converter_weight_name(self, weight_name)

    def _wrap_converter(fn_inner):
        extra_closure = _definition_namespace()
        ir = parse_func(fn_inner, extra_closure=extra_closure)
        if ir.body is None:
            raise TypeError(
                "tilefoundry.converter: a converter must have a real body, "
                "not `pass`"
            )

        object.__setattr__(ir, "name", f"{self.name}.converter[{weight_name}]")
        verify_function(ir)
        self.add_converter(weight_name, ir)
        return ir

    return _wrap_converter


HirFunction.converter = _converter


def prim_func(fn=None, *, target=None):
    """Decorator: parse a ``@prim_func`` function into a ``tir.PrimFunction``.

    The decorated name binds to the resulting IR node. ``target`` (a Target
    object) selects the compilation target; omitted, it uses the
    normal compile-entry default.
    """
    if target is not None:
        target_instance(target)
    resolved_target = target

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
