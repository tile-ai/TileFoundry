"""Define parser-backed ``@func`` and ``@prim_func`` decorators.

The surface follows [parser §1](docs/spec/parser.md#1-dsl-syntax). A decorator
returns the parsed and verified IR node, not the original Python function.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, ClassVar, Literal

from tilefoundry.ir.core.module import Module
from tilefoundry.ir.core.pattern import DimVarRangePat, Pattern
from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.hir.specialize import DISPLAY_NAME
from tilefoundry.ir.hir.verify import verify_function
from tilefoundry.ir.tir.intrinsic import intrinsic as _intrinsic
from tilefoundry.ir.tir.verify import verify_prim_function
from tilefoundry.module import UNDECLARED, _Entry, enclosing_declaration
from tilefoundry.parser import parse_prim_func
from tilefoundry.parser.hir_parser import _parse_func
from tilefoundry.target.base import target_instance


class ParsedFuncKind(StrEnum):
    """The three parser-time Function roles."""

    KERNEL = "kernel"
    VARIANT = "variant"
    CONVERTER = "converter"


@dataclass(frozen=True)
class NamingRule:
    """Binding-name constraints for one parser-time Function role."""

    binding_unique: bool
    allow_underscore: bool

    def check(
        self,
        binding_name: str,
        entry: _Entry | None,
        *,
        kind: ParsedFuncKind,
        base_name: str | None = None,
    ) -> None:
        scope = _binding_scope(kind, entry, base_name)
        if not self.allow_underscore and binding_name == "_":
            raise ValueError(f"{scope}: a {kind.value} binding may not be named '_'")
        if self.binding_unique and entry is not None and binding_name in entry.bound:
            raise ValueError(f"{scope}: duplicate {kind.value} binding {binding_name!r}")


@dataclass(frozen=True)
class HandleRule:
    """Handle construction and uniqueness scope for one Function role."""

    of: Callable[[object, object], str]
    unique_within: Literal["module", "base"]

    def check(
        self,
        kind: ParsedFuncKind,
        fn: object,
        key: object,
        entry: _Entry | None,
        base: object | None,
    ) -> None:
        if self.unique_within == "module":
            handle = self.of(fn, key)
            if entry is not None and entry.bound.get(handle) is kind:
                scope = _binding_scope(kind, entry, None)
                raise ValueError(f"{scope}: duplicate {kind.value} binding {handle!r}")
            return
        if base is None:
            return
        handle = self.of(base, key)
        if kind is ParsedFuncKind.VARIANT:
            keys = (variant.specializations[0] for variant in getattr(base, "variants", ()))
        else:
            keys = (weight_name for weight_name, _ in getattr(base, "converters", ()))
        if any(self.of(base, existing_key) == handle for existing_key in keys):
            scope = _binding_scope(kind, entry, base.name)
            raise ValueError(f"{scope}: duplicate {kind.value} handle {handle!r}")


class ParsedFuncRules:
    """Parser-time naming and handle rules for parsed Functions."""

    NAMING: ClassVar[dict[ParsedFuncKind, NamingRule]] = {
        ParsedFuncKind.KERNEL: NamingRule(binding_unique=True, allow_underscore=False),
        ParsedFuncKind.VARIANT: NamingRule(binding_unique=True, allow_underscore=False),
        ParsedFuncKind.CONVERTER: NamingRule(binding_unique=False, allow_underscore=True),
    }

    HANDLE: ClassVar[dict[ParsedFuncKind, HandleRule]] = {
        ParsedFuncKind.KERNEL: HandleRule(lambda fn, key: fn.name, "module"),
        ParsedFuncKind.VARIANT: HandleRule(
            lambda fn, key: f"{fn.name}${key.dim_var}${key.lo}_{key.hi}", "base"
        ),
        ParsedFuncKind.CONVERTER: HandleRule(
            lambda fn, key: f"{fn.name}.converter[{key}]", "base"
        ),
    }

    @classmethod
    def check(
        cls,
        kind: ParsedFuncKind,
        ir: object,
        binding_name: str,
        key: object,
        *,
        entry: _Entry | None,
        base: object | None = None,
    ) -> None:
        cls.HANDLE[kind].check(kind, ir, key, entry, base)
        cls.NAMING[kind].check(
            binding_name,
            entry,
            kind=kind,
            base_name=getattr(base, "name", None),
        )


def _binding_scope(
    kind: ParsedFuncKind, entry: _Entry | None, base_name: str | None
) -> str:
    if entry is not None:
        module_name = entry.owner_name or "<module>"
        if kind is ParsedFuncKind.VARIANT:
            return f"@module {module_name!r} base {base_name!r}"
        return f"@module {module_name!r}"
    if kind is ParsedFuncKind.VARIANT:
        return f"base {base_name!r}"
    return "standalone @func"


def _register(
    kind: ParsedFuncKind,
    ir: HirFunction,
    binding_name: str,
    key: object,
    *,
    base: HirFunction | None = None,
) -> None:
    """Validate and record a parser-time Function in its enclosing Module."""
    entry = _enclosing_declaration()
    ParsedFuncRules.check(kind, ir, binding_name, key, entry=entry, base=base)
    if entry is None:
        return
    entry.bound[binding_name] = kind
    if kind is not ParsedFuncKind.KERNEL:
        entry.owned.add(id(ir))


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


def _enclosing_declaration():
    """The ``@module`` declaration whose class body encloses this decorator."""
    frame = sys._getframe(1)
    here = __file__
    while frame is not None and frame.f_code.co_filename == here:
        frame = frame.f_back
    return enclosing_declaration(frame)


def _enclosing_topologies() -> tuple | None:
    """Find the declaration belonging to the enclosing ``@module`` class body."""
    entry = _enclosing_declaration()
    return entry.topologies if entry is not None else None


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
        ParsedFuncRules.NAMING[ParsedFuncKind.KERNEL].check(
            fn_inner.__name__,
            _enclosing_declaration(),
            kind=ParsedFuncKind.KERNEL,
        )
        extra_closure = _definition_namespace()
        parse_topologies = declared_topologies
        if parse_topologies is None:
            parse_topologies = _enclosing_topologies()
        ir = _parse_func(
            fn_inner, topologies=parse_topologies or (),
            extra_closure=extra_closure,
            in_module_body=_enclosing_declaration() is not None,
        )
        verify_function(ir)
        _register(ParsedFuncKind.KERNEL, ir, fn_inner.__name__, None)
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
    ``base.variants``. The identifier becomes the variant's display label and the
    variant's ``name`` is the base's either way.
    Legal only before ``base`` enters a ``Module`` (a later call raises).
    """
    pat = _validate_one_pattern(pattern)

    def _wrap_variant(fn_inner):
        ParsedFuncRules.NAMING[ParsedFuncKind.VARIANT].check(
            fn_inner.__name__,
            _enclosing_declaration(),
            kind=ParsedFuncKind.VARIANT,
            base_name=self.name,
        )
        extra_closure = _definition_namespace()
        ir = _parse_func(
            fn_inner, topologies=_enclosing_topologies() or (),
            specializations=(pat,), extra_closure=extra_closure,
            in_module_body=_enclosing_declaration() is not None,
        )
        if ir.body is None:
            raise TypeError(
                "tilefoundry.specialize: a variant must have a real body, not "
                "`pass` (only the base prototype declares a `pass` body)"
            )

        object.__setattr__(ir, DISPLAY_NAME, fn_inner.__name__)
        object.__setattr__(ir, "name", self.name)
        verify_function(ir)
        _register(
            ParsedFuncKind.VARIANT,
            ir,
            fn_inner.__name__,
            pat,
            base=self,
        )
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
        ir = _parse_func(
            fn_inner, extra_closure=extra_closure,
            in_module_body=_enclosing_declaration() is not None,
        )
        if ir.body is None:
            raise TypeError(
                "tilefoundry.converter: a converter must have a real body, "
                "not `pass`"
            )

        object.__setattr__(ir, "name", f"{self.name}.converter[{weight_name}]")
        verify_function(ir)
        _register(
            ParsedFuncKind.CONVERTER,
            ir,
            fn_inner.__name__,
            weight_name,
            base=self,
        )
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
