"""Define parser-backed ``@func`` and ``@prim_func`` decorators.

The surface follows [parser §2](docs/spec/parser.md#2-syntax-and-rules). A decorator
returns the parsed and verified IR node, not the original Python function.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import StrEnum
from types import FunctionType
from typing import Any, Callable, ClassVar, Literal, Mapping

from tilefoundry.ir.core.module import Module
from tilefoundry.ir.core.pattern import DimVarRangePat, Pattern
from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.hir.specialize import DISPLAY_NAME
from tilefoundry.ir.hir.verify import verify_function
from tilefoundry.ir.tir.intrinsic import intrinsic as _intrinsic
from tilefoundry.ir.tir.verify import verify_prim_function
from tilefoundry.module import UNDECLARED, _Entry
from tilefoundry.parser import FuncParserContext, FunctionRole, parse_function
from tilefoundry.parser.ast_pattern import LexicalScope, module_context_for_frame
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
    """Compatibility no-op; FunctionPattern owns validation and registration."""
    return None


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
    return module_context_for_frame(frame)


def _enclosing_topologies() -> tuple | None:
    """Find the declaration belonging to the enclosing ``@module`` class body."""
    entry = _enclosing_declaration()
    return entry.topologies if entry is not None else None


def _parse_authored(
    fn_inner: FunctionType,
    *,
    dialect: Literal["hir", "tir"],
    role: FunctionRole,
    binding_name: str,
    base: HirFunction | None = None,
    key: object | None = None,
    target: object | None = None,
    topologies: tuple | None = None,
    module_context=None,
    closure: Mapping[str, Any] | None = None,
):
    """Build one typed context and route every authored function through the API."""
    module_context = module_context if module_context is not None else _enclosing_declaration()
    closure = dict(closure) if closure is not None else _definition_namespace()
    if fn_inner.__globals__ is not None:
        closure.update(fn_inner.__globals__)
    if fn_inner.__closure__ is not None:
        for name, cell in zip(fn_inner.__code__.co_freevars, fn_inner.__closure__):
            try:
                closure[name] = cell.cell_contents
            except ValueError:
                pass
    use_owner_context = module_context is not None and (
        role is not FunctionRole.ROOT or (target is None and topologies is None)
    )
    if use_owner_context:
        context = module_context.function_context(
            dialect=dialect,
            role=role,
            binding_name=binding_name,
            closure=closure,
            base=base,
            key=key,
        )
    else:
        topology_scope = {
            getattr(topology, "name", str(index)): topology
            for index, topology in enumerate(topologies or ())
        }
        context = FuncParserContext(
            dialect=dialect,
            role=role,
            closure=closure,
            topologies=topology_scope,
            module_scope=LexicalScope(),
            base=base,
            key=key,
            target=target if dialect == "tir" else None,
            binding_name=binding_name,
        )
    return parse_function(fn_inner, context)


def _capture_function_closure(fn_inner: FunctionType) -> dict[str, Any]:
    closure = _definition_namespace()
    closure.update(fn_inner.__globals__)
    if fn_inner.__closure__ is not None:
        for name, cell in zip(fn_inner.__code__.co_freevars, fn_inner.__closure__):
            try:
                closure[name] = cell.cell_contents
            except ValueError:
                pass
    return closure


@dataclass
class _DeferredFunction:
    module_context: Any
    fn_inner: FunctionType
    dialect: Literal["hir", "tir"]
    role: FunctionRole
    binding_name: str
    closure: Mapping[str, Any]
    base: object | None = None
    key: object | None = None
    parsed: object | None = None
    _tilefoundry_deferred: bool = True

    def parse(self):
        base = self.base.parsed if isinstance(self.base, _DeferredFunction) else self.base
        if self.role is FunctionRole.CONVERTER:
            _validate_converter_weight_name(base, self.key)
        self.parsed = _parse_authored(
            self.fn_inner,
            dialect=self.dialect,
            role=self.role,
            binding_name=self.binding_name,
            base=base,
            key=self.key,
            module_context=self.module_context,
            closure=self.closure,
        )
        if self.dialect == "hir":
            if self.role is FunctionRole.VARIANT:
                setattr(self.parsed, DISPLAY_NAME, self.binding_name)
                self.parsed.name = base.name
            elif self.role is FunctionRole.CONVERTER:
                self.parsed.name = f"{base.name}.converter[{self.key}]"
        return self.parsed

    def specialize(self, pattern: Any):
        pat = _validate_one_pattern(pattern)

        def _wrap_variant(fn_inner):
            declaration = _DeferredFunction(
                self.module_context,
                fn_inner,
                "hir",
                FunctionRole.VARIANT,
                fn_inner.__name__,
                _capture_function_closure(fn_inner),
                base=self,
                key=pat,
            )
            self.module_context.declarations.append(declaration)
            return declaration

        return _wrap_variant

    def converter(self, weight_name: str):
        def _wrap_converter(fn_inner):
            declaration = _DeferredFunction(
                self.module_context,
                fn_inner,
                "hir",
                FunctionRole.CONVERTER,
                fn_inner.__name__,
                _capture_function_closure(fn_inner),
                base=self,
                key=weight_name,
            )
            self.module_context.declarations.append(declaration)
            return declaration

        return _wrap_converter


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
        module_context = _enclosing_declaration()
        if module_context is not None and not declares_context:
            declaration = _DeferredFunction(
                module_context,
                fn_inner,
                "hir",
                FunctionRole.ROOT,
                fn_inner.__name__,
                _capture_function_closure(fn_inner),
            )
            module_context.declarations.append(declaration)
            return declaration
        ir = _parse_authored(
            fn_inner,
            dialect="hir",
            role=FunctionRole.ROOT,
            binding_name=fn_inner.__name__,
            target=resolved_target,
            topologies=declared_topologies,
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
        ir = _parse_authored(
            fn_inner,
            dialect="hir",
            role=FunctionRole.VARIANT,
            binding_name=fn_inner.__name__,
            base=self,
            key=pat,
            topologies=_enclosing_topologies(),
        )
        if ir.body is None:
            raise TypeError(
                "tilefoundry.specialize: a variant must have a real body, not "
                "`pass` (only the base prototype declares a `pass` body)"
            )

        setattr(ir, DISPLAY_NAME, fn_inner.__name__)
        ir.name = self.name
        verify_function(ir)
        _register(ParsedFuncKind.VARIANT, ir, fn_inner.__name__, pat, base=self)
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
        ir = _parse_authored(
            fn_inner,
            dialect="hir",
            role=FunctionRole.CONVERTER,
            binding_name=fn_inner.__name__,
            base=self,
            key=weight_name,
            topologies=_enclosing_topologies(),
        )
        if ir.body is None:
            raise TypeError(
                "tilefoundry.converter: a converter must have a real body, "
                "not `pass`"
            )

        ir.name = f"{self.name}.converter[{weight_name}]"
        verify_function(ir)
        _register(ParsedFuncKind.CONVERTER, ir, fn_inner.__name__, weight_name, base=self)
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
        module_context = _enclosing_declaration()
        if module_context is not None and resolved_target is None:
            declaration = _DeferredFunction(
                module_context,
                fn_inner,
                "tir",
                FunctionRole.ROOT,
                fn_inner.__name__,
                _capture_function_closure(fn_inner),
            )
            module_context.declarations.append(declaration)
            return declaration
        ir = _parse_authored(
            fn_inner,
            dialect="tir",
            role=FunctionRole.ROOT,
            binding_name=fn_inner.__name__,
            target=resolved_target,
        )
        verify_prim_function(ir)
        return ir

    if fn is not None:
        return _wrap(fn)
    return _wrap


intrinsic = _intrinsic

__all__ = ["func", "prim_func", "intrinsic"]
