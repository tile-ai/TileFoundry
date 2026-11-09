"""Define parser-backed ``@func`` and ``@prim_func`` decorators.

The surface follows [parser §2](docs/spec/parser.md#2-syntax-and-rules). A decorator
returns the parsed and verified IR node, not the original Python function.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from types import FunctionType
from typing import Any, Literal, Mapping

from tilefoundry import services, verifier
from tilefoundry import target as tilefoundry_target
from tilefoundry.ir import core, pattern, tir, types

from .ast_pattern import (
    FuncContext,
    FunctionRole,
    LexicalScope,
    module_context_for_frame,
    target_hardware_context,
)
from .module_decorator import UNDECLARED
from .parser_visitor import parse_function


def validate_one_pattern(declared: Any) -> pattern.Pattern:
    if not isinstance(declared, pattern.Pattern):
        raise TypeError(
            f"tilefoundry.specialize: pattern must be a Pattern instance, got "
            f"{type(declared).__name__}"
        )
    if not isinstance(declared, pattern.Range):
        raise TypeError(
            f"tilefoundry.specialize: only pattern.Range is supported for v0, "
            f"got {type(declared).__name__}"
        )
    return declared


def validate_converter_weight_name(base: core.Function, weight_name: str) -> None:
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
        f"tilefoundry.converter: {base.name!r} has no ConstTensor param named {weight_name!r}"
    )


def definition_namespace() -> dict[str, Any]:
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


def enclosing_declaration():
    """The ``@module`` declaration whose class body encloses this decorator."""
    frame = sys._getframe(1)
    here = __file__
    while frame is not None and frame.f_code.co_filename == here:
        frame = frame.f_back
    return module_context_for_frame(frame)


def enclosing_topologies() -> tuple | None:
    """Find the declaration belonging to the enclosing ``@module`` class body."""
    entry = enclosing_declaration()
    return entry.topologies if entry is not None else None


def parse_authored(
    fn_inner: FunctionType,
    *,
    dialect: Literal["hir", "tir"],
    role: FunctionRole,
    binding_name: str,
    base: core.Function | None = None,
    key: object | None = None,
    target: object | None = None,
    topologies: tuple | None = None,
    mesh: types.Mesh | None = None,
    output_count: int = 1,
    module_context=None,
    closure: Mapping[str, Any] | None = None,
):
    """Build one typed context and route every authored function through the API."""
    module_context = module_context if module_context is not None else enclosing_declaration()
    closure = dict(closure) if closure is not None else definition_namespace()
    if fn_inner.__globals__ is not None:
        closure.update(fn_inner.__globals__)
    if fn_inner.__closure__ is not None:
        for name, cell in zip(fn_inner.__code__.co_freevars, fn_inner.__closure__):
            try:
                closure[name] = cell.cell_contents
            except ValueError:
                pass
    use_owner_context = module_context is not None and (
        role is not FunctionRole.ROOT or (target is None and topologies is None) or dialect == "tir"
    )
    if use_owner_context:
        context = module_context.function_context(
            dialect=dialect,
            role=role,
            binding_name=binding_name,
            closure=closure,
            base=base,
            key=key,
            mesh=mesh,
            output_count=output_count,
        )
        if dialect == "tir" and target is not None:
            context = replace(context, target=target, hardware_context=target_hardware_context(target))
    else:
        topology_scope = {
            getattr(topology, "name", str(index)): topology
            for index, topology in enumerate(topologies or ())
        }
        context = FuncContext(
            dialect=dialect,
            role=role,
            closure=closure,
            topologies=topology_scope,
            module_scope=LexicalScope(),
            base=base,
            key=key,
            target=target,
            mesh=mesh,
            output_count=output_count,
            hardware_context=target_hardware_context(target),
            binding_name=binding_name,
        )
    return parse_function(fn_inner, context)


def capture_function_closure(fn_inner: FunctionType) -> dict[str, Any]:
    closure = definition_namespace()
    closure.update(fn_inner.__globals__)
    if fn_inner.__closure__ is not None:
        for name, cell in zip(fn_inner.__code__.co_freevars, fn_inner.__closure__):
            try:
                closure[name] = cell.cell_contents
            except ValueError:
                pass
    return closure


def validate_mesh(mesh: types.Mesh | None, decorator: str) -> None:
    """Share the explicit execution-mesh check across authoring decorators."""
    if mesh is not None and not isinstance(mesh, types.Mesh):
        raise TypeError(f"tilefoundry.{decorator}: mesh must be a Mesh, got {type(mesh).__name__}")


@dataclass
class DeferredFunction:
    module_context: Any
    fn_inner: FunctionType
    dialect: Literal["hir", "tir"]
    role: FunctionRole
    binding_name: str
    closure: Mapping[str, Any]
    mesh: types.Mesh | None = None
    target: object | None = None
    base: object | None = None
    key: object | None = None
    output_count: int = 1
    parsed: object | None = None
    is_deferred: bool = True

    def parse(self):
        base = self.base.parsed if isinstance(self.base, DeferredFunction) else self.base
        if self.role is FunctionRole.CONVERTER:
            validate_converter_weight_name(base, self.key)
        self.parsed = parse_authored(
            self.fn_inner,
            dialect=self.dialect,
            role=self.role,
            binding_name=self.binding_name,
            base=base,
            key=self.key,
            mesh=self.mesh,
            output_count=self.output_count,
            target=self.target,
            module_context=self.module_context,
            closure=self.closure,
        )
        if self.role is FunctionRole.VARIANT:
            self.parsed.relabel(base.name, display_name=self.binding_name)
        elif self.dialect == "hir":
            if self.role is FunctionRole.CONVERTER:
                self.parsed.relabel(f"{base.name}.converter[{self.key}]")
        return self.parsed

    def specialize(self, declared: Any, *, mesh: types.Mesh | None = None):
        pat = validate_one_pattern(declared)
        validate_mesh(mesh, "specialize")

        def wrap_variant(fn_inner):
            declaration = DeferredFunction(
                self.module_context,
                fn_inner,
                self.dialect,
                FunctionRole.VARIANT,
                fn_inner.__name__,
                capture_function_closure(fn_inner),
                base=self,
                key=pat,
                mesh=mesh,
                target=self.target,
            )
            self.module_context.declarations.append(declaration)
            return declaration

        return wrap_variant

    def converter(self, weight_name: str):
        def wrap_converter(fn_inner):
            declaration = DeferredFunction(
                self.module_context,
                fn_inner,
                "hir",
                FunctionRole.CONVERTER,
                fn_inner.__name__,
                capture_function_closure(fn_inner),
                base=self,
                key=weight_name,
            )
            self.module_context.declarations.append(declaration)
            return declaration

        return wrap_converter


def func(fn=None, *, topologies=UNDECLARED, target=None, mesh=None):
    """Decorator: parse an ``@func``-decorated function into HIR.

    Plain ``@func`` inherits its owning module's topology. Supplying a target or
    topology makes an implicit single-function module with its own execution
    domain. ``mesh`` names the function's outer execution domain in both its
    signature and body. A ``pass`` body declares a dispatch prototype whose
    implementations are registered through :meth:`Function.specialize`.
    """
    if target is not None:
        tilefoundry_target.instance(target)
    validate_mesh(mesh, "func")
    resolved_target = target
    declares_context = resolved_target is not None or topologies is not UNDECLARED
    declared_topologies = None if topologies is UNDECLARED else tuple(topologies)

    def wrap(fn_inner):
        module_context = enclosing_declaration()
        if module_context is not None and not declares_context:
            declaration = DeferredFunction(
                module_context,
                fn_inner,
                "hir",
                FunctionRole.ROOT,
                fn_inner.__name__,
                capture_function_closure(fn_inner),
                mesh=mesh,
            )
            module_context.declarations.append(declaration)
            return declaration
        ir = parse_authored(
            fn_inner,
            dialect="hir",
            role=FunctionRole.ROOT,
            binding_name=fn_inner.__name__,
            target=resolved_target,
            topologies=declared_topologies,
            mesh=mesh,
        )
        verifier.verify_function(ir)
        if not declares_context:
            return ir
        return core.Module(
            name=ir.name,
            functions=(ir,),
            entry=ir.name,
            target=resolved_target,
            topologies=declared_topologies,
        )

    if fn is not None:
        return wrap(fn)
    return wrap


def specialize_function(
    self: core.Function, declared: Any, *, mesh: types.Mesh | None = None
):
    """``@base.specialize(pattern.Range(...))`` — register a shape variant.

    Parses the decorated ``def`` into a variant ``hir.Function`` and appends it to
    ``base.variants``. The identifier becomes the variant's display label and the
    variant's ``name`` is the base's either way.
    Legal only before ``base`` enters a ``Module`` (a later call raises).
    """
    pat = validate_one_pattern(declared)
    validate_mesh(mesh, "specialize")

    def wrap_variant(fn_inner):
        dialect = "tir" if isinstance(self, core.PrimFunction) else "hir"
        ir = parse_authored(
            fn_inner,
            dialect=dialect,
            role=FunctionRole.VARIANT,
            binding_name=fn_inner.__name__,
            base=self,
            key=pat,
            topologies=enclosing_topologies(),
            mesh=mesh,
        )
        if ir.body is None:
            raise TypeError(
                "tilefoundry.specialize: a variant must have a real body, not "
                "`pass` (only the base prototype declares a `pass` body)"
            )

        if dialect == "tir" and hasattr(pat, "dim_var"):
            name = pattern.mangle_variant_name(self.name, (pat,))
        else:
            name = self.name
        ir.relabel(name, display_name=fn_inner.__name__)
        if dialect == "hir":
            verifier.verify_function(ir)
        else:
            verifier.verify_prim_function(ir)
        return ir

    return wrap_variant


services.registry.register(services.specialize, specialize_function)


def convert_function(self: core.Function, weight_name: str):
    """``@base.converter(weight_name)`` — register a per-weight offline converter.

    ``@base.converter(weight_name)`` — register a per-weight offline
    converter. See [runtime §1.3](docs/spec/runtime.md#13-runtimedecoratorpy).
    """
    validate_converter_weight_name(self, weight_name)

    def wrap_converter(fn_inner):
        ir = parse_authored(
            fn_inner,
            dialect="hir",
            role=FunctionRole.CONVERTER,
            binding_name=fn_inner.__name__,
            base=self,
            key=weight_name,
            topologies=enclosing_topologies(),
        )
        if ir.body is None:
            raise TypeError("tilefoundry.converter: a converter must have a real body, not `pass`")

        ir.relabel(f"{self.name}.converter[{weight_name}]")
        verifier.verify_function(ir)
        return ir

    return wrap_converter


services.registry.register(services.converter, convert_function)


def prim_func(fn=None, *, target=None, output_count=1):
    """Decorator: parse a ``@prim_func`` function into a ``tir.PrimFunction``.

    The decorated name binds to the resulting IR node. ``target`` (a Target
    object) selects the compilation target; omitted, it uses the
    normal compile-entry default. ``output_count`` identifies the trailing
    output parameters and defaults to one.
    """
    if target is not None:
        tilefoundry_target.instance(target)
    resolved_target = target

    def wrap(fn_inner):
        module_context = enclosing_declaration()
        if module_context is not None:
            declaration = DeferredFunction(
                module_context,
                fn_inner,
                "tir",
                FunctionRole.ROOT,
                fn_inner.__name__,
                capture_function_closure(fn_inner),
                output_count=output_count,
                target=resolved_target,
            )
            module_context.declarations.append(declaration)
            return declaration
        ir = parse_authored(
            fn_inner,
            dialect="tir",
            role=FunctionRole.ROOT,
            binding_name=fn_inner.__name__,
            target=resolved_target,
            output_count=output_count,
        )
        if module_context is None:
            verifier.verify_prim_function(ir)
        return ir

    if fn is not None:
        return wrap(fn)
    return wrap


intrinsic = tir.intrinsic.intrinsic

__all__ = ["func", "prim_func", "intrinsic"]
