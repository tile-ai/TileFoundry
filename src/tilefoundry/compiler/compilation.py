"""The ``build`` and ``compile`` verbs and the module normalisation they share.

All accept ``Module`` exclusively.
"""

from __future__ import annotations

import tempfile

from tilefoundry import codegen, dump, passes, runtime
from tilefoundry import target as tilefoundry_target
from tilefoundry.ir import core
from tilefoundry.target import cpu

from .options import Options


def normalize_to_module(fn_or_mod: core.Function | core.Module) -> core.Module:
    """Normalise a ``Function`` or ``Module`` into a compile-ready ``Module``.

    - ``Function`` → the implicit single-function ``Module`` that owns it. It
      declares no execution context: a Function that needs one is authored
      inside the ``Module`` that declares it.
    - ``Module`` → validated and returned as the compile unit.

    Raises ``TypeError`` for unsupported input types.
    """
    if isinstance(fn_or_mod, core.Function):
        return core.Module(
            name=fn_or_mod.name,
            functions=(fn_or_mod,),
            entry=fn_or_mod.name,
        )
    if isinstance(fn_or_mod, core.Module):
        if fn_or_mod.entry is not None:
            fn_or_mod.entry_function()
        return fn_or_mod
    raise TypeError(
        f"normalize_to_module: expected Function or Module, got {type(fn_or_mod).__name__}"
    )


def resolve_options(
    operation: str,
    mod: core.Module,
    target: tilefoundry_target.Target | None,
    options: Options | None,
) -> Options:
    """The options of one compilation: *options*, or the defaults for the resolved target.

    The target is *target*, else the Module's, else ``options.target``, else the host CPU
    for ``jit``.
    A *target* or Module target that conflicts with ``options.target`` raises.
    """
    if target is not None:
        target = tilefoundry_target.instance(target)
    else:
        try:
            target = mod.resolve_target()
        except ValueError:
            if options is not None:
                target = options.target
            elif operation == "jit":
                target = cpu.Target()
            else:
                raise
    if options is None:
        return Options(target=target)
    if options.target != target:
        raise ValueError(
            f"tilefoundry.{operation}: options target {tilefoundry_target.summary(options.target)} "
            f"conflicts with the resolved Target {tilefoundry_target.summary(target)}"
        )
    return options


def preflight(mod: core.Module) -> None:
    """Check the authored namespace before lowering or native compilation."""
    implementation = tilefoundry_target.bindings.get((type(mod.target), runtime.CompiledModule), runtime.CompiledModule)
    runtime.validate_names((*[function.name for function in mod.functions], *mod.children), implementation, mod.name)
    if mod.entry is None and mod.children:
        raise ValueError(f"{mod.name}: compiled module tree is not decoded yet")
    for child in mod.children.values():
        for node in child if isinstance(child, tuple) else (child,):
            preflight(node)


def build_module(mod: core.Module, *, options: Options, declaration: dict) -> runtime.CompiledModule:
    """Link a lowered root using the declaration captured before transformation."""
    if mod.entry is not None:
        entry = mod.entry_function()
        if not isinstance(entry, core.PrimFunction):
            raise ValueError(f"tilefoundry.build: {mod.name}.{entry.name}: entry requires lowering")
        if not isinstance(entry.target, cpu.Target) and entry.target != options.target:
            raise ValueError(f"tilefoundry.build: {mod.name}.{entry.name}: entry Target conflicts with the root Target")
    workdir = tempfile.mkdtemp(prefix=f"tilefoundry_build_{mod.name}_")
    return runtime.load_linked_module(codegen.link_program(
        mod, workdir=workdir, options=options.codegen, declaration=declaration,
    ))


def build(
    mod: core.Module,
    /,
    *,
    target: tilefoundry_target.Target | None = None,
    options: Options | None = None,
) -> "runtime.CompiledModule":
    """Build a lowered root and bind its standalone compiled library."""
    if not isinstance(mod, core.Module):
        raise TypeError(f"tilefoundry.build: expected Module, got {type(mod).__name__}.")
    try:
        module_target = mod.resolve_target()
    except ValueError as error:
        raise ValueError("tilefoundry.build: module has no Target; declare one on the Module") from error
    if target is not None and tilefoundry_target.instance(target) != module_target:
        raise ValueError(f"tilefoundry.build: explicit target {tilefoundry_target.summary(target)} conflicts with the Module Target {tilefoundry_target.summary(module_target)}")
    options = resolve_options("build", mod, target, options)
    preflight(mod)
    return build_module(mod, options=options, declaration=codegen.artifact.describe_root(mod))


def compile(
    mod: core.Module,
    /,
    *,
    target: tilefoundry_target.Target | None = None,
    options: Options | None = None,
) -> "runtime.CompiledModule":
    """Capture the root declaration, lower its entry and build its library."""
    if not isinstance(mod, core.Module):
        raise TypeError(f"tilefoundry.compile: expected Module, got {type(mod).__name__}.")
    options = resolve_options("compile", mod, target, options)
    preflight(mod)
    declaration = codegen.artifact.describe_root(mod, target=options.target)
    with dump.Scope(flags=options.dump):
        prepared = passes.default_pipeline(mod, options.passes).run(mod) if mod.entry is not None else mod
        return build_module(prepared, options=options, declaration=declaration)


__all__ = ["build", "compile", "normalize_to_module", "resolve_options"]
