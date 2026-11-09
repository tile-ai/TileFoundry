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


def build(
    mod: core.Module,
    /,
    *,
    target: tilefoundry_target.Target | None = None,
    options: Options | None = None,
) -> "runtime.Module":
    """Codegen + compile + load *mod* and return a fully-loaded ``runtime.Module``.

    *mod* must be a ``Module``. *target* defaults to ``mod.metadata["target"]``.
    Raises ``ValueError`` if missing or if explicit *target* conflicts. *options*
    gives the codegen stage its part.
    """
    if not isinstance(mod, core.Module):
        raise TypeError(f"tilefoundry.build: expected Module, got {type(mod).__name__}.")
    if target is not None:
        target = tilefoundry_target.instance(target)
    try:
        module_target = mod.resolve_target()
    except ValueError as error:
        raise ValueError(
            "tilefoundry.build: module has no Target; declare one on the Module"
        ) from error
    if target is not None and target != module_target:
        raise ValueError(
            f"tilefoundry.build: explicit target {tilefoundry_target.summary(target)} "
            f"conflicts with the Module Target {tilefoundry_target.summary(module_target)}"
        )

    entry = mod.entry_function()
    if not (isinstance(entry, core.PrimFunction) and isinstance(entry.target, cpu.Target)):
        raise ValueError(f"tilefoundry.build: Module.entry {entry.name!r} is not a CPU host entry")
    workdir = tempfile.mkdtemp(prefix=f"tilefoundry_build_{mod.entry}_split_")
    codegen_options = codegen.Options() if options is None else options.codegen
    return runtime.load_linked_module(
        codegen.link_program(mod, workdir=workdir, options=codegen_options)
    )


def compile(
    mod: core.Module,
    /,
    *,
    target: tilefoundry_target.Target | None = None,
    options: Options | None = None,
) -> "runtime.Module":
    """Run the shared pass pipeline, then build the resulting Module.

    *mod* must be a ``Module``.  Meshes are derived from the IR body. Each stage
    receives its part of *options*.
    """
    options = resolve_options("compile", mod, target, options)
    with dump.Scope(flags=options.dump):
        prepared = passes.default_pipeline(mod, options.passes).run(mod)
        return build(prepared, target=target, options=options)


__all__ = ["build", "compile", "normalize_to_module", "resolve_options"]
