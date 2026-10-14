"""Top-level ``tilefoundry.build`` / ``tilefoundry.compile`` entries.

Three public verbs, all accept ``Module`` exclusively.
"""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass

from tilefoundry.codegen.link_module import link_program
from tilefoundry.inspection import as_script as _as_script
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.passes.pass_manager import PassManager
from tilefoundry.passes.transforms import InsertHostEntryPass
from tilefoundry.runtime.loader import load_linked_module
from tilefoundry.target import Target, default_target
from tilefoundry.target.base import _target_summary, target_instance


@dataclass(frozen=True)
class CompilerOptions:
    """Minimal compiler options for cache-key and compile configuration.

    Serialisation is deterministic: ``target`` + extra fields as
    sorted key-value pairs join by null separator.
    """

    target: Target

    def __post_init__(self) -> None:
        target_instance(self.target)

    def canonical_text(self) -> str:
        """Deterministic text serialisation for cache-key computation."""
        parts = [f"target={self.target!r}"]
        return "\0".join(parts)


def normalize_to_module(fn_or_mod: HirFunction | Module) -> Module:
    """Normalise a ``Function`` or ``Module`` into a compile-ready ``Module``.

    - ``Function`` → the implicit single-function ``Module`` that owns it. It
      declares no execution context: a Function that needs one is authored
      inside the ``Module`` that declares it.
    - ``Module`` → validated and returned as the compile unit.

    Raises ``TypeError`` for unsupported input types.
    """
    if isinstance(fn_or_mod, HirFunction):
        return Module(
            name=fn_or_mod.name,
            functions=(fn_or_mod,),
            entry=fn_or_mod.name,
        )
    if isinstance(fn_or_mod, Module):
        fn_or_mod.entry_function()
        return fn_or_mod
    raise TypeError(
        f"normalize_to_module: expected Function or Module, got {type(fn_or_mod).__name__}"
    )


def _build_default_pipeline() -> PassManager:
    pm = PassManager()
    pm.add(InsertHostEntryPass())
    return pm


def build(
    mod: Module,
    /,
    *,
    target: Target | None = None,
) -> "RuntimeModule":
    """Codegen + compile + load *mod* and return a fully-loaded ``RuntimeModule``.

    *mod* must be a ``Module``. *target* defaults to ``mod.metadata["target"]``.
    Raises ``ValueError`` if missing or if explicit *target* conflicts.
    """
    if not isinstance(mod, Module):
        raise TypeError(f"tilefoundry.build: expected Module, got {type(mod).__name__}.")
    if target is not None:
        target = target_instance(target)
    try:
        module_target = mod.resolve_target()
    except ValueError as error:
        raise ValueError(
            "tilefoundry.build: module has no Target; declare one on the Module"
        ) from error
    if target is not None and target != module_target:
        raise ValueError(
            f"tilefoundry.build: explicit target {_target_summary(target)} "
            f"conflicts with the Module Target {_target_summary(module_target)}"
        )

    workdir = tempfile.mkdtemp(prefix=f"tilefoundry_build_{mod.entry}_split_")
    return load_linked_module(link_program(mod, workdir=workdir))


def compile(
    mod: Module,
    /,
    *,
    target: Target | None = None,
) -> "RuntimeModule":
    """``build(mod, target=target)`` -- the full compile entry.

    *mod* must be a ``Module``.  Meshes are derived from the IR body.
    """
    prepared = _build_default_pipeline().run(mod)
    return build(prepared, target=target)


def _canonical_module_text(mod: Module) -> str:
    """Produce canonical text for cache-key: entry-function source + topologies.

    Entry source and declared topologies distinguish independently compiled
    execution domains without attaching parent context to shared definitions.
    """
    fn_text = _as_script(mod.entry_function())

    topologies = mod.effective_topologies()
    if topologies:
        topo_lines = []
        for t in sorted(topologies, key=lambda t: t.name):
            topo_lines.append(f"Topology({t.name!r}, {t.size})")
        fn_text += "\n" + "\n".join(topo_lines)
    return fn_text


def jit(
    fn_or_mod,
    /,
    *,
    target: Target | None = None,
    options: CompilerOptions | None = None,
    **kwargs,
) -> "RuntimeModule":
    """JIT-compile a ``hir.Function`` or ``Module`` to a ``RuntimeModule``.

    A ``Module`` is the compilation unit; a Function becomes a context-free
    single-function Module, so execution context belongs to its owner Module.
    The cache key is canonical module text, Target text, and options text -- no
    Python object identity participates.
    """
    if kwargs:
        bad = ", ".join(kwargs.keys())
        raise TypeError(
            f"tilefoundry.jit: unexpected keyword argument(s): {bad}. "
            f"Accepted parameters are: fn_or_mod, target, options."
        )

    if not isinstance(fn_or_mod, (HirFunction, Module)):
        raise TypeError(
            f"tilefoundry.jit: expected Function or Module, got {type(fn_or_mod).__name__}"
        )

    mod = normalize_to_module(fn_or_mod)

    if target is None:
        try:
            target = mod.resolve_target()
        except ValueError:
            target = default_target()
    else:
        target = target_instance(target)
    if options is None:
        options = CompilerOptions(target=target)
    elif options.target != target:
        raise ValueError(
            f"tilefoundry.jit: options target {_target_summary(options.target)} "
            f"conflicts with the resolved Target {_target_summary(target)}"
        )

    canonical_text = _canonical_module_text(mod)
    payload = canonical_text + "\0" + repr(target) + "\0" + options.canonical_text()
    key = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    if key not in _jit_cache:
        _jit_cache[key] = compile(mod, target=target)
    return _jit_cache[key]


_jit_cache: dict[str, "RuntimeModule"] = {}


def _jit_cache_clear() -> None:
    """Clear the jit cache (for testing)."""
    _jit_cache.clear()


def _jit_cache_info() -> dict:
    """Return cache stats dict."""
    return {"size": len(_jit_cache)}


jit.cache_clear = _jit_cache_clear  # type: ignore[attr-defined]
jit.cache_info = _jit_cache_info  # type: ignore[attr-defined]


def _jit_cache_key_payload(
    fn_or_mod: HirFunction | Module,
    target: Target | None = None,
    options: CompilerOptions | None = None,
) -> tuple[str, str, str]:
    r"""For testing only: return ``(module_text, target_repr, options_text)``.

    The actual cache key is ``sha256(text + "\0" + target + "\0" + opts)``.
    """
    mod = normalize_to_module(fn_or_mod)
    if target is None:
        try:
            target = mod.resolve_target()
        except ValueError:
            target = default_target()
    else:
        target = target_instance(target)
    if options is None:
        options = CompilerOptions(target=target)
    elif options.target != target:
        raise ValueError("CompilerOptions Target conflicts with cache-key Target")
    return (
        _canonical_module_text(mod),
        repr(target),
        options.canonical_text(),
    )


__all__ = [
    "build",
    "compile",
    "jit",
    "normalize_to_module",
    "CompilerOptions",
]
