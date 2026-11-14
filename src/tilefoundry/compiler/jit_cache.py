"""The ``jit`` entry: compile once per module text and options, then reuse."""

from __future__ import annotations

import hashlib

from tilefoundry import inspection, runtime
from tilefoundry.analysis import communication
from tilefoundry.interfaces import valuation
from tilefoundry.ir import core, expr

from .compilation import compile, normalize_to_module, resolve_options
from .options import Options


def canonical_module_text(mod: core.Module) -> str:
    """Include every emitted definition, physical binding and execution domain.

    Callee changes can alter buffer effects even when the entry is unchanged.
    """
    fn_text = inspection.as_script(mod)
    for path, frame in valuation.collect_functions(mod).items():
        bindings = communication.buffer_regions(frame.function)
        selected = expr.get_metadata(frame.function, communication.BufferBindings)
        if selected is not None:
            fn_text += "\n" + path.qualified_name + ": " + repr((tuple(sorted(bindings.items())), tuple(sorted(selected.uninitialized))))

    topologies = mod.effective_topologies()
    if topologies:
        topo_lines = []
        for t in sorted(topologies, key=lambda t: t.name):
            topo_lines.append(f"Topology({t.name!r}, {t.size})")
        fn_text += "\n" + "\n".join(topo_lines)
    return fn_text


def cache_key(mod: core.Module, options: Options) -> str:
    """The key of *mod* compiled with *options*."""
    payload = canonical_module_text(mod) + "\0" + repr(options)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Jit:
    """Compile a ``hir.Function`` or ``Module`` once per module text and options.

    The cache key is the canonical module text and the ``repr`` of the options,
    which names every stage's option -- no Python object identity participates.
    """

    def __init__(self) -> None:
        self.cache: dict[str, runtime.Module] = {}

    def __call__(
        self,
        fn_or_mod,
        /,
        *,
        target=None,
        options: Options | None = None,
        **kwargs,
    ) -> "runtime.Module":
        """JIT-compile *fn_or_mod* to a ``runtime.Module``.

        A ``Module`` is the compilation unit; a Function becomes a context-free
        single-function Module, so execution context belongs to its owner Module.
        """
        if kwargs:
            bad = ", ".join(kwargs.keys())
            raise TypeError(
                f"tilefoundry.jit: unexpected keyword argument(s): {bad}. "
                f"Accepted parameters are: fn_or_mod, target, options."
            )

        if not isinstance(fn_or_mod, (core.Function, core.Module)):
            raise TypeError(
                f"tilefoundry.jit: expected Function or Module, got {type(fn_or_mod).__name__}"
            )

        mod = normalize_to_module(fn_or_mod)
        options = resolve_options("jit", mod, target, options)
        key = cache_key(mod, options)
        if key not in self.cache:
            self.cache[key] = compile(mod, options=options)
        return self.cache[key]

    def cache_clear(self) -> None:
        """Clear the jit cache (for testing)."""
        self.cache.clear()

    def cache_info(self) -> dict:
        """Return cache stats dict."""
        return {"size": len(self.cache)}


jit = Jit()

__all__ = ["jit"]
