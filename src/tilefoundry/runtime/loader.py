"""Load compiled entries while retaining their libraries until process exit.

Global registrations outlive individual runtime modules, so the loader keeps
all loaded code alive for the full lifetime of those process-global names.
"""

from __future__ import annotations

from dataclasses import replace

from tilefoundry.runtime.module import CompiledModule

_LIBRARIES: list = []


def load_linked_module(linked: "LinkedModule") -> CompiledModule:
    """Load *linked*'s shared library and bind its entry into a ``CompiledModule``."""
    import tvm_ffi  # noqa: PLC0415

    entry = linked.functions[0]
    loaded = tvm_ffi.load_module(str(linked.library_path))
    _LIBRARIES.append(loaded)
    try:
        entry_callable = tvm_ffi.get_global_func(entry.name)
    except ValueError as e:
        raise RuntimeError(
            f"load_linked_module: library {linked.library_path} has no registration {entry.name!r}"
        ) from e
    return CompiledModule(type=replace(entry.signature, name=entry.name), fn=entry_callable)


__all__ = ["load_linked_module"]
