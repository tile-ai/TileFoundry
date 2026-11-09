"""Load compiled entries while retaining their libraries until process exit.

Global registrations outlive individual runtime modules, so the loader keeps
all loaded code alive for the full lifetime of those process-global names.
"""

from __future__ import annotations

from dataclasses import replace

from tilefoundry import utils

from .module import CompiledModule

tvm_ffi = utils.LazyModule("tvm_ffi")


class Libraries:
    """The shared libraries loaded so far, kept alive for the life of the process."""

    def __init__(self) -> None:
        self.loaded: list = []

    def keep(self, library) -> None:
        """Hold *library* so its global registrations stay valid."""
        self.loaded.append(library)


libraries = Libraries()


def load_linked_module(linked: "LinkedModule") -> CompiledModule:
    """Load *linked*'s shared library and bind its entry into a ``CompiledModule``."""
    entry = linked.functions[0]
    loaded = tvm_ffi.load_module(str(linked.library_path))
    libraries.keep(loaded)
    try:
        entry_callable = tvm_ffi.get_global_func(entry.name)
    except ValueError as e:
        raise RuntimeError(
            f"load_linked_module: library {linked.library_path} has no registration {entry.name!r}"
        ) from e
    return CompiledModule(type=replace(entry.signature, name=entry.name), fn=entry_callable)


__all__ = ["load_linked_module"]
