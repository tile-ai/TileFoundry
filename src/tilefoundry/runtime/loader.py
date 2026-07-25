"""Runtime loader — turn a ``LinkedModule`` into a callable ``CompiledModule``."""
from __future__ import annotations

from tilefoundry.runtime.module import CompiledModule


def load_linked_module(linked: "LinkedModule") -> CompiledModule:
    """Load *linked*'s shared library and bind its entry into a ``CompiledModule``."""
    import tvm_ffi  # noqa: PLC0415

    loaded = tvm_ffi.load_module(str(linked.library_path))
    try:
        entry_callable = getattr(loaded, linked.entry.name)
    except AttributeError as e:
        raise RuntimeError(
            f"load_linked_module: library {linked.library_path} has no "
            f"symbol {linked.entry.name!r}"
        ) from e
    return CompiledModule(type=linked.entry, fn=entry_callable)


__all__ = ["load_linked_module"]
