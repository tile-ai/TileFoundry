"""Runtime loader — turn a ``LinkedModule`` into a callable ``CompiledModule``."""

from __future__ import annotations

from tilefoundry.runtime.module import CompiledModule


def load_linked_module(linked: "LinkedModule") -> CompiledModule:
    """Load *linked*'s shared library and bind its entry into a ``CompiledModule``."""
    import tvm_ffi  # noqa: PLC0415

    if linked.entry is None:
        raise ValueError("load_linked_module: linked program has no runtime entry")
    loaded = tvm_ffi.load_module(str(linked.library_path))
    try:
        entry_callable = tvm_ffi.get_global_func(linked.entry.name)
    except ValueError as e:
        raise RuntimeError(
            f"load_linked_module: library {linked.library_path} has no registration {linked.entry.name!r}"
        ) from e
    return CompiledModule(type=linked.entry, fn=entry_callable, library=loaded)


__all__ = ["load_linked_module"]
