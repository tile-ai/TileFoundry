"""Runtime loader — turn a ``LinkedModule`` into a callable ``RuntimeModule``."""
from __future__ import annotations

from tilefoundry.runtime.function import CompiledFunction
from tilefoundry.runtime.module import RuntimeModule


class _LoadedModule(RuntimeModule):
    """One compiled entry as a ``RuntimeModule``: ``forward`` delegates to the
    bound ``CompiledFunction`` (weights are ordinary entry arguments on the
    compiled path, so ``load`` is the inherited no-op)."""

    def __init__(self, name: str, fn: CompiledFunction) -> None:
        super().__init__(name=name, entry=name)
        self.fn = fn

    def forward(self, *args):
        return self.fn(*args)


def load_linked_module(linked: "LinkedModule") -> RuntimeModule:
    """Load *linked*'s shared library and bind its entry into a ``RuntimeModule``."""
    # noqa lazy: tvm_ffi is an optional runtime dep, imported at load time only.
    import tvm_ffi  # noqa: PLC0415

    loaded = tvm_ffi.load_module(str(linked.library_path))
    try:
        entry_callable = getattr(loaded, linked.entry.name)
    except AttributeError as e:
        raise RuntimeError(
            f"load_linked_module: library {linked.library_path} has no "
            f"symbol {linked.entry.name!r}"
        ) from e
    return _LoadedModule(
        linked.entry.name,
        CompiledFunction(type=linked.entry, entry=entry_callable),
    )


__all__ = ["load_linked_module"]
