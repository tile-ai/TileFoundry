"""Runtime loader — turn a ``LinkedModule`` into a callable ``RuntimeModule``."""
from __future__ import annotations

from tilefoundry.runtime.module import RuntimeFunction, RuntimeModule


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
    return RuntimeModule(
        source=linked.source,
        kernels=linked.kernels,
        launch_config=linked.launch_config,
        functions={
            linked.entry.name: RuntimeFunction(type=linked.entry, _entry=entry_callable),
        },
        entry=linked.entry.name,
    )


__all__ = ["load_linked_module"]
