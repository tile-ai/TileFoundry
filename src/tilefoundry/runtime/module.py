"""``RuntimeModule`` — the runtime twin of an ir ``Module``, authored like a
``torch.nn.Module`` (subclass; build children in ``__init__``; write the step
in ``forward``; resolve weights in ``load``). ``CompiledModule`` is the
compiled-path ``RuntimeModule`` the loader binds. See docs/spec/runtime.md §1.1.
"""
from __future__ import annotations

from typing import Callable

from tilefoundry.runtime.function import EntryABI
from tilefoundry.runtime.resource import RuntimeResource

__all__ = ["CompiledModule", "RuntimeModule"]


class RuntimeModule:
    """Base runtime module: explicit child registration + recursive load.

    Subclasses override ``forward`` (the computation/orchestration — kernel
    implementations are ``RuntimeFunction`` attributes called from it) and
    ``load`` (own weight/state resolution, ending with ``super().load``).
    """

    def __init__(
        self, name: str, entry: str | None = None, modules: tuple["RuntimeModule", ...] = ()
    ) -> None:
        # ``entry`` mirrors the ir Module's entry name (metadata for the
        # correspondence; ``forward`` is what actually runs the step).
        self.name = name
        self.entry = entry
        self.modules = tuple(modules)

    def forward(self, *args):
        raise NotImplementedError(
            f"RuntimeModule {self.name!r}: subclass must implement forward()"
        )

    def __call__(self, *args):
        return self.forward(*args)

    def load(self, resource: RuntimeResource) -> None:
        """Recurse ``load`` into every child under its own name prefix.

        A subclass resolves its own tensors first, then calls
        ``super().load(resource)``; the base itself owns nothing.
        """
        for child in self.modules:
            child.load(resource.subtree(child.name))


class CompiledModule(RuntimeModule):
    """One compiled entry as a ``RuntimeModule`` (produced by
    ``runtime.loader.load_linked_module``). Weights are ordinary entry
    arguments on the compiled path, so ``load`` is the inherited no-op and
    ``modules`` is always empty.

    ``forward`` is the one place output allocation happens (the trailing
    ``type.output_count`` params are the outputs): ``rm(x)`` allocates them and
    returns them, ``rm(x, out)`` writes into the ones given.
    """

    def __init__(self, type: EntryABI, fn: Callable) -> None:
        super().__init__(name=type.name, entry=type.name)
        self.type = type
        self.fn = fn

    def forward(self, *args):
        n_in = self.type.input_count
        if len(args) == len(self.type.params):
            outs = args[n_in:]
            self.fn(*args)
        elif len(args) == n_in:
            outs = self._alloc_outputs(args)
            self.fn(*args, *outs)
        else:
            raise TypeError(
                f"{self.type.name}: expected {n_in} inputs (auto-alloc) or "
                f"{len(self.type.params)} inputs+outputs (pre-alloc), got {len(args)}"
            )
        return outs[0] if len(outs) == 1 else tuple(outs)

    def _alloc_outputs(self, args) -> tuple:
        """Empty output tensors per ``output_params``, on the inputs' device."""
        import torch  # noqa: PLC0415 -- optional runtime dep, needed only to allocate

        from tilefoundry.evaluator.value import to_torch_dtype  # noqa: PLC0415

        device = next((a.device for a in args if isinstance(a, torch.Tensor)), None)
        if device is None:
            raise TypeError(
                f"{self.type.name}: cannot infer device for auto-alloc; "
                f"no torch.Tensor in inputs"
            )
        return tuple(
            torch.empty(p.type.shape, dtype=to_torch_dtype(p.type.dtype), device=device)
            for p in self.type.output_params
        )
