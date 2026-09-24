"""Provide runtime twins of IR modules.

``CompiledModule`` is the compiled-path variant bound by the loader. See
[runtime §1.1](docs/spec/runtime.md#11-runtimemodulepy).
"""

from __future__ import annotations

from typing import Callable

from tilefoundry.codegen.signature import CallableSignature
from tilefoundry.ir.types import Placement
from tilefoundry.runtime.resource import RuntimeResource

__all__ = ["CompiledModule", "RuntimeModule"]


class RuntimeModule:
    """Base runtime module: explicit child registration + recursive load.

    Subclasses override ``forward`` (the step) and ``load`` (own weights,
    ending with ``super().load``).
    """

    name: str
    """Mirrors the IR ``Module`` node name."""

    entry: str | None
    """Mirrors the IR ``Module`` entry."""

    modules: tuple["RuntimeModule", ...]
    """The children, registered explicitly in ``__init__``."""

    def __init__(
        self, name: str, entry: str | None = None, modules: tuple["RuntimeModule", ...] = ()
    ) -> None:
        self.name = name
        self.entry = entry
        self.modules = tuple(modules)

    @property
    def module(self):
        """Module.

        The authored ``Module`` this twin stands for, or ``None`` when there is
        no such thing to name -- a compiled entry or a hand-written subclass.
        """
        return None

    def forward(self, *args):
        raise NotImplementedError(f"RuntimeModule {self.name!r}: subclass must implement forward()")

    def __call__(self, *args):
        return self.forward(*args)

    def load(self, resource: RuntimeResource, *, placement: "Placement | None" = None) -> None:
        """Recurse ``load`` into every child under its own name prefix.

        Recurse ``load`` into every child under its own name prefix. Weight
        values are read lazily by each runtime function on first use, and
        *placement* -- which program of the mesh this process is -- reaches
        every child unchanged, because one process is one program throughout
        the tree it loaded.
        """
        for child in self.modules:
            child.load(resource.subtree(child.name), placement=placement)


class CompiledModule(RuntimeModule):
    """One compiled entry as a ``RuntimeModule``.

    One compiled entry as a ``RuntimeModule``: ``load`` is the inherited
    no-op and ``modules`` is empty (weights are ordinary entry args).
    ``forward``: ``rm(x)`` allocates the trailing ``output_count`` outputs;
    ``rm(x, out)`` writes into the ones given.
    """

    def __init__(self, type: CallableSignature, fn: Callable) -> None:
        super().__init__(name=type.name, entry=type.name)
        self.type = type
        self.fn = fn
        self._placement: "Placement | None" = None

    def load(self, resource: RuntimeResource, *, placement: "Placement | None" = None) -> None:
        """Remember which program this is; a compiled entry holds no weights."""
        self._placement = placement
        super().load(resource, placement=placement)

    def _program_ids(self) -> tuple:
        """The ids the entry needs told, ahead of its own arguments.

        A level no register on the device answers for has to be told, so its
        id travels with the call; each leading parameter states which level it
        carries, which is how a ``Placement`` is keyed. A program whose levels
        are all answered on the device is called as it was written.
        """
        if not self.type.leading:
            return ()
        topology_levels = tuple(p.topology_level for p in self.type.leading)
        if self._placement is None:
            raise ValueError(
                f"{self.type.name}: needs the ids of {topology_levels} and no Placement "
                f"says which program this is; pass placement= to load()"
            )
        return tuple(int(self._placement.ids[topology_level]) for topology_level in topology_levels)

    def forward(self, *args):
        leading = self._program_ids()
        n_in = self.type.input_count
        if len(args) == len(self.type.params):
            outs = args[n_in:]
            self.fn(*leading, *args)
        elif len(args) == n_in:
            outs = self._alloc_outputs(args)
            self.fn(*leading, *args, *outs)
        else:
            raise TypeError(
                f"{self.type.name}: expected {n_in} inputs (auto-alloc) or "
                f"{len(self.type.params)} inputs+outputs (pre-alloc), got {len(args)}"
            )
        return outs[0] if len(outs) == 1 else tuple(outs)

    def _alloc_outputs(self, args) -> tuple:
        import torch  # noqa: PLC0415

        from tilefoundry.evaluator.value import to_torch_dtype  # noqa: PLC0415

        device = next((a.device for a in args if isinstance(a, torch.Tensor)), None)
        if device is None:
            raise TypeError(
                f"{self.type.name}: cannot infer device for auto-alloc; no torch.Tensor in inputs"
            )
        return tuple(
            torch.empty(p.type.shape, dtype=to_torch_dtype(p.type.dtype), device=device)
            for p in self.type.output_params
        )
