"""Provide runtime twins of IR modules.

``CompiledModule`` is the compiled-path variant bound by the loader. See
[runtime §1.1](docs/spec/runtime.md#11-runtimemodulepy).
"""

from __future__ import annotations

import pickle
import shutil
from pathlib import Path
from typing import Callable

from tilefoundry import codegen, utils
from tilefoundry.interfaces import evaluator
from tilefoundry.ir import core, types

from .resource import Resource

__all__ = ["CompiledModule", "Module"]

tvm_ffi = utils.LazyModule("tvm_ffi")


class Module:
    """Base runtime module: explicit child registration + recursive load.

    Subclasses override ``forward`` (the step) and ``load`` (own weights,
    ending with ``super().load``).
    """

    name: str
    """Mirrors the IR ``Module`` node name."""

    entry: str | None
    """Mirrors the IR ``Module`` entry."""

    children: dict[str, Module | tuple[Module, ...]]
    """Instance slots, registered explicitly in ``__init__``."""

    module: "core.Module | None" = None
    """The authored ``Module`` this twin stands for, or ``None`` when there is
    no such thing to name -- a compiled entry or a hand-written subclass."""

    def __init__(
        self, name: str, entry: str | None = None,
        children: dict[str, Module | tuple[Module, ...]] | None = None,
    ) -> None:
        self.name = name
        self.entry = entry
        self.children = {} if children is None else dict(children)

    def forward(self, *args):
        raise NotImplementedError(f"runtime.Module {self.name!r}: subclass must implement forward()")

    def __call__(self, *args):
        return self.forward(*args)

    def load(self, resource: Resource, *, placement: "types.Placement | None" = None) -> None:
        """Recurse ``load`` into each slot under its key and optional index.

        Weight
        values are read lazily by each runtime function on first use, and
        *placement* -- which unit of the mesh this process is -- reaches
        every child unchanged, because one process is one unit throughout
        the tree it loaded.
        """
        for key, child in self.children.items():
            base = resource.subtree(key)
            if isinstance(child, tuple):
                for index, instance in enumerate(child):
                    instance.load(base.subtree(str(index)), placement=placement)
            else:
                child.load(base, placement=placement)


class CompiledModule(Module):
    """One compiled entry as a ``Module``.

    One compiled entry as a ``Module``: ``load`` is the inherited
    no-op and ``children`` is empty (weights are ordinary entry args).
    ``forward``: ``rm(x)`` allocates the trailing ``output_count`` outputs;
    ``rm(x, out)`` writes into the ones given.
    """

    def __init__(self, type: codegen.CallableSignature, fn: Callable, library_path: Path) -> None:
        super().__init__(name=type.name, entry=type.name)
        self.type = type
        self.fn = fn
        self.library_path = library_path
        self.placement: "types.Placement | None" = None

    @classmethod
    def create(cls, target, mod: core.Module, options) -> CompiledModule:
        """Refuse an entry that no class bound to *target* builds."""
        raise ValueError(f"tilefoundry.build: Module.entry {mod.entry!r} is not a CPU host entry")

    @classmethod
    def from_library(
        cls, path: str | Path, type: codegen.CallableSignature | None = None
    ) -> CompiledModule:
        """Load the library at *path* for the life of the process and bind its entry.

        The registration is process-global and outlives this module, so the
        library stays loaded. *type* is the entry's signature, named by its
        registration; it defaults to the one ``export_library`` wrote beside *path*.
        """
        path = Path(path)
        if type is None:
            type = pickle.loads(signature_file(path).read_bytes())
        tvm_ffi.load_module(str(path), keep_module_alive=True)
        try:
            fn = tvm_ffi.get_global_func(type.name)
        except ValueError as error:
            raise RuntimeError(f"library {path} has no registration {type.name!r}") from error
        return cls(type, fn, path)

    def export_library(self, path: str | Path) -> None:
        """Copy the library to *path* and pickle the entry signature beside it."""
        path = Path(path)
        shutil.copyfile(self.library_path, path)
        signature_file(path).write_bytes(pickle.dumps(self.type))

    def load(self, resource: Resource, *, placement: "types.Placement | None" = None) -> None:
        """Remember which unit this is; a compiled entry holds no weights."""
        self.placement = placement
        super().load(resource, placement=placement)

    def unit_ids(self) -> tuple:
        """The ids the entry needs told, ahead of its own arguments.

        A topology no register on the device answers for has to be told, so its
        id travels with the call; each leading parameter states which topology it
        carries, which is how a ``Placement`` is keyed. A program whose topologies
        are all answered on the device is called as it was written.
        """
        if not self.type.leading:
            return ()
        topologies = tuple(p.topology for p in self.type.leading)
        if self.placement is None:
            raise ValueError(
                f"{self.type.name}: needs the ids of {topologies} and no Placement "
                f"says which unit this is; pass placement= to load()"
            )
        return tuple(int(self.placement.ids[topology]) for topology in topologies)

    def forward(self, *args):
        leading = self.unit_ids()
        n_in = self.type.input_count
        if len(args) == len(self.type.params):
            outs = args[n_in:]
            self.fn(*leading, *args)
        elif len(args) == n_in:
            outs = self.alloc_outputs(args)
            self.fn(*leading, *args, *outs)
        else:
            raise TypeError(
                f"{self.type.name}: expected {n_in} inputs (auto-alloc) or "
                f"{len(self.type.params)} inputs+outputs (pre-alloc), got {len(args)}"
            )
        return outs[0] if len(outs) == 1 else tuple(outs)

    def alloc_outputs(self, args) -> tuple:
        device = next((a.device for a in args if isinstance(a, utils.torch.Tensor)), None)
        if device is None:
            raise TypeError(
                f"{self.type.name}: cannot infer device for auto-alloc; no torch.Tensor in inputs"
            )
        return tuple(
            utils.torch.empty(p.type.shape, dtype=evaluator.to_torch_dtype(p.type.dtype), device=device)
            for p in self.type.output_params
        )


def signature_file(library: Path) -> Path:
    """The file beside *library* that holds its pickled entry signature."""
    return library.with_name(library.name + ".pkl")
