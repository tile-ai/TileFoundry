"""Give a device-only module the CPU entry that calls into it.

The entry is synthesized, never retargeted: which target a function records is
what its author wrote, and a pass that rewrites it decides in the wrong place.
Launch geometry is settled here and written into the ``Launch``, so nothing
downstream derives it a second time. A launch-provided (dynamic) grid is not
supported by the synthesized entry.
"""

from __future__ import annotations

from dataclasses import replace

from tilefoundry.ir.core import Var
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.tir.launch import launch_call
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Sequential
from tilefoundry.passes.pass_base import ModulePass
from tilefoundry.target import CpuTarget, CudaTarget

_DEFAULT_ENTRY_NAME = "main"


def _fresh_entry_name(taken: set[str]) -> str:
    if _DEFAULT_ENTRY_NAME not in taken:
        return _DEFAULT_ENTRY_NAME
    candidate = f"_tilefoundry_{_DEFAULT_ENTRY_NAME}"
    i = 0
    while candidate in taken:
        i += 1
        candidate = f"_tilefoundry_{_DEFAULT_ENTRY_NAME}_{i}"
    return candidate


def _device_callee(module: Module) -> PrimFunction:
    """The one device function a synthesized entry can be written against.

    Refuses rather than guesses: a second device function would leave the
    entry's parameters undecided, and a CPU function that is not the entry
    means the author meant some other function to be called first.
    """
    if any(isinstance(fn.target, CpuTarget) for fn in module.functions):
        raise ValueError(
            "insert_default_host_entry: module has a CPU function that is not "
            "the entry; refusing to guess the host entry"
        )
    device_fns = [
        fn
        for fn in module.functions
        if isinstance(fn, PrimFunction) and isinstance(fn.target, CudaTarget)
    ]
    if len(device_fns) != 1:
        raise ValueError(
            f"insert_default_host_entry: expected exactly one CUDA device "
            f"prim_function and no CPU entry, found {len(device_fns)} device "
            f"functions"
        )
    return device_fns[0]


def _product(values: tuple[object, ...]) -> object:
    result: object = 1
    for value in values:
        if value is None:
            raise ValueError(
                "insert_default_host_entry: topology extent cannot be None; "
                "declare a DimVar-backed host-computable extent"
            )
        result = result * value
    return result


def _launch_config_of(
    module: Module,
) -> tuple[tuple[object, object, object], tuple[object, object, object]]:
    """Derive the implicit Launch expressions solely from module topologies."""
    topologies = module.effective_topologies()
    grid = _product(tuple(t.size for t in topologies if t.name == "cta"))
    block = _product(tuple(t.size for t in topologies if t.name == "thread"))
    return (grid, 1, 1), (block, 1, 1)


def insert_default_host_entry(module: Module) -> Module:
    """Return a module whose entry is host-callable, synthesizing one if it is not.

    A CPU entry passes through. Otherwise the device entry gains a CPU entry
    that launches it with its own parameters. Whether that call reads as one
    launch or as a dispatch over variants is the callee's shape to state, so
    a prototype and a lone kernel take the same route here.
    """
    if isinstance(module.entry_function().target, CpuTarget):
        return module

    device_fn = _device_callee(module)
    entry_params = tuple(Var(type=p.type, name=p.name) for p in device_fn.params)
    grid, block = _launch_config_of(module)
    launch = launch_call(device_fn, entry_params, grid, block)
    name = _fresh_entry_name({fn.name for fn in module.functions})
    entry = PrimFunction(
        name=name,
        params=entry_params,
        body=Sequential(body=(launch,)),
        output_count=device_fn.output_count,
        target=CpuTarget(),
    )
    return replace(module, functions=(*module.functions, entry), entry=name)


class InsertHostEntryPass(ModulePass):
    """Give a device-only module a host-callable entry.

    The only pass a TIR module needs: everything else it is made of was
    settled before it became TIR.
    """

    name: str = "insert_host_entry"
    requires: tuple[str, ...] = ()

    def run(self, module: Module) -> Module:
        return insert_default_host_entry(module)


__all__ = ["InsertHostEntryPass", "insert_default_host_entry"]
