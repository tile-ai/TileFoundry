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
from tilefoundry.ir.core.module import (
    Module,
    called_functions,
    child_module_of,
    function_selectors,
    module_functions,
    owning_module,
    target_of,
)
from tilefoundry.ir.hir.destination import destination_params, fresh_name
from tilefoundry.ir.hir.function import Function
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
    """Keep CPU entries or synthesize entries according to the module's IR kind."""
    if isinstance(target_of(module, module.entry_function()), CpuTarget):
        return module
    functions = module_functions(module)
    if all(isinstance(fn, PrimFunction) for fn in functions):
        return entry_for_one_device_function(module)
    if all(isinstance(fn, Function) for fn in functions):
        return entries_for_hir_functions(module)
    raise TypeError("insert_default_host_entry: mixed HIR/TIR device functions are unsupported")


def entry_for_one_device_function(module: Module) -> Module:
    """Preserve the authored TIR single-device-function normalization."""
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


def reject_omitted_child_weights(module: Module) -> None:
    """Reject device calls that depend on implicit child-owner weight injection."""
    functions = function_selectors(module)
    selectors = {id(fn): selector for selector, fn in functions}
    for selector, fn in functions:
        for caller in (fn, *fn.variants):
            for callee in called_functions(caller):
                if child_module_of(module, caller, callee) is not None:
                    weights = [param.name for param in callee.params if param.is_const]
                    if weights:
                        raise ValueError(
                            f"CUDA skeleton: {selector} calls "
                            f"{selectors[id(callee)]} with omitted weights {weights}; "
                            "declare the weights as parameters or call from orchestration"
                        )


def host_entry_for(module: Module, selector: str, fn: Function, taken: set[str]) -> PrimFunction:
    """Launch one HIR function with its declared inputs and flattened destinations."""
    params = tuple(Var(name=p.name, type=p.type, is_const=p.is_const) for p in destination_params(fn))
    owner = owning_module(module, fn)
    grid, block = _launch_config_of(owner)
    launch = launch_call(fn, params, grid, block, symbol=selector)
    return PrimFunction(
        name=fresh_name(f"{fn.name}_host", taken),
        params=params,
        body=Sequential(body=(launch,)),
        output_count=len(params) - len(fn.params),
        target=CpuTarget(),
    )


def entries_for_hir_functions(module: Module) -> Module:
    """Append a host entry per HIR prototype, retaining the authored root entry."""
    reject_omitted_child_weights(module)
    taken = {fn.name for fn in module.functions}
    entries = tuple(
        host_entry_for(module, selector, fn, taken)
        for selector, fn in function_selectors(module)
    )
    return replace(module, functions=(*module.functions, *entries))


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
