"""Module transform: synthesize a default host (CPU) entry for a lone device
kernel.

When a module has no host-callable (CPU-target) entry and exactly one CUDA
device ``PrimFunction``, build a ``PrimFunction(target="cpu")`` whose signature
mirrors the device function's parameters and whose body is a single ``Launch``
of that device function. Grid / block are taken from the existing static
launch-config derivation and frozen as ``i64`` constants (dynamic grid is a
later concern).

The transform returns a new ``Module`` (the device function object is reused,
but ``functions`` / ``entry`` are rebuilt). It is not wired into the production
compile path.
"""
from __future__ import annotations

from dataclasses import replace

from tilefoundry.codegen.cuda.emit import _derive_launch_config
from tilefoundry.codegen.cuda.tir.prim_function import _is_dispatch_entry_shape
from tilefoundry.ir.core import Var
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.tir.launch import launch_call
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Sequential
from tilefoundry.target import CpuTarget

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


def insert_default_host_entry(module: Module) -> Module:
    """Return a module whose entry is host-callable (CPU-target):

    - a CPU entry is already present → unchanged;
    - the entry is a dispatch entry (host-only ``DispatchCall``) → retarget it
      to CPU in place of the original (``module.entry`` name unchanged);
    - exactly one CUDA device kernel with no CPU entry → synthesize a CPU entry
      whose body is a single ``Launch`` of that kernel.
    """
    entry_fn = module.entry_function()
    if entry_fn.target.name == "cpu":
        return module
    # The entry is not CPU-target. A CPU function that is not the entry leaves
    # the module without a host-callable entry; v1 does not guess which one to
    # promote.
    if any(fn.target.name == "cpu" for fn in module.functions):
        raise ValueError(
            "insert_default_host_entry: module has a CPU function that is not "
            "the entry; refusing to guess the host entry"
        )
    # Dispatch entry: it is host-only (no device kernel), so retarget it to CPU.
    # The launched variants remain CUDA device functions.
    if _is_dispatch_entry_shape(entry_fn):
        cpu_entry = replace(entry_fn, target=CpuTarget())
        new_functions = tuple(
            cpu_entry if fn is entry_fn else fn for fn in module.functions
        )
        return replace(module, functions=new_functions)
    device_fns = [
        fn
        for fn in module.functions
        if isinstance(fn, PrimFunction) and fn.target.name == "cuda"
    ]
    if len(device_fns) != 1:
        raise ValueError(
            f"insert_default_host_entry: expected exactly one CUDA device "
            f"prim_function (or a dispatch entry) and no CPU entry, found "
            f"{len(device_fns)} device functions"
        )
    device_fn = device_fns[0]
    # Mirror the device parameters with fresh Var identities; pre-allocated
    # outputs stay as tensor parameters (no auto-alloc).
    entry_params = tuple(Var(type=p.type, name=p.name) for p in device_fn.params)
    grid, block = _derive_launch_config(device_fn.body)
    if grid[0] is None:
        raise ValueError(
            f"insert_default_host_entry: device function {device_fn.name!r} has "
            f"a launch-provided (dynamic) CTA extent; the implicit host entry "
            f"cannot derive its grid — launch it from an explicit host entry"
        )
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


__all__ = ["insert_default_host_entry"]
