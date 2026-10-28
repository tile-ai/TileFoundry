"""Give launchable HIR Module entries CPU wrappers before lowering.

Launch geometry and the analyzed shared-memory peak are written into each
``Launch``. A launch-provided dynamic grid is unsupported by synthesized entries.
"""

from __future__ import annotations

from tilefoundry.analysis.metadata import RegionMemoryMetadata
from tilefoundry.ir.core import Var, get_metadata
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.module_utils import list_call_paths
from tilefoundry.ir.tir.launch import launch_call
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Sequential
from tilefoundry.ir.types import StorageKind, TensorType, TupleType, UnitType
from tilefoundry.ir.types.layout import Layout
from tilefoundry.passes.pass_base import ModulePass
from tilefoundry.passes.pass_manager import PassPoint, register_pass
from tilefoundry.target import CpuTarget
from tilefoundry_nvidia.cuda import CudaTarget


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


def insert_default_host_entry(module: Module) -> Module:
    """Keep CPU entries or synthesize entries according to the module's IR kind."""
    entry = module.entry_function()
    runs_on = entry.target if isinstance(entry, PrimFunction) else module.resolve_target()
    if isinstance(runs_on, CpuTarget):
        return module
    functions = tuple(frame.function for frame in list_call_paths(module).values())
    if all(isinstance(fn, PrimFunction) for fn in functions):
        return module
    if all(isinstance(fn, Function) for fn in functions):
        return entries_for_hir_functions(module)
    raise TypeError("insert_default_host_entry: mixed HIR/TIR device functions are unsupported")


def _smem_peak(fn: Function) -> int:
    """The analyzed shared-memory high-water mark of *fn*, and 0 before analysis."""
    region = get_metadata(fn, RegionMemoryMetadata)
    peaks = () if region is None else region.peaks
    return next((peak.peak_bytes for peak in peaks if peak.memory_level == "smem"), 0)


def _launchable(ty) -> bool:
    if isinstance(ty, TupleType):
        return all(_launchable(field) for field in ty.fields)
    return not isinstance(ty, TensorType) or ty.storage in (StorageKind.GMEM, StorageKind.UMAT)


def _fresh_name(stem: str, taken: set[str]) -> str:
    """Reserve *stem*, or its first free ``_1``, ``_2``, ... suffix, in *taken*."""
    candidate = stem
    index = 0
    while candidate in taken:
        index += 1
        candidate = f"{stem}_{index}"
    taken.add(candidate)
    return candidate


def _entry_params(fn: Function) -> tuple[Var, ...]:
    """*fn*'s declared inputs, then one GMEM destination per tensor result leaf.

    An HIR result has a type and no name, so each destination is named ``out``
    with the declared parameter names reserved.
    """
    taken = {param.name for param in fn.params}

    def destinations(ty):
        if isinstance(ty, TensorType):
            yield Var(
                _fresh_name("out", taken),
                type=TensorType(
                    tuple(ty.shape), ty.dtype, Layout(tuple(ty.shape)), StorageKind.GMEM
                ),
            )
        elif isinstance(ty, TupleType):
            for field in ty.fields:
                yield from destinations(field)
        elif not isinstance(ty, UnitType):
            raise TypeError(
                f"insert_default_host_entry: unsupported result type {ty!r} in {fn.name!r}"
            )

    inputs = tuple(Var(name=p.name, type=p.type, is_const=p.is_const) for p in fn.params)
    return (*inputs, *destinations(fn.return_type))


def host_entry_for(frame, selector: str, fn: Function, taken: set[str]) -> PrimFunction:
    """Launch one Module entry with its declared inputs and flattened destinations."""
    params = _entry_params(fn)
    grid = (_product(tuple(t.size for t in frame.topologies if t.name == "cta")), 1, 1)
    block = (_product(tuple(t.size for t in frame.topologies if t.name == "thread")), 1, 1)
    launch = launch_call(
        fn, params, grid, block, symbol=selector, dynamic_smem=_smem_peak(fn)
    )
    return PrimFunction(
        name=_fresh_name(f"{fn.name}_host", taken),
        params=params,
        body=Sequential(body=(launch,)),
        output_count=len(params) - len(fn.params),
        target=CpuTarget(),
    )


def entries_for_hir_functions(module: Module) -> Module:
    """Append a host entry per Module entry, retaining the authored root entry.

    Python begins a kernel invocation only by entering a Module; any other
    function runs as a device call inside one, or is wrapped by Python itself.
    """
    target = module.resolve_target()
    if not isinstance(target, CudaTarget):
        raise ValueError(
            f"insert_default_host_entry: HIR entries launch on a CUDA target, not {target!r}"
        )
    taken = {fn.name for fn in module.functions}
    entries = []
    for call_path, frame in list_call_paths(module).items():
        fn = frame.function
        if (
            isinstance(fn, Function)
            and frame.module.entry == fn.name
            and not any(part.isdecimal() for part in call_path.split(".")[:-1])
            and _launchable(fn.return_type)
        ):
            entries.append(host_entry_for(frame, call_path, fn, taken))
    module.functions = (*module.functions, *entries)
    return module


@register_pass(CudaTarget, PassPoint.BEFORE_LOWER)
class InsertHostEntryPass(ModulePass):
    """Give launchable HIR Module entries a host-callable entry before lowering."""

    name: str = "insert_host_entry"
    requires: tuple[str, ...] = ()

    def run(self, module: Module) -> Module:
        return insert_default_host_entry(module)


__all__ = ["InsertHostEntryPass", "insert_default_host_entry"]
