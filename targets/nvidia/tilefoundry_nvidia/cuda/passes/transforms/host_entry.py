"""Give launchable HIR Module entries CPU wrappers before lowering.

Launch geometry and the analyzed shared-memory peak are written into each
``Launch``. A launch-provided dynamic grid is unsupported by synthesized entries.
"""

from __future__ import annotations

from tilefoundry import analysis, target
from tilefoundry.interfaces import valuation
from tilefoundry.ir import core, expr, tir, types
from tilefoundry.passes import transforms
from tilefoundry.target import cpu
from tilefoundry_nvidia.cuda import analysis as cuda_analysis
from tilefoundry_nvidia.cuda import target as cuda_target
from tilefoundry_nvidia.cuda.ir.tir import sync


def product(values: tuple[object, ...]) -> object:
    result: object = 1
    for value in values:
        if value is None:
            raise ValueError(
                "insert_default_host_entry: topology extent cannot be None; "
                "declare a DimVar-backed host-computable extent"
            )
        result = result * value
    return result


def insert_default_host_entry(module: core.Module) -> core.Module:
    """Keep CPU entries or synthesize entries according to the module's IR kind."""
    entry = module.entry_function()
    runs_on = entry.target if isinstance(entry, core.PrimFunction) else module.resolve_target()
    if isinstance(runs_on, cpu.Target):
        return module
    functions = tuple(frame.function for frame in valuation.collect_functions(module).values())
    if all(isinstance(fn, core.PrimFunction) for fn in functions):
        return module
    if all(isinstance(fn, core.Function) for fn in functions):
        return entries_for_hir_functions(module)
    raise TypeError("insert_default_host_entry: mixed HIR/TIR device functions are unsupported")


def smem_peak(fn: core.Function) -> int:
    """The analyzed shared-memory high-water mark of *fn*, and 0 before analysis.

    It ends after the per-warp mbarrier words when the ``sync`` plan places them.
    """
    region = expr.get_metadata(fn, analysis.RegionMemoryMetadata)
    peaks = () if region is None else region.peaks
    peak = next((peak.peak_bytes for peak in peaks if peak.memory_level == "smem"), 0)
    plan = expr.get_metadata(fn, cuda_analysis.SyncPlanMetadata)
    if plan is None or plan.barriers is None:
        return peak
    offset, words = plan.barriers
    return offset + sync.MBARRIER_BYTES * words


def launchable(ty) -> bool:
    if isinstance(ty, types.TupleType):
        return all(launchable(field) for field in ty.fields)
    return not isinstance(ty, types.TensorType) or ty.storage in (expr.StorageKind.GMEM, expr.StorageKind.UMAT)


def fresh_name(stem: str, taken: set[str]) -> str:
    """Reserve *stem*, or its first free ``_1``, ``_2``, ... suffix, in *taken*."""
    candidate = stem
    index = 0
    while candidate in taken:
        index += 1
        candidate = f"{stem}_{index}"
    taken.add(candidate)
    return candidate


def entry_params(fn: core.Function) -> tuple[expr.Var, ...]:
    """*fn*'s declared inputs, then one GMEM destination per tensor result leaf.

    An HIR result has a type and no name, so each destination is named ``out``
    with the declared parameter names reserved.
    """
    taken = {param.name for param in fn.params}

    def destinations(ty):
        if isinstance(ty, types.TensorType):
            yield expr.Var(
                fresh_name("out", taken),
                type=types.TensorType(
                    tuple(ty.shape), ty.dtype, types.Layout(tuple(ty.shape)), expr.StorageKind.GMEM
                ),
            )
        elif isinstance(ty, types.TupleType):
            for field in ty.fields:
                yield from destinations(field)
        elif not isinstance(ty, types.UnitType):
            raise TypeError(
                f"insert_default_host_entry: unsupported result type {ty!r} in {fn.name!r}"
            )

    inputs = tuple(expr.Var(name=p.name, type=p.type, is_const=p.is_const) for p in fn.params)
    return (*inputs, *destinations(fn.return_type))


def host_entry_for(frame, ref: core.FunctionRef, fn: core.Function, taken: set[str]) -> core.PrimFunction:
    """Launch one Module entry with its declared inputs and flattened destinations."""
    params = entry_params(fn)
    grid = (product(tuple(t.size for t in frame.topologies if t.name == "cta")), 1, 1)
    block = (product(tuple(t.size for t in frame.topologies if t.name == "thread")), 1, 1)
    launch = tir.launch.launch_call(
        fn, params, grid, block, ref=ref, dynamic_smem=smem_peak(fn)
    )
    return core.PrimFunction(
        name=fresh_name(f"{fn.name}_host", taken),
        params=params,
        body=core.Sequential(body=(launch,)),
        output_count=len(params) - len(fn.params),
        target=cpu.Target(),
    )


def entries_for_hir_functions(module: core.Module) -> core.Module:
    """Append host entries and select the root host as the Module entry.

    Python begins a kernel invocation only by entering a Module; any other
    function runs as a device call inside one, or is wrapped by Python itself.
    """
    tgt = module.resolve_target()
    if not isinstance(tgt, cuda_target.Target):
        raise ValueError(
            f"insert_default_host_entry: HIR entries launch on a CUDA target, not {tgt!r}"
        )
    taken = {fn.name for fn in module.functions}
    entries = []
    root = None
    for ref, frame in valuation.collect_functions(module).items():
        fn = frame.function
        if (
            isinstance(fn, core.Function)
            and frame.module.entry == fn.name
            and not any(isinstance(segment, int) for segment in ref.path)
            and launchable(fn.return_type)
        ):
            entry = host_entry_for(frame, ref, fn, taken)
            entries.append(entry)
            if ref.path == () and ref.name == module.entry:
                root = entry
    module.replace_functions((*module.functions, *entries), entry=None if root is None else root.name)
    return module


@target.bind(cuda_target.Target, transforms.InsertHostEntryPass)
class CudaInsertHostEntryPass(transforms.InsertHostEntryPass):
    """Give launchable HIR Module entries a host-callable entry before lowering."""

    def run(self, module: core.Module) -> core.Module:
        return insert_default_host_entry(module)


__all__ = ["CudaInsertHostEntryPass", "insert_default_host_entry"]
