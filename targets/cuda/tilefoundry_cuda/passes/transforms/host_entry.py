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
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.module_utils import list_call_paths
from tilefoundry.ir.tir.launch import launch_call
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Sequential
from tilefoundry.ir.types import StorageKind, TensorType, TupleType, UnitType
from tilefoundry.ir.types.layout import Layout
from tilefoundry.ir.visitor import ExprVisitor
from tilefoundry.passes.pass_base import ModulePass
from tilefoundry.passes.pass_manager import PassPoint, register_pass
from tilefoundry.target import CpuTarget
from tilefoundry_cuda import CudaTarget

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
    entry = module.entry_function()
    runs_on = entry.target if isinstance(entry, PrimFunction) else module.resolve_target()
    if isinstance(runs_on, CpuTarget):
        return module
    functions = tuple(frame.function for frame in list_call_paths(module).values())
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
    frames = list_call_paths(module)
    selectors = {id(frame.function): path for path, frame in frames.items()}

    class Check(ExprVisitor[None]):
        def default_visit_leaf(self, expr, operands, frame):
            return None

        def visit_Call(self, call, frame):
            self.visit_operands(call, frame)
            inner = frame._callee_context(call) if frame is not None else None
            if inner is None or isinstance(call.target, Function):
                return
            callee = inner.function
            weights = [param.name for param in callee.params if param.is_const]
            if weights:
                raise ValueError(
                    f"CUDA skeleton: {selectors[id(frame.function)]} calls "
                    f"{selectors[id(callee)]} with omitted weights {weights}; "
                    "declare the weights as parameters or call from orchestration"
                )

    for frame in frames.values():
        if isinstance(frame.function, Function):
            for function in (frame.function, *frame.function.variants):
                if function.body is not None:
                    Check().visit(function.body, frame)


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
    launch = launch_call(fn, params, grid, block, symbol=selector)
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
    reject_omitted_child_weights(module)
    taken = {fn.name for fn in module.functions}
    entries = []
    for call_path, frame in list_call_paths(module).items():
        fn = frame.function
        if isinstance(fn, Function) and frame.module.entry == fn.name:
            entries.append(host_entry_for(frame, call_path, fn, taken))
    return replace(module, functions=(*module.functions, *entries))


@register_pass(CudaTarget, PassPoint.BEFORE_CODEGEN)
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
