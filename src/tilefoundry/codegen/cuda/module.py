"""CUDA device linkable-module emitter (split pipeline).

Emits the device ``.cu`` translation unit: the ``__global__`` kernels (identical
to the single-source path) plus, per kernel, an ``extern "C"`` launch shim that
performs the ``<<<grid, block, smem, stream>>>`` launch. Grid / block / smem /
stream arrive as plain C ABI arguments from the host module, so all CUDA types
stay inside this translation unit.
"""
from __future__ import annotations

from tilefoundry.codegen.cuda.context import CodegenContext
from tilefoundry.codegen.cuda.templates import render
from tilefoundry.codegen.cuda.tir.prim_function import (
    _compute_kernel_fields,
    _is_dispatch_entry_shape,
)
from tilefoundry.codegen.linkable import LinkableFunction, LinkableModule
from tilefoundry.ir.tir.prim_function import PrimFunction


def shim_symbol(kernel_name: str) -> str:
    """The ``extern "C"`` launch-shim symbol for *kernel_name* (a plain C
    identifier; mangled-variant ``$`` is replaced)."""
    return "tilefoundry_" + kernel_name.replace("$", "__") + "_launch"


# Launch-config ABI appended to every shim signature, in call order.
_LAUNCH_ABI_PARAMS = (
    "int grid_x, int grid_y, int grid_z, "
    "int block_x, int block_y, int block_z, "
    "int dynamic_smem, void* stream"
)


def _emit_kernel_and_shim(fields) -> str:
    shim_params = []
    shim_casts = []
    call_args = []
    for p in fields.params:
        if fields.param_kinds[p.name] == "tensor":
            cpp = fields.param_cpp_types[p.name]
            shim_params.append(f"void* {p.name}")
            shim_casts.append(f"{cpp}* {p.name}_p = static_cast<{cpp}*>({p.name});")
            call_args.append(f"{p.name}_p")
        else:
            shim_params.append(f"long long {p.name}")
            shim_casts.append(f"int {p.name}_i = static_cast<int>({p.name});")
            call_args.append(f"{p.name}_i")
    shim_params_sig = ", ".join((*shim_params, _LAUNCH_ABI_PARAMS))
    return render(
        "device_kernel.cu.j2",
        kernel_name=fields.kernel_name,
        kernel_params_sig=fields.kernel_params_sig,
        param_wrappers=fields.param_wrappers,
        kernel_body=fields.kernel_body,
        shim_name=shim_symbol(fields.kernel_name),
        shim_params_sig=shim_params_sig,
        shim_casts=shim_casts,
        kernel_call_args=", ".join(call_args),
    )


def emit_cuda_module(cuda_fns: tuple[PrimFunction, ...]) -> LinkableModule:
    """Emit the device ``.cu`` linkable module for *cuda_fns* (the CUDA-target
    PrimFunctions of a module)."""
    from tilefoundry.codegen.cuda.emit import _topology_shape_specializations  # noqa: PLC0415

    ctx = CodegenContext()
    kernel_texts = []
    all_fields = []
    for fn in cuda_fns:
        # A dispatch entry is host-only (its dispatch lowers in the host
        # module) and carries no device kernel — skip it here.
        if _is_dispatch_entry_shape(fn):
            continue
        fields = _compute_kernel_fields(fn, ctx)
        if fields.entry_host_only:
            # Contains a DispatchCall but not the recognized host-only dispatch
            # shape — not a device kernel and not a dispatch entry we can lower.
            raise NotImplementedError(
                f"emit_cuda_module: cuda function {fn.name!r} has a DispatchCall "
                f"but is not a recognized dispatch entry; cannot emit a device "
                f"kernel for it"
            )
        kernel_texts.append(_emit_kernel_and_shim(fields))
        all_fields.append(fields)
    if not all_fields:
        raise ValueError("emit_cuda_module: no CUDA device kernels")
    # The file-level program_shape specializations are shared by every kernel
    # in the module, so all kernels must agree on the (grid, block) topology
    # shape — otherwise the spec for one kernel would be silently wrong.
    base = all_fields[0]
    for f in all_fields[1:]:
        if (f.grid, f.block) != (base.grid, base.block):
            raise ValueError(
                "emit_cuda_module: kernels disagree on launch topology — "
                f"{base.kernel_name!r} has (grid={base.grid}, block={base.block}) "
                f"but {f.kernel_name!r} has (grid={f.grid}, block={f.block})"
            )
    specs = _topology_shape_specializations(base.grid, base.block)
    # A module that emits a grid barrier gets its own internal-linkage counter
    # pair defined in this source; the runtime header carries only the helper,
    # so multiple modules in one image never collide on a shared global symbol.
    uses_grid_barrier = any("grid_barrier(" in text for text in kernel_texts)
    source = render(
        "cuda_module.cu.j2",
        topology_shape_specializations=specs,
        kernels="\n".join(kernel_texts),
        dynamic_cta=base.grid[0] is None,
        uses_grid_barrier=uses_grid_barrier,
    )
    functions = tuple(
        LinkableFunction(name=f.kernel_name, source=text)
        for f, text in zip(all_fields, kernel_texts)
    )
    return LinkableModule(
        target="cuda", language="cu", source=source, functions=functions
    )


__all__ = ["emit_cuda_module", "shim_symbol"]
