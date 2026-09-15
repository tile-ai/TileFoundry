"""The device ``.cu`` translation unit: one kernel and one shim per function.

Two passes over each function and nothing between them: one writes the
declaration a caller sees without entering the body, the other writes the
definition by walking it. What is not a function -- the includes, the program
dimensions this unit specializes, the device state something asked for while
it was being written -- settles only once every body has been walked, so it is
assembled last.
"""

from __future__ import annotations

import math

from tilefoundry.codegen.cuda.abi import (
    PROGRAM_META,
    PROGRAM_META_CTYPE,
    kernel_arguments,
    kernel_signature,
)
from tilefoundry.codegen.cuda.context import CudaCodegenContext, topology_scope_str
from tilefoundry.codegen.cuda.templates import render
from tilefoundry.codegen.linkable import LinkableFunction, LinkableModule
from tilefoundry.codegen.registry import CodeGenerator
from tilefoundry.codegen.signature import CallableSignature
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.types.shape_helpers import static_dim_value
from tilefoundry.target import Target

Geometry = tuple[tuple[object, object, object], tuple[object, object, object]]


def _emit_function(fn: PrimFunction, ctx: CudaCodegenContext) -> LinkableFunction:
    """*fn* in both of its positions, written from the one convention it is called by.

    A caller can name the shim and nothing else, so that is what the
    declaration states; the kernel is defined beside it and named only here.
    """
    shim = ctx.signature_of(fn)
    kernel = kernel_signature(shim, fn.name)
    return LinkableFunction(
        name=fn.name,
        declaration=f'extern "C" void {shim.name}({ctx.parameters(shim, exported=True)});',
        definition=ctx.capture(lambda scope: _define(fn, shim, kernel, scope)),
    )


def _define(
    fn: PrimFunction,
    shim: CallableSignature,
    kernel: CallableSignature,
    ctx: CudaCodegenContext,
) -> None:
    """The kernel and the shim that launches it, in that order."""
    _define_kernel(fn, kernel, ctx)
    ctx.blank()
    _define_shim(shim, kernel, ctx)


def _define_kernel(fn: PrimFunction, kernel: CallableSignature, ctx: CudaCodegenContext) -> None:
    """The ``__global__``: what it was told, then what it was written to do."""
    ctx.reset_barrier_ids()
    ctx.emit(f"__global__ void {kernel.name}({ctx.parameters(kernel)}) {{")
    ctx.indent()
    if kernel.leading:
        ctx.emit(f"tilefoundry::program_meta({PROGRAM_META.name});")
    ctx.emit_node(fn)
    ctx.dedent()
    ctx.emit("}")


def _define_shim(
    shim: CallableSignature, kernel: CallableSignature, ctx: CudaCodegenContext
) -> None:
    """The ``extern "C"`` entry: the geometry it was handed, spent on one launch."""
    ctx.emit(f'extern "C" void {shim.name}({ctx.parameters(shim, exported=True)}) {{')
    ctx.indent()
    _tell_the_block(shim, ctx)
    grid_x, block_x, smem = (signature.name for signature in shim.trailing)
    ctx.emit(f"dim3 grid({grid_x}, 1, 1);")
    ctx.emit(f"dim3 block({block_x}, 1, 1);")
    ctx.emit(f"{kernel.name}<<<grid, block, {smem}, nullptr>>>({kernel_arguments(kernel, ctx)});")
    ctx.dedent()
    ctx.emit("}")


def _tell_the_block(shim: CallableSignature, ctx: CudaCodegenContext) -> None:
    """Gather the ids the shim was told into the block the kernel takes.

    Which levels these are is the signature's answer, so the shim states no
    level of its own: one line per id the convention put in front.
    """
    if not shim.leading:
        return
    ctx.emit(f"{PROGRAM_META_CTYPE} {PROGRAM_META.name}{{}};")
    for signature in shim.leading:
        scope = topology_scope_str(signature.topology_level)
        ctx.emit(
            f"{PROGRAM_META.name}.program_id[int({scope})] = static_cast<int>({signature.name});"
        )


def _one_geometry(kernels: tuple[PrimFunction, ...], ctx: CudaCodegenContext) -> Geometry:
    """The geometry this unit runs at, which every kernel in it states the same.

    A unit specializes ``program_dim`` once, so two kernels launched at
    different geometries cannot share one: they are two units.
    """
    by_kernel = {fn.name: _launched_at(fn, ctx) for fn in kernels}
    distinct = set(by_kernel.values())
    if len(distinct) > 1:
        raise ValueError(
            f"emit_cuda_module: kernels disagree on launch geometry ({by_kernel}); "
            f"one translation unit runs every kernel in it at one geometry"
        )
    return distinct.pop()


def _launched_at(fn: PrimFunction, ctx: CudaCodegenContext) -> Geometry:
    """The geometry *fn*'s call states, which is the only thing that states one."""
    geometry = ctx.launches.get(id(fn))
    if geometry is None:
        raise ValueError(
            f"emit_cuda_module: nothing launches {fn.name!r}, so no call "
            f"states the geometry its unit runs at"
        )
    return geometry


def _program_dims(geometry: Geometry) -> list[dict[str, str]]:
    """One instance count per level: what ``program_dim`` states for this unit.

    ``program_shape`` is derived from these, so a unit specializes the counts
    and nothing else. A launch-provided grid has no count here; the runtime
    states that one when it is asked.
    """
    grid, block = geometry
    dims = []
    static_grid = tuple(static_dim_value(value) for value in grid)
    if all(value is not None for value in static_grid):
        dims.append({"scope": topology_scope_str("cta"), "count": str(math.prod(static_grid))})
    elif static_grid[1:] != (1, 1):
        raise ValueError("codegen: only grid.x may be a host-computed dynamic expression")
    static_block = tuple(static_dim_value(value) for value in block)
    if any(value is None for value in static_block):
        raise ValueError("codegen: thread topology extents must be compile-time constants")
    dims.append({"scope": topology_scope_str("thread"), "count": str(math.prod(static_block))})
    return dims


def emit_cuda_module(
    module: Module, cuda_fns: tuple[PrimFunction, ...], target: Target, ctx: CudaCodegenContext
) -> LinkableModule:
    """Emit the device ``.cu`` linkable module for one topology domain."""
    kernels = tuple(fn for fn in cuda_fns if fn.target == target)
    if not kernels:
        raise ValueError(f"emit_cuda_module: module {module.name!r} has no CUDA device kernels")
    functions = tuple(_emit_function(fn, ctx) for fn in kernels)
    geometry = _one_geometry(kernels, ctx)
    preamble = render(
        "cuda_preamble.cu.j2",
        program_topology=topology_scope_str(
            module.effective_topologies()[0].name if module.effective_topologies() else "cta"
        ),
        program_dims=_program_dims(geometry),
        dynamic_cta=static_dim_value(geometry[0][0]) is None,
        grid_barrier_state=ctx.needs_grid_barrier_state,
    )
    return LinkableModule(target="cuda", language="cu", preamble=preamble, functions=functions)


CUDA_CODE_GENERATOR = CodeGenerator(emit_cuda_module)


__all__ = ["CUDA_CODE_GENERATOR", "emit_cuda_module"]
