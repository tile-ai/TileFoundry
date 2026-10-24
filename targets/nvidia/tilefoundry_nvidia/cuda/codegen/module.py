"""The device ``.cu`` translation unit: one kernel and one launcher per function.

Two passes over each function and nothing between them: one writes the
declaration a caller sees without entering the body, the other writes the
definition by walking it. What is not a function -- the includes, the program
dimensions this unit specializes, the device state something asked for while
it was being written -- settles only once every body has been walked, so it is
assembled last.
"""

from __future__ import annotations

import math
import os
import shutil
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from tilefoundry.codegen.linkable_function import LinkableFunction
from tilefoundry.codegen.linkable_module import LinkableModule
from tilefoundry.codegen.signature import CallableSignature
from tilefoundry.ir.core.module import ModuleFunction
from tilefoundry.ir.types.utils import static_dim_value
from tilefoundry.target.base import UnsupportedCapabilityError, bind, target_summary
from tilefoundry_nvidia.cuda import CudaTarget
from tilefoundry_nvidia.cuda.codegen.abi import (
    PROGRAM_META,
    PROGRAM_META_CTYPE,
    check_parameter_names,
    kernel_arguments,
    kernel_signature,
)
from tilefoundry_nvidia.cuda.codegen.context import CudaCodegenContext, topology_scope_str
from tilefoundry_nvidia.cuda.codegen.hir.function import device_functions
from tilefoundry_nvidia.cuda.codegen.templates import render

Geometry = tuple[tuple[object, object, object], tuple[object, object, object]]


def _emit_function(fn: ModuleFunction, ctx: CudaCodegenContext) -> LinkableFunction:
    """*fn* in both of its positions, written from the one convention it is called by.

    A caller can name the launcher and nothing else, so that is what the
    declaration states; the kernel is defined beside it and named only here.
    """
    launcher = ctx.signature_of(fn)
    kernel = kernel_signature(launcher, ctx.codegen_context.symbols.by_function[id(fn)].symbol)
    return LinkableFunction(
        name=fn.name,
        declaration=f'extern "C" const char* {launcher.name}({ctx.parameters(launcher, exported=True)});',
        definition=ctx.capture(lambda scope: _define(fn, launcher, kernel, scope)),
    )


def _define(
    fn: ModuleFunction,
    launcher: CallableSignature,
    kernel: CallableSignature,
    ctx: CudaCodegenContext,
) -> None:
    """The kernel and the launcher that launches it, in that order."""
    _define_kernel(fn, kernel, ctx)
    ctx.blank()
    _define_launcher(launcher, kernel, ctx)


def _define_kernel(fn: ModuleFunction, kernel: CallableSignature, ctx: CudaCodegenContext) -> None:
    """The ``__global__``: what it was told, then what it was written to do."""
    ctx.reset_barrier_ids()
    ctx.emit(f"__global__ void {kernel.name}({ctx.parameters(kernel)}) {{")
    ctx.indent()
    if kernel.leading:
        ctx.emit(f"tilefoundry::program_meta({PROGRAM_META.name});")
    ctx.emit_node(fn)
    ctx.dedent()
    ctx.emit("}")


def _define_launcher(
    launcher: CallableSignature, kernel: CallableSignature, ctx: CudaCodegenContext
) -> None:
    """The ``extern "C"`` entry: the geometry it was handed, spent on one launch.

    It returns the CUDA error text when the device refuses the launch and null otherwise, so
    the host raises without an exception crossing the C ABI.
    """
    ctx.emit(f'extern "C" const char* {launcher.name}({ctx.parameters(launcher, exported=True)}) {{')
    ctx.indent()
    _tell_the_block(launcher, ctx)
    grid_x, block_x, smem = (signature.name for signature in launcher.trailing)
    ctx.emit(f"dim3 grid({grid_x}, 1, 1);")
    ctx.emit(f"dim3 block({block_x}, 1, 1);")
    ctx.emit(f"{kernel.name}<<<grid, block, {smem}, nullptr>>>({kernel_arguments(kernel, ctx)});")
    ctx.emit("cudaError_t status = cudaGetLastError();")
    ctx.emit("return status == cudaSuccess ? nullptr : cudaGetErrorString(status);")
    ctx.dedent()
    ctx.emit("}")


def _tell_the_block(launcher: CallableSignature, ctx: CudaCodegenContext) -> None:
    """Gather the ids the launcher was told into the block the kernel takes.

    Which levels these are is the signature's answer, so the launcher states no
    level of its own: one line per id the convention put in front.
    """
    if not launcher.leading:
        return
    ctx.emit(f"{PROGRAM_META_CTYPE} {PROGRAM_META.name}{{}};")
    for signature in launcher.leading:
        scope = topology_scope_str(signature.topology_level)
        ctx.emit(
            f"{PROGRAM_META.name}.program_id[int({scope})] = static_cast<int>({signature.name});"
        )


def _one_geometry(kernels: tuple[ModuleFunction, ...], ctx: CudaCodegenContext) -> Geometry:
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


def _launched_at(fn: ModuleFunction, ctx: CudaCodegenContext) -> Geometry:
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


def _cutlass_include() -> Path:
    configured = os.environ.get("TILEFOUNDRY_CUTLASS_INCLUDE")
    if configured is not None:
        path = Path(configured).resolve()
        if not path.is_dir():
            raise RuntimeError(f"TILEFOUNDRY_CUTLASS_INCLUDE={configured!r} is not a directory")
        return path
    checked = tuple(
        parent / "third_party/cutlass/include" for parent in Path(__file__).resolve().parents
    )
    for path in checked:
        if path.is_dir():
            return path
    raise RuntimeError(
        "TILEFOUNDRY_CUTLASS_INCLUDE is not set; no local CUTLASS directory found; checked: "
        + ", ".join(map(str, checked))
    )


@bind(CudaTarget, LinkableModule)
@dataclass(frozen=True, kw_only=True)
class CudaLinkableModule(LinkableModule):
    """A device translation unit and the architecture used to compile it."""

    arch: str

    emits = frozenset({"full", "skeleton"})

    @classmethod
    def create(cls, target, module, functions, ctx):
        """Emit the device translation unit using target-owned state."""
        if ctx.codegen_context is None:
            raise ValueError("CUDA module emission requires a CodegenContext")
        emit = ctx.codegen_context.options.emit
        if emit not in cls.emits:
            raise UnsupportedCapabilityError(
                f"{target_summary(target)}: code generator does not support emit={emit!r}"
            )
        for symbol in ctx.codegen_context.symbols.by_function.values():
            if symbol.target == target:
                check_parameter_names(symbol.export_name, symbol.callable.params)
        if ctx.codegen_context.options.emit == "skeleton" and target.device_count is not None:
            raise UnsupportedCapabilityError(
                f"{target_summary(target)}: skeleton emission supports only single-device CUDA "
                "targets without device_count"
            )
        kernels = tuple(fn for fn in functions if ctx.codegen_context.targets[id(fn)] == target)
        if not kernels:
            raise ValueError(f"emit_cuda_module: module {module.name!r} has no CUDA device kernels")
        functions = (
            *device_functions(kernels, ctx),
            *(_emit_function(fn, ctx) for fn in kernels),
        )
        geometry = _one_geometry(kernels, ctx)
        preamble = render(
            "cuda_preamble.cu.j2",
            program_topology=topology_scope_str(
                ctx.codegen_context.topology_domains[id(kernels[0])].topologies[0].name
                if ctx.codegen_context.topology_domains[id(kernels[0])].topologies
                else "cta"
            ),
            program_dims=_program_dims(geometry),
            dynamic_cta=static_dim_value(geometry[0][0]) is None,
            grid_barrier_state=ctx.needs_grid_barrier_state,
        )
        return cls(
            target="cuda", language="cu", preamble=preamble, functions=functions, arch=target.arch
        )

    def link(self, project: "CMakeProject") -> None:
        """Add CUDA source, compiler settings, and package-owned include paths."""
        if shutil.which("nvcc") is None:
            raise RuntimeError("link_modules: 'nvcc' not on PATH")
        index = sum(name.startswith("device_") for name in project.sources)
        source = f"device_{index}.cu"
        project.add_source(source, self.source)
        project.enable_language("CUDA")
        project.include(files(__package__).joinpath("include"), _cutlass_include())
        project.append(
            "set_target_properties(${PROJECT_NAME} PROPERTIES\n"
            f'  CUDA_ARCHITECTURES "{self.arch.removeprefix("sm_")}"\n'
            "  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF)\n"
            f"set_source_files_properties({source} PROPERTIES\n"
            "  # Device headers need C++20; older CMake rejects the CUDA20 dialect.\n"
            "  # A per-source option follows the standard flag; nvcc uses the last -std.\n"
            '  COMPILE_OPTIONS "-std=c++20;-Wno-deprecated-gpu-targets")\n'
        )
