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

from tilefoundry import codegen, target
from tilefoundry.interfaces import valuation
from tilefoundry.ir import core, expr
from tilefoundry_nvidia.cuda import target as cuda_target

from .abi import (
    PROGRAM_META,
    PROGRAM_META_CTYPE,
    check_parameter_names,
    kernel_arguments,
    kernel_signature,
)
from .context import Context, topology_scope_str
from .hir import device_functions
from .templates import render

Geometry = tuple[tuple[object, object, object], tuple[object, object, object]]

DEFAULT_DYNAMIC_SMEM = 48 * 1024


def reject_omitted_child_weights(module: core.Module) -> None:
    """Reject device calls that depend on implicit child-owner weight injection."""
    frames = valuation.collect_functions(module)
    selectors = {id(frame.function): ref.qualified_name for ref, frame in frames.items()}

    class Check(core.ExprVisitor[None]):
        def default_visit_leaf(self, node, operands, frame):
            return None

        def visit_Call(self, call, frame):
            self.visit_operands(call, frame)
            inner = frame.callee_context(call) if frame is not None else None
            if inner is None or isinstance(call.target, core.Function):
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
        if isinstance(frame.function, core.Function):
            for function in (frame.function, *frame.function.variants):
                if function.body is not None:
                    Check().visit(function.body, frame)


def emit_function(fn: core.ModuleFunction, ctx: Context) -> codegen.LinkableFunction:
    """*fn* in both of its positions, written from the one convention it is called by.

    A caller can name the launcher and nothing else, so that is what the
    declaration states; the kernel is defined beside it and named only here.
    """
    launcher = ctx.signature_of(fn)
    kernel = kernel_signature(launcher, ctx.codegen_context.symbols.by_function[id(fn)].symbol)
    return codegen.LinkableFunction(
        name=fn.name,
        declaration=f'extern "C" const char* {launcher.name}({ctx.parameters(launcher, exported=True)});',
        definition=ctx.capture(lambda scope: define(fn, launcher, kernel, scope)),
    )


def define(
    fn: core.ModuleFunction,
    launcher: codegen.CallableSignature,
    kernel: codegen.CallableSignature,
    ctx: Context,
) -> None:
    """The kernel and the launcher that launches it, in that order."""
    define_kernel(fn, kernel, ctx)
    ctx.blank()
    define_launcher(launcher, kernel, ctx)


def define_kernel(fn: core.ModuleFunction, kernel: codegen.CallableSignature, ctx: Context) -> None:
    """The ``__global__``: what it was told, then what it was written to do."""
    ctx.reset_barrier_ids()
    ctx.emit(f"__global__ void {kernel.name}({ctx.parameters(kernel)}) {{")
    ctx.indent()
    ctx.declare_smem_base(lambda scope: write_kernel_body(fn, kernel, scope))
    ctx.dedent()
    ctx.emit("}")


def write_kernel_body(
    fn: core.ModuleFunction, kernel: codegen.CallableSignature, ctx: Context
) -> None:
    """What the kernel was told, then what it was written to do."""
    if kernel.leading:
        ctx.emit(f"tilefoundry::program_meta({PROGRAM_META.name});")
    ctx.emit_node(fn)


def define_launcher(
    launcher: codegen.CallableSignature, kernel: codegen.CallableSignature, ctx: Context
) -> None:
    """The ``extern "C"`` entry: the geometry it was handed, spent on one launch.

    CUDA grants a kernel 48 KiB of dynamic shared memory unless it opts in to more,
    so a larger request first raises the kernel's limit to what the launch asks for.
    It returns the CUDA error text when the device refuses either step and null
    otherwise, so the host raises without an exception crossing the C ABI.
    """
    ctx.emit(f'extern "C" const char* {launcher.name}({ctx.parameters(launcher, exported=True)}) {{')
    ctx.indent()
    tell_the_block(launcher, ctx)
    grid_x, block_x, smem = (signature.name for signature in launcher.trailing)
    ctx.emit(f"dim3 grid({grid_x}, 1, 1);")
    ctx.emit(f"dim3 block({block_x}, 1, 1);")
    ctx.emit(f"if ({smem} > {DEFAULT_DYNAMIC_SMEM}) {{")
    ctx.indent()
    ctx.emit(
        f"cudaError_t raised = cudaFuncSetAttribute({kernel.name}, "
        f"cudaFuncAttributeMaxDynamicSharedMemorySize, {smem});"
    )
    ctx.emit("if (raised != cudaSuccess) return cudaGetErrorString(raised);")
    ctx.dedent()
    ctx.emit("}")
    ctx.emit(f"{kernel.name}<<<grid, block, {smem}, nullptr>>>({kernel_arguments(kernel, ctx)});")
    ctx.emit("cudaError_t status = cudaGetLastError();")
    ctx.emit("return status == cudaSuccess ? nullptr : cudaGetErrorString(status);")
    ctx.dedent()
    ctx.emit("}")


def tell_the_block(launcher: codegen.CallableSignature, ctx: Context) -> None:
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


def one_geometry(kernels: tuple[core.ModuleFunction, ...], ctx: Context) -> Geometry:
    """The geometry this unit runs at, which every kernel in it states the same.

    A unit specializes ``program_dim`` once, so two kernels launched at
    different geometries cannot share one: they are two units.
    """
    by_kernel = {fn.name: launched_at(fn, ctx) for fn in kernels}
    distinct = set(by_kernel.values())
    if len(distinct) > 1:
        raise ValueError(
            f"emit_cuda_module: kernels disagree on launch geometry ({by_kernel}); "
            f"one translation unit runs every kernel in it at one geometry"
        )
    return distinct.pop()


def launched_at(fn: core.ModuleFunction, ctx: Context) -> Geometry:
    """The geometry *fn*'s call states, which is the only thing that states one."""
    geometry = ctx.launches.get(id(fn))
    if geometry is None:
        raise ValueError(
            f"emit_cuda_module: nothing launches {fn.name!r}, so no call "
            f"states the geometry its unit runs at"
        )
    return geometry


def program_dimensions(geometry: Geometry) -> list[dict[str, str]]:
    """One instance count per level: what ``program_dim`` states for this unit.

    ``program_shape`` is derived from these, so a unit specializes the counts
    and nothing else. A launch-provided grid has no count here; the runtime
    states that one when it is asked.
    """
    grid, block = geometry
    dims = []
    static_grid = tuple(expr.static_dim_value(value) for value in grid)
    if all(value is not None for value in static_grid):
        dims.append({"scope": topology_scope_str("cta"), "count": str(math.prod(static_grid))})
    elif static_grid[1:] != (1, 1):
        raise ValueError("codegen: only grid.x may be a host-computed dynamic expression")
    static_block = tuple(expr.static_dim_value(value) for value in block)
    if any(value is None for value in static_block):
        raise ValueError("codegen: thread topology extents must be compile-time constants")
    dims.append({"scope": topology_scope_str("thread"), "count": str(math.prod(static_block))})
    return dims


def cutlass_include() -> Path:
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


@target.bind(cuda_target.Target, codegen.LinkableModule)
@dataclass(frozen=True, kw_only=True)
class CudaLinkableModule(codegen.LinkableModule):
    """A device translation unit and the nvcc architecture options that compile it."""

    gpu_architecture: str
    gpu_code: tuple[str, ...]

    emits = frozenset({"full", "skeleton"})

    @classmethod
    def create(cls, tgt, module, functions, ctx):
        """Emit the device translation unit using target-owned state."""
        if ctx.codegen_context is None:
            raise ValueError("CUDA module emission requires a CodegenContext")
        emit = ctx.codegen_context.options.emit
        if emit not in cls.emits:
            raise target.UnsupportedCapabilityError(
                f"{target.summary(tgt)}: code generator does not support emit={emit!r}"
            )
        for symbol in ctx.codegen_context.symbols.by_function.values():
            if symbol.target == tgt:
                check_parameter_names(symbol.export_name, symbol.callable.params)
        if ctx.codegen_context.options.emit == "skeleton" and tgt.device_count is not None:
            raise target.UnsupportedCapabilityError(
                f"{target.summary(tgt)}: skeleton emission supports only single-device CUDA "
                "targets without device_count"
            )
        if emit == "skeleton":
            reject_omitted_child_weights(module)
        kernels = tuple(fn for fn in functions if ctx.codegen_context.targets[id(fn)] == tgt)
        if not kernels:
            raise ValueError(f"emit_cuda_module: module {module.name!r} has no CUDA device kernels")
        functions = (
            *device_functions(kernels, ctx),
            *(emit_function(fn, ctx) for fn in kernels),
        )
        geometry = one_geometry(kernels, ctx)
        preamble = render(
            "cuda_preamble.cu.j2",
            program_topology=topology_scope_str(
                ctx.codegen_context.topology_domains[id(kernels[0])].topologies[0].name
                if ctx.codegen_context.topology_domains[id(kernels[0])].topologies
                else "cta"
            ),
            program_dims=program_dimensions(geometry),
            dynamic_cta=expr.static_dim_value(geometry[0][0]) is None,
            grid_barrier_state=ctx.needs_grid_barrier_state,
        )
        return cls(
            target="cuda",
            language="cu",
            preamble=preamble,
            functions=functions,
            gpu_architecture=tgt.architecture.gpu_architecture,
            gpu_code=tgt.architecture.gpu_code,
        )

    def link(self, project: "CMakeProject") -> None:
        """Add CUDA source, compiler settings, and package-owned include paths."""
        if shutil.which("nvcc") is None:
            raise RuntimeError("link_modules: 'nvcc' not on PATH")
        index = sum(name.startswith("device_") for name in project.sources)
        source = f"device_{index}.cu"
        project.add_source(source, self.source)
        project.enable_language("CUDA")
        project.include(files(__package__).joinpath("include"), cutlass_include())
        project.append(
            "set_target_properties(${PROJECT_NAME} PROPERTIES\n"
            "  CUDA_ARCHITECTURES OFF\n"
            "  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF)\n"
            f"set_source_files_properties({source} PROPERTIES\n"
            "  # Device headers need C++20; older CMake rejects the CUDA20 dialect.\n"
            "  # A per-source option follows the standard flag; nvcc uses the last -std.\n"
            '  COMPILE_OPTIONS "-std=c++20;-Wno-deprecated-gpu-targets;'
            f'--gpu-architecture={self.gpu_architecture};--gpu-code={",".join(self.gpu_code)}")\n'
        )


__all__ = ["CudaLinkableModule"]
