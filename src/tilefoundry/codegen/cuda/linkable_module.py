"""CUDA translation units contribute their own build settings."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from tilefoundry.codegen.cuda import module as emission
from tilefoundry.codegen.cuda.abi import check_parameter_names
from tilefoundry.codegen.cuda.context import topology_scope_str
from tilefoundry.codegen.cuda.hir.function import device_functions
from tilefoundry.codegen.cuda.templates import render
from tilefoundry.codegen.linkable_module import LinkableModule
from tilefoundry.ir.types.utils import static_dim_value
from tilefoundry.target import CudaTarget
from tilefoundry.target.base import UnsupportedCapabilityError, bind, target_summary


def _cutlass_include() -> Path:
    configured = os.environ.get("TILEFOUNDRY_CUTLASS_INCLUDE")
    if configured is not None:
        path = Path(configured).resolve()
        if not path.is_dir():
            raise RuntimeError(f"TILEFOUNDRY_CUTLASS_INCLUDE={configured!r} is not a directory")
        return path
    checked = tuple(parent / "third_party/cutlass/include" for parent in Path(__file__).resolve().parents)
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
            *(emission._emit_function(fn, ctx) for fn in kernels),
        )
        geometry = emission._one_geometry(kernels, ctx)
        preamble = render(
            "cuda_preamble.cu.j2",
            program_topology=topology_scope_str(
                ctx.codegen_context.topology_domains[id(kernels[0])].topologies[0].name
                if ctx.codegen_context.topology_domains[id(kernels[0])].topologies else "cta"
            ),
            program_dims=emission._program_dims(geometry),
            dynamic_cta=static_dim_value(geometry[0][0]) is None,
            grid_barrier_state=ctx.needs_grid_barrier_state,
        )
        return cls(target="cuda", language="cu", preamble=preamble, functions=functions, arch=target.arch)

    def link(self, project: "CMakeProject") -> None:
        """Add CUDA source, compiler settings, and package-owned include paths."""
        if shutil.which("nvcc") is None:
            raise RuntimeError("link_modules: 'nvcc' not on PATH")
        index = sum(name.startswith("device_") for name in project.sources)
        source = f"device_{index}.cu"
        project.add_source(source, self.source)
        project.enable_language("CUDA")
        project.include(files("tilefoundry.codegen.cuda").joinpath("include"), _cutlass_include())
        project.append(
            'set_target_properties(${PROJECT_NAME} PROPERTIES\n'
            f'  CUDA_ARCHITECTURES "{self.arch.removeprefix("sm_")}"\n'
            '  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF)\n'
            f'set_source_files_properties({source} PROPERTIES\n'
            '  # Device headers need C++20; older CMake rejects the CUDA20 dialect.\n'
            '  # A per-source option follows the standard flag; nvcc uses the last -std.\n'
            '  COMPILE_OPTIONS "-std=c++20;-Wno-deprecated-gpu-targets")\n'
        )
