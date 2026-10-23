"""Linker — the final stage of codegen.

Separately compiles each target's :class:`~tilefoundry.codegen.linkable_module.LinkableModule`
translation unit and links them into one host-callable shared library, returning
a ``LinkedModule`` (artifact + export metadata) for the runtime loader
to turn into a ``RuntimeModule``.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from tilefoundry.codegen.context_builder import EmissionGroup, build_codegen_context
from tilefoundry.codegen.cpu.context import CpuCodegenContext
from tilefoundry.codegen.link_function import link_function
from tilefoundry.codegen.linked_function import LinkedFunction
from tilefoundry.codegen.linked_module import LinkedModule
from tilefoundry.codegen.options import CodegenOptions
from tilefoundry.dump import DumpFlags, DumpScope, dump
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.target import CpuTarget
from tilefoundry.target.base import UnsupportedCapabilityError, target_summary

_TILEFOUNDRY_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_INCLUDE = _TILEFOUNDRY_ROOT / "include"
_DEFAULT_CUTLASS_INCLUDE = _TILEFOUNDRY_ROOT / "third_party" / "cutlass" / "include"
_CMAKE_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates" / "cmake"


def _render_cmakelists(
    *,
    name: str,
    includes: list[str],
    device_options: str,
    cuda_arch: str,
    device_sources: list[str],
) -> str:
    """Render the split-pipeline CMake project from its Jinja template."""
    # noqa lazy: jinja2 is already a codegen dep; import here keeps the linker

    from jinja2 import (  # noqa: PLC0415
        Environment,
        FileSystemLoader,
        StrictUndefined,
    )

    env = Environment(
        loader=FileSystemLoader(str(_CMAKE_TEMPLATE_DIR)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    return env.get_template("CMakeLists.txt.j2").render(
        name=name,
        includes=includes,
        device_options=device_options,
        cuda_arch=cuda_arch,
        device_sources=device_sources,
    )


def _tvm_ffi_include() -> Path:
    # noqa lazy: tvm_ffi is an optional runtime dep; only required when

    import tvm_ffi  # noqa: PLC0415
    return Path(tvm_ffi.__file__).resolve().parent / "include"


def link_modules(
    modules,
    *,
    workdir: str | Path,
    lib_name: str,
    functions: tuple[LinkedFunction, ...],
    nvcc: str = "nvcc",
    host_cxx: str = "g++",
    extra_nvcc_flags: tuple[str, ...] = (),
    cuda_arch: str = "90",
    include_dirs: tuple[Path, ...] = (),
) -> LinkedModule:
    """Link modules.

    Separately compile device ``cu`` modules and one host ``cpp`` module, then
    link them into one host-callable shared library; return the ``LinkedModule``.

    Requires at least one ``cu`` module and exactly one ``cpp`` module. Each device
    module compiles with nvcc, the host module with a plain host compiler; the
    final link runs through nvcc (to pull in the CUDA runtime) and statically
    links libstdc++ so the ``.so`` keeps the GLIBCXX-independent load behaviour.
    """
    modules = tuple(modules)
    if (
        len(modules) < 2
        or sum(m.language == "cpp" for m in modules) != 1
        or sum(m.language == "cu" for m in modules) < 1
    ):
        received = ", ".join(f"({m.target}, {m.language})" for m in modules)
        raise ValueError(
            f"link_modules: requires one 'cpp' and at least one 'cu' "
            f"module, got [{received}]"
        )
    cu = [m for m in modules if m.language == "cu"]
    cpp = [m for m in modules if m.language == "cpp"]
    for tool in (nvcc, host_cxx, "cmake"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"link_modules: {tool!r} not on PATH")

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    for index, device in enumerate(cu):
        (workdir / f"device_{index}.cu").write_text(device.source)
    (workdir / "host.cpp").write_text(cpp[0].source)






    tvm_inc = _tvm_ffi_include()
    includes = [_DEFAULT_INCLUDE, _DEFAULT_CUTLASS_INCLUDE, tvm_inc, *include_dirs]
    device_options = ";".join(("-Wno-deprecated-gpu-targets", *extra_nvcc_flags))
    cmake_text = _render_cmakelists(
        name=lib_name,
        includes=[str(p) for p in includes],
        device_options=device_options,
        cuda_arch=cuda_arch,
        device_sources=[f"device_{i}.cu" for i in range(len(cu))],
    )
    (workdir / "CMakeLists.txt").write_text(cmake_text)

    build_dir = workdir / "build"



    configure_cmd = [
        "cmake", "-G", "Ninja", "-S", str(workdir), "-B", str(build_dir),
        f"-DCMAKE_CUDA_COMPILER={nvcc}", f"-DCMAKE_CXX_COMPILER={host_cxx}",
    ]
    build_cmd = ["cmake", "--build", str(build_dir), "--verbose"]
    lib = build_dir / f"lib{lib_name}.so"

    with DumpScope("build"):
        dump("CMakeLists.txt", cmake_text, DumpFlags.BUILD_LOG)
        for index, device in enumerate(cu):
            dump(f"module.device_{index}.cu", device.source, DumpFlags.BUILD_LOG)
        dump("module.host.cpp", cpp[0].source, DumpFlags.BUILD_LOG)
        for step, cmd in (("cmake-configure", configure_cmd), ("cmake-build", build_cmd)):
            dump(f"{step}.cmd.txt", " ".join(cmd) + "\n", DumpFlags.BUILD_LOG)
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            dump(f"{step}.stdout", proc.stdout, DumpFlags.BUILD_LOG)
            dump(f"{step}.stderr", proc.stderr, DumpFlags.BUILD_LOG)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"link_modules: {step} failed (rc={proc.returncode})\n"
                    f"cmd: {' '.join(cmd)}\n"
                    f"stdout:\n{proc.stdout}\n"
                    f"stderr:\n{proc.stderr}"
                )
    if not lib.exists():
        raise RuntimeError(f"link_modules: cmake build did not produce {lib}")

    source = (
        f"// cpu module\n{cpp[0].source}\n"
        f"// cuda modules\n{''.join(m.source for m in cu)}"
    )
    return LinkedModule(
        library_path=lib,
        source=source,
        functions=functions,
    )


def link_program(
    mod: Module,
    *,
    workdir: str | Path,
    options: CodegenOptions = CodegenOptions(),
    entries=None,
) -> LinkedModule:
    """Emit and link host/device units directly in workdir, without loading."""
    codegen = build_codegen_context(mod, options)
    cpu_entry = mod.entry_function()
    entries = (cpu_entry,) if entries is None else tuple(entries)
    if not entries:
        raise ValueError("link_program: expected at least one CPU host entry")
    for entry in entries:
        if not (isinstance(entry, PrimFunction) and isinstance(entry.target, CpuTarget)):
            raise ValueError(f"link_program: entry {entry.name!r} is not a CPU host entry")

    generators = {
        target: target.get_code_generator()
        for target in dict.fromkeys(group.target for group in codegen.groups)
    }
    for group in codegen.groups:
        if options.emit not in generators[group.target].emits:
            raise UnsupportedCapabilityError(
                f"{target_summary(group.target)}: code generator does not support "
                f"emit={options.emit!r}"
            )

    device_modules = []
    for group in codegen.groups:
        view = codegen.for_group(group)
        if not isinstance(group.target, CpuTarget):
            device_modules.append(
                generators[group.target].emit(group.owner, group.functions, group.target, view)
            )
    view = codegen.for_group(EmissionGroup(mod, entries[0].target, entries))
    context = CpuCodegenContext(
        symbols=view.symbols,
        target=view.target,
        resolved_launches=view.resolved_launches,
        codegen_context=view.codegen,
    )
    host_module = generators[entries[0].target].emit(mod, entries, entries[0].target, context)

    units = (*device_modules, host_module)
    functions = tuple(link_function(codegen.symbols.by_function[id(entry)]) for entry in entries)
    device_targets = tuple(
        group.target for group in codegen.groups if not isinstance(group.target, CpuTarget)
    )
    if not device_targets:
        raise ValueError(f"link_program: module {mod.name!r} has no CUDA device functions")
    linked_module = link_modules(
        units,
        workdir=workdir,
        lib_name=mod.name,
        functions=functions,
        cuda_arch=device_targets[0].arch.removeprefix("sm_"),
    )
    return linked_module


__all__ = ["link_program", "link_modules"]
