"""Generate and link a Module tree through its targets' build services."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import replace
from importlib.resources import files
from pathlib import Path

from tilefoundry.codegen.context_builder import build_codegen_context
from tilefoundry.codegen.link_function import link_function
from tilefoundry.codegen.linked_function import LinkedFunction
from tilefoundry.codegen.linked_module import LinkedModule
from tilefoundry.codegen.options import CodegenOptions
from tilefoundry.dump import DumpFlags, DumpScope, dump
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.target import CpuTarget
from tilefoundry.target.base import UnsupportedCapabilityError, target_summary
from tilefoundry.target.services import CMakeFragment


def _default_cmake(target, units, workdir) -> CMakeFragment:
    sources = []
    for index, unit in enumerate(units):
        name = ("host" if index == 0 else f"host_{index}") if isinstance(target, CpuTarget) else f"{target.name}_{index}"
        source = f"{name}.{unit.language}"
        (workdir / source).write_text(unit.source)
        sources.append(source)
    return CMakeFragment((), tuple(sources))


def link_modules(
    units, *, workdir: str | Path, lib_name: str, functions: tuple[LinkedFunction, ...],
    host_cxx: str = "g++",
) -> LinkedModule:
    """Build all translation units in one target-configured CMake project."""
    units = tuple(units)
    import tvm_ffi  # noqa: PLC0415
    from jinja2 import Environment, FileSystemLoader, StrictUndefined  # noqa: PLC0415

    for tool in ("cmake", host_cxx):
        if shutil.which(tool) is None:
            raise RuntimeError(f"link_modules: {tool!r} not on PATH")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    by_target = {}
    for target, unit in units:
        by_target.setdefault(target, []).append(unit)
    fragments = [
        (target.get_code_generator().cmake or _default_cmake)(target, tuple(group), workdir)
        for target, group in by_target.items()
    ]
    includes = (
        files("tilefoundry.runtime").joinpath("include"),
        Path(tvm_ffi.__file__).resolve().parent / "include",
    )
    env = Environment(
        loader=FileSystemLoader(str(files("tilefoundry.codegen").joinpath("templates/cmake"))),
        undefined=StrictUndefined, keep_trailing_newline=True,
    )
    cmake_text = env.get_template("CMakeLists.txt.j2").render(
        name=lib_name, includes=includes, fragments=fragments,
        languages=tuple(dict.fromkeys(language for fragment in fragments for language in fragment.languages)),
        sources=tuple(source for fragment in fragments for source in fragment.sources),
    )
    (workdir / "CMakeLists.txt").write_text(cmake_text)
    build_dir = workdir / "build"
    configure_cmd = [
        "cmake", "-G", "Ninja", "-S", str(workdir), "-B", str(build_dir),
        f"-DCMAKE_CXX_COMPILER={host_cxx}",
    ]
    build_cmd = ["cmake", "--build", str(build_dir), "--verbose"]
    lib = build_dir / f"lib{lib_name}.so"
    with DumpScope("build"):
        dump("CMakeLists.txt", cmake_text, DumpFlags.BUILD_LOG)
        for fragment in fragments:
            for source in fragment.sources:
                dump(f"module.{source}", (workdir / source).read_text(), DumpFlags.BUILD_LOG)
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
    return LinkedModule(
        library_path=lib,
        source="\n".join(f"// {target.name} module\n{unit.source}" for target, unit in units),
        functions=functions,
    )


def link_program(
    mod: Module, *, workdir: str | Path,
    options: CodegenOptions = CodegenOptions(), entries=None,
) -> LinkedModule:
    """Generate code for the whole Module tree and link it into one library."""
    codegen = build_codegen_context(mod, options)
    entries = (mod.entry_function(),) if entries is None else tuple(entries)
    if not entries:
        raise ValueError("link_program: expected at least one CPU host entry")
    for entry in entries:
        if not (isinstance(entry, PrimFunction) and isinstance(entry.target, CpuTarget)):
            raise ValueError(f"link_program: entry {entry.name!r} is not a CPU host entry")
    groups = tuple(
        replace(group, functions=tuple(fn for fn in entries if id(fn) in {id(member) for member in group.functions}))
        if isinstance(group.target, CpuTarget) else group
        for group in codegen.groups
    )
    groups = tuple(group for group in groups if group.functions)
    generators = {group.target: group.target.get_code_generator() for group in groups}
    for target, generator in generators.items():
        if options.emit not in generator.emits:
            raise UnsupportedCapabilityError(
                f"{target_summary(target)}: code generator does not support emit={options.emit!r}"
            )
    units = tuple(
        (group.target, generators[group.target].emit(
            group.owner, group.functions, group.target, codegen.for_group(group),
        )) for group in groups
    )
    functions = tuple(link_function(codegen.symbols.by_function[id(entry)]) for entry in entries)
    return link_modules(units, workdir=workdir, lib_name=mod.name, functions=functions)


__all__ = ["link_program", "link_modules"]
