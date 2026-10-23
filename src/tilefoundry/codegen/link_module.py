"""Generate and link a Module tree through its targets' build services."""

from __future__ import annotations

import shutil
import subprocess
from importlib.resources import files
from pathlib import Path

from tilefoundry.codegen.context import EmitContext
from tilefoundry.codegen.context_builder import EmissionGroup, build_codegen_context
from tilefoundry.codegen.linkable_module import LinkableModule
from tilefoundry.codegen.linked_function import LinkedFunction
from tilefoundry.codegen.linked_module import LinkedModule
from tilefoundry.codegen.options import CodegenOptions
from tilefoundry.dump import DumpFlags, DumpScope, dump
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.target import CpuTarget


class CMakeProject:
    """One shared project with target-independent build, dump, and link behavior."""

    def __init__(self, workdir: str | Path, lib_name: str, host_cxx: str = "g++") -> None:
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.lib_name = lib_name
        self.host_cxx = host_cxx
        self.sources: dict[str, str] = {}
        self.languages: list[str] = []
        self.includes: list[str] = []
        self.appended: list[str] = []

    def add_source(self, name: str, text: str) -> None:
        """Write one translation unit under its unique project-relative name."""
        if name in self.sources:
            raise ValueError(f"CMakeProject: duplicate source {name!r}")
        (self.workdir / name).write_text(text)
        self.sources[name] = text

    def enable_language(self, language: str) -> None:
        """Add one language to the CMake project."""
        if language != "CXX" and language not in self.languages:
            self.languages.append(language)

    def include(self, *dirs) -> None:
        """Add include directories without repeating an existing path."""
        for directory in dirs:
            path = str(directory)
            if path not in self.includes:
                self.includes.append(path)

    def append(self, cmake_text: str) -> None:
        """Append target-owned commands after the shared library declaration."""
        self.appended.append(cmake_text)

    def build(self, functions: tuple[LinkedFunction, ...]) -> LinkedModule:
        """Configure and build this project once, returning its library and exports."""
        import tvm_ffi  # noqa: PLC0415
        from jinja2 import Environment, FileSystemLoader, StrictUndefined  # noqa: PLC0415

        for tool in ("cmake", self.host_cxx):
            if shutil.which(tool) is None:
                raise RuntimeError(f"link_modules: {tool!r} not on PATH")
        self.include(
            files("tilefoundry.runtime").joinpath("include"),
            Path(tvm_ffi.__file__).resolve().parent / "include",
        )
        env = Environment(
            loader=FileSystemLoader(str(files("tilefoundry.codegen").joinpath("templates/cmake"))),
            undefined=StrictUndefined, keep_trailing_newline=True,
        )
        cmake_text = env.get_template("CMakeLists.txt.j2").render(
            name=self.lib_name, includes=self.includes, appended=self.appended,
            languages=self.languages, sources=self.sources,
        )
        (self.workdir / "CMakeLists.txt").write_text(cmake_text)
        build_dir = self.workdir / "build"
        configure_cmd = [
            "cmake", "-G", "Ninja", "-S", str(self.workdir), "-B", str(build_dir),
            f"-DCMAKE_CXX_COMPILER={self.host_cxx}",
        ]
        build_cmd = ["cmake", "--build", str(build_dir), "--verbose"]
        lib = build_dir / f"lib{self.lib_name}.so"
        with DumpScope("build"):
            dump("CMakeLists.txt", cmake_text, DumpFlags.BUILD_LOG)
            for name, source in self.sources.items():
                dump(f"module.{name}", source, DumpFlags.BUILD_LOG)
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
            source="\n".join(f"// {name}\n{source}" for name, source in self.sources.items()),
            functions=functions,
        )


def link_modules(
    modules: tuple[LinkableModule, ...], *, workdir: str | Path, lib_name: str,
    functions: tuple[LinkedFunction, ...], host, host_cxx: str = "g++",
) -> LinkedModule:
    """Let each translation unit join one project, then build the shared library."""
    project = CMakeProject(workdir, lib_name, host_cxx)
    for module in modules:
        module.link(project)
    return host.resolve(LinkedModule, project, functions)


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
    groups = (
        *(group for group in codegen.groups if not isinstance(group.target, CpuTarget)),
        EmissionGroup(mod, entries[0].target, entries),
    )
    modules = []
    for group in groups:
        ctx = group.target.resolve(EmitContext, codegen.for_group(group))
        modules.append(group.target.resolve(LinkableModule, group.owner, group.functions, ctx))
    functions = tuple(
        entry.target.resolve(LinkedFunction, codegen.symbols.by_function[id(entry)]) for entry in entries
    )
    return link_modules(
        tuple(modules), workdir=workdir, lib_name=mod.name, functions=functions, host=entries[0].target,
    )


__all__ = ["CMakeProject", "link_program", "link_modules"]
