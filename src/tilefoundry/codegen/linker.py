"""Linker — the final stage of codegen.

Separately compiles each target's :class:`~tilefoundry.codegen.linkable.LinkableModule`
translation unit and links them into one host-callable shared library, returning
a ``LinkedModule`` (artifact + host-visible ABI metadata) for the runtime loader
to turn into a ``RuntimeModule``.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from tilefoundry.dump import DumpFlags, DumpScope, dump
from tilefoundry.runtime.module import CallableType, KernelInfo, LaunchConfig

_TILEFOUNDRY_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_INCLUDE = _TILEFOUNDRY_ROOT / "include"
_DEFAULT_CUTLASS_INCLUDE = _TILEFOUNDRY_ROOT / "third_party" / "cutlass" / "include"
_CMAKE_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates" / "cmake"


@dataclass(frozen=True)
class LinkedModule:
    """Linked .so + host-visible ABI metadata. Consumed by the runtime loader."""
    library_path: Path
    source: str
    entry: CallableType
    launch_config: LaunchConfig
    kernels: tuple[KernelInfo, ...]


def _render_cmakelists(*, name: str, includes: list[str], device_options: str, cuda_arch: str) -> str:
    """Render the split-pipeline CMake project from its Jinja template."""
    # noqa lazy: jinja2 is already a codegen dep; import here keeps the linker
    # load path light and avoids a hard import at module load.
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
        name=name, includes=includes, device_options=device_options, cuda_arch=cuda_arch
    )


def _tvm_ffi_include() -> Path:
    # noqa lazy: tvm_ffi is an optional runtime dep; only required when
    # actually launching kernels, so avoid an import-time hard dep.
    import tvm_ffi  # noqa: PLC0415
    return Path(tvm_ffi.__file__).resolve().parent / "include"


def link_modules(
    modules,
    *,
    workdir: str | Path,
    lib_name: str,
    entry: CallableType,
    launch_config: LaunchConfig,
    kernels: tuple[KernelInfo, ...],
    nvcc: str = "nvcc",
    host_cxx: str = "g++",
    extra_nvcc_flags: tuple[str, ...] = (),
    cuda_arch: str = "90",
    include_dirs: tuple[Path, ...] = (),
) -> LinkedModule:
    """Separately compile a device ``cu`` module and a host ``cpp`` module, then
    link them into one host-callable shared library; return the ``LinkedModule``.

    Requires exactly one ``cu`` module and exactly one ``cpp`` module. The device
    module compiles with nvcc, the host module with a plain host compiler; the
    final link runs through nvcc (to pull in the CUDA runtime) and statically
    links libstdc++ so the ``.so`` keeps the GLIBCXX-independent load behaviour.
    """
    modules = tuple(modules)
    if sorted(m.language for m in modules) != ["cpp", "cu"]:
        received = ", ".join(f"({m.target}, {m.language})" for m in modules)
        raise ValueError(
            f"link_modules: requires exactly one 'cu' and one 'cpp' "
            f"module, got [{received}]"
        )
    cu = [m for m in modules if m.language == "cu"]
    cpp = [m for m in modules if m.language == "cpp"]
    for tool in (nvcc, host_cxx, "cmake"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"link_modules: {tool!r} not on PATH")

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "device.cu").write_text(cu[0].source)
    (workdir / "host.cpp").write_text(cpp[0].source)

    # Device include set + nvcc flags. The arch is parameterized: the CMake
    # target's ``CUDA_ARCHITECTURES {cuda_arch}`` emits both ``compute_<arch>``
    # PTX and ``sm_<arch>`` SASS; ``-fPIC`` by ``POSITION_INDEPENDENT_CODE``;
    # ``-static-libstdc++`` by the template's link options. The host compiler
    # reuses the same include set.
    tvm_inc = _tvm_ffi_include()
    includes = [_DEFAULT_INCLUDE, _DEFAULT_CUTLASS_INCLUDE, tvm_inc, *include_dirs]
    device_options = ";".join(("-Wno-deprecated-gpu-targets", *extra_nvcc_flags))
    cmake_text = _render_cmakelists(
        name=lib_name,
        includes=[str(p) for p in includes],
        device_options=device_options,
        cuda_arch=cuda_arch,
    )
    (workdir / "CMakeLists.txt").write_text(cmake_text)

    build_dir = workdir / "build"
    # No CMAKE_BUILD_TYPE: keep the prior compile semantics (no -O3 / -DNDEBUG
    # injected) — nvcc still optimizes device code by default, the host wrapper
    # keeps its asserts.
    configure_cmd = [
        "cmake", "-G", "Ninja", "-S", str(workdir), "-B", str(build_dir),
        f"-DCMAKE_CUDA_COMPILER={nvcc}", f"-DCMAKE_CXX_COMPILER={host_cxx}",
    ]
    build_cmd = ["cmake", "--build", str(build_dir), "--verbose"]
    lib = build_dir / f"lib{lib_name}.so"

    with DumpScope("build"):
        dump("CMakeLists.txt", cmake_text, DumpFlags.BUILD_LOG)
        dump("module.device.cu", cu[0].source, DumpFlags.BUILD_LOG)
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
        f"// cuda module\n{cu[0].source}"
    )
    return LinkedModule(
        library_path=lib,
        source=source,
        entry=entry,
        launch_config=launch_config,
        kernels=kernels,
    )


__all__ = ["LinkedModule", "link_modules"]
