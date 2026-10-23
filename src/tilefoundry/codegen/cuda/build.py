"""CUDA sources and compiler settings in the shared CMake project."""

from __future__ import annotations

import os
import shutil
from importlib.resources import files
from pathlib import Path

from tilefoundry.target.services import CMakeFragment


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


def cuda_cmake(target, units, workdir: Path) -> CMakeFragment:
    """Write device sources and return their additions to one CMake target."""
    if shutil.which("nvcc") is None:
        raise RuntimeError("link_modules: 'nvcc' not on PATH")
    sources = tuple(f"device_{index}.cu" for index in range(len(units)))
    for source, unit in zip(sources, units):
        (workdir / source).write_text(unit.source)
    includes = (files("tilefoundry.codegen.cuda").joinpath("include"), _cutlass_include())
    text = (
        'set_target_properties(${PROJECT_NAME} PROPERTIES\n'
        f'  CUDA_ARCHITECTURES "{target.arch.removeprefix("sm_")}"\n'
        '  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF)\n'
        f'set_source_files_properties({" ".join(sources)} PROPERTIES\n'
        '  COMPILE_OPTIONS "-std=c++20;-Wno-deprecated-gpu-targets")\n'
        'target_include_directories(${PROJECT_NAME} PRIVATE\n'
        + "".join(f'  "{path}"\n' for path in includes) + ')\n'
    )
    return CMakeFragment(("CUDA",), sources, text)
