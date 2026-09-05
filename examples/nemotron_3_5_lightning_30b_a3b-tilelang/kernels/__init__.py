"""Build ``nemotron.cu`` on first import and hand back the loaded extension.

The build is cached under ``kernels/.build`` keyed by nvcc's own timestamp
check, so a run after an edit recompiles and a run after nothing does not.

Unlike ``granite_4_0_h_small-cuda``, this kernel does not carry its own
primitives: it includes ``tilefoundry/runtime/cuda/runtime.cuh`` and calls the
public ``tilefoundry::ops`` entries. That header is not a standalone translation
unit -- it includes each ``ops/<op>.cuh`` in-context inside its own namespace --
so the two include roots below are the repository's ``include/`` and the
vendored CuTe, the same pair ``tilefoundry.codegen.linker`` puts on its nvcc
line.
"""
from __future__ import annotations

import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BUILD = _HERE / ".build"
_ROOT = _HERE.parents[2]

#: sm_90a: the H200 is Hopper, and the `elect.sync`, `mbarrier.*` and
#: `cp.async.bulk` instructions this kernel is made of are the `a` variants'.
_ARCH = os.environ.get("NEMOTRON_CUDA_ARCH", "90a")

_ext = None


def ops():
    """The compiled extension module, built on the first call."""
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load  # noqa: PLC0415

        _BUILD.mkdir(exist_ok=True)
        _ext = load(
            name="nemotron_kernels",
            sources=[str(_HERE / "nemotron.cu")],
            build_directory=str(_BUILD),
            extra_include_paths=[
                str(_ROOT / "include"),
                str(_ROOT / "third_party" / "cutlass" / "include"),
            ],
            extra_cuda_cflags=[
                "-O3",
                # No `--use_fast_math`: it swaps in the approximate
                # transcendentals, and this kernel's exponentials are compared
                # elementwise against torch's accurate ones.
                "-lineinfo",
                "-std=c++20",
                f"-gencode=arch=compute_{_ARCH},code=sm_{_ARCH}",
                "--expt-relaxed-constexpr",
            ],
            extra_cflags=["-O3", "-std=c++20"],
            verbose=os.environ.get("NEMOTRON_BUILD_VERBOSE", "") == "1",
        )
    return _ext


__all__ = ["ops"]
