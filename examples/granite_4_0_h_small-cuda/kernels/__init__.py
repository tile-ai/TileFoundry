"""Build ``granite.cu`` on first import and hand back the loaded extension.

The build is cached under ``kernels/.build`` keyed by nvcc's own timestamp
check, so a run after an edit recompiles and a run after nothing does not.
"""
from __future__ import annotations

import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BUILD = _HERE / ".build"

#: sm_90a: the H200 is Hopper, and the `a` variants are what the architecture's
#: own instructions need.
_ARCH = os.environ.get("GRANITE_CUDA_ARCH", "90a")

_ext = None


def ops():
    """The compiled extension module, built on the first call."""
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load  # noqa: PLC0415

        _BUILD.mkdir(exist_ok=True)
        _ext = load(
            name="granite_kernels",
            sources=[str(_HERE / "granite.cu")],
            build_directory=str(_BUILD),
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-lineinfo",
                f"-gencode=arch=compute_{_ARCH},code=sm_{_ARCH}",
                "--expt-relaxed-constexpr",
            ],
            extra_cflags=["-O3"],
            verbose=os.environ.get("GRANITE_BUILD_VERBOSE", "") == "1",
        )
    return _ext


__all__ = ["ops"]
