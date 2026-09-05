"""Compile ``device_ops.cu`` against the CUDA runtime headers, once per session.

The runtime's op headers are not standalone translation units -- ``runtime.cuh``
includes each ``ops/<op>.cuh`` in-context inside ``namespace tilefoundry::ops``
-- so a caller includes ``runtime.cuh`` and adds two include roots: the
repository's ``include/`` and the vendored CuTe. Those are the same two roots
``tilefoundry.codegen.linker`` puts on its own nvcc line, recomputed here rather
than imported because the linker's are private to codegen and this compile
deliberately does not go through codegen.
"""
from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_HERE = Path(__file__).resolve().parent
_ARCH = os.environ.get("TILEFOUNDRY_TEST_CUDA_ARCH", "90a")

_ext = None


def device_ops():
    """The compiled extension, built on the first call.

    ``sm_90a`` by default: the ``elect.sync``, ``mbarrier.*`` and
    ``cp.async.bulk`` instructions these ops are made of are Hopper's, and the
    ``a`` variants are what the architecture's own instructions need.
    """
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load  # noqa: PLC0415

        _ext = load(
            name="tilefoundry_device_ops_test",
            sources=[str(_HERE / "device_ops.cu")],
            extra_include_paths=[
                str(_ROOT / "include"),
                str(_ROOT / "third_party" / "cutlass" / "include"),
            ],
            extra_cuda_cflags=[
                "-O2",
                "-std=c++20",
                f"-gencode=arch=compute_{_ARCH},code=sm_{_ARCH}",
                "--expt-relaxed-constexpr",
            ],
            extra_cflags=["-O2", "-std=c++20"],
            verbose=os.environ.get("TILEFOUNDRY_TEST_BUILD_VERBOSE", "") == "1",
        )
    return _ext


__all__ = ["device_ops"]
