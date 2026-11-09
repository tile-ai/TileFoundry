"""Carry a CuTe XOR swizzle from the IR to shared memory and back.

See [shard §4.1](docs/spec/shard.md#41-swizzle).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import tilefoundry
from tests.core._corpus import reload
from tilefoundry import cli, inspection
from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda
from tilefoundry_nvidia.cuda.codegen.tir import memory

SWIZZLED_SQUARE = Path(__file__).parents[2] / "fixtures" / "ops" / "swizzled_square.py"
_CUDA = cuda.Target("nvidia.h200_sxm")
_ROWS, _COLS = 128, 4
_THREADS = Mesh((Topology("thread", _ROWS),), Layout((_ROWS,), (1,)), ("t",))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_swizzled_smem_load_executes() -> None:
    """One kernel through print, parse, codegen, nvcc and the device.

    The emitted layout is asserted as well, because running the kernel cannot
    tell us about it: a swizzle permutes addresses, so writing and reading
    through the same layout returns the same numbers whether it is applied or
    dropped entirely.
    """
    namespace = cli.load_namespace(str(SWIZZLED_SQUARE))[0]
    SwizzledSquare, _swizzled_rows = namespace["SwizzledSquare"], namespace["_swizzled_rows"]
    source = inspection.as_script(SwizzledSquare)
    restored = reload(source, "SwizzledSquare")
    assert inspection.as_script(restored) == source

    preamble, _ = memory.render_shard_layout_value("tile", _swizzled_rows(_THREADS))
    assert any(
        "cute::make_composed_layout(cute::Swizzle<2, 2, 2>{}, cute::Int<0>{}, "
        "cute::make_layout(cute::make_shape(cute::Int<128>{}, cute::Int<4>{}), "
        "cute::make_stride(cute::Int<4>{}, cute::Int<1>{})))" in line
        for line in preamble
    ), preamble

    runtime_module = tilefoundry.compile(restored, target=_CUDA)
    torch.manual_seed(0)
    src = torch.randn(_ROWS, _COLS, dtype=torch.float32, device="cuda")
    dst = torch.zeros_like(src)
    runtime_module(src, dst)
    torch.cuda.synchronize()

    assert torch.equal(dst, src * src)
