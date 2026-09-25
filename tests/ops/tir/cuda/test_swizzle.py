"""Carry a CuTe XOR swizzle from the IR to shared memory and back.

See [shard §4.1](docs/spec/shard.md#41-swizzle).
"""

from __future__ import annotations

import pytest
import torch

import tilefoundry
import tilefoundry.codegen.cuda  # noqa: F401 -- trigger emitter autodiscovery
from tests._source import import_dsl
from tilefoundry import module, prim_func
from tilefoundry.codegen.cuda.tir.memory.tensor_view import render_shard_layout_value
from tilefoundry.dsl import T, Tensor
from tilefoundry.inspection import as_script
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.types import ComposedLayout, Layout, Mesh, ShardLayout, Split, Swizzle, Topology
from tilefoundry.target import CpuTarget, CudaTarget

_CUDA = CudaTarget("nvidia.h200_sxm")
_ROWS, _COLS = 128, 4
_SWIZZLE = Swizzle(2, 2, 2)
_THREADS = Mesh((Topology("thread", _ROWS),), Layout((_ROWS,), (1,)), ("t",))


def _rows(mesh: Mesh) -> ShardLayout:
    """One row per thread, addressed as the tile is laid out."""
    return ShardLayout(Layout((_ROWS, _COLS), (_COLS, 1)), (Split(0),), mesh)


def _swizzled_rows(mesh: Mesh) -> ShardLayout:
    """The same rows, reached through the swizzle.

    ``Swizzle(2, 2, 2)`` XORs Y bits 4-5 onto Z bits 2-3, so over this
    (128, 4) f32 tile thread ``t`` holds the row ``t ^ ((t >> 2) & 3)``: the
    address really moves, and the four elements of a row stay together.
    """
    return ShardLayout(
        ComposedLayout(inner=_SWIZZLE, offset=0, outer=Layout((_ROWS, _COLS), (_COLS, 1))),
        (Split(0),),
        mesh,
    )


@module(entry="swizzled_square_host", target=_CUDA, topologies=(Topology("thread", _ROWS),))
class SwizzledSquare:
    """Square a tile that is staged through swizzled shared memory."""

    @prim_func(target=_CUDA)
    def swizzled_square_device(
        src: Tensor[(_ROWS, _COLS), "f32"],
        dst: Tensor[(_ROWS, _COLS), "f32"],
    ):
        with Mesh((Topology("thread", _ROWS),), Layout((_ROWS,), (1,)), ("t",)) as threads:
            src_view = T.tensor_view(T.ptr_of(src), layout=_rows(threads))
            dst_view = T.tensor_view(T.ptr_of(dst), layout=_rows(threads))
            tile = T.alloc_tensor(Tensor[(_ROWS, _COLS), "f32", _swizzled_rows(threads), "smem"])
            fragment = T.alloc_tensor(Tensor[(_ROWS, _COLS), "f32", _rows(threads), "rmem"])
            T.copy(src_view, tile)
            T.sync(threads)
            T.copy(tile, fragment)
            T.binary(fragment, fragment, fragment, kind=BinaryKind.MUL)
            T.copy(fragment, dst_view)

    @prim_func(target=CpuTarget())
    def swizzled_square_host(
        src: Tensor[(_ROWS, _COLS), "f32"],
        dst: Tensor[(_ROWS, _COLS), "f32"],
    ):
        launch(  # noqa: F821
            swizzled_square_device,  # noqa: F821
            src,
            dst,
            grid=(1, 1, 1),
            block=(_ROWS, 1, 1),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_swizzled_smem_load_compiles_and_executes() -> None:
    """One kernel through print, parse, codegen, nvcc and the device.

    The emitted layout is asserted as well, because running the kernel cannot
    tell us about it: a swizzle permutes addresses, so writing and reading
    through the same layout returns the same numbers whether it is applied or
    dropped entirely.
    """
    source = as_script(SwizzledSquare)
    restored = import_dsl(source, name="SwizzledSquare")
    assert as_script(restored) == source

    preamble, _ = render_shard_layout_value("tile", _swizzled_rows(_THREADS))
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
