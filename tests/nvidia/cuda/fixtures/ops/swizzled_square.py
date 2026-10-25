"""Square a (128, 4) f32 tile staged through XOR-swizzled shared memory."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.target import CpuTarget
from tilefoundry_nvidia import CudaTarget

_CUDA = CudaTarget("nvidia.h200_sxm")
_ROWS, _COLS = 128, 4
_SWIZZLE = Swizzle(2, 2, 2)


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
