"""``ops::copy`` across the storage boundaries and both vector widths, one kernel.

See [runtime §2.6](docs/spec/runtime.md#26-cudaops).
"""

from __future__ import annotations

import pytest
import torch

import tilefoundry
import tilefoundry_cuda.codegen  # noqa: F401 -- trigger emitter autodiscovery
from tests.fixtures.tir.layouts import split_pairs, split_rows, split_short_rows
from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget
from tilefoundry_cuda import CudaTarget

_CUDA = CudaTarget("nvidia.h200_sxm")


@module(entry="copy_storage_host", target=_CUDA, topologies=(Topology("thread", 128),))
class CopyStorage:
    """Five operand pairs in one device function, one per thing that can differ."""

    @prim_func(target=_CUDA)
    def copy_storage_device(
        a_smem: Tensor[(128, 4), "f32"],
        b_smem: Tensor[(128, 4), "f32"],
        a_rmem: Tensor[(128, 4), "f32"],
        b_rmem: Tensor[(128, 4), "f32"],
        a_bcast: Tensor[(32, 4), "f32"],
        b_bcast: Tensor[(32, 4), "f32"],
        a_wide: Tensor[(128, 4), "f32"],
        b_wide: Tensor[(128, 4), "f32"],
        a_narrow: Tensor[(128, 2), "f32"],
        b_narrow: Tensor[(128, 2), "f32"],
    ):
        with Mesh((Topology("thread", 128),), Layout(shape=(128,), strides=(1,)), ("t",)) as m:
            smem_src = T.tensor_view(T.ptr_of(a_smem), layout=split_rows(m))
            smem_dst = T.tensor_view(T.ptr_of(b_smem), layout=split_rows(m))
            smem_tile = T.alloc_tensor(Tensor[(128, 4), "f32", split_rows(m), "smem"])
            T.copy(smem_src, smem_tile)
            T.sync(m)
            T.copy(smem_tile, smem_dst)
            rmem_src = T.tensor_view(T.ptr_of(a_rmem), layout=split_rows(m))
            rmem_dst = T.tensor_view(T.ptr_of(b_rmem), layout=split_rows(m))
            rmem_tile = T.alloc_tensor(Tensor[(128, 4), "f32", split_rows(m), "smem"])
            rmem_frag = T.alloc_tensor(Tensor[(128, 4), "f32", split_rows(m), "rmem"])
            T.copy(rmem_src, rmem_tile)
            T.sync(m)
            T.copy(rmem_tile, rmem_frag)
            T.copy(rmem_frag, rmem_dst)
            wide_src = T.tensor_view(T.ptr_of(a_wide), layout=split_rows(m))
            wide_dst = T.tensor_view(T.ptr_of(b_wide), layout=split_rows(m))
            wide_frag = T.alloc_tensor(Tensor[(128, 4), "f32", split_rows(m), "rmem"])
            T.copy(wide_src, wide_frag)
            T.copy(wide_frag, wide_dst)
            narrow_src = T.tensor_view(T.ptr_of(a_narrow), layout=split_pairs(m))
            narrow_dst = T.tensor_view(T.ptr_of(b_narrow), layout=split_pairs(m))
            narrow_frag = T.alloc_tensor(Tensor[(128, 2), "f32", split_pairs(m), "rmem"])
            T.copy(narrow_src, narrow_frag)
            T.copy(narrow_frag, narrow_dst)
        with Mesh((Topology("thread", 32),), Layout(shape=(32,), strides=(1,)), ("t",)) as mb:
            bcast_src = T.tensor_view(T.ptr_of(a_bcast), layout=split_short_rows(mb))
            bcast_dst = T.tensor_view(T.ptr_of(b_bcast), layout=split_short_rows(mb))
            bcast_frag = T.alloc_tensor(Tensor[(32, 4), "f32", split_short_rows(mb), "rmem"])
            T.copy(bcast_src, bcast_frag)
            T.copy(bcast_frag, bcast_dst)

    @prim_func(target=CpuTarget())
    def copy_storage_host(
        a_smem: Tensor[(128, 4), "f32"],
        b_smem: Tensor[(128, 4), "f32"],
        a_rmem: Tensor[(128, 4), "f32"],
        b_rmem: Tensor[(128, 4), "f32"],
        a_bcast: Tensor[(32, 4), "f32"],
        b_bcast: Tensor[(32, 4), "f32"],
        a_wide: Tensor[(128, 4), "f32"],
        b_wide: Tensor[(128, 4), "f32"],
        a_narrow: Tensor[(128, 2), "f32"],
        b_narrow: Tensor[(128, 2), "f32"],
    ):
        launch(  # noqa: F821
            copy_storage_device,  # noqa: F821
            a_smem,
            b_smem,
            a_rmem,
            b_rmem,
            a_bcast,
            b_bcast,
            a_wide,
            b_wide,
            a_narrow,
            b_narrow,
            grid=(1, 1, 1),
            block=(128, 1, 1),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_every_crossing_round_trips_the_input_bit_for_bit() -> None:
    """One compile behind five assertions, each naming the pair that failed."""
    rm = tilefoundry.compile(CopyStorage, target=_CUDA)
    torch.manual_seed(0)
    a_smem = torch.randn(128, 4, dtype=torch.float32, device="cuda")
    b_smem = torch.zeros_like(a_smem)
    a_rmem = torch.randn(128, 4, dtype=torch.float32, device="cuda")
    b_rmem = torch.zeros_like(a_rmem)
    a_bcast = torch.arange(128, dtype=torch.float32, device="cuda").view(32, 4)
    b_bcast = torch.full((32, 4), -1.0, dtype=torch.float32, device="cuda")
    a_wide = torch.randn(128, 4, dtype=torch.float32, device="cuda")
    b_wide = torch.zeros_like(a_wide)
    a_narrow = torch.randn(128, 2, dtype=torch.float32, device="cuda")
    b_narrow = torch.zeros_like(a_narrow)
    rm(
        a_smem,
        b_smem,
        a_rmem,
        b_rmem,
        a_bcast,
        b_bcast,
        a_wide,
        b_wide,
        a_narrow,
        b_narrow,
    )
    torch.cuda.synchronize()
    assert torch.equal(b_smem, a_smem), (
        "gmem->smem, smem->gmem: shared memory belongs to the CTA, so the "
        "allocation is sized to the whole (128, 4) tile while local() offsets "
        "each thread into its own row -- the pair of facts that a buffer sized "
        "to one instance's share gets wrong for every instance but the first"
    )
    assert torch.equal(b_rmem, a_rmem), (
        "smem->rmem, rmem->gmem: a thread's registers are its own, so local() "
        "applies no offset to the fragment; the shared side does carry one, "
        "which is what makes this crossing different from the global one"
    )
    assert torch.equal(b_bcast, a_bcast), (
        "broadcast register tile: local() hands a register engine back by "
        "reference for a broadcast ShardLayout exactly as for a split one, so a "
        "value written into the tile has to be there to read back -- "
        f"{int((b_bcast == -1.0).sum())} of 128 cells never written is a lost "
        "write rather than a wrong offset, which would move data, not delete it"
    )
    assert torch.equal(b_wide, a_wide), (
        "gmem->rmem at 128 bits: a static contiguous run of four floats is what "
        "selects the vector load, and it must copy through the fragment "
        "unchanged"
    )
    assert torch.equal(b_narrow, a_narrow), (
        "gmem->rmem at 64 bits: a sub-128-bit fragment falls back to the element "
        "loop, which must answer identically to the vector path"
    )


@module(
    target=_CUDA,
    topologies=(Topology("cta", 1), Topology("thread", 256)),
)
class _RegisterRoundTrip:
    """The optimize tutorial's register round trip, at its own mesh and shard.

    A (2, 5120) bf16 window on cta 1 x thread (2, 4, 32): the row axis over
    ``x``, the 40 blocks over ``y``, the 128-wide block over ``t`` -- a
    40-element fragment per thread, and both crossings of the plain/shard
    boundary, since the tutorial's output copy keeps its window plain.
    """

    @prim_func(target=_CUDA)
    def round_trip(a: Tensor[(2, 5120), "bf16"], out: Tensor[(2, 5120), "bf16"]):
        with Mesh((Topology("cta", 1),), layout=(1,), names=("tile",)) as _cta:
            with Mesh(
                (Topology("thread", 256),), layout=(2, 4, 32), names=("x", "y", "t")
            ) as scope:
                value = T.alloc_tensor(
                    tensor_type=Tensor[
                        (2, 5120),
                        "bf16",
                        (2 @ scope.x, 40 @ scope.y, 128 @ scope.t),
                        "rmem",
                    ]
                )
                dst_view = T.tensor_view(
                    T.ptr_of(value),
                    layout=(2 @ scope.x, 40 @ scope.y, 128 @ scope.t),
                    shape=(2, 5120),
                )
                T.copy(a, dst_view)
                src_view = T.tensor_view(
                    T.ptr_of(value),
                    layout=(2 @ scope.x, 40 @ scope.y, 128 @ scope.t),
                    shape=(2, 5120),
                )
                T.copy(src_view, out)


@module(entry="round_trip_host", target=_CUDA)
class RegisterRoundTrip:
    """Launch the one CTA the shard is written for."""

    round_trip = _RegisterRoundTrip

    @prim_func(target=CpuTarget())
    def round_trip_host(a: Tensor[(2, 5120), "bf16"], out: Tensor[(2, 5120), "bf16"]):
        launch(  # noqa: F821
            round_trip.round_trip,  # noqa: F821
            a,
            out,
            grid=(1, 1, 1),
            block=(256, 1, 1),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_a_plain_window_round_trips_through_thread_fragments() -> None:
    """A copy with one plain and one sharded operand moves the instance's slice.

    ``local_tensor`` projects each operand through its own shard layout, so a
    plain operand used to stay whole for every instance: the gmem->rmem leg
    read the window's first 40 elements into every fragment, and the rmem->gmem
    leg tiled the fragment across the whole window (or, vectorized, read it out
    of bounds -- the tutorial's ``Invalid __local__ read of size 2``). The
    plain side is the slice the shard side's layout hands this instance.
    """
    rm = tilefoundry.compile(RegisterRoundTrip, target=_CUDA)
    torch.manual_seed(0)
    a = torch.randn(2, 5120, dtype=torch.bfloat16, device="cuda")
    out = torch.zeros_like(a)
    rm(a, out)
    torch.cuda.synchronize()
    assert torch.equal(out, a), (
        "plain->shard->plain round trip: each thread holds the 40 elements its "
        "(x, y, t) coordinate names, so the window must reassemble exactly -- "
        "every thread copying the window's head, or tiling its fragment across "
        "the whole window, is the boundary projected wrong"
    )


@module(
    target=_CUDA,
    topologies=(Topology("thread", 32),),
)
class _OneWay:
    """A plain (4, 64) tile entering a column-pair shard, one direction only.

    Lane ``t`` of 32 owns columns ``2t`` and ``2t+1`` of every row: the
    plain->shard leg takes the ``local_as`` projection, the shard->shard leg
    the path that predates it, so a slice placed wrong on the way in lands in
    the wrong output columns directly.
    """

    @prim_func(target=_CUDA)
    def one_way(a: Tensor[(4, 64), "f32"], out: Tensor[(4, 64), "f32"]):
        with Mesh((Topology("thread", 32),), layout=(32,), names=("t",)) as m:
            src = T.tensor_view(T.ptr_of(a), layout=((4, 64), (64, 1)))
            reg = T.alloc_tensor(
                tensor_type=Tensor[(4, 64), "f32", ((4, 32 @ m.t, 2), (64, 2, 1)), "rmem"]
            )
            dst = T.tensor_view(T.ptr_of(out), layout=((4, 32 @ m.t, 2), (64, 2, 1)))
            T.copy(src, reg)
            T.copy(reg, dst)


@module(entry="one_way_host", target=_CUDA)
class OneWay:
    """Launch the one warp the shard is written for."""

    one_way = _OneWay

    @prim_func(target=CpuTarget())
    def one_way_host(a: Tensor[(4, 64), "f32"], out: Tensor[(4, 64), "f32"]):
        launch(  # noqa: F821
            one_way.one_way,  # noqa: F821
            a,
            out,
            grid=(1, 1, 1),
            block=(32, 1, 1),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_a_plain_window_enters_thread_fragments_in_row_major_order() -> None:
    """The projection's row-major steps must not come out transposed.

    CuTe decodes an integer index first-mode-fastest, so composing the
    instance's row-major window and origin into the plain tensor's rank-2
    layout unreversed hands every lane a permutation of its slice: lane 1 of
    a (2, 4) tile reads element (1, 0) where it owns (0, 2). A round trip
    hides it -- write and read share the permutation -- so this test copies
    in once and compares against the input.
    """
    rm = tilefoundry.compile(OneWay, target=_CUDA)
    a = torch.arange(4 * 64, dtype=torch.float32, device="cuda").view(4, 64)
    out = torch.zeros_like(a)
    rm(a, out)
    torch.cuda.synchronize()
    assert torch.equal(out, a), (
        "plain->shard with arange input: lane t must hold columns 2t and "
        "2t+1 of each row, so a transposed or otherwise permuted slice shows "
        "up as out != a exactly where the projection misread the index"
    )
