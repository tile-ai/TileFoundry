"""``ops::copy`` across the storage boundaries and both vector widths, one kernel.

See [runtime §2.6](docs/spec/runtime.md#26-cudaops).
"""

from __future__ import annotations

import pytest
import torch

import tilefoundry
import tilefoundry.codegen.cuda  # noqa: F401 -- trigger emitter autodiscovery
from tests.fixtures.tir.layouts import broadcast_run, split_pairs, split_rows, split_short_rows
from tilefoundry import module, prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.types import Layout, Mesh, Topology
from tilefoundry.target import CpuTarget, CudaTarget

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
            bcast_frag = T.alloc_tensor(Tensor[(4,), "f32", broadcast_run(mb), "rmem"])
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
