"""``ops::copy`` across the storage boundaries and both vector widths, one kernel."""

from __future__ import annotations

from tests.core.fixtures.tir.layouts import split_pairs, split_rows, split_short_rows
from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget
from tilefoundry_nvidia import CudaTarget

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
