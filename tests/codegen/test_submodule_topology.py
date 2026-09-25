"""End-to-end compilation of child CUDA Modules with independent topologies."""

from __future__ import annotations

import pytest
import torch

import tilefoundry
from tilefoundry import module, prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- bind bare launch
from tilefoundry.ir.types import Layout, Mesh, ShardLayout, Split, Topology
from tilefoundry.target import CpuTarget, CudaTarget

_CUDA = CudaTarget("nvidia.h200_sxm")


@module(topologies=(Topology("thread", 32),))
class _Left:
    @prim_func(target=_CUDA)
    def left_device(x: Tensor[(32,), "f32"], out: Tensor[(32,), "f32"]):
        with Mesh((Topology("thread", 32),), Layout((32,), (1,))) as thread:
            layout = ShardLayout(Layout((32,), (1,)), (Split(0),), thread)
            src = T.tensor_view(T.ptr_of(x), layout=layout)
            dst = T.tensor_view(T.ptr_of(out), layout=layout)
            T.copy(src, dst)
            T.sync(thread)


@module(topologies=(Topology("thread", 128),))
class _Right:
    @prim_func(target=_CUDA)
    def right_device(x: Tensor[(128,), "f32"], out: Tensor[(128,), "f32"]):
        with Mesh((Topology("thread", 128),), Layout((128,), (1,))) as thread:
            layout = ShardLayout(Layout((128,), (1,)), (Split(0),), thread)
            src = T.tensor_view(T.ptr_of(x), layout=layout)
            dst = T.tensor_view(T.ptr_of(out), layout=layout)
            T.copy(src, dst)
            T.sync(thread)


@module(entry="host", target=_CUDA)
class _Parent:
    left = _Left
    right = _Right

    @prim_func(target=CpuTarget())
    def host(
        x0: Tensor[(32,), "f32"],
        o0: Tensor[(32,), "f32"],
        x1: Tensor[(128,), "f32"],
        o1: Tensor[(128,), "f32"],
    ):
        launch(left.left_device, x0, o0, grid=(1, 1, 1), block=(32, 1, 1))
        launch(right.right_device, x1, o1, grid=(1, 1, 1), block=(128, 1, 1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_parent_links_two_topology_domains() -> None:
    torch.manual_seed(0)
    x0 = torch.randn(32, device="cuda")
    x1 = torch.randn(128, device="cuda")
    o0 = torch.empty_like(x0)
    o1 = torch.empty_like(x1)
    runtime = tilefoundry.compile(_Parent, target=_CUDA)
    runtime(x0, o0, x1, o1)
    torch.cuda.synchronize()
    torch.testing.assert_close(o0, x0)
    torch.testing.assert_close(o1, x1)
