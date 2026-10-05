"""Every reduce tier in one kernel, each forced there by its layouts alone.

See [runtime §2.6](docs/spec/runtime.md#26-cudaops).
"""

from __future__ import annotations

import isl
import pytest
import torch

import tilefoundry
import tilefoundry.codegen.cuda  # noqa: F401 -- trigger emitter autodiscovery
from tilefoundry import module, prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.core import Call, Var, VerifyError
from tilefoundry.ir.core.kinds import ReduceKind
from tilefoundry.ir.isl_interop import shape_to_isl_set
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.reduce import Reduce
from tilefoundry.ir.tir.stmts import Evaluate, Return, Sequential
from tilefoundry.ir.types import (
    ComposedLayout,
    DType,
    Layout,
    Mesh,
    ShardLayout,
    Split,
    Swizzle,
    Topology,
    UnitType,
    make_tensor_type,
)
from tilefoundry.ir.types.shard_layout import Broadcast
from tilefoundry.ir.types.stride import compact_row_major
from tilefoundry.target import CpuTarget, CudaTarget
from tilefoundry.visitor_registry.access_relation import relation_of, relations_of
from tilefoundry.visitor_registry.contexts import TypeInferContext
from tilefoundry.visitor_registry.verify import verify_prim_function

_CUDA = CudaTarget("nvidia.h200_sxm")
_MESH = Mesh((Topology("thread", 32),), Layout((32,), (1,)), ("t",))
_OTHER_MESH = Mesh((Topology("thread", 32),), Layout((4, 8), (8, 1)), ("w", "t"))


def _sharded(
    n: int,
    *,
    dtype=DType.f32,
    storage="rmem",
    mesh: Mesh = _MESH,
    layout=None,
):
    return make_tensor_type(
        (n,),
        dtype,
        storage=storage,
        layout=ShardLayout(layout or Layout((n,), (1,)), (Broadcast(),), mesh),
    )


def _pf(*types, axes: tuple = (0,)) -> PrimFunction:
    args = tuple(Var(type=type_, name=f"a{index}") for index, type_ in enumerate(types))
    return PrimFunction(
        name="fn",
        params=args,
        body=Sequential(
            body=(
                Evaluate(
                    callable=Reduce(axes=axes, keepdim=True, kind=ReduceKind.MEAN),
                    args=args,
                ),
                Return(),
            )
        ),
    )


_THREADS_128 = Mesh((Topology("thread", 128),), Layout((128,), (1,)), ("t",))


def _split(shape: tuple, axis: int):
    """A source cut along *axis* over the 128-thread mesh."""
    strides = (shape[1], 1)
    return make_tensor_type(
        shape,
        DType.f32,
        storage="rmem",
        layout=ShardLayout(Layout(shape, strides), (Split(axis),), _THREADS_128),
    )


def _replicated(shape: tuple, storage: str = "rmem"):
    return make_tensor_type(
        shape,
        DType.f32,
        storage=storage,
        layout=ShardLayout(Layout(shape, compact_row_major(shape)), (Broadcast(),), _THREADS_128),
    )


@pytest.mark.parametrize(
    ("operands", "axes", "refused"),
    [
        pytest.param(
            (_sharded(8), _sharded(1), _sharded(4, storage="rmem")),
            (0,),
            r"workspace .* does not match StorageKind.SMEM",
            id="outside_shared_memory",
        ),
        pytest.param(
            (_split((128, 8), 0), _replicated((1, 8)), _replicated((2,), "smem")),
            (0,),
            r"workspace \(2,\) must be rank-1 with at least 4 slots",
            id="fewer_slots_than_warps",
        ),
        pytest.param(
            (_split((128, 8), 0), _replicated((1, 8)), _replicated((2, 2), "smem")),
            (0,),
            r"workspace \(2, 2\) must be rank-1",
            id="not_rank_one",
        ),
        pytest.param(
            (_split((128, 8), 0), _replicated((1, 8)), _replicated((128,), "smem")),
            (0,),
            None,
            id="split_axis_0_wider_workspace",
        ),
        pytest.param(
            (_split((8, 128), 1), _replicated((8, 1)), _replicated((4,), "smem")),
            (1,),
            None,
            id="split_axis_1",
        ),
    ],
)
def test_a_workspace_holds_one_slot_per_warp(operands, axes, refused) -> None:
    """A workspace is shared memory with one slot per warp of the source's mesh.

    What every source coordinate reaches there is those slots, all of them and
    no others, however wide the workspace was allocated: which warp posts its
    partial where is the runtime's choice, not a coordinate of the source.
    """
    function = _pf(*operands, axes=axes)
    if refused is not None:
        with pytest.raises(VerifyError, match=refused):
            verify_prim_function(function)
        return
    verify_prim_function(function)
    (statement,) = (stmt for stmt in function.body.body if isinstance(stmt, Evaluate))
    call = Call(type=UnitType(), target=statement.callable, args=statement.args)
    reached = relation_of(relations_of(call, TypeInferContext()).inputs[2].pattern)
    source = shape_to_isl_set(tuple(operands[0].shape), {})
    assert reached.is_equal(isl.map.from_domain_and_range(source, isl.set("{ [s] : 0 <= s < 4 }")))


def test_refuses_a_dtype_that_does_not_fold_in_float() -> None:
    with pytest.raises(
        VerifyError,
        match=r"could not satisfy src_dtype in \{f32, f16, bf16\}",
    ):
        verify_prim_function(_pf(_sharded(8, dtype=DType.fp8e4m3), _sharded(1)))


def test_refuses_a_swizzled_source_layout() -> None:
    swizzled = ComposedLayout(Swizzle(2, 2, 2), 0, Layout((8,), (1,)))
    with pytest.raises(
        VerifyError,
        match=r"does not match Layout\(\(\*_\,\), \(\*_\,\)\)",
    ):
        verify_prim_function(_pf(_sharded(8, layout=swizzled), _sharded(1)))


def test_refuses_operands_on_different_meshes() -> None:
    with pytest.raises(VerifyError, match=r"does not match \*mesh_shape"):
        verify_prim_function(_pf(_sharded(8), _sharded(1, mesh=_OTHER_MESH)))


@module(topologies=(Topology("thread", 128),))
class _Plain:
    @prim_func(target=_CUDA)
    def plain(a_plain: Tensor[(128, 8), "f32"], out_plain: Tensor[(128,), "f32"]):
        with Mesh((Topology("thread", 128),), Layout(shape=(128,), strides=(1,)), ("t",)) as mp:
            plain_src = T.tensor_view(
                T.ptr_of(a_plain), layout=ShardLayout(Layout((128, 8), (8, 1)), (Split(0),), mp)
            )
            plain_dst = T.tensor_view(
                T.ptr_of(out_plain), layout=ShardLayout(Layout((128,), (1,)), (Split(0),), mp)
            )
            T.reduce(
                plain_src,
                plain_dst,
                axes=(1,),
                keepdim=False,
                kind=ReduceKind.MEAN,
            )


@module(topologies=(Topology("thread", 32),))
class _Warp:
    @prim_func(target=_CUDA)
    def intra_warp(a_warp: Tensor[(32, 4), "f32"], out_warp: Tensor[(1,), "f32"]):
        with Mesh((Topology("thread", 32),), Layout(shape=(32,), strides=(1,)), ("t",)) as mw:
            warp_src = T.tensor_view(
                T.ptr_of(a_warp), layout=ShardLayout(Layout((32, 4), (4, 1)), (Split(0),), mw)
            )
            warp_dst = T.tensor_view(
                T.ptr_of(out_warp), layout=ShardLayout(Layout((1,), (1,)), (Broadcast(),), mw)
            )
            T.reduce(
                warp_src,
                warp_dst,
                axes=(1,),
                keepdim=False,
                kind=ReduceKind.ABS_MAX,
            )


@module(topologies=(Topology("thread", 128),))
class _Cta:
    @prim_func(target=_CUDA)
    def intra_cta(a_cta: Tensor[(4, 32, 8), "f32"], out_cta: Tensor[(1,), "f32"]):
        with Mesh(
            (Topology("thread", 128),), Layout(shape=(4, 32), strides=(32, 1)), ("w", "t")
        ) as mc:
            cta_src = T.tensor_view(
                T.ptr_of(a_cta),
                layout=ShardLayout(Layout((4, 32, 8), (256, 8, 1)), (Split(0), Split(1)), mc),
            )
            cta_dst = T.tensor_view(
                T.ptr_of(out_cta),
                layout=ShardLayout(Layout((1,), (1,)), (Broadcast(), Broadcast()), mc),
            )
            cta_ws = T.alloc_tensor(Tensor[(4,), "f32", None, "smem"])
            T.reduce(
                cta_src,
                cta_dst,
                cta_ws,
                axes=(2,),
                keepdim=False,
                kind=ReduceKind.MEAN,
            )


@module(topologies=(Topology("thread", 128),))
class _Cross:
    @prim_func(target=_CUDA)
    def cross_warp(a_cross: Tensor[(4, 32), "f32"], out_cross: Tensor[(1, 32), "f32"]):
        with Mesh(
            (Topology("thread", 128),), Layout(shape=(4, 32), strides=(32, 1)), ("w", "t")
        ) as mx:
            cross_src = T.tensor_view(
                T.ptr_of(a_cross),
                layout=ShardLayout(Layout((4, 32), (32, 1)), (Split(0), Split(1)), mx),
            )
            cross_dst = T.tensor_view(
                T.ptr_of(out_cross),
                layout=ShardLayout(Layout((1, 32), (32, 1)), (Broadcast(), Split(1)), mx),
            )
            cross_ws = T.alloc_tensor(Tensor[(128,), "f32", None, "smem"])
            T.reduce(
                cross_src,
                cross_dst,
                cross_ws,
                axes=(0,),
                keepdim=True,
                kind=ReduceKind.ABS_MAX,
            )


@module(entry="reduce_tiers_host", target=_CUDA)
class ReduceTiers:
    plain = _Plain
    warp = _Warp
    cta = _Cta
    cross = _Cross

    @prim_func(target=CpuTarget())
    def reduce_tiers_host(
        a_plain: Tensor[(128, 8), "f32"],
        out_plain: Tensor[(128,), "f32"],
        a_warp: Tensor[(32, 4), "f32"],
        out_warp: Tensor[(1,), "f32"],
        a_cta: Tensor[(4, 32, 8), "f32"],
        out_cta: Tensor[(1,), "f32"],
        a_cross: Tensor[(4, 32), "f32"],
        out_cross: Tensor[(1, 32), "f32"],
    ):
        launch(plain.plain, a_plain, out_plain, grid=(1, 1, 1), block=(128, 1, 1))
        launch(warp.intra_warp, a_warp, out_warp, grid=(1, 1, 1), block=(32, 1, 1))
        launch(cta.intra_cta, a_cta, out_cta, grid=(1, 1, 1), block=(128, 1, 1))
        launch(cross.cross_warp, a_cross, out_cross, grid=(1, 1, 1), block=(128, 1, 1))


_ENTRY = r"tilefoundry::ops::reduce<tilefoundry::ops::"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_every_tier_answers_what_torch_answers() -> None:
    """One compile behind four assertions, each naming the tier that failed."""
    rm = tilefoundry.compile(ReduceTiers, target=_CUDA)
    torch.manual_seed(0)
    a_plain = torch.randn(128, 8, dtype=torch.float32, device="cuda")
    out_plain = torch.zeros(128, dtype=torch.float32, device="cuda")
    a_warp = torch.randn(32, 4, dtype=torch.float32, device="cuda")
    out_warp = torch.zeros(1, dtype=torch.float32, device="cuda")
    a_cta = torch.randn(4, 32, 8, dtype=torch.float32, device="cuda")
    out_cta = torch.zeros(1, dtype=torch.float32, device="cuda")
    a_cross = torch.randn(4, 32, dtype=torch.float32, device="cuda")
    out_cross = torch.zeros(1, 32, dtype=torch.float32, device="cuda")
    rm(a_plain, out_plain, a_warp, out_warp, a_cta, out_cta, a_cross, out_cross)
    torch.cuda.synchronize()
    assert torch.allclose(out_plain, a_plain.mean(dim=1), rtol=1e-6, atol=1e-6), (
        "plain tier (mesh_reduced false: dst splits the axis src splits): mean "
        "divides by the reduced span alone -- 8 -- which is the only count this "
        "tier has, so a tier that had brought a mesh extent into the divisor "
        "answers 8 or 32 times small"
    )
    assert torch.allclose(out_warp, a_warp.abs().max().reshape(1), rtol=0, atol=0), (
        "intra-warp tier (warps_per_group == 1 on a flat 32-lane mesh): a fold "
        "that added the lanes' maxima instead of maxing them answers roughly 32 "
        "times large"
    )
    assert torch.allclose(out_cta, a_cta.mean().reshape(1), rtol=1e-5, atol=1e-6), (
        "intra-CTA tier (lane_reduced with warps_per_group == 4): mean's divisor "
        "is the product of all three counts the tier knows -- span 8, lanes 32, "
        "warps 4 -- so the 1024 it must divide by is exactly what the greedy "
        "warp walk this dispatch replaced got wrong"
    )
    assert torch.allclose(out_cross[0], a_cross.abs().amax(dim=0), rtol=0, atol=0), (
        "cross-warp tier (warps_per_group == 4 with lane_reduced false): absmax "
        "is what shows the fold combining with the reduction's own operator "
        "rather than adding"
    )
