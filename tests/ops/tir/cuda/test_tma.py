"""Cover the CUDA staging-copy definition: direction, dtype, shape, and the call.

See [tir §2.3](docs/spec/tir.md#23-tir-ops).
"""

from __future__ import annotations

import pytest
import torch

import tilefoundry
import tilefoundry.codegen.cuda  # noqa: F401 -- trigger emitter autodiscovery
from tilefoundry import module, prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.core import Var, VerifyError
from tilefoundry.ir.tir.cuda.memory.tma import TmaCopy
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, Return, Sequential
from tilefoundry.ir.tir.verify import verify_prim_function
from tilefoundry.ir.types import DType, make_tensor_type
from tilefoundry.ir.types.shard import Layout, Mesh, ShardLayout, Topology
from tilefoundry.ir.types.shard.shard_layout import Broadcast
from tilefoundry.target import CpuTarget, CudaTarget

_BAR = make_tensor_type((1,), DType.from_name("i64"), storage="smem")


def _pf(src, dst, bar=_BAR) -> PrimFunction:
    args = tuple(Var(type=t, name=n) for t, n in ((src, "src"), (dst, "dst"), (bar, "bar")))
    return PrimFunction(
        name="fn",
        params=args,
        body=Sequential(body=(Evaluate(callable=TmaCopy(), args=args), Return())),
    )


def _ty(n, dtype=DType.from_name("f32"), storage="gmem"):
    return make_tensor_type((n,), dtype, storage=storage)


def test_accepts_a_whole_grain_gmem_to_smem_run() -> None:
    """The shape this op is built for: a gmem run into a shared tile."""
    verify_prim_function(_pf(_ty(8), _ty(8, storage="smem")))


def test_refuses_the_wrong_direction() -> None:
    """Only gmem into smem is this instruction."""
    with pytest.raises(VerifyError, match="source must be gmem"):
        verify_prim_function(_pf(_ty(8, storage="smem"), _ty(8, storage="smem")))
    with pytest.raises(VerifyError, match="destination must be smem"):
        verify_prim_function(_pf(_ty(8), _ty(8, storage="gmem")))


def test_refuses_a_barrier_outside_shared_memory() -> None:
    with pytest.raises(VerifyError, match="barrier must be smem"):
        verify_prim_function(
            _pf(
                _ty(8),
                _ty(8, storage="smem"),
                make_tensor_type((1,), DType.from_name("i64"), storage="gmem"),
            )
        )


def test_refuses_a_dtype_change() -> None:
    """A staging copy moves bytes; it does not convert them."""
    with pytest.raises(VerifyError, match="dtype mismatch"):
        verify_prim_function(_pf(_ty(8), _ty(8, DType.from_name("bf16"), storage="smem")))


def test_refuses_a_shape_change() -> None:
    with pytest.raises(VerifyError, match="shape mismatch"):
        verify_prim_function(_pf(_ty(8), _ty(4, storage="smem")))


@pytest.mark.parametrize(("n", "dtype"), [(5, DType.from_name("f32")), (1, DType.from_name("f32")), (7, DType.from_name("bf16"))])
def test_admits_a_transfer_off_the_sixteen_byte_grain(n, dtype) -> None:
    """The grain belongs to one instruction, and the op does not name one.

    See [runtime §3](docs/spec/runtime.md#3-runtime-ops).
    """
    verify_prim_function(_pf(_ty(n, dtype), _ty(n, dtype, storage="smem")))


@module(entry="tma_tiers_host")
class TmaTiers:
    """Both staging tiers in one device function, one operand pair each."""

    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def tma_tiers_device(
        bulk_a: Tensor[(256,), "f32"],
        bulk_b: Tensor[(256,), "f32"],
        odd_a: Tensor[(5,), "f32"],
        odd_b: Tensor[(5,), "f32"],
    ):
        with Mesh((Topology("thread", 128),), Layout(shape=(128,), strides=(1,)), ("t",)) as m:
            bulk_view = T.tensor_view(
                bulk_a,
                layout=ShardLayout(
                    layout=Layout(shape=(256,), strides=(1,)), attrs=(Broadcast(),), mesh=m
                ),
            )
            bulk_stage = T.alloc_tensor(
                Tensor[(256,), 'f32', ShardLayout(
                        layout=Layout(shape=(256,), strides=(1,)),
                        attrs=(Broadcast(),),
                        mesh=m,
                    ), 'smem']
            )
            bulk_bar = T.alloc_tensor(
                Tensor[(1,), 'i64', None, 'smem']
            )
            T.mbarrier_init(bulk_bar, arrive_count=1)
            T.sync(m)
            T.tma_copy(bulk_view, bulk_stage, bulk_bar)
            T.mbarrier_wait_parity(bulk_bar, 0)
            T.copy(bulk_stage, bulk_b)
        with Mesh((Topology("thread", 128),), Layout(shape=(128,), strides=(1,)), ("t",)) as mo:
            odd_view = T.tensor_view(
                odd_a,
                layout=ShardLayout(
                    layout=Layout(shape=(5,), strides=(1,)), attrs=(Broadcast(),), mesh=mo
                ),
            )
            odd_stage = T.alloc_tensor(
                Tensor[(5,), 'f32', ShardLayout(
                        layout=Layout(shape=(5,), strides=(1,)),
                        attrs=(Broadcast(),),
                        mesh=mo,
                    ), 'smem']
            )
            odd_bar = T.alloc_tensor(
                Tensor[(1,), 'i64', None, 'smem']
            )
            T.mbarrier_init(odd_bar, arrive_count=1)
            T.sync(mo)
            T.tma_copy(odd_view, odd_stage, odd_bar)
            T.mbarrier_wait_parity(odd_bar, 0)
            T.copy(odd_stage, odd_b)

    @prim_func(target=CpuTarget())
    def tma_tiers_host(
        bulk_a: Tensor[(256,), "f32"],
        bulk_b: Tensor[(256,), "f32"],
        odd_a: Tensor[(5,), "f32"],
        odd_b: Tensor[(5,), "f32"],
    ):
        launch(  # noqa: F821
            tma_tiers_device,  # noqa: F821
            bulk_a,
            bulk_b,
            odd_a,
            odd_b,
            grid=(1, 1, 1),
            block=(128, 1, 1),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_both_tiers_stage_the_input_unchanged() -> None:
    """One compile behind two assertions, each naming the tier that failed."""
    rm = tilefoundry.compile(TmaTiers, target=CudaTarget("nvidia.h200_sxm"))
    torch.manual_seed(0)
    bulk_a = torch.randn(256, dtype=torch.float32, device="cuda")
    bulk_b = torch.zeros(256, dtype=torch.float32, device="cuda")
    odd_a = torch.randn(5, dtype=torch.float32, device="cuda")
    odd_b = torch.zeros(5, dtype=torch.float32, device="cuda")
    rm(bulk_a, bulk_b, odd_a, odd_b)
    torch.cuda.synchronize()
    assert torch.equal(bulk_b, bulk_a), (
        "bulk tier (two contiguous runs of the same element type): one_run_v is "
        "cosize == size on the projected view, and a broadcast operand's "
        "projection is the allocation's own layout, so bulk_eligible_v is true "
        "at compile time. 256 floats is 1024 bytes, a whole number of grains, so "
        "the elected lane issues the instruction and the block waits on the phase"
    )
    assert torch.equal(odd_b, odd_a), (
        "strided tier (a byte count the bulk instruction refuses): five floats "
        "is 20 bytes and bytes & 15 is not zero, which is the check Bulk makes "
        "before issuing and now the only route into Strided by construction, "
        "the static tier having been retired -- every instance of the "
        "destination mesh strides the run, then one arrival says the tile is "
        "readable. Same entry, same barrier, same answer"
    )
