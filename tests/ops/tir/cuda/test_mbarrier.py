"""Cover the CUDA mbarrier definitions and the instructions they emit.

See [tir §2.3](docs/spec/tir.md#23-tir-ops).
"""

from __future__ import annotations

import pytest

import tilefoundry
import tilefoundry.codegen.cuda  # noqa: F401 -- trigger emitter autodiscovery
from tilefoundry import module, prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.core import Var, VerifyError
from tilefoundry.ir.tir.cuda.sync.mbarrier import (
    MBarrierArriveExpectTx,
    MBarrierInit,
    MBarrierInvalidate,
    MBarrierWaitParity,
)
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, Return, Sequential
from tilefoundry.ir.tir.verify import verify_prim_function
from tilefoundry.ir.types import DType, Layout, Mesh, Topology, make_tensor_type
from tilefoundry.target import CpuTarget, CudaTarget

_SMEM_BAR = make_tensor_type((1,), DType.from_name("i64"), storage="smem")
_GMEM_BAR = make_tensor_type((1,), DType.from_name("i64"), storage="gmem")
_PHASE = make_tensor_type((), DType.from_name("i32"), storage="rmem")


def _pf(op, *types) -> PrimFunction:
    args = tuple(Var(type=t, name=f"a{i}") for i, t in enumerate(types))
    return PrimFunction(
        name="fn",
        params=args,
        body=Sequential(body=(Evaluate(callable=op, args=args), Return())),
    )


def test_init_accepts_a_shared_barrier_and_a_positive_count() -> None:
    verify_prim_function(_pf(MBarrierInit(arrive_count=1), _SMEM_BAR))


@pytest.mark.parametrize("count", [0, -1])
def test_init_refuses_a_non_positive_arrive_count(count: int) -> None:
    """A non-positive arrive count is refused."""
    with pytest.raises(VerifyError, match="arrive_count must be a positive int"):
        verify_prim_function(_pf(MBarrierInit(arrive_count=count), _SMEM_BAR))


def test_invalidate_accepts_a_shared_barrier() -> None:
    verify_prim_function(_pf(MBarrierInvalidate(), _SMEM_BAR))


@pytest.mark.parametrize(
    "stated",
    [
        pytest.param(_pf(MBarrierInit(arrive_count=1), _GMEM_BAR), id="init"),
        pytest.param(_pf(MBarrierArriveExpectTx(tx_bytes=16), _GMEM_BAR), id="arrive_expect_tx"),
        pytest.param(_pf(MBarrierWaitParity(), _GMEM_BAR, _PHASE), id="wait_parity"),
        pytest.param(_pf(MBarrierInvalidate(), _GMEM_BAR), id="invalidate"),
    ],
)
def test_every_entry_refuses_a_barrier_outside_shared_memory(stated) -> None:
    """A barrier outside shared memory is refused."""
    with pytest.raises(VerifyError, match="barrier must be smem"):
        verify_prim_function(stated)


def test_arrive_expect_tx_accepts_a_positive_byte_count() -> None:
    verify_prim_function(_pf(MBarrierArriveExpectTx(tx_bytes=4096), _SMEM_BAR))


@pytest.mark.parametrize("tx", [0, -16])
def test_arrive_expect_tx_refuses_a_non_positive_byte_count(tx: int) -> None:
    with pytest.raises(VerifyError, match="tx_bytes must be a positive int"):
        verify_prim_function(_pf(MBarrierArriveExpectTx(tx_bytes=tx), _SMEM_BAR))


def test_wait_parity_accepts_a_shared_barrier_and_a_phase() -> None:
    verify_prim_function(_pf(MBarrierWaitParity(), _SMEM_BAR, _PHASE))


@module(entry="mbarrier_ring_host", target=CudaTarget("nvidia.h200_sxm"))
class MBarrierRing:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def mbarrier_ring_device(a: Tensor[(4,), "f32"]):
        with Mesh((Topology("thread", 128),), Layout(shape=(128,), strides=(1,)), ("t",)) as m:
            bar = T.alloc_tensor(
                Tensor[(1,), "i64", None, "smem"]
            )
            T.mbarrier_init(bar, arrive_count=1)
            T.sync(m)
            T.mbarrier_arrive_expect_tx(bar, tx_bytes=1024)
            T.mbarrier_wait_parity(bar, 0)
            T.mbarrier_invalidate(bar)

    @prim_func(target=CpuTarget())
    def mbarrier_ring_host(a: Tensor[(4,), "f32"]):
        launch(mbarrier_ring_device, a, grid=(1, 1, 1), block=(128, 1, 1))  # noqa: F821

