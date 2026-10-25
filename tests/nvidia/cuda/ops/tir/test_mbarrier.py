"""Cover the CUDA mbarrier definitions and the instructions they emit.

See [tir §2.3](docs/spec/tir.md#23-tir-ops).
"""

from __future__ import annotations

import pytest

import tilefoundry_nvidia.cuda.codegen  # noqa: F401 -- trigger emitter autodiscovery
from tilefoundry.dsl import *
from tilefoundry.ir.core import Var, VerifyError
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, MeshScope, Return, Sequential
from tilefoundry.visitor_registry.verify import verify_prim_function
from tilefoundry_nvidia import CudaTarget
from tilefoundry_nvidia.cuda.ir.tir.sync.mbarrier import (
    MBarrierArriveExpectTx,
    MBarrierInit,
    MBarrierInvalidate,
    MBarrierWaitParity,
)

_SMEM_BAR = make_tensor_type((1,), DType.from_name("i64"), storage="smem")
_GMEM_BAR = make_tensor_type((1,), DType.from_name("i64"), storage="gmem")
_PHASE = make_tensor_type((), DType.from_name("i32"), storage="rmem")


def _pf(op, *types) -> PrimFunction:
    args = tuple(Var(type=t, name=f"a{i}") for i, t in enumerate(types))
    mesh = Mesh((Topology("thread", 32),), Layout((32,), (1,)))
    return PrimFunction(
        target=CudaTarget("nvidia.h200_sxm"),
        name="fn",
        params=args,
        body=Sequential(
            body=(
                MeshScope(
                    mesh=mesh,
                    binding=Var(name="thread", type=mesh),
                    body=Sequential(body=(Evaluate(callable=op, args=args),)),
                ),
                Return(),
            )
        ),
    )


def test_mbarrier_init_accepts_shared_barrier() -> None:
    verify_prim_function(_pf(MBarrierInit(arrive_count=1), _SMEM_BAR))


@pytest.mark.parametrize("count", [0, -1])
def test_mbarrier_init_refuses_nonpositive_count(count: int) -> None:
    """A non-positive arrive count is refused."""
    with pytest.raises(VerifyError, match="arrive_count: could not satisfy arrive_count > 0"):
        verify_prim_function(_pf(MBarrierInit(arrive_count=count), _SMEM_BAR))


def test_mbarrier_invalidate_accepts_shared_barrier() -> None:
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
def test_mbarrier_refuses_gmem_barrier(stated) -> None:
    """A barrier outside shared memory is refused."""
    with pytest.raises(VerifyError, match="barrier: gmem does not match smem"):
        verify_prim_function(stated)


def test_expect_tx_accepts_positive_bytes() -> None:
    verify_prim_function(_pf(MBarrierArriveExpectTx(tx_bytes=4096), _SMEM_BAR))


@pytest.mark.parametrize("tx", [0, -16])
def test_expect_tx_refuses_nonpositive_bytes(tx: int) -> None:
    with pytest.raises(VerifyError, match="tx_bytes: could not satisfy tx_bytes > 0"):
        verify_prim_function(_pf(MBarrierArriveExpectTx(tx_bytes=tx), _SMEM_BAR))


def test_wait_parity_accepts_shared_barrier() -> None:
    verify_prim_function(_pf(MBarrierWaitParity(), _SMEM_BAR, _PHASE))
