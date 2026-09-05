"""Cover the CUDA mbarrier definitions: shared-memory residence and counts.

These ops have no CUDA emit; the definition layer is what is verifiable here.
The implementations they name are exercised as a producer/consumer pipeline in
``tests/integration/test_runtime_device_ops.py``.

See [tir §2.3](docs/spec/tir.md#23-tir-ops).
"""

from __future__ import annotations

import pytest

from tilefoundry.ir.core import Var, VerifyError
from tilefoundry.ir.tir.cuda.sync.mbarrier import (
    MBarrierArrive,
    MBarrierArriveExpectTx,
    MBarrierExpectTx,
    MBarrierInit,
    MBarrierInvalidate,
    MBarrierWaitParity,
)
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, Return, Sequential
from tilefoundry.ir.tir.verify import verify_prim_function
from tilefoundry.ir.types import DType, make_tensor_type

_SMEM_BAR = make_tensor_type((1,), DType.i64, storage="smem")
_GMEM_BAR = make_tensor_type((1,), DType.i64, storage="gmem")
_PHASE = make_tensor_type((), DType.i32, storage="rmem")


def _pf(op, *types) -> PrimFunction:
    args = tuple(Var(type=t, name=f"a{i}") for i, t in enumerate(types))
    return PrimFunction(
        name="fn",
        params=args,
        body=Sequential(body=(Evaluate(callable=op, args=args), Return())),
    )


_BARRIER_ONLY = [
    pytest.param(MBarrierArrive(), id="arrive"),
    pytest.param(MBarrierInvalidate(), id="invalidate"),
]


def test_init_accepts_a_shared_barrier_and_a_positive_count() -> None:
    verify_prim_function(_pf(MBarrierInit(arrive_count=1), _SMEM_BAR))


@pytest.mark.parametrize("count", [0, -1])
def test_init_refuses_a_non_positive_arrive_count(count: int) -> None:
    """A non-positive arrive count is refused.

    A phase needing zero arrivals is complete before anything is produced, which
    turns every consumer's wait into a no-op.
    """
    with pytest.raises(VerifyError, match="arrive_count must be a positive int"):
        verify_prim_function(_pf(MBarrierInit(arrive_count=count), _SMEM_BAR))


@pytest.mark.parametrize("op", _BARRIER_ONLY)
def test_barrier_only_entries_accept_a_shared_barrier(op) -> None:
    verify_prim_function(_pf(op, _SMEM_BAR))


@pytest.mark.parametrize("op", _BARRIER_ONLY)
def test_barrier_only_entries_refuse_a_global_barrier(op) -> None:
    with pytest.raises(VerifyError, match="barrier must be smem"):
        verify_prim_function(_pf(op, _GMEM_BAR))


def test_init_refuses_a_global_barrier() -> None:
    """A barrier outside shared memory is refused.

    The instructions take a shared-window address, so a barrier in global memory
    is not a slower barrier -- it is not one at all.
    """
    with pytest.raises(VerifyError, match="barrier must be smem"):
        verify_prim_function(_pf(MBarrierInit(arrive_count=1), _GMEM_BAR))


@pytest.mark.parametrize(
    "make", [MBarrierArriveExpectTx, MBarrierExpectTx], ids=["arrive_expect_tx", "expect_tx"]
)
def test_expect_tx_entries_accept_a_positive_byte_count(make) -> None:
    verify_prim_function(_pf(make(tx_bytes=4096), _SMEM_BAR))


@pytest.mark.parametrize(
    "make", [MBarrierArriveExpectTx, MBarrierExpectTx], ids=["arrive_expect_tx", "expect_tx"]
)
@pytest.mark.parametrize("tx", [0, -16])
def test_expect_tx_entries_refuse_a_non_positive_byte_count(make, tx: int) -> None:
    with pytest.raises(VerifyError, match="tx_bytes must be a positive int"):
        verify_prim_function(_pf(make(tx_bytes=tx), _SMEM_BAR))


def test_wait_parity_accepts_a_shared_barrier_and_a_phase() -> None:
    verify_prim_function(_pf(MBarrierWaitParity(), _SMEM_BAR, _PHASE))


def test_wait_parity_refuses_a_global_barrier() -> None:
    with pytest.raises(VerifyError, match="barrier must be smem"):
        verify_prim_function(_pf(MBarrierWaitParity(), _GMEM_BAR, _PHASE))
