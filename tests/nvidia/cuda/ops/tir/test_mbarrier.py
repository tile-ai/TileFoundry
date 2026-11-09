"""Cover the CUDA mbarrier definitions and the instructions they emit.

See [tir §2.3](docs/spec/tir.md#23-tir-ops).
"""

from __future__ import annotations

import pytest

from tilefoundry import verifier
from tilefoundry.dsl import *
from tilefoundry.ir import core, expr
from tilefoundry_nvidia import cuda
from tilefoundry_nvidia.cuda.ir.tir import sync

_SMEM_BAR = TensorType.from_shape((1,), DType.from_name("i64"), storage="smem")
_GMEM_BAR = TensorType.from_shape((1,), DType.from_name("i64"), storage="gmem")
_PHASE = TensorType.from_shape((), DType.from_name("i32"), storage="rmem")


def _pf(op, *types) -> core.PrimFunction:
    args = tuple(expr.Var(type=t, name=f"a{i}") for i, t in enumerate(types))
    mesh = Mesh((Topology("thread", 32),), Layout((32,), (1,)))
    return core.PrimFunction(
        target=cuda.Target("nvidia.h200_sxm"),
        name="fn",
        params=args,
        body=core.Sequential(
            body=(
                core.MeshScope(
                    mesh=mesh,
                    binding=expr.Var(name="thread", type=mesh),
                    body=core.Sequential(body=(core.Evaluate(callable=op, args=args),)),
                ),
                core.Return(),
            )
        ),
    )


def test_mbarrier_init_accepts_shared_barrier() -> None:
    verifier.verify_prim_function(_pf(sync.MBarrierInit(arrive_count=1), _SMEM_BAR))


@pytest.mark.parametrize("count", [0, -1])
def test_mbarrier_init_refuses_nonpositive_count(count: int) -> None:
    """A non-positive arrive count is refused."""
    with pytest.raises(expr.VerifyError, match="arrive_count: could not satisfy arrive_count > 0"):
        verifier.verify_prim_function(_pf(sync.MBarrierInit(arrive_count=count), _SMEM_BAR))


def test_mbarrier_invalidate_accepts_shared_barrier() -> None:
    verifier.verify_prim_function(_pf(sync.MBarrierInvalidate(), _SMEM_BAR))


@pytest.mark.parametrize(
    "stated",
    [
        pytest.param(_pf(sync.MBarrierInit(arrive_count=1), _GMEM_BAR), id="init"),
        pytest.param(_pf(sync.MBarrierArriveExpectTx(tx_bytes=16), _GMEM_BAR), id="arrive_expect_tx"),
        pytest.param(_pf(sync.MBarrierWaitParity(), _GMEM_BAR, _PHASE), id="wait_parity"),
        pytest.param(_pf(sync.MBarrierInvalidate(), _GMEM_BAR), id="invalidate"),
    ],
)
def test_mbarrier_refuses_gmem_barrier(stated) -> None:
    """A barrier outside shared memory is refused."""
    with pytest.raises(expr.VerifyError, match="barrier: gmem does not match smem"):
        verifier.verify_prim_function(stated)


def test_expect_tx_accepts_positive_bytes() -> None:
    verifier.verify_prim_function(_pf(sync.MBarrierArriveExpectTx(tx_bytes=4096), _SMEM_BAR))


@pytest.mark.parametrize("tx", [0, -16])
def test_expect_tx_refuses_nonpositive_bytes(tx: int) -> None:
    with pytest.raises(expr.VerifyError, match="tx_bytes: could not satisfy tx_bytes > 0"):
        verifier.verify_prim_function(_pf(sync.MBarrierArriveExpectTx(tx_bytes=tx), _SMEM_BAR))


def test_wait_parity_accepts_shared_barrier() -> None:
    verifier.verify_prim_function(_pf(sync.MBarrierWaitParity(), _SMEM_BAR, _PHASE))
