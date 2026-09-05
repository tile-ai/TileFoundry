"""Cover the CUDA warp-op definitions: what they accept and what they refuse.

These ops have no CUDA emit, so what is verifiable here is the definition layer
-- the operand storage and dtype agreements, and the attribute ranges the
hardware imposes. The implementations they name are exercised against torch in
``tests/integration/test_runtime_device_ops.py``.

See [tir §2.3](docs/spec/tir.md#23-tir-ops).
"""

from __future__ import annotations

import pytest

from tilefoundry.ir.core import Var, VerifyError
from tilefoundry.ir.tir.cuda.warp import (
    ShuffleElect,
    ShuffleXor,
    WarpReduce,
    WarpReduceKind,
)
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, Return, Sequential
from tilefoundry.ir.tir.verify import verify_prim_function
from tilefoundry.ir.types import DType, make_tensor_type


def _pf(op, *types) -> PrimFunction:
    args = tuple(Var(type=t, name=f"a{i}") for i, t in enumerate(types))
    return PrimFunction(
        name="fn",
        params=args,
        body=Sequential(body=(Evaluate(callable=op, args=args), Return())),
    )


def _reg(dtype: DType = DType.f32):
    return make_tensor_type((), dtype, storage="rmem")


def test_shuffle_xor_accepts_a_register_pair_and_an_in_warp_mask() -> None:
    verify_prim_function(_pf(ShuffleXor(lane_mask=16), _reg(), _reg()))


@pytest.mark.parametrize("lane_mask", [0, 32, 64, -1])
def test_shuffle_xor_refuses_a_mask_outside_the_warp(lane_mask: int) -> None:
    """A mask outside 1..31 is refused.

    A mask of 0 exchanges a lane with itself and 32 or more names a lane that is not in this
    warp; neither is a butterfly step.
    """
    with pytest.raises(VerifyError, match="lane_mask must be in 1..31"):
        verify_prim_function(_pf(ShuffleXor(lane_mask=lane_mask), _reg(), _reg()))


def test_shuffle_xor_refuses_an_operand_outside_registers() -> None:
    smem = make_tensor_type((), DType.f32, storage="smem")
    with pytest.raises(VerifyError, match="src must be rmem"):
        verify_prim_function(_pf(ShuffleXor(lane_mask=1), smem, _reg()))
    with pytest.raises(VerifyError, match="dst must be rmem"):
        verify_prim_function(_pf(ShuffleXor(lane_mask=1), _reg(), smem))


def test_shuffle_xor_refuses_a_dtype_change() -> None:
    with pytest.raises(VerifyError, match="dtype mismatch"):
        verify_prim_function(_pf(ShuffleXor(lane_mask=1), _reg(DType.f32), _reg(DType.bf16)))


def test_shuffle_elect_accepts_a_positive_width() -> None:
    verify_prim_function(_pf(ShuffleElect(width=256), _reg()))


@pytest.mark.parametrize("width", [0, -8])
def test_shuffle_elect_refuses_a_non_positive_width(width: int) -> None:
    with pytest.raises(VerifyError, match="width must be a positive int"):
        verify_prim_function(_pf(ShuffleElect(width=width), _reg()))


@pytest.mark.parametrize("kind", list(WarpReduceKind))
def test_warp_reduce_accepts_every_combine(kind: WarpReduceKind) -> None:
    verify_prim_function(_pf(WarpReduce(kind=kind), _reg(), _reg()))


def test_warp_reduce_refuses_a_kind_that_is_not_the_enum() -> None:
    with pytest.raises(VerifyError, match="kind must be WarpReduceKind"):
        verify_prim_function(_pf(WarpReduce(kind="sum"), _reg(), _reg()))


def test_warp_reduce_refuses_a_dtype_change() -> None:
    with pytest.raises(VerifyError, match="dtype mismatch"):
        verify_prim_function(
            _pf(WarpReduce(kind=WarpReduceKind.SUM), _reg(DType.f32), _reg(DType.bf16))
        )
