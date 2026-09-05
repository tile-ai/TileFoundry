"""Cover the CUDA bulk-copy definition: direction, dtype, shape and grain.

The op has no CUDA emit; the definition layer is what is verifiable here. The
implementation it names is exercised as a staged ring in
``tests/integration/test_runtime_device_ops.py``.

See [tir §2.3](docs/spec/tir.md#23-tir-ops).
"""

from __future__ import annotations

import pytest

from tilefoundry.ir.core import Var, VerifyError
from tilefoundry.ir.tir.cuda.memory.tma import TmaBulkCopy
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, Return, Sequential
from tilefoundry.ir.tir.verify import verify_prim_function
from tilefoundry.ir.types import DType, make_tensor_type

_BAR = make_tensor_type((1,), DType.i64, storage="smem")


def _pf(src, dst, bar=_BAR) -> PrimFunction:
    args = tuple(
        Var(type=t, name=n) for t, n in ((src, "src"), (dst, "dst"), (bar, "bar"))
    )
    return PrimFunction(
        name="fn",
        params=args,
        body=Sequential(body=(Evaluate(callable=TmaBulkCopy(), args=args), Return())),
    )


def _ty(n, dtype=DType.f32, storage="gmem"):
    return make_tensor_type((n,), dtype, storage=storage)


def test_accepts_a_whole_grain_gmem_to_smem_run() -> None:
    """8 f32 is 32 bytes: two whole 16-byte grains."""
    verify_prim_function(_pf(_ty(8), _ty(8, storage="smem")))


def test_refuses_the_wrong_direction() -> None:
    """Only gmem into smem is this instruction.

    This stages global into shared; the reverse is a different instruction, not this one with
    its operands swapped.
    """
    with pytest.raises(VerifyError, match="source must be gmem"):
        verify_prim_function(_pf(_ty(8, storage="smem"), _ty(8, storage="smem")))
    with pytest.raises(VerifyError, match="destination must be smem"):
        verify_prim_function(_pf(_ty(8), _ty(8, storage="gmem")))


def test_refuses_a_barrier_outside_shared_memory() -> None:
    with pytest.raises(VerifyError, match="barrier must be smem"):
        verify_prim_function(
            _pf(_ty(8), _ty(8, storage="smem"), make_tensor_type((1,), DType.i64, storage="gmem"))
        )


def test_refuses_a_dtype_change() -> None:
    """A bulk copy moves bytes; it does not convert them."""
    with pytest.raises(VerifyError, match="dtype mismatch"):
        verify_prim_function(_pf(_ty(8), _ty(8, DType.bf16, storage="smem")))


def test_refuses_a_shape_change() -> None:
    with pytest.raises(VerifyError, match="shape mismatch"):
        verify_prim_function(_pf(_ty(8), _ty(4, storage="smem")))


@pytest.mark.parametrize(
    ("n", "dtype", "byte_count"),
    [(5, DType.f32, 20), (1, DType.f32, 4), (7, DType.bf16, 14)],
)
def test_refuses_a_transfer_off_the_sixteen_byte_grain(n, dtype, byte_count) -> None:
    """A transfer off the 16-byte grain is refused.

    The instruction has no defined behaviour off the grain, so this is rejected rather than
    rounded up to the next whole one.
    """
    with pytest.raises(VerifyError, match=f"multiple of 16 bytes, got {byte_count}"):
        verify_prim_function(_pf(_ty(n, dtype), _ty(n, dtype, storage="smem")))


def test_a_bf16_run_is_measured_in_bytes_not_elements() -> None:
    """The grain is counted in bytes, not elements.

    8 bf16 is 16 bytes -- one grain -- while 8 f32 is two, so the check reads the dtype's width
    rather than the element count.
    """
    verify_prim_function(_pf(_ty(8, DType.bf16), _ty(8, DType.bf16, storage="smem")))
