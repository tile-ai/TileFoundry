"""Cover the CUDA staging-copy definition: direction, dtype and shape.

The op has no CUDA emit; the definition layer is what is verifiable here. The
implementation it names is exercised as a staged ring in
``tests/integration/test_runtime_device_ops.py``.

See [tir §2.3](docs/spec/tir.md#23-tir-ops).
"""

from __future__ import annotations

import pytest

from tilefoundry.ir.core import Var, VerifyError
from tilefoundry.ir.tir.cuda.memory.tma import TmaCopy
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
        body=Sequential(body=(Evaluate(callable=TmaCopy(), args=args), Return())),
    )


def _ty(n, dtype=DType.f32, storage="gmem"):
    return make_tensor_type((n,), dtype, storage=storage)


def test_accepts_a_whole_grain_gmem_to_smem_run() -> None:
    """The shape this op is built for: a gmem run into a shared tile."""
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
    """A staging copy moves bytes; it does not convert them."""
    with pytest.raises(VerifyError, match="dtype mismatch"):
        verify_prim_function(_pf(_ty(8), _ty(8, DType.bf16, storage="smem")))


def test_refuses_a_shape_change() -> None:
    with pytest.raises(VerifyError, match="shape mismatch"):
        verify_prim_function(_pf(_ty(8), _ty(4, storage="smem")))


@pytest.mark.parametrize(("n", "dtype"), [(5, DType.f32), (1, DType.f32), (7, DType.bf16)])
def test_admits_a_transfer_off_the_sixteen_byte_grain(n, dtype) -> None:
    """The grain belongs to one instruction, and the op does not name one.

    20 bytes cannot be a ``cp.async.bulk``, but it is a perfectly good staging
    copy on the element path. Rejecting it here would be the definition layer
    carrying a tier that [runtime §3](docs/spec/runtime.md#3-runtime-ops) puts
    behind the entry.
    """
    verify_prim_function(_pf(_ty(n, dtype), _ty(n, dtype, storage="smem")))
