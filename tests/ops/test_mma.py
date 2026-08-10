"""HIR matrix-multiply-accumulate ops typeinfer: the accumulator fragment in the accumulator dtype.

HIR matrix-multiply-accumulate ops typeinfer: the (M, N) accumulator
fragment in the accumulator dtype.
"""

from __future__ import annotations

import pytest

from tests.ops.typeinfer_utils import TypeInferCase, infer_call, run_typeinfer_case
from tilefoundry.ir.hir.cuda.nn.mma import Mma_SM80_16x8x16, Wgmma_SM90_64x128x16
from tilefoundry.ir.types import DType, make_tensor_type
from tilefoundry.ir.types.shard import Layout
from tilefoundry.ir.types.storage import StorageKind

_BF = DType.bf16
_RMEM = StorageKind.RMEM


CASES = [
    TypeInferCase(
        "mma_sm80_16x8x16",
        Mma_SM80_16x8x16(dtype_a=_BF, dtype_b=_BF, dtype_acc=DType.f32),
        (
            make_tensor_type((16, 16), _BF, storage=_RMEM),
            make_tensor_type((16, 8), _BF, storage=_RMEM),
        ),
        make_tensor_type((16, 8), DType.f32, storage=_RMEM),
    ),
    TypeInferCase(
        "wgmma_sm90_64x128x16",
        Wgmma_SM90_64x128x16(dtype_a=_BF, dtype_b=_BF, dtype_acc=DType.f32),
        (
            make_tensor_type((64, 16), _BF, storage=_RMEM),
            make_tensor_type((16, 128), _BF, storage=_RMEM),
        ),
        make_tensor_type((64, 128), DType.f32, storage=_RMEM),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_mma_typeinfer(case):
    run_typeinfer_case(case)


def test_mma_layout_describes_the_accumulator_result():
    a = make_tensor_type(
        (16, 16),
        _BF,
        storage=_RMEM,
        layout=Layout(shape=(16, 16), strides=(16, 1)),
    )
    b = make_tensor_type(
        (16, 8),
        _BF,
        storage=_RMEM,
        layout=Layout(shape=(16, 8), strides=(8, 1)),
    )

    result = infer_call(Mma_SM80_16x8x16(dtype_a=_BF, dtype_b=_BF, dtype_acc=DType.f32), a, b)

    assert result.layout == Layout(shape=(16, 8), strides=(8, 1))
