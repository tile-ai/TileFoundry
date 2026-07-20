"""HIR matrix-multiply-accumulate ops typeinfer: the (M, N) accumulator
fragment in the accumulator dtype."""
from __future__ import annotations

import pytest

from tests.ops.typeinfer_utils import TypeInferCase, run_typeinfer_case
from tilefoundry.ir.hir.cuda.nn.mma import Mma_SM80_16x8x16, Wgmma_SM90_64x128x16
from tilefoundry.ir.types import DType, make_tensor_type
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
