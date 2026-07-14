"""HIR matrix-multiply-accumulate ops typeinfer: the (M, N) accumulator
fragment in the accumulator dtype."""
from __future__ import annotations

import pytest

from tests.ops.typeinfer_utils import TypeInferCase, run_typeinfer_case, ten
from tilefoundry.ir.hir.cuda.nn.mma import Mma, Mma_SM80_16x8x16, Wgmma_SM90_64x128x16
from tilefoundry.ir.target.storage import StorageKind
from tilefoundry.ir.types import DType

_BF = DType.bf16
_RMEM = StorageKind.RMEM


def _rmem(shape):
    return ten(shape, _BF, storage=_RMEM)


CASES = [
    TypeInferCase(
        "mma_sm80_16x8x16",
        Mma_SM80_16x8x16(dtype_a=_BF, dtype_b=_BF, dtype_acc=DType.f32),
        (_rmem((16, 16)), _rmem((16, 8))),
        ten((16, 8), DType.f32, storage=_RMEM),
    ),
    TypeInferCase(
        "wgmma_sm90_64x128x16",
        Wgmma_SM90_64x128x16(dtype_a=_BF, dtype_b=_BF, dtype_acc=DType.f32),
        (_rmem((64, 16)), _rmem((16, 128))),
        ten((64, 128), DType.f32, storage=_RMEM),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_mma_typeinfer(case):
    run_typeinfer_case(case)


def test_mma_concrete_classes_share_marker_base() -> None:
    """Concrete classes inherit from the abstract ``Mma`` marker."""
    assert issubclass(Mma_SM80_16x8x16, Mma)
    assert issubclass(Wgmma_SM90_64x128x16, Mma)
