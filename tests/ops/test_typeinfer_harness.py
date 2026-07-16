"""Declarative typeinfer harness — runner mechanics over a real op."""
from __future__ import annotations

import pytest

from tests.ops.typeinfer_utils import (
    STORAGES,
    ExpectedError,
    TypeInferCase,
    run_typeinfer_case,
    tensor_grid,
)
from tilefoundry.ir.core.kinds import BinaryKind, UnaryKind
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.math.unary import Unary
from tilefoundry.ir.types import DType, make_tensor_type

_ADD = Binary(kind=BinaryKind.ADD)

CASES = [
    TypeInferCase(
        name="same_shape",
        op=_ADD,
        inputs=(make_tensor_type((4, 8), DType.f32), make_tensor_type((4, 8), DType.f32)),
        expected=make_tensor_type((4, 8), DType.f32),
    ),
    TypeInferCase(
        name="broadcast_size1",
        op=_ADD,
        inputs=(make_tensor_type((4, 8), DType.f32), make_tensor_type((1, 8), DType.f32)),
        expected=make_tensor_type((4, 8), DType.f32),
    ),
    TypeInferCase(
        name="dtype_mismatch_errors",
        op=_ADD,
        inputs=(make_tensor_type((4, 8), DType.f32), make_tensor_type((4, 8), DType.bf16)),
        expected=ExpectedError(match="dtype mismatch"),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_typeinfer_cases(case):
    run_typeinfer_case(case)


# Combination builder: sweep STORAGES through a passthrough op. Unary
# preserves shape / dtype / storage, so each grid entry round-trips.
_SQUARE = Unary(kind=UnaryKind.SQUARE)

STORAGE_CASES = [
    TypeInferCase(
        name=f"unary_passthrough_{t.storage.name.lower()}",
        op=_SQUARE,
        inputs=(t,),
        expected=t,
    )
    for t in tensor_grid((4, 8), DType.f32)
]


def test_storage_grid_covers_all_storages():
    assert {c.inputs[0].storage.name.lower() for c in STORAGE_CASES} == set(STORAGES)


@pytest.mark.parametrize("case", STORAGE_CASES, ids=lambda c: c.name)
def test_typeinfer_storage_grid(case):
    run_typeinfer_case(case)
