"""TupleGetItem typeinfer: extracts the field at ``index`` from a TupleType."""
from __future__ import annotations

import pytest

from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    run_typeinfer_case,
    ten,
)
from tilefoundry.ir.core import Op
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.target.storage import StorageKind
from tilefoundry.ir.types import DType, TupleType


def _scalar(dtype):
    return ten((), dtype, storage=StorageKind.RMEM)


CASES = [
    TypeInferCase(
        "index_1_of_2",
        TupleGetItem(index=1),
        (TupleType(fields=(_scalar(DType.f32), _scalar(DType.i32))),),
        _scalar(DType.i32),
    ),
    TypeInferCase(
        "non_tuple_input",
        TupleGetItem(index=0),
        (_scalar(DType.f32),),
        ExpectedError(match="non-TupleType", exc=Exception),
    ),
    TypeInferCase(
        "index_out_of_range",
        TupleGetItem(index=5),
        (TupleType(fields=(_scalar(DType.f32),)),),
        ExpectedError(match="out of range", exc=Exception),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_tuple_get_item_typeinfer(case):
    run_typeinfer_case(case)


def test_tuple_get_item_is_op_subclass():
    assert issubclass(TupleGetItem, Op)
