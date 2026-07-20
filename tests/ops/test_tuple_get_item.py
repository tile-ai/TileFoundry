"""TupleGetItem typeinfer: extracts the field at ``index`` from a TupleType."""
from __future__ import annotations

import pytest

from tests.ops.typeinfer_utils import TypeInferCase, run_typeinfer_case
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.types import DType, TupleType, make_tensor_type
from tilefoundry.ir.types.storage import StorageKind


def _scalar(dtype):
    return make_tensor_type((), dtype, storage=StorageKind.RMEM)


CASES = [
    TypeInferCase(
        "index_1_of_2",
        TupleGetItem(index=1),
        (TupleType(fields=(_scalar(DType.f32), _scalar(DType.i32))),),
        _scalar(DType.i32),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_tuple_get_item_typeinfer(case):
    run_typeinfer_case(case)
