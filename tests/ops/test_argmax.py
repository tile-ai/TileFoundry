"""ArgMax typeinfer: the reduced axis is dropped and the result is i64."""

from __future__ import annotations

import pytest

from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    infer_call,
    run_typeinfer_case,
)
from tilefoundry.ir.hir.tensor.argmax import ArgMax
from tilefoundry.ir.types import DType, make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.shard import Layout, ShardLayout, make_mesh
from tilefoundry.ir.types.shard.shard_layout import (
    Partial,
    Split,
    layout_axis_to_tensor_axis,
)

_I64 = DType.i64

CASES = [
    TypeInferCase(
        "default_axis_last",
        ArgMax(),
        (make_tensor_type((1, 151936), DType.f32),),
        make_tensor_type((1,), _I64),
    ),
    TypeInferCase(
        "axis_out_of_range",
        ArgMax(axis=3),
        (make_tensor_type((4,), DType.f32),),
        ExpectedError(match="out of range"),
    ),
    TypeInferCase(
        "partial_input_rejected",
        ArgMax(),
        (make_shard_tensor_type((4, 256), mesh=make_mesh((4,)), attrs=(Partial("max"),)),),
        ExpectedError(match="x carries Partial"),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_argmax_typeinfer(case):
    run_typeinfer_case(case)


def test_argmax_layout_describes_result_and_preserves_surviving_split():
    plain = make_tensor_type((4, 256), DType.f32, layout=Layout(shape=(4, 256), strides=(256, 1)))
    assert infer_call(ArgMax(axis=-1), plain).layout == Layout(shape=(4,), strides=(1,))

    sharded = make_shard_tensor_type((4, 256), mesh=make_mesh((4,)), attrs=(Split(0),))
    result = infer_call(ArgMax(axis=-1), sharded)
    assert isinstance(result.layout, ShardLayout)
    assert result.layout.attrs == (Split(0),)
    assert layout_axis_to_tensor_axis(result.layout.layout.shape, result.shape)[0] == 0
