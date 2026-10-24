"""ArgMax typeinfer: the reduced axis is dropped and the result is i64."""

from __future__ import annotations

import torch

from tests.core.ops.ir.op_case import ExpectedError, OpCase, infer_call, op_case_test
from tilefoundry.ir.hir.tensor.argmax import ArgMax
from tilefoundry.ir.types import (
    DType,
    Layout,
    Mesh,
    ShardLayout,
    TensorType,
    Topology,
    make_shard_tensor_type,
    make_tensor_type,
)
from tilefoundry.ir.types.int_tuple import compatible_profile, mode_axes
from tilefoundry.ir.types.shard_layout import Partial, Split

_I64 = DType.i64


def test_argmax_layout_describes_result_and_preserves_surviving_split():
    plain = make_tensor_type((4, 256), DType.f32, layout=Layout(shape=(4, 256), strides=(256, 1)))
    assert infer_call(ArgMax(axis=-1), plain).layout == Layout(shape=(4,), strides=(1,))
    sharded = make_shard_tensor_type(
        (4, 256), mesh=Mesh((Topology("gpu", 4),), Layout((4,), (1,)), ("g",)), attrs=(Split(0),)
    )
    result = infer_call(ArgMax(axis=-1), sharded)
    assert isinstance(result.layout, ShardLayout)
    assert result.layout.attrs == (Split(0),)
    assert mode_axes(compatible_profile(result.layout.layout.shape, result.shape))[0] == 0


def _make_argmax_matches_torch_reference() -> OpCase:
    torch.manual_seed(0)
    x = torch.randn(1, 151936)
    return OpCase("greedy_sampling", ArgMax(), (x,), value=torch.argmax(x, dim=-1))


CASES = (
    _make_argmax_matches_torch_reference(),
    OpCase(
        "default_axis_last",
        ArgMax(),
        (make_tensor_type((1, 151936), DType.f32),),
        type=make_tensor_type((1,), _I64),
    ),
    OpCase(
        "axis_out_of_range",
        ArgMax(axis=3),
        (make_tensor_type((4,), DType.f32),),
        error=ExpectedError(match="out of range"),
    ),
    OpCase(
        "partial_input_rejected",
        ArgMax(),
        (
            make_shard_tensor_type(
                (4, 256),
                mesh=Mesh((Topology("gpu", 4),), Layout((4,), (1,)), ("g",)),
                attrs=(Partial("max"),),
            ),
        ),
        error=ExpectedError(match="x carries Partial"),
    ),
    OpCase(
        "reduction_axis_split_rejected",
        ArgMax(axis=-1),
        (
            make_shard_tensor_type(
                (4, 256),
                mesh=Mesh((Topology("gpu", 4),), Layout((4,), (1,)), ("g",)),
                attrs=(Split(1),),
            ),
        ),
        error=ExpectedError(match="reduction axis 1.*Split-sharded.*Reshard"),
    ),
    OpCase(
        "nested_modes_fall_back_to_flat_axis_projection",
        ArgMax(axis=1),
        (
            TensorType(
                (8, 512),
                DType.f32,
                ShardLayout(
                    Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))),
                    (Split(1),),
                    Mesh((Topology("thread", 8),), Layout((8,), (1,)), ("lane",)),
                ),
                "rmem",
            ),
        ),
        error=ExpectedError(match="reduction axis 1.*Split-sharded.*Reshard"),
    ),
)
test_op_case = op_case_test(CASES)
