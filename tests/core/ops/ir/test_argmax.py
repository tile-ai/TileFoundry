"""ArgMax typeinfer: the reduced axis is dropped and the result is i64."""

from __future__ import annotations

import torch

from tests.core.ops.ir.op_case import ExpectedError, OpCase, infer_call, op_case_test
from tilefoundry.ir import expr, types, utils
from tilefoundry.ir.hir import tensor
from tilefoundry.ir.types import int_tuple

_I64 = expr.DType.i64


def test_argmax_layout_keeps_surviving_split():
    plain = types.TensorType.from_shape((4, 256), expr.DType.f32, layout=types.Layout(shape=(4, 256), strides=(256, 1)))
    assert infer_call(tensor.ArgMax(axis=-1), plain).layout == types.Layout(shape=(4,), strides=(1,))
    sharded = utils.make_shard_tensor_type(
        (4, 256), mesh=types.Mesh((types.Topology("gpu", 4),), types.Layout((4,), (1,)), ("g",)), attrs=(types.Split(0),)
    )
    result = infer_call(tensor.ArgMax(axis=-1), sharded)
    assert isinstance(result.layout, types.ShardLayout)
    assert result.layout.attrs == (types.Split(0),)
    assert int_tuple.mode_axes(int_tuple.compatible_profile(result.layout.layout.shape, result.shape))[0] == 0


def _make_argmax_matches_torch_reference() -> OpCase:
    torch.manual_seed(0)
    x = torch.randn(1, 151936)
    return OpCase("greedy_sampling", tensor.ArgMax(), (x,), value=torch.argmax(x, dim=-1))


CASES = (
    _make_argmax_matches_torch_reference(),
    OpCase(
        "default_axis_last",
        tensor.ArgMax(),
        (types.TensorType.from_shape((1, 151936), expr.DType.f32),),
        type=types.TensorType.from_shape((1,), _I64),
    ),
    OpCase(
        "axis_out_of_range",
        tensor.ArgMax(axis=3),
        (types.TensorType.from_shape((4,), expr.DType.f32),),
        error=ExpectedError(match="out of range"),
    ),
    OpCase(
        "partial_input_rejected",
        tensor.ArgMax(),
        (
            utils.make_shard_tensor_type(
                (4, 256),
                mesh=types.Mesh((types.Topology("gpu", 4),), types.Layout((4,), (1,)), ("g",)),
                attrs=(types.Partial("max"),),
            ),
        ),
        error=ExpectedError(match="x carries Partial"),
    ),
    OpCase(
        "reduction_axis_split_rejected",
        tensor.ArgMax(axis=-1),
        (
            utils.make_shard_tensor_type(
                (4, 256),
                mesh=types.Mesh((types.Topology("gpu", 4),), types.Layout((4,), (1,)), ("g",)),
                attrs=(types.Split(1),),
            ),
        ),
        error=ExpectedError(match="reduction axis 1.*Split-sharded.*Reshard"),
    ),
    OpCase(
        "nested_modes_fall_back_to_flat_axis_projection",
        tensor.ArgMax(axis=1),
        (
            types.TensorType(
                (8, 512),
                expr.DType.f32,
                types.ShardLayout(
                    types.Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))),
                    (types.Split(1),),
                    types.Mesh((types.Topology("thread", 8),), types.Layout((8,), (1,)), ("lane",)),
                ),
                "rmem",
            ),
        ),
        error=ExpectedError(match="reduction axis 1.*Split-sharded.*Reshard"),
    ),
)
test_op_case = op_case_test(CASES)
