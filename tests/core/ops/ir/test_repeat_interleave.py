"""RepeatInterleave's layouts, per-unit values and fail-closed boundaries.

Each unit repeats its own elements, so the result keeps the input's attrs. A
Split on the repeated axis is valid only in contiguous blocks; a cyclic Split
fails closed. Out-of-range axes are named rather than normalized. How Varying
reads depends on the target's storage owners, so those rows are in
tests/nvidia/cuda/ops/ir/test_varying_operands.py.
"""

from __future__ import annotations

import pytest
import torch

from tests.core.ops.ir.op_case import ExpectedError, OpCase, make_call, op_case_test
from tilefoundry import core
from tilefoundry.interfaces import evaluator, typeinfer
from tilefoundry.ir import core as ir_core
from tilefoundry.ir import expr, types, utils
from tilefoundry.ir.hir import tensor

_F = expr.DType.f32
_M = types.Mesh((types.Topology("gpu", 4),), types.Layout((4,), (1,)), ("g",))
_M2 = types.Mesh((types.Topology("gpu", 4),), types.Layout((2, 2), (1, 2)), ("a", "b"))
_CYCLIC = types.TensorType(
    (8,), _F, types.ShardLayout(types.Layout((2, 4), (4, 1)), (types.Split(1),), _M), "gmem"
)
CASES = (
    OpCase(
        "varying_keeps_its_mesh_on_the_new_shape",
        tensor.RepeatInterleave(repeats=2, axis=1),
        (utils.make_shard_tensor_type((4, 8), mesh=_M, attrs=(types.Varying(),)),),
        type=utils.make_shard_tensor_type((4, 16), mesh=_M, attrs=(types.Varying(),)),
    ),
    OpCase(
        "split_on_another_axis_keeps_its_attrs",
        tensor.RepeatInterleave(repeats=2, axis=1),
        (utils.make_shard_tensor_type((4, 8), mesh=_M, attrs=(types.Split(0),)),),
        type=utils.make_shard_tensor_type((4, 16), mesh=_M, attrs=(types.Split(0),)),
    ),
    OpCase(
        "split_in_blocks_on_the_repeated_axis_keeps_its_attrs",
        tensor.RepeatInterleave(repeats=2, axis=0),
        (utils.make_shard_tensor_type((8,), mesh=_M, attrs=(types.Split(0),)),),
        type=utils.make_shard_tensor_type((16,), mesh=_M, attrs=(types.Split(0),)),
    ),
    OpCase(
        "partial_keeps_its_attrs",
        tensor.RepeatInterleave(repeats=2, axis=1),
        (utils.make_shard_tensor_type((4, 8), mesh=_M, attrs=(types.Partial(),)),),
        type=utils.make_shard_tensor_type((4, 16), mesh=_M, attrs=(types.Partial(),)),
    ),
    OpCase(
        "cyclic_split_on_the_repeated_axis_fails_closed",
        tensor.RepeatInterleave(repeats=2, axis=0),
        (_CYCLIC,),
        error=ExpectedError(match="axis 0 is split, but not in contiguous blocks"),
    ),
    OpCase(
        "splits_keep_their_tensor_axes_on_a_two_axis_mesh",
        tensor.RepeatInterleave(repeats=2, axis=1),
        (utils.make_shard_tensor_type((4, 6, 8), mesh=_M2, attrs=(types.Split(0), types.Split(2))),),
        type=utils.make_shard_tensor_type((4, 12, 8), mesh=_M2, attrs=(types.Split(0), types.Split(2))),
    ),
    OpCase(
        "cyclic_split_on_another_axis_fails_closed",
        tensor.RepeatInterleave(repeats=2, axis=1),
        (types.TensorType((8, 3), _F, types.ShardLayout(types.Layout((2, 4, 3), (12, 3, 1)), (types.Split(1),), _M), "gmem"),),
        error=ExpectedError(match="axis 0 is split, but not in contiguous blocks"),
    ),
    OpCase(
        "axis_out_of_range",
        tensor.RepeatInterleave(repeats=2, axis=5),
        (types.TensorType.from_shape((4,), _F),),
        error=ExpectedError(match="out of range"),
    ),
)
test_op_case = op_case_test(CASES)


@pytest.mark.parametrize(
    ("shape", "attrs", "axis"),
    [
        ((4, 8), (types.Split(0),), 1),
        ((8,), (types.Split(0),), 0),
        ((4, 8), (types.Partial(),), 1),
    ],
    ids=["split_other_axis", "split_blocks", "partial"],
)
def test_each_unit_repeats_its_own_elements(shape, attrs, axis):
    """Per-unit pieces repeated locally assemble to torch.repeat_interleave of the whole."""
    source_type = utils.make_shard_tensor_type(shape, mesh=_M, attrs=attrs)
    op = tensor.RepeatInterleave(repeats=2, axis=axis)
    call = make_call(op, (source_type,))
    result_type = typeinfer.TypeInferVisitor().visit(call, typeinfer.TypeInferContext())
    function = ir_core.Function.from_parts(
        name="repeat", params=call.args, body=call, return_type=result_type
    )
    whole = torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape)
    pieces = []
    for unit in range(4):
        if isinstance(attrs[0], types.Partial):
            piece = whole.clone() if unit == 0 else torch.zeros(shape)
        else:
            piece = core.ShardTensor(source_type, whole).to_local(types.Placement({"gpu": unit})).clone()
        pieces.append((types.Placement({"gpu": unit}), piece))
    result = evaluator.evaluate(function, core.ShardTensor(source_type, tuple(pieces)))
    expected = torch.repeat_interleave(whole, 2, dim=axis)
    torch.testing.assert_close(result.to_global().tensor, expected, rtol=0, atol=0)
