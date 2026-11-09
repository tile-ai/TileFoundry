"""Quant's exact per-token-group FP8 value, shape, and ownership contract."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from tests.core.ops.ir.op_case import (
    ExpectedError,
    OpCase,
    infer_call,
    op_case_test,
    raw_shard_tensor_type,
)
from tilefoundry.interfaces import evaluator, typeinfer
from tilefoundry.ir import core, expr, types, utils
from tilefoundry.ir.hir import tensor

_BF = expr.DType.bf16
_FP8 = expr.DType.fp8e4m3


def _reference(x: torch.Tensor, group: int) -> tuple[torch.Tensor, torch.Tensor]:
    grouped = x.float().reshape(*x.shape[:-1], x.shape[-1] // group, group)
    absmax = grouped.abs().amax(dim=-1)
    scale = torch.where(absmax == 0, torch.ones_like(absmax), absmax / 448.0)
    quantized = (
        (grouped / scale.unsqueeze(-1))
        .clamp(-448.0, 448.0)
        .reshape(x.shape)
        .to(torch.float8_e4m3fn)
    )
    return (quantized, scale)


def _quant_function(shape, group: int) -> tuple[core.Function, types.TupleType]:
    x = expr.Var(type=types.TensorType.from_shape(shape, expr.DType.f32), name="x")
    call = expr.Call(type=x.type, target=tensor.Quant(group=group), args=(x,))
    result_type = typeinfer.TypeInferVisitor().visit(call, typeinfer.TypeInferContext())
    call = replace(call, type=result_type)
    return (
        core.Function.from_parts(name="quant_eval", params=(x,), body=call, return_type=result_type),
        result_type,
    )


@pytest.mark.parametrize(
    "source",
    (
        torch.tensor([[224.0, -112.0, 7.0, -1.0, 3.0, -12.0, 96.0, -48.0]]),
        torch.zeros((2, 8), dtype=torch.float32),
    ),
    ids=("signed", "zero_groups"),
)
def test_quant_matches_exact_fp8e4m3_reference(source) -> None:
    function, _ = _quant_function(tuple(source.shape), 4)
    quantized, scale = evaluator.evaluate(function, source)
    expected_q, expected_scale = _reference(source, 4)
    assert quantized.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32
    torch.testing.assert_close(quantized.float(), expected_q.float(), atol=0, rtol=0)
    torch.testing.assert_close(scale, expected_scale, atol=0, rtol=0)
    assert not torch.isnan(quantized.float()).any()
    assert not torch.isnan(scale).any()
    if not torch.count_nonzero(source):
        torch.testing.assert_close(scale, torch.ones_like(scale), atol=0, rtol=0)


def test_quant_layouts_describe_results() -> None:
    source = types.TensorType.from_shape((2, 256), _BF, layout=types.Layout(shape=(2, 256), strides=(256, 1)))
    quantized, scale = infer_call(tensor.Quant(group=128), source).fields
    assert quantized.layout == types.Layout(shape=(2, 256), strides=(256, 1))
    assert quantized.layout is not source.layout
    assert scale.layout == types.Layout(shape=(2, 2), strides=(2, 1))


@pytest.mark.parametrize(
    ("shape", "split_axis", "expected_scale_shape"),
    (((4, 256), 0, (4, 2)), ((2, 1024), 1, (2, 8))),
    ids=("outer_axis", "whole_group_last_axis"),
)
def test_quant_propagates_representable_sharding(shape, split_axis, expected_scale_shape) -> None:
    source = utils.make_shard_tensor_type(
        shape,
        _BF,
        mesh=types.Mesh((types.Topology("gpu", 4),), types.Layout((4,), (1,)), ("g",)),
        attrs=(types.Split(split_axis),),
    )
    quantized, scale = infer_call(tensor.Quant(group=128), source).fields
    assert quantized.shape == shape
    assert scale.shape == expected_scale_shape
    for field in (quantized, scale):
        assert isinstance(field.layout, types.ShardLayout)
        assert field.layout is not source.layout
        assert types.split_target_axes(field.layout, field.shape) == (split_axis,)
        assert math.prod(field.layout.layout.shape) == math.prod(field.shape)


def test_quant_accepts_factorized_groups() -> None:
    source = raw_shard_tensor_type(
        (2, 1024),
        (2, 256, 4),
        (1024, 4, 1),
        (types.Split(1),),
        types.Mesh((types.Topology("gpu", 4),), types.Layout((4,), (1,)), ("g",)),
        dtype=_BF,
    )
    quantized, scale = infer_call(tensor.Quant(group=128), source).fields
    assert scale.shape == (2, 8)
    assert types.split_target_axes(quantized.layout, quantized.shape) == (1,)
    assert types.split_target_axes(scale.layout, scale.shape) == (1,)


def test_quant_preserves_broadcast_ownership() -> None:
    source = utils.make_shard_tensor_type(
        (2, 256),
        _BF,
        mesh=types.Mesh((types.Topology("gpu", 4),), types.Layout((4,), (1,)), ("g",)),
        attrs=(types.Broadcast(),),
    )
    quantized, scale = infer_call(tensor.Quant(group=128), source).fields
    assert quantized.layout == types.ShardLayout(
        types.Layout(quantized.shape), source.layout.attrs, source.layout.mesh
    )
    assert scale.layout == types.ShardLayout(
        types.Layout(scale.shape), source.layout.attrs, source.layout.mesh
    )


_SYMBOLIC_LAST = expr.DimVar("quant_symbolic_last", 1, 1024)


def test_quant_symbolic_scale_checks_divisibility() -> None:
    function, result_type = _quant_function((2, _SYMBOLIC_LAST), 4)
    scale_last = result_type.fields[1].shape[-1]
    assert isinstance(scale_last, expr.Call)
    assert isinstance(scale_last.target, expr.DimFloorDiv)
    assert scale_last.args[0] is _SYMBOLIC_LAST
    assert scale_last.args[1].value == 4
    source = torch.arange(-12, 12, dtype=torch.float32).reshape(2, 12)
    quantized, scale = evaluator.evaluate(function, source)
    expected_q, expected_scale = _reference(source, 4)
    torch.testing.assert_close(quantized.float(), expected_q.float(), atol=0, rtol=0)
    torch.testing.assert_close(scale, expected_scale, atol=0, rtol=0)
    with pytest.raises(evaluator.EvalError, match="runtime last dim 10.*group=4"):
        evaluator.evaluate(function, torch.zeros((2, 10)))


CASES = (
    OpCase(
        "rank2_per_token_group_128",
        tensor.Quant(),
        (types.TensorType.from_shape((1, 2048), _BF),),
        type=types.TupleType(
            fields=(types.TensorType.from_shape((1, 2048), _FP8), types.TensorType.from_shape((1, 16), expr.DType.f32))
        ),
    ),
    OpCase(
        "rank_zero_rejected",
        tensor.Quant(group=1),
        (types.TensorType.from_shape((), _BF),),
        error=ExpectedError(match="at least rank-1"),
    ),
    OpCase(
        "indivisible_last_dim",
        tensor.Quant(),
        (types.TensorType.from_shape((1, 100), _BF),),
        error=ExpectedError(match="not divisible by group"),
    ),
    OpCase(
        "unknown_scheme",
        tensor.Quant(scheme="block"),
        (types.TensorType.from_shape((1, 128), _BF),),
        error=ExpectedError(match="scheme must be 'per_token_group'.*block"),
    ),
    OpCase(
        "zero_group",
        tensor.Quant(group=0),
        (types.TensorType.from_shape((1, 128), _BF),),
        error=ExpectedError(match="positive non-boolean integer.*0"),
    ),
    OpCase(
        "negative_group",
        tensor.Quant(group=-4),
        (types.TensorType.from_shape((1, 128), _BF),),
        error=ExpectedError(match="positive non-boolean integer.*-4"),
    ),
    OpCase(
        "boolean_group",
        tensor.Quant(group=True),
        (types.TensorType.from_shape((1, 128), _BF),),
        error=ExpectedError(match="positive non-boolean integer.*True"),
    ),
    OpCase(
        "unsupported_target_dtype",
        tensor.Quant(target_dtype=expr.DType.f8e8m0),
        (types.TensorType.from_shape((1, 128), _BF),),
        error=ExpectedError(match="target_dtype must be fp8e4m3.*f8e8m0"),
    ),
    OpCase(
        "partial_input_rejected",
        tensor.Quant(),
        (
            utils.make_shard_tensor_type(
                (1, 2048),
                mesh=types.Mesh((types.Topology("gpu", 4),), types.Layout((4,), (1,)), ("g",)),
                attrs=(types.Partial("max"),),
            ),
        ),
        error=ExpectedError(match="x carries Partial"),
    ),
    OpCase(
        "last_split_through_group_rejected",
        tensor.Quant(group=128),
        (
            utils.make_shard_tensor_type(
                (2, 256),
                mesh=types.Mesh((types.Topology("gpu", 4),), types.Layout((4,), (1,)), ("g",)),
                attrs=(types.Split(1),),
            ),
        ),
        error=ExpectedError(match="last axis 1 Split cuts through group=128.*Reshard"),
    ),
    OpCase(
        "strided_last_split_rejected",
        tensor.Quant(group=128),
        (
            raw_shard_tensor_type(
                (2, 1024),
                (2, 4, 256),
                (1024, 256, 2),
                (types.Split(1),),
                types.Mesh((types.Topology("gpu", 4),), types.Layout((4,), (1,)), ("g",)),
            ),
        ),
        error=ExpectedError(match="last axis 1 Split cuts through group=128.*Reshard"),
    ),
    OpCase(
        "overlapping_last_split_rejected",
        tensor.Quant(group=128),
        (
            raw_shard_tensor_type(
                (2, 1024),
                (2, 4, 256),
                (1024, 128, 1),
                (types.Split(1),),
                types.Mesh((types.Topology("gpu", 4),), types.Layout((4,), (1,)), ("g",)),
            ),
        ),
        error=ExpectedError(match="last axis 1 Split cuts through group=128.*Reshard"),
    ),
)
test_op_case = op_case_test(CASES)
