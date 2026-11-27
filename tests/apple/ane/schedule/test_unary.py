"""Unary affine order and independent fp16 rounding boundaries."""

import pytest

from tilefoundry import core, utils
from tilefoundry.interfaces import evaluator
from tilefoundry.ir import expr, types
from tilefoundry_apple.ane.ir import tir


def evaluate(op, values):
    """Evaluate one Unary with typed source and destination buffers."""
    source = utils.torch.tensor(values, dtype=utils.torch.float16).reshape(1, 1, 1, -1)
    type_ = types.TensorType.from_shape(tuple(source.shape), dtype=expr.DType.f16, storage="gmem")
    destination = evaluator.TensorValue(core.ShardTensor(type_, utils.torch.empty_like(source)))
    ctx = evaluator.EvaluateContext(
        op=op,
        args=(destination, evaluator.TensorValue(core.ShardTensor(type_, source))),
        result_type=types.TupleType.Void,
    )
    handler = evaluator.registry.get(tir.Unary, None)
    assert handler is not None
    return handler(ctx).data.flatten()


@pytest.mark.parametrize("scale,bias,expected", (
    (2.0, 1.0, (-6, -2, 0, 2, 4, 6, 10, 18)),
    (3.0, 1.0, (-9, -3, 0, 3, 6, 9, 15, 27)),
    (0.5, 4.0, (0, 1, 1.5, 2, 2.5, 3, 4, 6)),
))
def test_bias_precedes_scale(scale, bias, expected):
    actual = evaluate(tir.Unary(scale=scale, bias=bias), (-4, -2, -1, 0, 1, 2, 4, 8))
    assert actual.tolist() == list(expected)


@pytest.mark.parametrize("source,pre_scale,bias,scale,expected", (
    (1 + 2**-10, 1.5, -1.5, 1.0, 2**-9),
    (2**-11, 1.0, 1.0, 3.0, 3 + 2**-8),
    (1 + 2**-10, 1.0, 0.0, 1.5, 1.5 + 2**-9),
))
def test_each_affine_step_rounds_to_fp16(source, pre_scale, bias, scale, expected):
    actual = evaluate(tir.Unary(pre_scale=pre_scale, bias=bias, scale=scale), (source,))
    assert actual.item() == expected


def test_negative_scale_still_implements_negation_with_zero_bias():
    assert evaluate(tir.Unary(scale=-1.0), (-4, -1, 0, 1, 4)).tolist() == [4, 1, 0, -1, -4]
