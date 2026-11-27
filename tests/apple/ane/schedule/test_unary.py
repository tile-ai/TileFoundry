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


@pytest.mark.parametrize("attributes,values,expected", (
    ({"scale": -2.0, "abs": True}, (-2, -1, 0, 1, 2), (4, 2, 0, 2, 4)),
    ({"scale": -2.0, "clamp": (-1.0, 3.0)}, (-2, -1, 0, 1, 2), (3, 2, 0, -1, -1)),
    ({"square": True}, (-2, -1, 0, 1, 2), (4, 1, 0, 1, 4)),
    ({"bias": -1.5, "abs": True}, (-1.0009765625, 2), (2.501953125, 0.5)),
    ({"bias": 1.5, "square": True}, (2**-11, 4.375), (2.25390625, 34.53125)),
    ({"pre_scale": -1.5, "abs": True}, (-2, 1), (3, 1.5)),
    ({"abs": True, "clamp": (-1.0, 2.0)}, (-6, -1, 0, 1, 6), (2, 1, 0, 1, 2)),
    ({"square": True, "clamp": (-1.0, 2.0)}, (-2, -1, 0, 1, 2), (2, 1, 0, 1, 2)),
    ({"pre_scale": -1.5, "bias": 0.75, "abs": True}, (2**-11,), (0.74951171875,)),
    ({"bias": -1.5, "abs": True, "clamp": (-0.5, 2.5)}, (-2, 1, 2, 6), (2.5, 0.5, 0.5, 2.5)),
    ({"scale": -2.0, "abs": True, "clamp": (-0.5, 2.5)}, (-2, -1, 0, 1, 2), (2.5, 2, 0, 2, 2.5)),
    ({"square": True, "clamp": (-1.0, 3.0)}, (-2, -1, 0, 1, 2), (3, 1, 0, 1, 3)),
    ({"pre_scale": -1.5, "bias": 0.75, "scale": -0.5, "clamp": (-3.0, -1.0)},
     (-6, -2, -1, 0, 2), (-3, -1.875, -1.125, -1, -1)),
    ({"pre_scale": -1.5, "bias": 0.75, "scale": -0.5, "abs": True},
     (-2, -1, 0, 1, 2), (1.875, 1.125, 0.375, 0.375, 1.125)),
    ({"pre_scale": -1.5, "bias": 0.75, "square": True},
     (-2, -1, 0, 1, 2, 2**-11), (14.0625, 5.0625, 0.5625, 0.5625, 5.0625, 0.56201171875)),
    ({"pre_scale": -1.5, "bias": 0.75, "square": True, "clamp": (-1.0, 3.0)},
     (-2, -1, 0, 1, 2, 2**-11), (3, 3, 0.5625, 0.5625, 3, 0.56201171875)),
))
def test_device_observed_unary_values(attributes, values, expected):
    assert evaluate(tir.Unary(**attributes), values).tolist() == list(expected)


def test_abs_with_square_has_no_decoded_evaluation():
    with pytest.raises(evaluator.EvalError, match="abs with square is not decoded yet"):
        evaluate(tir.Unary(abs=True, square=True), (1,))


def test_square_evaluation_cannot_silently_ignore_scale():
    with pytest.raises(evaluator.EvalError, match="square requires scale=1; hardware square mode ignores scale"):
        evaluate(tir.Unary(square=True, scale=-2.0), (1,))


def test_ambiguous_affine_abs_clamp_has_no_decoded_evaluation():
    with pytest.raises(evaluator.EvalError, match="affine combination with abs and clamp is not decoded yet"):
        evaluate(tir.Unary(pre_scale=-1.5, bias=0.75, scale=-0.5, abs=True, clamp=(-3.0, -1.0)), (1,))
