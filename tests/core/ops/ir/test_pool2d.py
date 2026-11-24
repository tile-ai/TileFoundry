"""Pool2D parsing, result shape, value against torch, shard relation, and projected cost."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tests.core._corpus import reload
from tests.core.ops.ir.op_case import Cost, ExpectedError, OpCase, infer_call, op_case_test
from tilefoundry import inspection
from tilefoundry.interfaces import cost
from tilefoundry.ir import expr, types, utils
from tilefoundry.ir.hir import nn

_F = expr.DType.f32
_MAX = expr.PoolKind.MAX
_AVG = expr.PoolKind.AVG
_MESH = types.Mesh((types.Topology("gpu", 2),), types.Layout((2,), (1,)), ("g",))
_SOURCE = """
from tilefoundry.dsl import *
from tilefoundry.target import cpu

@module(entry="pool", target=cpu.Target())
class Pools:
    @func
    def pool(x: Tensor[(1, 32, 16, 16), "f32"]) -> Tensor[(1, 32, 8, 8), "f32"]:
        return tf.{alias}(x, kernel_size=(2, 2), stride=(2, 2))
"""


def _pool(kind=_MAX, kernel_size=(2, 2), stride=(2, 2), **attrs) -> nn.Pool2D:
    return nn.Pool2D(kernel_size=kernel_size, stride=stride, kind=kind, **attrs)


def _torch_case(
    name, kind, input_shape, kernel_size, stride, padding=(0, 0), ceil_mode=False, include_pad=True
) -> OpCase:
    torch.manual_seed(0)
    input_ = torch.randn(*input_shape)
    op = _pool(
        kind,
        kernel_size,
        stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=include_pad,
    )
    if kind is _MAX:
        expected = F.max_pool2d(input_, kernel_size, stride, padding, ceil_mode=ceil_mode)
    else:
        expected = F.avg_pool2d(input_, kernel_size, stride, padding, ceil_mode, include_pad)
    return OpCase(name, op, (input_,), value=expected)


@pytest.mark.parametrize(("alias", "kind"), [("max_pool2d", _MAX), ("avg_pool2d", _AVG)])
def test_pool_alias_round_trips(alias, kind) -> None:
    printed = inspection.as_script(reload(_SOURCE.format(alias=alias), "Pools"))
    rebuilt = reload(printed, "Pools")

    assert f"tf.{alias}(x, kernel_size=(2, 2), stride=(2, 2))" in printed
    assert rebuilt.entry_function().body.target.kind is kind
    assert inspection.as_script(rebuilt) == printed


@pytest.mark.parametrize(
    ("input_shape", "kernel_size", "stride", "padding", "ceil_mode", "expected"),
    [
        ((1, 32, 16, 16), (2, 2), (2, 2), (0, 0), False, (1, 32, 8, 8)),
        ((1, 1, 5, 5), (2, 2), (2, 2), (1, 1), True, (1, 1, 3, 3)),
        ((1, 1, 9, 11), (3, 2), (2, 1), (1, 0), False, (1, 1, 5, 10)),
    ],
    ids=["window", "ceil_drops_padded_window", "asymmetric"],
)
def test_pool2d_shape_matches_torch(input_shape, kernel_size, stride, padding, ceil_mode, expected):
    op = _pool(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode)
    result = infer_call(op, types.TensorType.from_shape(input_shape, _F))
    reference = F.max_pool2d(
        torch.zeros(input_shape), kernel_size, stride, padding, ceil_mode=ceil_mode
    )
    assert result.shape == expected == tuple(reference.shape)


def test_pool2d_layout_describes_result() -> None:
    result = infer_call(_pool(), types.TensorType.from_shape((2, 4, 8, 8), _F))
    assert result.layout == types.Layout(shape=(2, 4, 4, 4), strides=(64, 16, 4, 1))


PROPAGATION_CASES = [
    pytest.param((types.Split(0),), (0,), id="batch_split"),
    pytest.param((types.Split(1),), (1,), id="channel_split"),
]


@pytest.mark.parametrize(("attrs", "targets"), PROPAGATION_CASES)
def test_pool2d_relation_propagates_surviving_splits(attrs, targets) -> None:
    input_ = utils.make_shard_tensor_type((2, 4, 8, 8), mesh=_MESH, attrs=attrs)
    result = infer_call(_pool(), input_)
    assert isinstance(result.layout, types.ShardLayout)
    assert result.layout != input_.layout
    assert result.layout.attrs == attrs
    assert types.split_target_axes(result.layout, result.shape) == targets


_X = types.TensorType.from_shape((1, 32, 16, 16), _F)
CASES = (
    _torch_case("max_padded", _MAX, (2, 3, 9, 11), (3, 2), (2, 1), padding=(1, 0)),
    _torch_case("avg_include_pad", _AVG, (2, 3, 8, 8), (3, 3), (1, 2), padding=(1, 1)),
    _torch_case(
        "avg_ceil",
        _AVG,
        (1, 2, 16, 16),
        (3, 3),
        (2, 2),
        padding=(1, 1),
        ceil_mode=True,
        include_pad=False,
    ),
    OpCase(
        "rank4_input",
        _pool(),
        (types.TensorType.from_shape((4, 7, 7)),),
        error=ExpectedError(match="expects rank-4 NCHW input.*\\(4, 7, 7\\)"),
    ),
    OpCase(
        "kernel_size_length",
        _pool(kernel_size=(2,)),
        (_X,),
        error=ExpectedError(match="kernel_size must be length-2.*\\(2,\\)"),
    ),
    OpCase(
        "padding_over_half_kernel",
        _pool(padding=(2, 0)),
        (_X,),
        error=ExpectedError(match="padding \\(2, 0\\) must be at most half of kernel_size \\(2, 2\\)"),
    ),
    OpCase(
        "output_height_positive",
        _pool(kernel_size=(3, 3), stride=(1, 1)),
        (types.TensorType.from_shape((1, 1, 2, 8)),),
        error=ExpectedError(match="output height extent must be positive, got 0"),
    ),
    OpCase(
        "static_spatial_extent",
        _pool(),
        (types.TensorType.from_shape((1, 1, expr.DimVar("height", 8, 64), 16)),),
        error=ExpectedError(match="input height and width must be static"),
    ),
    OpCase(
        "kind_type",
        nn.Pool2D(kernel_size=(2, 2), stride=(2, 2), kind="max"),
        (_X,),
        error=ExpectedError(match="kind must be PoolKind"),
    ),
    OpCase(
        "spatial_split_is_underivable",
        _pool(),
        (utils.make_shard_tensor_type((2, 4, 8, 8), mesh=_MESH, attrs=(types.Split(2),)),),
        error=ExpectedError(match="input 0.*non-projection.*Reshard"),
    ),
    OpCase(
        "partial_input",
        _pool(),
        (utils.make_shard_tensor_type((2, 4, 8, 8), mesh=_MESH, attrs=(types.Partial("max"),)),),
        error=ExpectedError(match="Partial\\(max\\).*does not commute"),
    ),
    OpCase(
        "global",
        _pool(),
        (_X,),
        cost=Cost(
            flops={_F: 32 * 8 * 8 * 2 * 2},
            traffic=(
                cost.TrafficBytes(read=32 * 16 * 16 * 4),
                cost.TrafficBytes(write=32 * 8 * 8 * 4),
            ),
        ),
    ),
)
test_op_case = op_case_test(CASES)
