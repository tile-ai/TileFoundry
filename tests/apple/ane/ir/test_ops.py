"""ANE values the fixture corpus does not reach, load-time rules, and where a MatMul lands."""

from __future__ import annotations

import itertools
from math import prod

import pytest

from tests.core._corpus import reload
from tests.core.ops.ir import op_case
from tilefoundry import utils
from tilefoundry.interfaces import evaluator
from tilefoundry.ir import expr, types
from tilefoundry.ir.hir import nn
from tilefoundry_apple import ane
from tilefoundry_apple.ane.ir.tir import evaluation

torch = utils.torch
F = utils.F

PROGRAM = """
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="main", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def main({params}):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as engine:
            return tf.schedule((tf.empty({dst}), {operands}), op=T.ane.{op})
"""


def tensor(shape, dtype="f16", storage="gmem", strides=None) -> str:
    """Source of a tensor type, row-major unless *strides* say otherwise."""
    strides = strides or tuple(prod(shape[axis + 1:]) for axis in range(len(shape)))
    return f'Tensor[{tuple(shape)}, "{dtype}", ({tuple(shape)}, {strides}), "{storage}"]'


def load(op: str, dst: str, *operands: str):
    """The module that applies *op* to a fresh *dst* and to parameters of the *operands* types."""
    names = tuple(f"x{index}" for index in range(len(operands)))
    params = ", ".join(f"{name}: {type_}" for name, type_ in zip(names, operands))
    return reload(PROGRAM.format(params=params, dst=dst, operands=", ".join(names), op=op), "Program")


def run(op: str, shape, *inputs, strides=None):
    """Evaluate *op* over *inputs* into a destination of *shape*.

    *strides* maps an operand position (0 is the destination) to its strides.
    """
    strides = strides or {}
    module = load(op, tensor(shape, strides=strides.get(0)),
                  *(tensor(value.shape, strides=strides.get(index + 1)) for index, value in enumerate(inputs)))
    return evaluator.evaluate(module, *(value.clone() for value in inputs))


def check(op: str, expected, *inputs, strides=None) -> None:
    assert torch.equal(run(op, expected.shape, *inputs, strides=strides), expected.to(torch.float16)), op


LINE = torch.arange(-4, 4).reshape(1, 1, 1, 8).half()
FEATURE = (torch.arange(24) % 5 + 1).reshape(1, 2, 3, 4).half()
MAP = (torch.arange(2 * 5 * 6) % 11 - 5).reshape(1, 2, 5, 6).half()


def test_f16_rounds_ties_away_from_zero():
    tie = torch.tensor(1.0 + 2**-11)
    assert tie.to(torch.float16) == 1.0
    assert evaluation.f16(tie) == 1.0 + 2**-10
    assert evaluation.f16(-tie) == -(1.0 + 2**-10)


def test_unary_applies_bias_then_abs_then_square_then_clamp():
    expected = ((1 - LINE).abs() ** 2).clip(0, 10)
    check("unary(scale=-1.0, bias=1.0, abs=True, square=True, clamp=(0.0, 10.0))", expected, LINE)


def test_max_pool_takes_a_partial_last_window():
    expected = F.max_pool2d(MAP.float(), (3, 2), (1, 2), (1, 0), ceil_mode=True)
    check("pool(kernel_size=(3, 2), stride=(1, 2), padding=(1, 0), ceil_mode=True, kind=PoolKind.MAX)", expected, MAP)


def test_average_pool_excluding_padding_multiplies_by_the_f16_reciprocal_of_the_cells_inside():
    padded = F.pad(MAP.float(), (1, 1, 1, 1))
    inside = F.pad(torch.ones(5, 6), (1, 1, 1, 1))
    expected = torch.zeros(1, 2, 3, 3)
    for row, col in itertools.product(range(3), range(3)):
        rows, cols = slice(2 * row, 2 * row + 3), slice(2 * col, 2 * col + 3)
        reciprocal = evaluation.f16(torch.tensor(1 / inside[rows, cols].sum().item()))
        expected[..., row, col] = evaluation.f16(padded[..., rows, cols].sum((-1, -2)) * reciprocal)
    attributes = "kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), count_include_pad=False"
    check(f"pool({attributes}, kind=PoolKind.AVG)", expected, MAP)


SUM = "axes=(2, 3), keepdim=True, kind=ReduceKind.SUM"
TOTAL = FEATURE.sum((2, 3), keepdim=True)


@pytest.mark.parametrize(
    ("attributes", "expected"),
    [
        ("axes=(1, 3), keepdim=True, kind=ReduceKind.SUM", FEATURE.sum((1, 3), keepdim=True)),
        ("axes=(2, 3), keepdim=False, kind=ReduceKind.SUM", FEATURE.sum((2, 3))),
        (f"{SUM}, square=True", (FEATURE * FEATURE).sum((2, 3), keepdim=True)),
        (f"{SUM}, post=T.ane.Post.RSQRT", TOTAL.rsqrt()),
    ],
)
def test_reduce(attributes, expected):
    check(f"reduce({attributes})", expected, FEATURE)


ROWS = (torch.arange(8 * 16) % 9 - 4).reshape(8, 16).half()
COLUMNS = (torch.arange(16 * 5) % 7 - 3).reshape(16, 5).half()
OFFSETS = (torch.arange(5) % 4 - 1).half()
IMAGE = (torch.arange(4 * 6 * 6) % 7 - 3).reshape(1, 4, 6, 6).half()
KERNELS = (torch.arange(6 * 2 * 3 * 3) % 5 - 2).reshape(6, 2, 3, 3).half()
BIASES = (torch.arange(6) % 4 - 1).half()


def test_conv_feature_form_strides_and_groups_as_f_conv2d():
    expected = F.conv2d(IMAGE.float(), KERNELS.float(), BIASES.float(), stride=2, padding=1, groups=2)
    check("conv(stride=(2, 2), padding=(1, 1), groups=2)", expected, IMAGE, KERNELS, BIASES)


def test_conv_matrix_form_adds_its_bias_to_each_column():
    expected = ROWS.float() @ COLUMNS.float() + OFFSETS.float()
    check("conv()", expected, ROWS, COLUMNS, OFFSETS, strides={1: (1, 8)})


@pytest.mark.parametrize(
    ("lhs", "rhs", "storage"),
    [
        ("smem", "gmem", expr.StorageKind.SMEM),
        ("gmem", "gmem", expr.StorageKind.GMEM),
    ],
)
def test_ane_matmul_result_follows_an_on_chip_operand(lhs, rhs, storage):
    lhs_type, rhs_type = (types.TensorType.from_shape(shape, expr.DType.f16, held)
                          for shape, held in (((4, 8), lhs), ((8, 6), rhs)))
    result = op_case.infer_call(nn.MatMul(), lhs_type, rhs_type, target=ane.Target())
    assert result.storage is storage


FEATURE_TYPE = tensor((1, 4, 6, 6))
SCALAR_TYPE = tensor((1, 4, 1, 1))
SINGLE = tensor((1, 4, 6, 6), "f32")
KERNEL_TYPE = tensor((6, 4, 3, 3))
CONVOLVED_TYPE = tensor((1, 6, 4, 4))
ADD = "binary(kind=BinaryKind.ADD)"
POOL = "pool(kernel_size=(2, 2), stride=(2, 2), kind=PoolKind.MAX)"
REDUCE = "reduce(axes=(2, 3), keepdim=True, kind=ReduceKind.SUM)"


@pytest.mark.parametrize(
    ("op", "dst", "operands", "word"),
    [
        (ADD, SINGLE, (SINGLE, SINGLE), "dtype"),
        ("unary(abs=True)", tensor((4, 6, 6)), (tensor((4, 6, 6)),), "shape"),
        (POOL, tensor((1, 4, 2, 3)), (FEATURE_TYPE,), "dst shape"),
        ("conv()", CONVOLVED_TYPE, (FEATURE_TYPE, tensor((6, 4, 3, 3), storage="smem")), "storage"),
        ("unary(abs=True)", FEATURE_TYPE, (tensor((1, 4, 6, 6), strides=(288, 72, 12, 2)),), "strides"),
        ("conv(stride=(3, 3))", CONVOLVED_TYPE, (FEATURE_TYPE, KERNEL_TYPE), "stride"),
        ("conv(padding=(3, 0))", tensor((1, 6, 10, 4)), (FEATURE_TYPE, KERNEL_TYPE), "padding"),
        ("conv(dilation=(2, 2))", CONVOLVED_TYPE, (FEATURE_TYPE, KERNEL_TYPE), "dilation"),
        ("conv(groups=3)", CONVOLVED_TYPE, (FEATURE_TYPE, KERNEL_TYPE), "groups"),
        ("conv()", tensor((8, 5)), (tensor((8, 16)), tensor((16, 5))), "layout"),
        ("bypass(perm=(0, 0, 1))", FEATURE_TYPE, (FEATURE_TYPE,), "perm"),
        (REDUCE.replace("(2, 3)", "(2,)"), SCALAR_TYPE, (FEATURE_TYPE,), "axes"),
        (REDUCE.replace("(2, 3)", "(3,)").replace("True", "False"), tensor((1, 4, 6)), (FEATURE_TYPE,), "keepdim"),
        (ADD.replace("ADD", "DIV"), FEATURE_TYPE, (FEATURE_TYPE, FEATURE_TYPE), "kind"),
        ("unary(clamp=(6.0, 0.0))", FEATURE_TYPE, (FEATURE_TYPE,), "clamp"),
    ],
)
def test_load_refuses(op, dst, operands, word):
    with pytest.raises(expr.VerifyError, match=word):
        load(op, dst, *operands)
