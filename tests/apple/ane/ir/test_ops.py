"""ANE instructions: values against torch, the rules checked at load, and where a MatMul lands."""

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
OTHER = (torch.arange(24) % 7 - 3).reshape(1, 2, 3, 4).half()
MAP = (torch.arange(2 * 5 * 6) % 11 - 5).reshape(1, 2, 5, 6).half()


def test_f16_rounds_ties_away_from_zero():
    tie = torch.tensor(1.0 + 2**-11)
    assert tie.to(torch.float16) == 1.0
    assert evaluation.f16(tie) == 1.0 + 2**-10
    assert evaluation.f16(-tie) == -(1.0 + 2**-10)


@pytest.mark.parametrize(
    ("attributes", "expected"),
    [
        ("abs=True", LINE.abs()),
        ("square=True", LINE * LINE),
        ("scale=-1.0", -LINE),
        ("pre_scale=2.0, bias=3.0", 2 * LINE + 3),
        ("pre_scale=2.0, square=True", (2 * LINE) ** 2),
        ("bias=3.0, clamp=(0.0, 6.0)", (LINE + 3).clip(0, 6)),
    ],
)
def test_unary_steps(attributes, expected):
    check(f"unary({attributes})", expected, LINE)


@pytest.mark.parametrize(
    ("kind", "torch_op"),
    [
        ("ADD", torch.add),
        ("SUB", torch.sub),
        ("MUL", torch.mul),
        ("MIN", torch.minimum),
        ("MAX", torch.maximum),
    ],
)
def test_binary_kinds(kind, torch_op):
    check(f"binary(kind=BinaryKind.{kind})", torch_op(FEATURE, OTHER), FEATURE, OTHER)


@pytest.mark.parametrize(
    ("kernel", "stride", "padding", "ceil_mode"),
    [
        ((2, 2), (2, 2), (0, 0), False),
        ((3, 3), (1, 1), (1, 1), False),
        ((3, 3), (2, 2), (0, 0), True),
        ((3, 2), (1, 2), (1, 0), True),
    ],
)
def test_max_pool(kernel, stride, padding, ceil_mode):
    expected = F.max_pool2d(MAP.float(), kernel, stride, padding, ceil_mode=ceil_mode)
    attributes = f"kernel_size={kernel}, stride={stride}, padding={padding}, ceil_mode={ceil_mode}"
    check(f"pool({attributes}, kind=PoolKind.MAX)", expected, MAP)


@pytest.mark.parametrize("include_pad", [True, False])
def test_average_pool_multiplies_by_the_f16_reciprocal(include_pad):
    padded = F.pad(MAP.float(), (1, 1, 1, 1))
    inside = F.pad(torch.ones(5, 6), (1, 1, 1, 1))
    expected = torch.zeros(1, 2, 3, 3)
    for row, col in itertools.product(range(3), range(3)):
        rows, cols = slice(2 * row, 2 * row + 3), slice(2 * col, 2 * col + 3)
        count = 9.0 if include_pad else inside[rows, cols].sum().item()
        reciprocal = evaluation.f16(torch.tensor(1 / count))
        expected[..., row, col] = evaluation.f16(padded[..., rows, cols].sum((-1, -2)) * reciprocal)
    attributes = f"kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), count_include_pad={include_pad}"
    check(f"pool({attributes}, kind=PoolKind.AVG)", expected, MAP)


SUM = "axes=(2, 3), keepdim=True, kind=ReduceKind.SUM"
TOTAL = FEATURE.sum((2, 3), keepdim=True)


@pytest.mark.parametrize(
    ("attributes", "expected"),
    [
        (SUM, TOTAL),
        ("axes=(2, 3), keepdim=True, kind=ReduceKind.MAX", FEATURE.amax((2, 3), keepdim=True)),
        ("axes=(2, 3), keepdim=True, kind=ReduceKind.MIN", FEATURE.amin((2, 3), keepdim=True)),
        ("axes=(3,), keepdim=True, kind=ReduceKind.SUM", FEATURE.sum(3, keepdim=True)),
        ("axes=(1, 2, 3), keepdim=True, kind=ReduceKind.SUM", FEATURE.sum((1, 2, 3), keepdim=True)),
        ("axes=(1, 3), keepdim=True, kind=ReduceKind.SUM", FEATURE.sum((1, 3), keepdim=True)),
        ("axes=(2, 3), keepdim=False, kind=ReduceKind.SUM", FEATURE.sum((2, 3))),
        (f"{SUM}, square=True", (FEATURE * FEATURE).sum((2, 3), keepdim=True)),
        (f"{SUM}, scale=0.25", 0.25 * TOTAL),
        (f"{SUM}, post=T.ane.Post.INVERSE", 1 / TOTAL),
        (f"{SUM}, post=T.ane.Post.SQRT", TOTAL.sqrt()),
        (f"{SUM}, post=T.ane.Post.RSQRT", TOTAL.rsqrt()),
    ],
)
def test_reduce(attributes, expected):
    check(f"reduce({attributes})", expected, FEATURE)


PERMUTED = (torch.arange(4096) - 2048).reshape(1, 32, 8, 16).half()


def test_bypass_relu_is_max_with_zero():
    check("bypass(nonlinear=T.ane.Nonlinear.RELU)", LINE.clamp(min=0), LINE)


@pytest.mark.parametrize("perm", [perm for perm in itertools.permutations(range(3)) if perm != (0, 1, 2)])
def test_bypass_permutes_channel_height_width(perm):
    expected = PERMUTED.permute(0, *(1 + axis for axis in perm))
    check(f"bypass(perm={perm})", expected, PERMUTED)


ROWS = (torch.arange(8 * 16) % 9 - 4).reshape(8, 16).half()
COLUMNS = (torch.arange(16 * 5) % 7 - 3).reshape(16, 5).half()
OFFSETS = (torch.arange(5) % 4 - 1).half()
CHANNEL_OUTER = {1: (1, 8)}


def test_bypass_relays_a_matrix_channel_outer():
    check("bypass(perm=(2, 1, 0))", ROWS, ROWS, strides={0: (1, 8)})


IMAGE = (torch.arange(4 * 6 * 6) % 7 - 3).reshape(1, 4, 6, 6).half()
KERNELS = (torch.arange(6 * 2 * 3 * 3) % 5 - 2).reshape(6, 2, 3, 3).half()
BIASES = (torch.arange(6) % 4 - 1).half()
CONVOLVED = F.conv2d(IMAGE.float(), KERNELS.float(), BIASES.float(), stride=2, padding=1, groups=2)
CONV = "stride=(2, 2), padding=(1, 1), groups=2"
PRODUCT = ROWS.float() @ COLUMNS.float()


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ("", CONVOLVED),
        (", nonlinear=T.ane.Nonlinear.RELU", CONVOLVED.relu()),
    ],
)
def test_conv_feature_form_is_f_conv2d(extra, expected):
    check(f"conv({CONV}{extra})", expected, IMAGE, KERNELS, BIASES)


@pytest.mark.parametrize(
    ("attributes", "operands", "expected"),
    [
        ("", (ROWS, COLUMNS), PRODUCT),
        ("", (ROWS, COLUMNS, OFFSETS), PRODUCT + OFFSETS.float()),
        ("nonlinear=T.ane.Nonlinear.RELU", (ROWS, COLUMNS, OFFSETS), (PRODUCT + OFFSETS.float()).relu()),
    ],
)
def test_conv_matrix_form_is_a_product(attributes, operands, expected):
    check(f"conv({attributes})", expected, *operands, strides=CHANNEL_OUTER)


@pytest.mark.parametrize(
    ("lhs", "rhs", "storage"),
    [
        ("smem", "gmem", expr.StorageKind.SMEM),
        ("gmem", "smem", expr.StorageKind.SMEM),
        ("gmem", "gmem", expr.StorageKind.GMEM),
    ],
)
def test_ane_matmul_result_follows_an_on_chip_operand(lhs, rhs, storage):
    lhs_type, rhs_type = (types.TensorType.from_shape(shape, expr.DType.f16, held)
                          for shape, held in (((4, 8), lhs), ((8, 6), rhs)))
    result = op_case.infer_call(nn.MatMul(), lhs_type, rhs_type, target=ane.Target())
    assert result.storage is storage


FEATURE_TYPE = tensor((1, 4, 6, 6))
POOLED_TYPE = tensor((1, 4, 3, 3))
SCALAR_TYPE = tensor((1, 4, 1, 1))
ADD = "binary(kind=BinaryKind.ADD)"
POOL = "pool(kernel_size=(2, 2), stride=(2, 2), kind=PoolKind.MAX)"
REDUCE = "reduce(axes=(2, 3), keepdim=True, kind=ReduceKind.SUM)"
HALF = tensor((1, 4, 6, 3))
SINGLE = tensor((1, 4, 6, 6), "f32")
KERNEL_TYPE = tensor((6, 4, 3, 3))
CONVOLVED_TYPE = tensor((1, 6, 4, 4))
SWAP = "bypass(perm=(2, 1, 0))"


@pytest.mark.parametrize(
    ("op", "dst", "operands", "word"),
    [
        (ADD, SINGLE, (SINGLE, SINGLE), "dtype"),
        ("unary(abs=True)", tensor((4, 6, 6)), (tensor((4, 6, 6)),), "shape"),
        (POOL, tensor((1, 70000, 1, 1)), (tensor((1, 70000, 2, 2)),), "shape"),
        (REDUCE, SCALAR_TYPE, (tensor((1, 4, 6, 6), storage="umat"),), "storage"),
        (POOL.replace("stride=(2, 2)", "stride=(3, 3)"), POOLED_TYPE, (FEATURE_TYPE,), "stride"),
        (POOL.replace("kind", "padding=(2, 0), kind"), POOLED_TYPE, (FEATURE_TYPE,), "padding"),
        (REDUCE.replace("(2, 3)", "(2,)"), SCALAR_TYPE, (FEATURE_TYPE,), "axes"),
        (REDUCE.replace("(2, 3)", "(3,)").replace("True", "False"), tensor((1, 4, 6)), (FEATURE_TYPE,), "keepdim"),
        (ADD.replace("ADD", "DIV"), FEATURE_TYPE, (FEATURE_TYPE, FEATURE_TYPE), "kind"),
        (REDUCE.replace("SUM", "MEAN"), SCALAR_TYPE, (FEATURE_TYPE,), "kind"),
        (ADD, FEATURE_TYPE, (FEATURE_TYPE, HALF), "shape"),
        (ADD, HALF, (FEATURE_TYPE, FEATURE_TYPE), "shape"),
        ("unary(clamp=(6.0, 0.0))", FEATURE_TYPE, (FEATURE_TYPE,), "clamp"),
        (POOL, tensor((1, 4, 2, 3)), (FEATURE_TYPE,), "shape"),
        (REDUCE, tensor((1, 4, 6, 1)), (FEATURE_TYPE,), "shape"),
        ("unary(abs=True)", HALF, (FEATURE_TYPE,), "shape"),
        ("conv()", tensor((1, 6, 4, 4), "f32"), (tensor((1, 4, 6, 6), "f32"), tensor((6, 4, 3, 3), "f32")), "dtype"),
        ("bypass()", SINGLE, (SINGLE,), "dtype"),
        ("conv()", tensor((6, 4, 4)), (tensor((4, 6, 6)), KERNEL_TYPE), "shape"),
        ("conv()", tensor((1, 6, 1, 1)), (tensor((1, 70000, 2, 2)), tensor((6, 70000, 2, 2))), "shape"),
        ("conv()", tensor((1, 6, 8, 8)), (tensor((1, 4, 40, 40)), tensor((6, 4, 33, 33))), "shape"),
        (SWAP, tensor((1, 2, 2, 20000)), (tensor((1, 20000, 2, 2)),), "shape"),
        ("conv()", CONVOLVED_TYPE, (FEATURE_TYPE, tensor((6, 4, 3, 3), storage="smem")), "storage"),
        ("conv()", CONVOLVED_TYPE, (FEATURE_TYPE, KERNEL_TYPE, tensor((6,), storage="smem")), "storage"),
        ("conv(stride=(3, 3))", CONVOLVED_TYPE, (FEATURE_TYPE, KERNEL_TYPE), "stride"),
        ("conv(padding=(3, 0))", tensor((1, 6, 10, 4)), (FEATURE_TYPE, KERNEL_TYPE), "padding"),
        ("conv(dilation=(2, 2))", CONVOLVED_TYPE, (FEATURE_TYPE, KERNEL_TYPE), "dilation"),
        ("conv(groups=3)", CONVOLVED_TYPE, (FEATURE_TYPE, KERNEL_TYPE), "groups"),
        ("conv()", tensor((8, 5)), (tensor((8, 16)), tensor((16, 5))), "layout"),
        ("bypass(perm=(0, 0, 1))", FEATURE_TYPE, (FEATURE_TYPE,), "perm"),
        ("bypass(perm=(1, 0, 2, 3))", FEATURE_TYPE, (FEATURE_TYPE,), "perm"),
        ("conv()", tensor((1, 6, 4, 3)), (FEATURE_TYPE, KERNEL_TYPE), "shape"),
        (SWAP, FEATURE_TYPE, (FEATURE_TYPE,), "shape"),
        ("conv()", CONVOLVED_TYPE, (tensor((8, 16), strides=(1, 8)), KERNEL_TYPE), "shape"),
        ("conv(stride=(2, 2))", tensor((8, 5)), (tensor((8, 16), strides=(1, 8)), tensor((16, 5))), "stride"),
        ("bypass()", tensor((8, 16), strides=(1, 8)), (tensor((8, 16)),), "perm"),
        (SWAP, tensor((8, 16)), (tensor((8, 16)),), "layout"),
    ],
)
def test_load_refuses(op, dst, operands, word):
    with pytest.raises(expr.VerifyError, match=word):
        load(op, dst, *operands)
