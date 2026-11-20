"""ANE processing-engine instructions: values against torch, and the rules checked at load."""

from __future__ import annotations

import itertools
from math import prod

import pytest

from tests.core._corpus import reload
from tilefoundry import utils
from tilefoundry.interfaces import evaluator
from tilefoundry.ir import expr
from tilefoundry_apple.ane.ir.tir import evaluation

torch = utils.torch
F = utils.F

PROGRAM = """
from tilefoundry.dsl import *
from tilefoundry.ir.expr import PoolKind
from tilefoundry_apple import ane


@module(entry="main", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def main({params}):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as engine:
            return tf.schedule((tf.empty({dst}), {operands}), op=T.ane.{op})
"""


def tensor(shape, dtype="f16", storage="gmem") -> str:
    """Source of a row-major tensor type."""
    strides = tuple(prod(shape[axis + 1:]) for axis in range(len(shape)))
    return f'Tensor[{tuple(shape)}, "{dtype}", ({tuple(shape)}, {strides}), "{storage}"]'


def load(op: str, dst: str, *operands: str):
    """The module that applies *op* to a fresh *dst* and to parameters of the *operands* types."""
    names = tuple(f"x{index}" for index in range(len(operands)))
    params = ", ".join(f"{name}: {type_}" for name, type_ in zip(names, operands))
    return reload(PROGRAM.format(params=params, dst=dst, operands=", ".join(names), op=op), "Program")


def run(op: str, shape, *inputs):
    """Evaluate *op* over *inputs* into a destination of *shape*."""
    module = load(op, tensor(shape), *(tensor(value.shape) for value in inputs))
    return evaluator.evaluate(module, *(value.clone() for value in inputs))


def check(op: str, expected, *inputs) -> None:
    assert torch.equal(run(op, expected.shape, *inputs), expected.to(torch.float16)), op


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


FEATURE_TYPE = tensor((1, 4, 6, 6))
POOLED_TYPE = tensor((1, 4, 3, 3))
SCALAR_TYPE = tensor((1, 4, 1, 1))
ADD = "binary(kind=BinaryKind.ADD)"
POOL = "pool(kernel_size=(2, 2), stride=(2, 2), kind=PoolKind.MAX)"
REDUCE = "reduce(axes=(2, 3), keepdim=True, kind=ReduceKind.SUM)"
HALF = tensor((1, 4, 6, 3))
SINGLE = tensor((1, 4, 6, 6), "f32")


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
    ],
)
def test_load_refuses(op, dst, operands, word):
    with pytest.raises(expr.VerifyError, match=word):
        load(op, dst, *operands)
