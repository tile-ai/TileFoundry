"""IndexAdd and IndexCopy on split operands: a Split off ``dim`` is written, one on it refused."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.core.ops.ir.op_case import infer_call
from tilefoundry import cli, runtime
from tilefoundry.dsl import *
from tilefoundry.interfaces import evaluator
from tilefoundry.ir import expr, types
from tilefoundry.ir.hir import tensor
from tilefoundry.target import cpu

FIXTURES = Path(__file__).parents[2] / "fixtures"
MESH = Mesh((Topology("gpu", 2),), Layout((2,)), ("shard",))
OWNERS = cpu.Target().storage_owners
OPS = (tensor.IndexAdd(dim=0), tensor.IndexCopy(dim=0))


def placed(shape, attr, dtype=DType.f32) -> types.TensorType:
    return TensorType(shape, dtype, ShardLayout.from_logical(shape, MESH, (attr,), storage="gmem", owners=OWNERS), "gmem")


INDEX = placed((3,), Broadcast(), DType.i64)


@pytest.mark.parametrize("op", OPS, ids=lambda op: type(op).__name__)
def test_split_off_dim_keeps_the_destination_layout(op) -> None:
    table = placed((5, 6), Split(1))
    assert infer_call(op, table, INDEX, placed((3, 6), Split(1))) == table


@pytest.mark.parametrize("op", OPS, ids=lambda op: type(op).__name__)
@pytest.mark.parametrize(
    ("table", "index", "rows", "message"),
    (
        (placed((6, 6), Split(0)), INDEX, placed((3, 6), Broadcast()), "dst is Split on dim 0"),
        (
            placed((5, 6), Broadcast()),
            placed((4,), Broadcast(), DType.i64),
            placed((4, 6), Split(0)),
            "src is Split on dim 0",
        ),
        (
            placed((5, 6), Broadcast()),
            placed((4,), Split(0), DType.i64),
            placed((4, 6), Broadcast()),
            "index is Split on dim 0",
        ),
        (placed((5, 6), Split(1)), INDEX, placed((3, 6), Broadcast()), "src must be split on the axes dst is"),
    ),
    ids=("dst on dim", "src on dim", "index", "src unlike dst"),
)
def test_a_split_the_index_values_decide_is_refused(op, table, index, rows, message) -> None:
    name = type(op).__name__
    with pytest.raises(expr.VerifyError, match=f"^{name}: {message}"):
        infer_call(op, table, index, rows)


@pytest.mark.parametrize(
    ("name", "reference"),
    (("IndexAddColumns", torch.Tensor.index_add), ("IndexCopyColumns", torch.Tensor.index_copy)),
)
def test_split_off_dim_evaluates_as_torch(name, reference) -> None:
    module = cli.load_authored_ir(f"{FIXTURES / 'logical' / 'index_write_split.py'}:{name}")
    generator = torch.Generator().manual_seed(0)
    table, rows = torch.randn((5, 6), generator=generator), torch.randn((3, 6), generator=generator)
    index = torch.tensor([4, 0, 2])
    got = evaluator.evaluate(module.load(runtime.DictResource({})), table, index, rows)
    torch.testing.assert_close(got, reference(table, 0, index, rows))
