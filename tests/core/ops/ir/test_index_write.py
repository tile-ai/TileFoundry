"""IndexAdd and IndexCopy on split operands: a Split off ``dim`` is written, one on it refused."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.core.ops.ir.op_case import infer_call
from tilefoundry import cli, runtime
from tilefoundry.dsl import *
from tilefoundry.interfaces import evaluator
from tilefoundry.ir import expr, utils
from tilefoundry.ir.hir import tensor
from tilefoundry.target import cpu

FIXTURES = Path(__file__).parents[2] / "fixtures"
MESH = Mesh((Topology("gpu", 2),), Layout((2,)), ("shard",))
OWNERS = cpu.Target().storage_owners
OPS = (tensor.IndexAdd(dim=0), tensor.IndexCopy(dim=0))


INDEX = utils.make_shard_tensor_type((3,), DType.i64, mesh=MESH, attrs=(Broadcast(),), owners=OWNERS)


@pytest.mark.parametrize("op", OPS, ids=lambda op: type(op).__name__)
def test_split_off_dim_keeps_the_destination_layout(op) -> None:
    table = utils.make_shard_tensor_type((5, 6), mesh=MESH, attrs=(Split(1),), owners=OWNERS)
    assert (
        infer_call(
            op,
            table,
            INDEX,
            utils.make_shard_tensor_type((3, 6), mesh=MESH, attrs=(Split(1),), owners=OWNERS),
        )
        == table
    )


@pytest.mark.parametrize("op", OPS, ids=lambda op: type(op).__name__)
def test_a_plain_table_takes_broadcast_rows(op) -> None:
    table = TensorType.from_shape((5, 6), DType.f32)
    assert (
        infer_call(
            op,
            table,
            INDEX,
            utils.make_shard_tensor_type((3, 6), mesh=MESH, attrs=(Broadcast(),), owners=OWNERS),
        )
        == table
    )


@pytest.mark.parametrize("op", OPS, ids=lambda op: type(op).__name__)
@pytest.mark.parametrize(
    ("table", "index", "rows", "message"),
    (
        (
            utils.make_shard_tensor_type((6, 6), mesh=MESH, attrs=(Split(0),), owners=OWNERS),
            INDEX,
            utils.make_shard_tensor_type((3, 6), mesh=MESH, attrs=(Broadcast(),), owners=OWNERS),
            "dst is Split on dim 0",
        ),
        (
            utils.make_shard_tensor_type((5, 6), mesh=MESH, attrs=(Broadcast(),), owners=OWNERS),
            utils.make_shard_tensor_type((4,), DType.i64, mesh=MESH, attrs=(Broadcast(),), owners=OWNERS),
            utils.make_shard_tensor_type((4, 6), mesh=MESH, attrs=(Split(0),), owners=OWNERS),
            "src is Split on dim 0",
        ),
        (
            utils.make_shard_tensor_type((5, 6), mesh=MESH, attrs=(Broadcast(),), owners=OWNERS),
            utils.make_shard_tensor_type((4,), DType.i64, mesh=MESH, attrs=(Split(0),), owners=OWNERS),
            utils.make_shard_tensor_type((4, 6), mesh=MESH, attrs=(Broadcast(),), owners=OWNERS),
            "index is Split on dim 0",
        ),
        (
            utils.make_shard_tensor_type((5, 6), mesh=MESH, attrs=(Split(1),), owners=OWNERS),
            INDEX,
            utils.make_shard_tensor_type((3, 6), mesh=MESH, attrs=(Broadcast(),), owners=OWNERS),
            "src must be split on the axes dst is",
        ),
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
