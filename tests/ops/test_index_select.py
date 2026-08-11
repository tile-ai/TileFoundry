"""Torch-compatible IndexSelect type, value, shard, and lowering boundaries."""

from __future__ import annotations

import pytest
import torch

from tests.evaluator.eval_utils import EvalCase, run_eval_case
from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    infer_call,
    run_typeinfer_case,
)
from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import index_select
from tilefoundry.ir.core import Call, Constant
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.tensor.index_select import IndexSelect
from tilefoundry.ir.tir.memory.tensor_view import TensorView
from tilefoundry.ir.tir.stmts import LetStmt
from tilefoundry.ir.types import DType, make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.dim import DimMul
from tilefoundry.ir.types.shard import make_mesh
from tilefoundry.ir.types.shard.shard_layout import Partial, ShardLayout, Split
from tilefoundry.passes.transforms import HirToTirPass
from tilefoundry.target import CudaTarget

_MESH = make_mesh((2,))


@pytest.mark.parametrize(
    ("attrs", "expected"),
    [
        pytest.param((Split(1),), (Split(1),), id="other_dim_split_survives"),
        pytest.param(
            (Split(0),),
            (Partial(reduction="sum"),),
            id="selected_dim_split_becomes_partial",
        ),
    ],
)
def test_index_select_carries_shard_state(attrs, expected):
    ty = infer_call(
        IndexSelect(dim=0),
        make_shard_tensor_type((6, 4, 8), mesh=_MESH, attrs=attrs),
        make_tensor_type((2,), DType.i32),
    )

    assert tuple(ty.shape) == (2, 4, 8)
    assert isinstance(ty.layout, ShardLayout) and ty.layout.attrs == expected


TYPEINFER_CASES = [
    TypeInferCase(
        "rank_two_index_rejected",
        IndexSelect(dim=0),
        (make_tensor_type((6, 4), DType.f32), make_tensor_type((1, 2), DType.i64)),
        ExpectedError(match="index must be 1-D"),
    ),
    TypeInferCase(
        "float_index_rejected",
        IndexSelect(dim=1),
        (make_tensor_type((6, 4), DType.f32), make_tensor_type((2,), DType.f32)),
        ExpectedError(match="index must have dtype i32 or i64"),
    ),
]


@pytest.mark.parametrize("case", TYPEINFER_CASES, ids=lambda case: case.name)
def test_index_select_typeinfer(case):
    run_typeinfer_case(case)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_index_select_matches_torch(dtype):
    torch.manual_seed(0)
    x = torch.randn(3, 5)
    index = torch.tensor([4, 1, 4], dtype=dtype)
    run_eval_case(
        EvalCase(
            "",
            IndexSelect(dim=-1),
            (x, index),
            torch.index_select(x, -1, index),
        )
    )


@func
def _one_index_select(
    x: Tensor[(4, 3), "f32"], index: Tensor[(1,), "i32"]
) -> Tensor[(1, 3), "f32"]:
    return index_select(x, index, dim=0)


@func
def _vector_index_select(
    x: Tensor[(4, 3), "f32"], index: Tensor[(2,), "i32"]
) -> Tensor[(2, 3), "f32"]:
    return index_select(x, index, dim=0)


def _module(fn) -> Module:
    return Module(name="t", functions=(fn,), entry=fn.name)


def test_one_element_index_select_lowers_as_a_view() -> None:
    pf = HirToTirPass().run(_module(_one_index_select)).functions[0]
    views = []

    def walk(stmt):
        if (
            isinstance(stmt, LetStmt)
            and isinstance(stmt.value, Call)
            and isinstance(stmt.value.target, TensorView)
            and len(stmt.value.args) == 2
        ):
            views.append(stmt.value)
        for attr in ("body", "stmts"):
            child = getattr(stmt, attr, None)
            if isinstance(child, (list, tuple)):
                for item in child:
                    walk(item)
            elif child is not None and hasattr(child, "__dict__"):
                walk(child)

    walk(pf.body)
    assert len(views) == 1
    coordinate = views[0].args[1]
    assert isinstance(coordinate, Call) and isinstance(coordinate.target, DimMul)
    assert isinstance(coordinate.args[1], Constant) and coordinate.args[1].value == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_one_element_index_select_gpu_uses_the_axis_stride() -> None:
    import tilefoundry  # noqa: PLC0415

    runtime = tilefoundry.compile(
        _module(_one_index_select), target=CudaTarget("nvidia.h200_sxm")
    )
    x = torch.arange(12, dtype=torch.float32, device="cuda").reshape(4, 3)
    index = torch.tensor([2], dtype=torch.int32, device="cuda")
    actual = runtime(x, index)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, x[2:3])


def test_vector_index_select_lowering_fails_closed() -> None:
    with pytest.raises(NotImplementedError, match="one-element index"):
        HirToTirPass().run(_module(_vector_index_select))
