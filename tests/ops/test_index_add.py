"""Whole-slice indexed writes: torch values, shared types, and touched traffic."""

from __future__ import annotations

import pytest
import torch

from tests.evaluator.eval_utils import EvalCase, run_eval_case
from tests.ops.cost_utils import CostCase, run_cost_case
from tests.ops.typeinfer_utils import ExpectedError, TypeInferCase, run_typeinfer_case
from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import index_add, index_copy
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.tensor.index_add import IndexAdd
from tilefoundry.ir.hir.tensor.index_copy import IndexCopy
from tilefoundry.ir.types import DType, make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.shard import make_mesh
from tilefoundry.ir.types.shard.shard_layout import Partial
from tilefoundry.passes.transforms import HirToTirPass
from tilefoundry.visitor_registry.contexts import TrafficBytes

_F32 = DType.f32
_I32 = DType.i32
_I64 = DType.i64
_MESH = make_mesh((2,))


TYPEINFER_CASES = [
    TypeInferCase(
        "index_add_result_is_dst",
        IndexAdd(dim=1),
        (
            make_tensor_type((2, 5, 3), _F32),
            make_tensor_type((4,), _I32),
            make_tensor_type((2, 4, 3), _F32),
        ),
        make_tensor_type((2, 5, 3), _F32),
    ),
    TypeInferCase(
        "index_rank_rejected",
        IndexAdd(),
        (
            make_tensor_type((5, 3), _F32),
            make_tensor_type((1, 2), _I64),
            make_tensor_type((2, 3), _F32),
        ),
        ExpectedError(match="IndexAdd: index must be 1-D"),
    ),
    TypeInferCase(
        "index_length_rejected",
        IndexAdd(),
        (
            make_tensor_type((5, 3), _F32),
            make_tensor_type((3,), _I64),
            make_tensor_type((2, 3), _F32),
        ),
        ExpectedError(match=r"index length 3 must equal src.shape\[0\] 2"),
    ),
    TypeInferCase(
        "src_rank_rejected",
        IndexAdd(),
        (
            make_tensor_type((5, 3), _F32),
            make_tensor_type((2,), _I64),
            make_tensor_type((2, 3, 1), _F32),
        ),
        ExpectedError(match="src rank 3 must equal dst rank 2"),
    ),
    TypeInferCase(
        "src_dtype_rejected",
        IndexAdd(),
        (
            make_tensor_type((5, 3), _F32),
            make_tensor_type((2,), _I64),
            make_tensor_type((2, 3), DType.f16),
        ),
        ExpectedError(match="dst/src dtype mismatch"),
    ),
    TypeInferCase(
        "non_indexed_shape_rejected",
        IndexAdd(),
        (
            make_tensor_type((5, 3), _F32),
            make_tensor_type((2,), _I64),
            make_tensor_type((2, 4), _F32),
        ),
        ExpectedError(match="outside dim 0; mismatch at dim 1"),
    ),
    TypeInferCase(
        "partial_src_rejected",
        IndexAdd(),
        (
            make_tensor_type((5, 3), _F32),
            make_tensor_type((2,), _I64),
            make_shard_tensor_type(
                (2, 3), _F32, mesh=_MESH, attrs=(Partial("sum"),)
            ),
        ),
        ExpectedError(match="src carries Partial"),
    ),
    TypeInferCase(
        "index_copy_requires_i64",
        IndexCopy(),
        (
            make_tensor_type((5, 3), _F32),
            make_tensor_type((2,), _I32),
            make_tensor_type((2, 3), _F32),
        ),
        ExpectedError(match="IndexCopy: index must have dtype i64"),
    ),
]


@pytest.mark.parametrize("case", TYPEINFER_CASES, ids=lambda case: case.name)
def test_index_write_typeinfer(case):
    run_typeinfer_case(case)


def test_index_add_repeated_index_accumulates_like_torch() -> None:
    torch.manual_seed(0)
    dst = torch.randn(5, 3)
    index = torch.tensor([1, 1, 4], dtype=torch.int32)
    src = torch.randn(3, 3)
    expected = dst.clone().index_add_(0, index, src)
    run_eval_case(EvalCase("", IndexAdd(), (dst, index, src), expected))


def test_index_copy_permutation_matches_torch() -> None:
    torch.manual_seed(1)
    dst = torch.randn(5, 3)
    index = torch.tensor([4, 0, 2], dtype=torch.int64)
    src = torch.randn(3, 3)
    expected = dst.clone().index_copy_(0, index, src)
    run_eval_case(EvalCase("", IndexCopy(), (dst, index, src), expected))


COST_CASES = [
    CostCase(
        name=f"index_add_dst_rows_{dst_rows}",
        op=IndexAdd(),
        inputs=(
            make_tensor_type((dst_rows, 4), _F32),
            make_tensor_type((2,), _I64),
            make_tensor_type((2, 4), _F32),
        ),
        flops={_F32: 8},
        traffic=(
            TrafficBytes(read=32),
            TrafficBytes(read=16),
            TrafficBytes(read=32),
            TrafficBytes(write=32),
        ),
    )
    for dst_rows in (8, 8192)
] + [
    CostCase(
        name="index_copy",
        op=IndexCopy(),
        inputs=(
            make_tensor_type((8192, 4), _F32),
            make_tensor_type((2,), _I64),
            make_tensor_type((2, 4), _F32),
        ),
        traffic=(
            TrafficBytes(),
            TrafficBytes(read=16),
            TrafficBytes(read=32),
            TrafficBytes(write=32),
        ),
    )
]


@pytest.mark.parametrize("case", COST_CASES, ids=lambda case: case.name)
def test_index_write_costs_only_touched_slices(case) -> None:
    run_cost_case(case)


@func
def _index_add_without_lowering(
    dst: Tensor[(5, 3), "f32"],
    index: Tensor[(2,), "i64"],
    src: Tensor[(2, 3), "f32"],
) -> Tensor[(5, 3), "f32"]:
    return index_add(dst, index, src, dim=0)


@func
def _index_copy_without_lowering(
    dst: Tensor[(5, 3), "f32"],
    index: Tensor[(2,), "i64"],
    src: Tensor[(2, 3), "f32"],
) -> Tensor[(5, 3), "f32"]:
    return index_copy(dst, index, src, dim=0)


@pytest.mark.parametrize(
    "fn", (_index_add_without_lowering, _index_copy_without_lowering)
)
def test_index_writes_have_no_hir_to_tir_lowering(fn) -> None:
    module = Module(name="t", functions=(fn,), entry=fn.name)
    with pytest.raises(TypeError, match="no lowering registered"):
        HirToTirPass().run(module)
