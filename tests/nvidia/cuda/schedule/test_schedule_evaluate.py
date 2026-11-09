"""Scheduling schedule evaluate checks."""

from __future__ import annotations

import pytest
import torch

from tests.nvidia.cuda.schedule._support import (
    _evaluate_call,
    _UnstatedInstruction,
)
from tilefoundry.interfaces import evaluator
from tilefoundry.ir import expr, hir, tir, types
from tilefoundry_nvidia.cuda.ir.tir import memory, nn


@pytest.mark.parametrize("dtype", (expr.DType.i32, expr.DType.i64))
def test_schedule_arange_evaluates_static_coordinates(dtype) -> None:
    type_ = types.TensorType((8,), dtype, types.Layout((8,), (1,)), expr.StorageKind.RMEM)
    torch_dtype = evaluator.to_torch_dtype(dtype)
    result = _evaluate_call(
        hir.schedule.ScheduleOp(op=tir.arange.Arange(start=3, step=2)),
        ((type_, torch.empty(8, dtype=torch_dtype)),),
        type_,
    )
    assert torch.equal(result, torch.tensor([3, 5, 7, 9, 11, 13, 15, 17], dtype=torch_dtype))


def test_schedule_copy_returns_source() -> None:
    layout = types.Layout((4,), (1,))
    source_type = types.TensorType((4,), expr.DType.bf16, layout, expr.StorageKind.GMEM)
    result_type = types.TensorType((4,), expr.DType.bf16, layout, expr.StorageKind.SMEM)
    source = torch.arange(4, dtype=torch.bfloat16)

    result = _evaluate_call(
        hir.schedule.ScheduleOp(op=tir.async_copy.CopyAsync()),
        ((source_type, source), (result_type, torch.zeros_like(source))),
        result_type,
    )

    assert torch.equal(result, source)


def test_reduce_store_adds_to_destination():
    cta = types.Mesh((types.Topology("cta", 2),), types.Layout((2,), (1,)))
    source_type = types.TensorType(
        (64, 64),
        expr.DType.f32,
        types.ShardLayout(types.Layout((64, 64), (64, 1)), (types.Partial("sum"),), cta),
        expr.StorageKind.SMEM,
    )
    result_type = types.TensorType((64, 64), expr.DType.f32, types.Layout((64, 64), (64, 1)), expr.StorageKind.GMEM)
    source = torch.arange(4096, dtype=torch.float32).reshape(64, 64)
    destination = torch.full_like(source, 7)
    result = _evaluate_call(
        hir.schedule.ScheduleOp(op=memory.CopyReduceAsyncTensor()),
        ((source_type, source), (result_type, destination)),
        result_type,
    )
    assert torch.equal(result, destination + source)


@pytest.mark.parametrize(
    ("atom", "dtype", "shape"),
    (
        pytest.param(nn.Mma(), expr.DType.bf16, (16, 16, 8), id="sm80_bf16"),
        pytest.param(
            nn.Wgmma(n=8, dtype="fp8e4m3", form=nn.Form.SS), expr.DType.fp8e4m3, (64, 32, 8), id="fp8_ss"
        ),
        pytest.param(
            nn.Wgmma(n=8, dtype="fp8e4m3", form=nn.Form.RS), expr.DType.fp8e4m3, (64, 32, 8), id="fp8_rs"
        ),
    ),
)
def test_schedule_mma_accumulates_exact_product(atom, dtype, shape) -> None:
    """Products are summed in the accumulator dtype, not rounded to the operand dtype first.

    Column 0 of lhs holds 2**-8, so each sum carries bits an operand-dtype result drops.
    """
    m, k, n = shape
    operand = evaluator.to_torch_dtype(dtype)
    acc_type = types.TensorType((m, n), expr.DType.f32, None, expr.StorageKind.RMEM)
    lhs_type = types.TensorType((m, k), dtype, None, expr.StorageKind.RMEM)
    rhs_type = types.TensorType((k, n), dtype, None, expr.StorageKind.RMEM)
    acc = torch.arange(m * n, dtype=torch.float32).reshape(m, n)
    lhs = (torch.arange(m * k).reshape(m, k) % 5).float()
    lhs[:, 0] = 2.0**-8
    lhs = lhs.to(operand)
    rhs = (torch.arange(k * n).reshape(k, n) % 3).to(operand)

    scheduled = _evaluate_call(
        hir.schedule.ScheduleOp(op=nn.TiledMma(atom=atom)),
        ((acc_type, acc), (lhs_type, lhs), (rhs_type, rhs)),
        acc_type,
    )

    exact = lhs.float() @ rhs.float()
    assert torch.equal(scheduled, acc + exact)
    assert not torch.equal(scheduled, acc + exact.to(operand).float())


def test_schedule_evaluation_rejects_unregistered_instruction() -> None:
    type_ = types.TensorType((4,), expr.DType.bf16, types.Layout((4,), (1,)), expr.StorageKind.RMEM)
    value = torch.arange(4, dtype=torch.bfloat16)

    with pytest.raises(evaluator.EvalError, match="no schedule evaluator registered"):
        _evaluate_call(hir.schedule.ScheduleOp(op=_UnstatedInstruction()), ((type_, value),), type_)
