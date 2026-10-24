"""Scheduling schedule evaluate checks."""

from __future__ import annotations

import pytest
import torch

from tests.nvidia.cuda.schedule._support import (
    _evaluate_call,
    _UnstatedInstruction,
)
from tilefoundry.evaluator import EvalError
from tilefoundry.evaluator.value import to_torch_dtype
from tilefoundry.ir.hir.schedule import ScheduleOp
from tilefoundry.ir.tir.arange import Arange as TirArange
from tilefoundry.ir.tir.async_copy import CopyAsync
from tilefoundry.ir.types import (
    DType,
    Layout,
    Mesh,
    Partial,
    ShardLayout,
    StorageKind,
    TensorType,
    Topology,
)
from tilefoundry_nvidia.cuda.ir.tir.memory.copy_reduce_async_tensor import CopyReduceAsyncTensor
from tilefoundry_nvidia.cuda.ir.tir.nn.mma import TiledMma
from tilefoundry_nvidia.cuda.ir.tir.nn.sm80_mma import Mma
from tilefoundry_nvidia.cuda.ir.tir.nn.wgmma import Form, Wgmma


@pytest.mark.parametrize("dtype", (DType.i32, DType.i64))
def test_schedule_arange_evaluates_static_coordinates(dtype) -> None:
    type_ = TensorType((8,), dtype, Layout((8,), (1,)), StorageKind.RMEM)
    torch_dtype = to_torch_dtype(dtype)
    result = _evaluate_call(
        ScheduleOp(op=TirArange(start=3, step=2)),
        ((type_, torch.empty(8, dtype=torch_dtype)),),
        type_,
    )
    assert torch.equal(result, torch.tensor([3, 5, 7, 9, 11, 13, 15, 17], dtype=torch_dtype))


def test_schedule_copy_evaluates_to_the_source_value() -> None:
    layout = Layout((4,), (1,))
    source_type = TensorType((4,), DType.bf16, layout, StorageKind.GMEM)
    result_type = TensorType((4,), DType.bf16, layout, StorageKind.SMEM)
    source = torch.arange(4, dtype=torch.bfloat16)

    result = _evaluate_call(
        ScheduleOp(op=CopyAsync()),
        ((source_type, source), (result_type, torch.zeros_like(source))),
        result_type,
    )

    assert torch.equal(result, source)


def test_schedule_reduce_store_adds_to_an_explicit_nonzero_destination():
    cta = Mesh((Topology("cta", 2),), Layout((2,), (1,)))
    source_type = TensorType(
        (64, 64),
        DType.f32,
        ShardLayout(Layout((64, 64), (64, 1)), (Partial("sum"),), cta),
        StorageKind.SMEM,
    )
    result_type = TensorType((64, 64), DType.f32, Layout((64, 64), (64, 1)), StorageKind.GMEM)
    source = torch.arange(4096, dtype=torch.float32).reshape(64, 64)
    destination = torch.full_like(source, 7)
    result = _evaluate_call(
        ScheduleOp(op=CopyReduceAsyncTensor()),
        ((source_type, source), (result_type, destination)),
        result_type,
    )
    assert torch.equal(result, destination + source)


@pytest.mark.parametrize(
    ("atom", "dtype", "shape"),
    (
        pytest.param(Mma(), DType.bf16, (16, 16, 8), id="sm80_bf16"),
        pytest.param(
            Wgmma(n=8, dtype="fp8e4m3", form=Form.SS), DType.fp8e4m3, (64, 32, 8), id="fp8_ss"
        ),
        pytest.param(
            Wgmma(n=8, dtype="fp8e4m3", form=Form.RS), DType.fp8e4m3, (64, 32, 8), id="fp8_rs"
        ),
    ),
)
def test_schedule_mma_evaluates_like_matmul_plus_accumulator(atom, dtype, shape) -> None:
    """Products are summed in the accumulator dtype, not rounded to the operand dtype first.

    Column 0 of lhs holds 2**-8, so each sum carries bits an operand-dtype result drops.
    """
    m, k, n = shape
    operand = to_torch_dtype(dtype)
    acc_type = TensorType((m, n), DType.f32, None, StorageKind.RMEM)
    lhs_type = TensorType((m, k), dtype, None, StorageKind.RMEM)
    rhs_type = TensorType((k, n), dtype, None, StorageKind.RMEM)
    acc = torch.arange(m * n, dtype=torch.float32).reshape(m, n)
    lhs = (torch.arange(m * k).reshape(m, k) % 5).float()
    lhs[:, 0] = 2.0**-8
    lhs = lhs.to(operand)
    rhs = (torch.arange(k * n).reshape(k, n) % 3).to(operand)

    scheduled = _evaluate_call(
        ScheduleOp(op=TiledMma(atom=atom)),
        ((acc_type, acc), (lhs_type, lhs), (rhs_type, rhs)),
        acc_type,
    )

    exact = lhs.float() @ rhs.float()
    assert torch.equal(scheduled, acc + exact)
    assert not torch.equal(scheduled, acc + exact.to(operand).float())


def test_schedule_evaluation_rejects_an_unregistered_instruction() -> None:
    type_ = TensorType((4,), DType.bf16, Layout((4,), (1,)), StorageKind.RMEM)
    value = torch.arange(4, dtype=torch.bfloat16)

    with pytest.raises(EvalError, match="no schedule evaluator registered"):
        _evaluate_call(ScheduleOp(op=_UnstatedInstruction()), ((type_, value),), type_)
