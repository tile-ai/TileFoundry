"""CUDA MatMul type inference across storage and architecture cases."""

from __future__ import annotations

from tests.core.ops.ir.op_case import ExpectedError, OpCase, op_case_test
from tests.core.ops.ir.test_matmul import _MM
from tilefoundry.ir import expr, types
from tilefoundry.ir.hir import nn
from tilefoundry_nvidia import cuda

_H200 = cuda.Target("nvidia.h200_sxm")
_SM89 = cuda.Target("nvidia.rtx_6000_ada")
CASES = (
    OpCase(
        "cuda_on_chip_result_is_where_the_mma_writes_it",
        nn.MatMul(out_dtype="f32"),
        (
            types.TensorType.from_shape((64, 16), expr.DType.bf16, "smem"),
            types.TensorType.from_shape((16, 32), expr.DType.bf16, "smem"),
        ),
        type=types.TensorType.from_shape((64, 32), expr.DType.f32, "rmem"),
        target=_H200,
    ),
    OpCase(
        "cuda_fp8_register_lhs_reads_like_rs",
        nn.MatMul(out_dtype="f32"),
        (
            types.TensorType.from_shape((64, 32), expr.DType.fp8e4m3, "rmem"),
            types.TensorType.from_shape((32, 32), expr.DType.fp8e4m3, "smem"),
        ),
        type=types.TensorType.from_shape((64, 32), expr.DType.f32, "rmem"),
        target=_H200,
    ),
    OpCase(
        "cuda_fp8_registers_read_through_the_sm89_atom",
        nn.MatMul(out_dtype="f32"),
        (
            types.TensorType.from_shape((16, 32), expr.DType.fp8e4m3, "rmem"),
            types.TensorType.from_shape((32, 8), expr.DType.fp8e4m3, "rmem"),
        ),
        type=types.TensorType.from_shape((16, 8), expr.DType.f32, "rmem"),
        target=_SM89,
    ),
    OpCase(
        "cuda_on_chip_without_an_mma_is_refused",
        _MM,
        (
            types.TensorType.from_shape((64, 16), expr.DType.f32, "smem"),
            types.TensorType.from_shape((16, 32), expr.DType.f32, "smem"),
        ),
        error=ExpectedError(
            match="no nvidia.h200_sxm MMA reads lhs f32 smem and rhs f32 smem into f32"
        ),
        target=_H200,
    ),
    OpCase(
        "cuda_logical_operands_keep_the_neutral_rule",
        _MM,
        (types.TensorType.from_shape((64, 16), expr.DType.bf16), types.TensorType.from_shape((16, 32), expr.DType.bf16)),
        type=types.TensorType.from_shape((64, 32), expr.DType.bf16),
        target=_H200,
    ),
)
test_op_case = op_case_test(CASES)
