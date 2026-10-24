"""CUDA MatMul type inference across storage and architecture cases."""

from __future__ import annotations

from tests.core.ops.ir.op_case import ExpectedError, OpCase, op_case_test
from tests.core.ops.ir.test_matmul import _MM
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.types import DType, make_tensor_type
from tilefoundry_nvidia import CudaTarget

_H200 = CudaTarget("nvidia.h200_sxm")
_SM89 = CudaTarget("nvidia.rtx_6000_ada")
CASES = (
    OpCase(
        "cuda_on_chip_result_is_where_the_mma_writes_it",
        MatMul(out_dtype="f32"),
        (
            make_tensor_type((64, 16), DType.bf16, "smem"),
            make_tensor_type((16, 32), DType.bf16, "smem"),
        ),
        type=make_tensor_type((64, 32), DType.f32, "rmem"),
        target=_H200,
    ),
    OpCase(
        "cuda_fp8_register_lhs_reads_like_rs",
        MatMul(out_dtype="f32"),
        (
            make_tensor_type((64, 32), DType.fp8e4m3, "rmem"),
            make_tensor_type((32, 32), DType.fp8e4m3, "smem"),
        ),
        type=make_tensor_type((64, 32), DType.f32, "rmem"),
        target=_H200,
    ),
    OpCase(
        "cuda_fp8_registers_read_through_the_sm89_atom",
        MatMul(out_dtype="f32"),
        (
            make_tensor_type((16, 32), DType.fp8e4m3, "rmem"),
            make_tensor_type((32, 8), DType.fp8e4m3, "rmem"),
        ),
        type=make_tensor_type((16, 8), DType.f32, "rmem"),
        target=_SM89,
    ),
    OpCase(
        "cuda_on_chip_without_an_mma_is_refused",
        _MM,
        (
            make_tensor_type((64, 16), DType.f32, "smem"),
            make_tensor_type((16, 32), DType.f32, "smem"),
        ),
        error=ExpectedError(
            match="no nvidia.h200_sxm MMA reads lhs f32 smem and rhs f32 smem into f32"
        ),
        target=_H200,
    ),
    OpCase(
        "cuda_logical_operands_keep_the_neutral_rule",
        _MM,
        (make_tensor_type((64, 16), DType.bf16), make_tensor_type((16, 32), DType.bf16)),
        type=make_tensor_type((64, 32), DType.bf16),
        target=_H200,
    ),
)
test_op_case = op_case_test(CASES)
