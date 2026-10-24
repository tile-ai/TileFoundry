"""MatMul's sharding invariants."""

from __future__ import annotations

import pytest

from tests.core.ops.ir.test_matmul import (
    _MM,
)
from tests.core.ops.ir.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    run_typeinfer_case,
)
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.types import (
    DType,
    make_tensor_type,
)
from tilefoundry_nvidia import CudaTarget

_H200 = CudaTarget("nvidia.h200_sxm")
_SM89 = CudaTarget("nvidia.rtx_6000_ada")


CASES = [
    TypeInferCase(
            name="cuda_on_chip_result_is_where_the_mma_writes_it",
            op=MatMul(out_dtype="f32"),
            inputs=(
                make_tensor_type((64, 16), DType.bf16, "smem"),
                make_tensor_type((16, 32), DType.bf16, "smem"),
            ),
            expected=make_tensor_type((64, 32), DType.f32, "rmem"),
            target=_H200,
        ),
    TypeInferCase(
            name="cuda_fp8_register_lhs_reads_like_rs",
            op=MatMul(out_dtype="f32"),
            inputs=(
                make_tensor_type((64, 32), DType.fp8e4m3, "rmem"),
                make_tensor_type((32, 32), DType.fp8e4m3, "smem"),
            ),
            expected=make_tensor_type((64, 32), DType.f32, "rmem"),
            target=_H200,
        ),
    TypeInferCase(
            name="cuda_fp8_registers_read_through_the_sm89_atom",
            op=MatMul(out_dtype="f32"),
            inputs=(
                make_tensor_type((16, 32), DType.fp8e4m3, "rmem"),
                make_tensor_type((32, 8), DType.fp8e4m3, "rmem"),
            ),
            expected=make_tensor_type((16, 8), DType.f32, "rmem"),
            target=_SM89,
        ),
    TypeInferCase(
            name="cuda_on_chip_without_an_mma_is_refused",
            op=_MM,
            inputs=(
                make_tensor_type((64, 16), DType.f32, "smem"),
                make_tensor_type((16, 32), DType.f32, "smem"),
            ),
            expected=ExpectedError(
                match="no nvidia.h200_sxm MMA reads lhs f32 smem and rhs f32 smem into f32"
            ),
            target=_H200,
        ),
    TypeInferCase(
            name="cuda_logical_operands_keep_the_neutral_rule",
            op=_MM,
            inputs=(make_tensor_type((64, 16), DType.bf16), make_tensor_type((16, 32), DType.bf16)),
            expected=make_tensor_type((64, 32), DType.bf16),
            target=_H200,
        ),
]

@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_matmul_typeinfer(case):
    run_typeinfer_case(case)
