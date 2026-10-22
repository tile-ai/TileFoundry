"""Declared tensor transfers fail explicitly without their runtime support."""

import pytest

from tilefoundry.codegen.cuda.context import CudaCodegenContext
from tilefoundry.codegen.emitter import CppEmitter
from tilefoundry.ir.tir.cuda.memory.copy_reduce_async_tensor import CopyReduceAsyncTensor
from tilefoundry.ir.tir.cuda.memory.stmatrix import StMatrix
from tilefoundry.ir.tir.stmts import Evaluate
from tilefoundry.target import CudaTarget


@pytest.mark.parametrize(
    "op, reason",
    (
        (StMatrix(), "stmatrix runtime support"),
        (CopyReduceAsyncTensor(), "host-encoded tensor maps"),
    ),
)
def test_declared_transfer_requires_runtime_support(op, reason):
    context = CudaCodegenContext(target=CudaTarget("nvidia.h200_sxm"))
    with pytest.raises(RuntimeError, match="no emitter without " + reason):
        CppEmitter(context=context).visit_Evaluate(Evaluate(op, ()))
