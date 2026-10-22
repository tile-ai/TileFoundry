"""A declared stmatrix store cannot silently turn into a generic copy."""

import pytest

from tilefoundry.codegen.cuda.context import CudaCodegenContext
from tilefoundry.codegen.emitter import CppEmitter
from tilefoundry.ir.tir.cuda.memory.stmatrix import StMatrix
from tilefoundry.ir.tir.stmts import Evaluate
from tilefoundry.target import CudaTarget


def test_stmatrix_requires_runtime_support():
    context = CudaCodegenContext(target=CudaTarget("nvidia.h200_sxm"))
    with pytest.raises(RuntimeError, match="no emitter without stmatrix runtime support"):
        CppEmitter(context=context).visit_Evaluate(Evaluate(StMatrix(), ()))
