"""Declared tensor transfers fail explicitly without their runtime support."""

import pytest

from tilefoundry import codegen
from tilefoundry.ir import core
from tilefoundry_nvidia import cuda
from tilefoundry_nvidia.cuda import codegen as cuda_codegen
from tilefoundry_nvidia.cuda.ir.tir import memory


@pytest.mark.parametrize(
    "op, reason",
    (
        (memory.StMatrix(), "stmatrix runtime support"),
        (memory.CopyReduceAsyncTensor(), "host-encoded tensor maps"),
    ),
)
def test_declared_transfer_requires_runtime_support(op, reason):
    context = cuda_codegen.Context(target=cuda.Target("nvidia.h200_sxm"))
    with pytest.raises(RuntimeError, match="no emitter without " + reason):
        codegen.CppEmitter(context=context).visit_Evaluate(core.Evaluate(op, ()))
