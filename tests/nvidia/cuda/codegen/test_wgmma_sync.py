"""The WGMMA group and async proxy fence Ops each emit their one PTX instruction."""

import pytest

from tilefoundry.codegen.emitter import CppEmitter
from tilefoundry.ir.tir.stmts import Evaluate
from tilefoundry_nvidia import CudaTarget
from tilefoundry_nvidia.cuda.codegen.context import CudaCodegenContext
from tilefoundry_nvidia.cuda.ir.tir.sync.wgmma import (
    FenceProxyAsync,
    WgmmaCommitGroup,
    WgmmaFence,
    WgmmaWaitGroup,
)


@pytest.mark.parametrize(
    "op, instruction",
    (
        (WgmmaFence(), "wgmma.fence.sync.aligned;"),
        (WgmmaCommitGroup(), "wgmma.commit_group.sync.aligned;"),
        (WgmmaWaitGroup(n=1), 'wgmma.wait_group.sync.aligned %0;\\n" :: "n"(1)'),
        (FenceProxyAsync(), "fence.proxy.async.shared::cta;"),
    ),
)
def test_emits_its_instruction(op, instruction):
    context = CudaCodegenContext(target=CudaTarget("nvidia.h200_sxm"))
    CppEmitter(context=context).visit_Evaluate(Evaluate(op, ()))
    assert instruction in context.source()
