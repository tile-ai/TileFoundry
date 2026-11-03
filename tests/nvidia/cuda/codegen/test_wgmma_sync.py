"""WGMMA group steps carry their architecture requirement into the CUDA build."""

import shutil
from pathlib import Path

import pytest

from tilefoundry.cli.source import load_authored_ir
from tilefoundry.codegen.context import EmitContext
from tilefoundry.codegen.context_builder import build_codegen_context
from tilefoundry.codegen.emitter import CppEmitter
from tilefoundry.codegen.link_module import link_program
from tilefoundry.codegen.linkable_module import LinkableModule
from tilefoundry.compile import default_pipeline
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


@pytest.fixture
def mixed_units():
    path = Path(__file__).parents[1] / "fixtures/codegen/wgmma_units.py"
    module = load_authored_ir(f"{path}:MixedUnits")
    return default_pipeline(module).run(module)


def test_each_unit_keeps_its_architecture(mixed_units):
    codegen = build_codegen_context(mixed_units)
    units = []
    for group in codegen.groups:
        if isinstance(group.target, CudaTarget):
            context = group.target.resolve(EmitContext, codegen.for_group(group))
            units.append(group.target.resolve(LinkableModule, group.owner, group.functions, context))
            assert group.target.arch == "sm_90"
    assert [unit.arch for unit in units] == ["sm_90a", "sm_90"]
    assert "wgmma.wait_group" in units[0].source
    assert "wgmma." not in units[1].source


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="requires the CUDA compiler")
def test_mixed_units_build(mixed_units, tmp_path):
    linked = link_program(mixed_units, workdir=tmp_path)
    assert linked.library_path.is_file()
    commands = (tmp_path / "build/build.ninja").read_text()
    assert "--generate-code=arch=compute_90a,code=[compute_90a,sm_90a]" in commands
    assert "--generate-code=arch=compute_90,code=[compute_90,sm_90]" in commands
