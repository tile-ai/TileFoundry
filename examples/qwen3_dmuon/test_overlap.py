"""The example compares execution choices derived from checked IR."""

from dataclasses import replace

import pytest

from tilefoundry import analysis, codegen
from tilefoundry.ir import core, expr, types
from tilefoundry_nvidia import cuda

from overlap import program, report


def test_chunk_overlap_trades_workspace_for_an_independent_execution_window():
    target = cuda.Target("nvidia.h200_sxm", device_count=2)
    serial, compact = program(target)
    pipelined, larger = program(target, pipelined=True)
    a = report(serial, compact, kernel_us=10, communication_us=20)
    b = report(pipelined, larger, kernel_us=10, communication_us=20)
    assert not a["overlap_candidates"] and len(b["overlap_candidates"]) == 2
    assert compact.peak_bytes == 256 and larger.peak_bytes == 384
    assert a["estimated_us"] == pytest.approx(80) and b["estimated_us"] == pytest.approx(60)
    assert "estimated_us" not in report(serial, compact)
    with pytest.raises(analysis.Error, match="infeasible"):
        program(target, pipelined=True, capacity_bytes=256)
    assert program(target, capacity_bytes=256)[1].peak_bytes == 256


def test_codegen_keeps_streams_and_checks_the_selected_workspace():
    target = cuda.Target("nvidia.h200_sxm", device_count=2)
    mod, _ = program(target, pipelined=True)
    host = mod.entry_function()
    scalar = types.TensorType.from_scalar(expr.DType.i32)
    bounds = tuple(expr.Constant(type=scalar, value=value) for value in (0, 2, 1))
    loop = core.For(expr.Var(name="step", type=scalar), *bounds, host.body)
    mod.replace_functions((mod.functions[0], replace(host, body=core.Sequential((loop,)))))
    root = codegen.build_context(mod, codegen.Options(communication=cuda.codegen.NCCL))
    sources = []
    for group in root.groups:
        ctx = group.target.resolve(codegen.EmitContext, root.for_group(group))
        sources.append(group.target.resolve(codegen.LinkableModule, group.owner, group.functions, ctx).source)
    emitted = "\n".join(sources)
    assert "reinterpret_cast<cudaStream_t>(stream)" in emitted
    assert "allocation binding mismatch" in emitted
    assert "cudaStreamWaitEvent" in emitted
