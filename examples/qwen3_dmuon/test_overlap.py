"""Equivalent gradient programs expose fusion, overlap and memory tradeoffs."""

import gzip
import json
from dataclasses import replace

import pytest
from benchmark_overlap import profile_summary, report
from overlap import MODES, program

from tilefoundry import analysis, codegen
from tilefoundry.ir import core
from tilefoundry.ir.tir import communication as C
from tilefoundry_nvidia import cuda


def test_fusion_removes_a_memory_pass_and_reduces_pipeline_workspace():
    target = cuda.Target("nvidia.h200_sxm", device_count=2)
    programs = {mode: program(target, mode=mode) for mode in MODES}
    facts = {mode: report(*selected) for mode, selected in programs.items()}
    assert [facts[mode]["workspace_bytes"] for mode in MODES] == [32768, 49152, 16384, 32768]
    assert [facts[mode]["local_kernel_bytes"] for mode in MODES] == [327680, 327680, 196608, 196608]
    assert [facts[mode]["local_kernel_launches"] for mode in MODES] == [8, 8, 4, 4]
    assert all(facts[mode]["collective_calls"] == 4 for mode in MODES)
    assert not facts["serial"]["overlap_candidates"]
    assert facts["pipelined"]["overlap_candidates"] and facts["fused_pipelined"]["overlap_candidates"]
    assert len(facts["pipelined"]["fusion_boundaries"]) == 4
    with pytest.raises(analysis.Error, match="infeasible"):
        program(target, mode="pipelined", capacity_bytes=32768)
    assert program(target, mode="fused_pipelined", capacity_bytes=32768)[1].peak_bytes == 32768


def test_missing_producer_wait_is_rejected_in_the_actual_candidate():
    mod, _ = program(cuda.Target("nvidia.h200_sxm"), mode="fused_pipelined")
    host = mod.entry_function()
    body = tuple(stmt for stmt in host.body.body if not (
        isinstance(stmt.callable, C.StreamWaitEvent) and stmt.args[0].name == "transfer"))
    bad = replace(host, body=core.Sequential(body))
    assert any(hazard.kind == "read_before_ready" for hazard in analysis.check_communication(bad, module=mod).hazards)


def test_codegen_realizes_fusion_as_the_existing_average_collective():
    mod, _ = program(cuda.Target("nvidia.h200_sxm"), mode="fused_pipelined")
    root = codegen.build_context(mod, codegen.Options(communication=cuda.codegen.NCCL))
    sources = []
    for group in root.groups:
        ctx = group.target.resolve(codegen.EmitContext, root.for_group(group))
        sources.append(group.target.resolve(codegen.LinkableModule, group.owner, group.functions, ctx).source)
    emitted = "\n".join(sources)
    assert "ncclAvg" in emitted
    assert "allocation binding mismatch" in emitted
    assert "cudaStreamWaitEvent" in emitted


def test_profile_uses_device_intersections_not_host_range_overlap(tmp_path):
    trace = tmp_path / "profile.json.gz"
    events = [dict(cat="user_annotation", name="bucket/serial/0", ts=0, dur=100),
              dict(cat="kernel", name="accumulate", ts=10, dur=20),
              dict(cat="kernel", name="nccl_AllReduce", ts=25, dur=20),
              dict(cat="kernel", name="normalize", ts=50, dur=10)]
    with gzip.open(trace, "wt") as output:
        json.dump(dict(traceEvents=events), output)
    result = profile_summary(trace, ("serial",), 1)["serial"]
    assert result["overlap_us"] == 5
    assert result["compute_us"] == 30
    assert result["nccl_overlap_fraction"] == 0.25
    with pytest.raises(ValueError, match="every synchronized bucket step"):
        profile_summary(trace, ("serial",), 2)
