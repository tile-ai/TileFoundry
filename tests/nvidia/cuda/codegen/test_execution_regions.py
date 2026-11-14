"""Generated kernels preserve disjoint byte regions in a shared workspace."""

import shutil
from dataclasses import replace

import pytest

from tilefoundry import analysis, codegen, runtime
from tilefoundry.dsl import B, Layout, Mesh, ShardLayout, T, Tensor, Topology, module, prim_func
from tilefoundry.ir import core, expr, tir, types
from tilefoundry.ir.tir import communication as C
from tilefoundry_nvidia import cuda


def program(target):
    @module(target=target)
    class Compute:
        @prim_func(target=target)
        def low(source: Tensor[(4,), "f32"], destination: Tensor[(8,), "f32"]):
            with Mesh((Topology("thread", 1),), Layout((1,))) as thread:
                src = T.tensor_view(T.ptr_of(source), shape=(4,), layout=ShardLayout(Layout((4,)), (B(),), thread))
                dst = T.tensor_view(T.ptr_of(destination), shape=(4,), layout=ShardLayout(Layout((4,)), (B(),), thread))
                T.copy(src, dst)

        @prim_func(target=target)
        def high(source: Tensor[(4,), "f32"], destination: Tensor[(8,), "f32"]):
            with Mesh((Topology("thread", 1),), Layout((1,))) as thread:
                src = T.tensor_view(T.ptr_of(source), shape=(4,), layout=ShardLayout(Layout((4,)), (B(),), thread))
                dst = T.tensor_view(T.ptr_of(destination[4:8]), shape=(4,),
                                    layout=ShardLayout(Layout((4,)), (B(),), thread))
                T.copy(src, dst)
    low, high = Compute.functions
    x, y = (expr.Var(name=name, type=low.params[0].type) for name in ("x", "y"))
    a, b = (expr.Var(name=name, type=low.params[1].type) for name in ("a", "b"))
    s0, s1 = (expr.Var(name=name, type=types.CommunicationHandleType("stream", name)) for name in ("s0", "s1"))
    body = core.Sequential((tir.launch.launch_call(low, (x, a), (1, 1, 1), (1, 1, 1), stream=s0),
                            tir.launch.launch_call(high, (y, b), (1, 1, 1), (1, 1, 1), stream=s1),
                            core.Evaluate(C.StreamSynchronize(), (s0,)), core.Evaluate(C.StreamSynchronize(), (s1,))))
    host = core.PrimFunction("host", (x, y, a, b, s0, s1), body, output_count=0)
    mod = core.Module(name="SharedRegions", functions=(low, high, host), entry="host", target=target)
    plan = analysis.plan_workspace(host, ("a", "b"), module=mod, capacity_bytes=32)
    mod.replace_functions((low, high, replace(host, metadata=(plan.bindings,))))
    return mod, plan


def test_partial_workspace_reaches_codegen_with_explicit_streams():
    mod, plan = program(cuda.Target("nvidia.h200_sxm"))
    assert plan.peak_bytes == 32
    report = analysis.check_communication(mod.entry_function(), module=mod)
    assert analysis.overlap_candidates(report) == (("kernel.0", "kernel.1"),)
    root = codegen.build_context(mod, codegen.Options(communication=cuda.codegen.NCCL))
    for group in root.groups:
        ctx = group.target.resolve(codegen.EmitContext, root.for_group(group))
        group.target.resolve(codegen.LinkableModule, group.owner, group.functions, ctx)


def test_native_shared_workspace_retains_both_concurrent_slices(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("tvm_ffi")
    if not torch.cuda.is_available() or shutil.which("nvcc") is None:
        pytest.skip("CUDA execution and nvcc are required")
    base = cuda.Target("nvidia.h200_sxm")
    major, minor = torch.cuda.get_device_capability()
    arch = replace(base.architecture, name=f"sm_{major}{minor}", gpu_architecture=f"compute_{major}{minor}",
                   gpu_code=(f"sm_{major}{minor}",))
    mod, plan = program(cuda.Target(base.device, arch))
    linked = codegen.link_program(mod, workdir=tmp_path, options=codegen.Options(communication=cuda.codegen.NCCL))
    fn = runtime.load_linked_module(linked)
    arena = torch.empty(plan.peak_bytes // 4, device="cuda")
    compute, transfer = torch.cuda.Stream(), torch.cuda.Stream()
    for step in range(3):
        x = torch.arange(4, dtype=torch.float32, device="cuda") + step
        y = x + 10
        compute.wait_stream(torch.cuda.current_stream())
        transfer.wait_stream(torch.cuda.current_stream())
        fn(x, y, arena, arena, compute.cuda_stream, transfer.cuda_stream)
        torch.testing.assert_close(arena.cpu(), torch.cat((x, y)).cpu(), rtol=0, atol=0)
