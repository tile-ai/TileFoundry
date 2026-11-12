"""Native communication emission uses the same semantic and completion contracts."""

import pytest

from tilefoundry import cli, codegen, inspection, verifier
from tilefoundry.ir import core, expr, types
from tilefoundry.ir.hir import sharding
from tilefoundry.ir.tir import communication as C
from tilefoundry_nvidia import cuda


def module(synchronize=True):
    mesh = types.Mesh((types.Topology("gpu", 8),), types.Layout((8,)), ("rank",))
    ty = types.TensorType.from_shape((8,), storage=expr.StorageKind.GMEM)
    a, b = (expr.Var(name=name, type=ty) for name in ("a", "b"))
    comm = expr.Var(name="comm", type=types.CommunicationHandleType("communicator", "data", mesh))
    stream = expr.Var(name="stream", type=types.CommunicationHandleType("stream", "stream"))
    op = sharding.Collective(participants=mesh, context="data", kind="all_reduce", reduction="sum")
    statements = [core.Evaluate(C.CommEnqueue(operation=op), (a, b, comm, stream))]
    if synchronize:
        statements.append(core.Evaluate(C.StreamSynchronize(), (stream,)))
    fn = core.PrimFunction("forward", (a, b, comm, stream), core.Sequential(tuple(statements)), output_count=0)
    return core.Module(name="native", functions=(fn,), entry="forward", target=cuda.Target("nvidia.h200_sxm", device_count=8),
                       topologies=(types.Topology("gpu", 8),))


def source(mod):
    root = codegen.build_context(mod)
    group = next(g for g in root.groups if g.functions)
    ctx = group.target.resolve(codegen.EmitContext, root.for_group(group))
    return group.target.resolve(codegen.LinkableModule, mod, group.functions, ctx).source


def test_communication_only_module_emits_native_host_calls():
    emitted = source(module())
    assert "runtime/cuda/nccl.h" in emitted
    assert "ncclSum" in emitted and "ncclFloat32" in emitted
    assert "cudaStreamSynchronize" in emitted
    assert "long long comm" in emitted
    assert "tilefoundry_gpu_program_id == 7" in emitted


def test_unsafe_asynchronous_return_is_rejected_before_codegen():
    with pytest.raises(ValueError, match="pending communication"):
        source(module(synchronize=False))


def test_host_communication_source_roundtrip_and_dynamic_exchange(tmp_path):
    mod = module()
    fn = mod.entry_function()
    a, b, comm, stream = fn.params
    event = expr.Var(name="ready", type=types.CommunicationHandleType("event", "ready"))
    consumer = expr.Var(name="consumer", type=types.CommunicationHandleType("stream", "consumer"))
    counts = expr.Var(name="counts", type=types.TensorType.from_shape((8, 8), dtype=expr.DType.i64, storage=expr.StorageKind.HOST))
    fn.params += (event, consumer, counts)
    exchange = sharding.Exchange(participants=comm.type.participants, context="data", receive_capacity=8)
    fn.body = core.Sequential((
        core.Evaluate(C.CommGroupStart(), ()), fn.body.body[0], core.Evaluate(C.CommGroupEnd(), ()),
        core.Evaluate(C.EventRecord(), (event, stream)), core.Evaluate(C.StreamWaitEvent(), (consumer, event)),
        core.Evaluate(C.CommEnqueue(operation=exchange), (b, a, comm, consumer, counts)),
        core.Evaluate(C.EventRecord(), (event, consumer)), core.Evaluate(C.EventSynchronize(), (event,)),
        core.Evaluate(C.ExportEvent(), (event,)),
    ))
    printed = inspection.tir_module_to_python(mod)
    path = tmp_path / "protocol.py"
    path.write_text(printed)
    restored = cli.load_authored_ir(f"{path}:native")
    assert inspection.tir_module_to_python(restored) == printed
    assert "cudaStreamWaitEvent" in source(restored)


def test_communication_statement_cannot_be_authored_inside_a_kernel():
    mod = module()
    fn = mod.entry_function()
    fn.target = mod.target
    with pytest.raises(expr.VerifyError, match="host scope"):
        verifier.verify_prim_function(fn, module_fns=mod)
