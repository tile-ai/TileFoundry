"""Mixed execution uses one completion relation for checking and placement."""

from dataclasses import replace

import pytest

from tilefoundry import analysis, jit, target
from tilefoundry.ir import core, expr, tir, types
from tilefoundry.ir.hir import sharding
from tilefoundry.ir.tir import communication as C
from tilefoundry.ir.tir import memory


class Device(target.Target):
    name = "execution_test"


def model(wait=True, reuse=False):
    ty = types.TensorType.from_shape((4,), storage=expr.StorageKind.GMEM)
    x, a, b, c, y = (expr.Var(name=name, type=ty) for name in ("x", "a", "b", "c", "y"))
    source, destination = (expr.Var(name=name, type=ty) for name in ("source", "destination"))
    kernel = core.PrimFunction("copy", (source, destination),
                              core.Sequential((core.Evaluate(memory.Copy(), (source, destination)),)), target=Device())
    compute, transfer, ready, done = (
        expr.Var(name=name, type=types.CommunicationHandleType(kind, name))
        for name, kind in (("compute", "stream"), ("transfer", "stream"), ("ready", "event"), ("done", "event"))
    )
    mesh = types.Mesh((types.Topology("gpu", 2),), types.Layout((2,)), ("rank",))
    communicator = expr.Var(name="comm", type=types.CommunicationHandleType("communicator", "data", mesh))
    collective = sharding.Collective(participants=mesh, context="data", kind="all_reduce", reduction="sum")
    def launch(src, dst):
        return tir.launch.launch_call(kernel, (src, dst), (1, 1, 1), (1, 1, 1), stream=compute)
    statements = [launch(x, a), core.Evaluate(C.EventRecord(), (ready, compute))]
    if wait:
        statements.append(core.Evaluate(C.StreamWaitEvent(), (transfer, ready)))
    statements += [core.Evaluate(C.CommEnqueue(operation=collective), (a, b, communicator, transfer)),
                   core.Evaluate(C.EventRecord(), (done, transfer))]
    if reuse:
        statements.append(launch(x, c))
    statements += [core.Evaluate(C.StreamWaitEvent(), (compute, done)), launch(b, y)]
    if not reuse:
        statements.append(launch(x, c))
    statements.append(core.Evaluate(C.StreamSynchronize(), (compute,)))
    host = core.PrimFunction("host", (x, a, b, c, y, compute, transfer, ready, done, communicator),
                            core.Sequential(tuple(statements)), output_count=0)
    return core.Module(name="mixed", functions=(kernel, host), entry="host", target=Device())


def test_mixed_kernel_dependencies_and_source_retention():
    mod = model()
    report = analysis.check_communication(mod.entry_function(), module=mod)
    assert not report.hazards
    assert len(report.operations) == 4
    bad = model(wait=False)
    assert any(h.kind == "read_before_ready" for h in analysis.check_communication(bad.entry_function(), module=bad).hazards)
    regions = {"a": analysis.BufferRegion("slot"), "c": analysis.BufferRegion("slot")}
    assert not analysis.check_communication(mod.entry_function(), regions, module=mod).hazards
    overlapping = model(reuse=True)
    assert any(h.kind == "overwrite_live_value" for h in analysis.check_communication(
        overlapping.entry_function(), regions, module=overlapping).hazards)


def test_allocation_uses_completion_order_and_enforces_capacity():
    serial, overlapping = model(), model(reuse=True)
    compact = analysis.plan_workspace(serial.entry_function(), ("a", "b", "c"), module=serial, capacity_bytes=32)
    assert compact.peak_bytes == 32
    plan = analysis.plan_workspace(overlapping.entry_function(), ("a", "b", "c"), module=overlapping)
    assert ("a", "c") in plan.conflicts and ("b", "c") in plan.conflicts
    assert plan.peak_bytes == 48
    with pytest.raises(analysis.Error, match="infeasible"):
        analysis.plan_workspace(overlapping.entry_function(), ("a", "b", "c"), module=overlapping, capacity_bytes=32)
    selected = replace(serial.entry_function(), metadata=(compact.bindings,))
    assert not analysis.check_communication(selected, module=serial).hazards


def test_workspace_never_reclaims_initialized_caller_inputs():
    mod = model()
    with pytest.raises(ValueError, match="initial input"):
        analysis.plan_workspace(mod.entry_function(), ("x",), module=mod)


def test_aliases_within_an_operation_are_not_inferred_safe():
    mod = model()
    regions = {"a": analysis.BufferRegion("slot"), "b": analysis.BufferRegion("slot")}
    assert any(h.kind == "overlapping_operands" for h in analysis.check_communication(
        mod.entry_function(), regions, module=mod).hazards)


def test_unknown_effects_and_implicit_streams_are_rejected():
    mod = model()
    fn = mod.entry_function()
    with pytest.raises(ValueError, match="owning module"):
        analysis.check_communication(fn)
    first = fn.body.body[0]
    implicit = core.Evaluate(tir.launch.Launch(), first.args)
    with pytest.raises(ValueError, match="explicit declared launch stream"):
        analysis.check_communication(replace(fn, body=core.Sequential((implicit, *fn.body.body[1:]))), module=mod)
    unknown = core.PrimFunction("unknown", (), core.Sequential((core.Evaluate(expr.Op(), ()),)))
    with pytest.raises(ValueError, match="unmodeled kernel effect"):
        analysis.kernel_effects.summarize(unknown)


def test_partial_parameter_views_cannot_be_treated_as_full_definitions():
    mod = model()
    kernel = mod.functions[0]
    source, destination = kernel.params
    view_type = types.TensorType.from_shape((2,), storage=expr.StorageKind.GMEM)
    source = expr.Var(name=source.name, type=view_type)
    pointer = expr.Call(memory.PtrOf(), (destination,), type=types.PointerType(destination.type.dtype, destination.type.storage))
    view = expr.Call(memory.TensorView(layout=view_type.layout, shape=view_type.shape), (pointer,), type=view_type)
    partial = replace(kernel, params=(source, destination), body=core.Sequential((core.Evaluate(memory.Copy(), (source, view)),)))
    with pytest.raises(ValueError, match="full compact parameter views"):
        analysis.kernel_effects.summarize(partial)


def test_concurrent_readers_and_exported_kernel_completion():
    mod = model(reuse=True)
    fn = mod.entry_function()
    report = analysis.check_communication(fn, module=mod)
    assert not report.hazards
    producer, transfer, independent, consumer = report.operations
    assert transfer.complete not in report.predecessors[independent.read]
    assert independent.complete not in report.predecessors[transfer.read]
    event, stream = fn.params[7], fn.params[5]
    exported = replace(fn, body=core.Sequential((*fn.body.body[:-1],
                       core.Evaluate(C.EventRecord(), (event, stream)), core.Evaluate(C.ExportEvent(), (event,)))))
    assert analysis.check_communication(exported, module=mod).exported == (event.name,)


def test_replayed_mixed_regions_have_distinct_occurrences_and_event_generations():
    mod = model()
    fn = mod.entry_function()
    scalar = types.TensorType.from_scalar(expr.DType.i32)
    bounds = tuple(expr.Constant(type=scalar, value=value) for value in (0, 2, 1))
    loop = core.For(expr.Var(name="replay", type=scalar), *bounds, fn.body)
    repeated = replace(fn, body=core.Sequential((loop,)))
    report = analysis.check_communication(repeated, module=mod)
    assert not report.hazards
    assert len({operation.name for operation in report.operations}) == 8
    first, replay = report.operations[0], report.operations[4]
    assert first.complete in report.predecessors[replay.read]
    assert analysis.plan_workspace(repeated, ("a", "b", "c"), module=mod, capacity_bytes=32).peak_bytes == 32
    event, stream = fn.params[7], fn.params[5]
    stale = replace(fn, body=core.Sequential((
        core.Evaluate(C.EventRecord(), (event, stream)), core.Evaluate(C.ExportEvent(), (event,)), loop,
    )))
    with pytest.raises(ValueError, match="recorded again"):
        analysis.check_communication(stale, module=mod)


def test_jit_does_not_reuse_an_artifact_with_different_physical_bindings(monkeypatch):
    mod = model()
    fn = mod.entry_function()
    plan = analysis.plan_workspace(fn, ("a", "b", "c"), module=mod)
    mod.replace_functions((mod.functions[0], replace(fn, metadata=(plan.bindings,))))
    monkeypatch.setattr("tilefoundry.compiler.jit_cache.compile", lambda *args, **kwargs: object())
    monkeypatch.setattr(jit, "cache", {})
    first = jit(mod)
    assert jit(mod) is first
    changed = analysis.BufferBindings(tuple((name, replace(region, offset=region.offset * 2))
                                            for name, region in plan.bindings.regions))
    mod.replace_functions((mod.functions[0], replace(fn, metadata=(changed,))))
    assert jit(mod) is not first
