"""Region proofs preserve logical values while permitting independent storage use."""

from dataclasses import replace

import isl
import pytest

from tests.core.analysis.test_execution import Device, model
from tilefoundry import analysis
from tilefoundry.interfaces import access
from tilefoundry.ir import core, dim, expr, tir, types
from tilefoundry.ir.hir import sharding, tensor
from tilefoundry.ir.tir import communication as C
from tilefoundry.ir.tir import memory


def view(parameter, count, offset=0, stride=1):
    layout = types.ComposedLayout(None, offset, types.Layout((count,), (stride,)))
    held = types.TensorType.from_shape((count,), storage=parameter.type.storage, layout=layout)
    pointer = expr.Call(memory.PtrOf(), (parameter,), type=types.PointerType(held.dtype, held.storage))
    return expr.Call(memory.TensorView(layout=layout, shape=held.shape), (pointer,), type=held)


def partial_kernel(offset=0, stride=1, size=8, count=4):
    source = expr.Var(name="source", type=types.TensorType.from_shape((count,)))
    destination = expr.Var(name="destination", type=types.TensorType.from_shape((size,)))
    output = view(destination, count, offset, stride)
    return core.PrimFunction("partial", (source, destination),
                             core.Sequential((core.Evaluate(memory.Copy(), (source, output)),)), target=Device())


def overlap_model(offset=4):
    mod = model()
    fn = mod.entry_function()
    x, a, b, c, y, compute, transfer, ready, done, communicator = fn.params
    large = expr.Var(name="large", type=types.TensorType.from_shape((8,)))
    kernel = partial_kernel(offset)
    enqueue = next(stmt for stmt in fn.body.body if isinstance(stmt.callable, C.CommEnqueue))
    body = core.Sequential((enqueue, tir.launch.launch_call(kernel, (x, large), (1, 1, 1), (1, 1, 1), stream=compute),
                            core.Evaluate(C.StreamSynchronize(), (compute,)), core.Evaluate(C.StreamSynchronize(), (transfer,))))
    host = replace(fn, params=(*fn.params, large), body=body)
    mod.replace_functions((kernel, host))
    return mod


def test_disjoint_kernel_slice_can_overlap_an_inflight_transfer():
    aliases = {"a": analysis.BufferRegion("arena"), "large": analysis.BufferRegion("arena")}
    mod = overlap_model()
    report = analysis.check_communication(mod.entry_function(), aliases, module=mod, uninitialized=("large",))
    assert not report.hazards
    assert analysis.overlap_candidates(report) == (("comm.0", "kernel.1"),)
    overlapping = overlap_model(0)
    bad = analysis.check_communication(overlapping.entry_function(), aliases, module=overlapping)
    assert any(h.kind == "overwrite_live_value" for h in bad.hazards)
    with pytest.raises(ValueError, match="buffer hazards"):
        analysis.overlap_candidates(bad)


def test_aliasing_never_changes_which_logical_version_a_later_read_uses():
    mod = model()
    fn, kernel = mod.entry_function(), mod.functions[0]
    x, a, _, c, y, compute, *_ = fn.params
    def launch(src, dst):
        return tir.launch.launch_call(kernel, (src, dst), (1, 1, 1), (1, 1, 1), stream=compute)
    host = replace(fn, body=core.Sequential((launch(x, a), launch(x, c), launch(a, y),
                                            core.Evaluate(C.StreamSynchronize(), (compute,)))))
    assert not analysis.check_communication(host, module=mod).hazards
    report = analysis.check_communication(host, {"a": analysis.BufferRegion("slot"), "c": analysis.BufferRegion("slot")}, module=mod)
    assert any(h.kind == "overwrite_live_value" for h in report.hazards)


def test_declared_outputs_remain_live_until_the_execution_boundary():
    mod = model()
    fn, kernel = mod.entry_function(), mod.functions[0]
    x, a, _, c, _, compute, *_ = fn.params
    body = core.Sequential((
        tir.launch.launch_call(kernel, (x, a), (1, 1, 1), (1, 1, 1), stream=compute),
        tir.launch.launch_call(kernel, (x, c), (1, 1, 1), (1, 1, 1), stream=compute),
        core.Evaluate(C.StreamSynchronize(), (compute,)),
    ))
    host = replace(fn, params=(x, compute, a, c), body=body, output_count=2)
    assert not analysis.check_communication(host, module=mod, uninitialized=("a", "c")).hazards
    aliases = {"a": analysis.BufferRegion("slot"), "c": analysis.BufferRegion("slot")}
    report = analysis.check_communication(host, aliases, module=mod, uninitialized=("a", "c"))
    assert any(h.kind == "overwrite_live_value" for h in report.hazards)


def test_partial_writes_do_not_initialize_the_rest_of_a_scratch_buffer():
    mod = model()
    fn = mod.entry_function()
    x, a, b, *_ = fn.params
    small = expr.Var(name="small", type=types.TensorType.from_shape((2,)))
    kernel = partial_kernel(size=4, count=2)
    launch = tir.launch.launch_call(kernel, (small, a), (1, 1, 1), (1, 1, 1), stream=fn.params[5])
    host = replace(fn, params=(*fn.params, small), body=core.Sequential((launch, *fn.body.body[1:])))
    mod.replace_functions((mod.functions[0], kernel, host))
    report = analysis.check_communication(host, module=mod, uninitialized=("a",))
    assert any(h.kind == "uninitialized_read" for h in report.hazards)
    selected = replace(host, metadata=(analysis.BufferBindings((), ("a",)),))
    assert any(h.kind == "uninitialized_read" for h in analysis.check_communication(selected, module=mod).hazards)
    with pytest.raises(ValueError, match="initial input"):
        analysis.plan_workspace(host, ("a",), module=mod)


@pytest.mark.parametrize("offset,stride", ((0, 2), (6, -2)))
def test_strided_regions_keep_holes_and_out_of_bounds_are_rejected(offset, stride):
    even = analysis.kernel_effects.summarize(partial_kernel(offset, stride)).effects[-1].region.relation.range()
    odd = analysis.kernel_effects.summarize(partial_kernel(offset + 1, stride)).effects[-1].region.relation.range()
    assert even.intersect(odd).is_empty()
    assert even.union(odd).is_equal(isl.set("{ [b] : 0 <= b < 32 }"))
    producer = analysis.BufferAccess("write", 0, 1)
    a = analysis.BufferLifetime("even", "arena", 0, 32, producer, region=even)
    b = analysis.BufferLifetime("odd", "arena", 0, 32, analysis.BufferAccess("other", 0, 1), region=odd)
    assert not analysis.check_buffer_lifetimes((a, b))
    with pytest.raises(ValueError, match="exceeds parameter"):
        analysis.kernel_effects.summarize(partial_kernel(offset + 2, stride))


def test_existing_allocator_can_overlay_live_buffers_with_disjoint_regions():
    mod = model()
    fn = mod.entry_function()
    x, a, b, c, y, compute, transfer, *_ = fn.params
    a, b = (replace(param, type=types.TensorType.from_shape((8,))) for param in (a, b))
    first = partial_kernel(0)
    second = replace(partial_kernel(4), name="second")
    body = core.Sequential((
        tir.launch.launch_call(first, (x, a), (1, 1, 1), (1, 1, 1), stream=compute),
        tir.launch.launch_call(second, (x, b), (1, 1, 1), (1, 1, 1), stream=transfer),
        core.Evaluate(C.StreamSynchronize(), (compute,)), core.Evaluate(C.StreamSynchronize(), (transfer,)),
    ))
    host = replace(fn, params=(x, a, b, *fn.params[3:]), body=body)
    mod.replace_functions((first, second, host))
    plan = analysis.plan_workspace(host, ("a", "b"), module=mod, capacity_bytes=32)
    assert plan.peak_bytes == 32
    assert {region.offset for _, region in plan.bindings.regions} == {0}
    assert plan.constraint_precision is analysis.Precision.EXACT
    mod.replace_functions((partial_kernel(0, 2), replace(partial_kernel(1, 2), name="second"), host))
    sparse = analysis.plan_workspace(host, ("a", "b"), module=mod, capacity_bytes=64)
    assert sparse.constraint_precision is analysis.Precision.UPPER_BOUND
    with pytest.raises(analysis.Error, match="conservative workspace constraints"):
        analysis.plan_workspace(host, ("a", "b"), module=mod, capacity_bytes=32)


def test_possible_writes_cannot_prove_initialization():
    class MaybeWrite(expr.Op):
        capability = expr.OpCapability(None)
        dst = expr.ParamDef(kind="input", effect=expr.MemoryEffect.WRITE)
    @access.register_relation(MaybeWrite)
    def relations(call, ctx):
        state = access.state(ctx.type_of(call.args[0]))
        return state, state
    mod = model()
    fn = mod.entry_function()
    destination = mod.functions[0].params[1]
    kernel = core.PrimFunction("maybe", (destination,), core.Sequential((core.Evaluate(MaybeWrite(), (destination,)),)), target=Device())
    first = tir.launch.launch_call(kernel, (fn.params[1],), (1, 1, 1), (1, 1, 1), stream=fn.params[5])
    host = replace(fn, body=core.Sequential((first, *fn.body.body[1:])))
    mod.replace_functions((mod.functions[0], kernel, host))
    report = analysis.check_communication(host, module=mod, uninitialized=("a",))
    assert any(h.kind == "uninitialized_read" for h in report.hazards)
    x, a, _, c, y, compute, *_ = fn.params
    copy = mod.functions[0]
    body = core.Sequential((
        tir.launch.launch_call(copy, (x, a), (1, 1, 1), (1, 1, 1), stream=compute),
        tir.launch.launch_call(copy, (x, c), (1, 1, 1), (1, 1, 1), stream=compute),
        first,
        tir.launch.launch_call(copy, (a, y), (1, 1, 1), (1, 1, 1), stream=compute),
        core.Evaluate(C.StreamSynchronize(), (compute,)),
    ))
    host = replace(fn, body=body)
    assert not analysis.check_communication(host, module=mod).hazards
    aliases = {"a": analysis.BufferRegion("slot"), "c": analysis.BufferRegion("slot")}
    report = analysis.check_communication(host, aliases, module=mod)
    assert any(h.kind == "overwrite_live_value" for h in report.hazards)


def test_rooted_collectives_do_not_initialize_nonrecipient_scratch():
    mod = model()
    fn = mod.entry_function()
    reduction = sharding.Collective(participants=fn.params[-1].type.participants, context="data", kind="reduce", root=0, reduction="sum")
    statements = tuple(replace(stmt, callable=C.CommEnqueue(operation=reduction))
                       if isinstance(stmt.callable, C.CommEnqueue) else stmt for stmt in fn.body.body)
    host = replace(fn, body=core.Sequential(statements))
    report = analysis.check_communication(host, module=mod, uninitialized=("b",))
    assert any(h.kind == "uninitialized_read" for h in report.hazards)


def test_static_replay_tracks_definitions_and_internal_reads():
    kernel = partial_kernel(size=4, count=1)
    source, destination = kernel.params
    index = expr.Var(name="index", type=types.TensorType.from_scalar(expr.DType.i64))
    copied = core.Evaluate(memory.Copy(), (source, view(destination, 1, index)))
    loop = core.For(index, dim.i64_const(0), dim.i64_const(4), dim.i64_const(1), core.Sequential((copied,)))
    read = core.Evaluate(memory.Copy(), (view(destination, 1), source))
    summary = analysis.kernel_effects.summarize(replace(kernel, body=core.Sequential((loop, read))))
    assert not any(effect.parameter == 1 and effect.mode & expr.MemoryEffect.READ for effect in summary.effects)
    written = next(effect for effect in summary.effects if effect.parameter == 1)
    assert written.region.relation.range().is_equal(isl.set("{ [b] : 0 <= b < 16 }"))


def test_dynamic_slice_does_not_become_a_fixed_pointer():
    kernel = partial_kernel(size=8, count=1)
    source, destination = kernel.params
    index = expr.Var(name="index", type=types.TensorType.from_scalar(expr.DType.i64))
    starts = expr.Tuple(elements=(index,), type=types.TupleType(fields=(index.type,)))
    sliced = expr.Call(tensor.Slice(sizes=(1,), strides=(1,)), (destination, starts), type=source.type)
    kernel = replace(kernel, params=(*kernel.params, index),
                     body=core.Sequential((core.Evaluate(memory.Copy(), (source, sliced)),)))
    effect = next(effect for effect in analysis.kernel_effects.summarize(kernel).effects if effect.parameter == 1)
    assert effect.region.may_access
    assert effect.region.relation.range().is_equal(isl.set("{ [b] : 0 <= b < 32 }"))
    reconstructed = replace(kernel, body=core.Sequential((core.Evaluate(memory.Copy(), (source, view(sliced, 1))),)))
    with pytest.raises(ValueError, match="unique known first byte"):
        analysis.kernel_effects.summarize(reconstructed)


def test_region_boundaries_preserve_external_values_and_intermediate_dependencies():
    mod = model()
    report = analysis.check_communication(mod.entry_function(), module=mod)
    producer, transfer, consumer, _ = report.operations
    region = analysis.region_boundary(report, (producer.name, transfer.name))
    assert region.inputs and region.outputs
    assert consumer.name in region.dependents
    with pytest.raises(ValueError, match="must execute between"):
        analysis.region_boundary(report, (producer.name, consumer.name))


def test_region_cannot_hide_an_exported_intermediate_completion():
    mod = model()
    host = mod.entry_function()
    body = (*host.body.body[:2], core.Evaluate(C.ExportEvent(), (host.params[7],)), *host.body.body[2:])
    report = analysis.check_communication(replace(host, body=core.Sequential(body)), module=mod)
    with pytest.raises(ValueError, match="intermediate completion"):
        analysis.region_boundary(report, tuple(operation.name for operation in report.operations[:2]))
